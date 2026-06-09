"""
03 — Heart Rate & HRV Processing
----------------------------------
Single-subject ECG processing for task-based cognitive research.
Designed for BIOPAC MP160 recordings via ECG100C amplifier module,
sampled at 1000 Hz. Outputs per-trial heart rate and HRV features
for downstream ML and group-level statistical analyses.

Pipeline steps:
    1.  Load raw ECG signal (.acq via bioread, or .csv/.txt)
    2.  Bandpass filter ECG (0.5–40 Hz)
    3.  R-peak detection (Pan-Tompkins algorithm via NeuroKit2)
    4.  Signal quality check — flag low-quality segments
    5.  Compute RR intervals and instantaneous heart rate
    6.  Align physiological data to task event triggers
    7.  Extract per-trial features over configurable time windows:
            Time domain HRV:      RMSSD, SDNN, MeanNN, pNN50
            Frequency domain HRV: HF power, LF power, LF/HF ratio
            Nonlinear HRV:        SD1, SD2 (Poincaré plot measures)
            Heart rate:           mean HR, HR range, HR trend
    8.  Compute Respiratory Sinus Arrhythmia (RSA) if respiration
        signal is provided (links to 04_respiration_processing.py)
    9.  Aggregate to participant means per condition

Design notes:
    - Per-trial HRV features require sufficient RR intervals per window.
      Short epochs (<15s) will yield unreliable frequency-domain HRV —
      time-domain measures (RMSSD, MeanNN) are more appropriate for
      short windows and are flagged accordingly in output.
    - Participant-level means are the correct statistical unit.
      Never compute group statistics from pooled trials.
    - Pre-stimulus HRV reflects autonomic baseline state and is a
      meaningful predictor of cognitive performance independently of
      the task response.

Input:
    {subject_id}_ecg.acq or {subject_id}_ecg.csv
    Optional: {subject_id}_events.csv — trigger onset times (seconds)
    Optional: {subject_id}_rsp_processed.csv — from 04_respiration_processing.py

Output:
    {subject_id}_rpeaks.csv             — R-peak locations and RR intervals
    {subject_id}_trial_hr_features.csv  — per-trial HRV and HR features
    {subject_id}_subject_hr_means.csv   — participant means per condition
    {subject_id}_ecg_report.html        — visual QC plots

Dependencies:
    neurokit2 >= 0.2, numpy, pandas, matplotlib, scipy
    Optional: bioread (for .acq files): pip install bioread

Install:
    pip install neurokit2 bioread
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "input_dir":  "/path/to/raw/physio",
    "output_dir": "/path/to/features/physio",

    # Recording parameters
    "sfreq":        1000,     # BIOPAC MP160 ECG100C default (Hz)
    "file_format":  "csv",    # "acq" for AcqKnowledge native, "csv" for exported text

    # For CSV/text imports — which column contains the ECG signal
    "ecg_column":   "ECG",

    # Filtering
    "ecg_filter_low":  0.5,   # Hz — removes baseline wander
    "ecg_filter_high": 40.0,  # Hz — removes high-frequency noise and EMG

    # R-peak detection method
    # Options: "pantompkins1985" (default, robust), "hamilton2002", "neurokit"
    "rpeak_method": "pantompkins1985",

    # Signal quality threshold — epochs below this are flagged
    # NeuroKit2 quality index: 0 (bad) to 1 (perfect)
    "quality_threshold": 0.5,

    # Trial epoch windows (seconds relative to event onset)
    # Short windows (<15s): use time-domain HRV only (RMSSD, MeanNN)
    # Longer windows (>30s): frequency-domain HRV becomes reliable
    "epoch_windows": {
        "pre_stimulus":  (-5.0, 0.0),   # 5s pre-stimulus autonomic baseline
        "post_stimulus": (0.0,  5.0),   # 5s post-stimulus response window
    },

    # HRV feature sets to compute per trial window
    # Frequency-domain HRV requires long epochs — set False for short trials
    "compute_time_domain_hrv":      True,
    "compute_frequency_domain_hrv": False,  # requires ~30s+ windows
    "compute_nonlinear_hrv":        True,
    "compute_rsa":                  True,   # requires respiration signal

    # Event trigger file — CSV with columns: onset_sec, condition
    # Set to None to process full recording without epoching
    "events_file": "{subject_id}_events.csv",

    # Respiration file from 04_respiration_processing.py (for RSA)
    "rsp_file": "{subject_id}_rsp_processed.csv",
}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_ecg(subject_id, config):
    """
    Load raw ECG signal from BIOPAC MP160 output.

    Supports two formats:
        .acq  — AcqKnowledge native binary format (requires bioread)
        .csv  — Text/CSV export from AcqKnowledge or similar

    Returns a 1D numpy array of the ECG signal and the sampling rate.
    """
    input_dir = config["input_dir"]

    if config["file_format"] == "acq":
        try:
            import bioread
            acq_path = os.path.join(input_dir, f"{subject_id}_ecg.acq")
            data     = bioread.read_file(acq_path)
            # Find ECG channel by name
            ecg_ch   = next(ch for ch in data.channels
                            if "ECG" in ch.name.upper() or "EKG" in ch.name.upper())
            ecg      = np.array(ecg_ch.data)
            sfreq    = ecg_ch.samples_per_second
            print(f"  Loaded .acq: {acq_path} | {len(ecg)/sfreq:.1f}s @ {sfreq} Hz")
        except ImportError:
            raise ImportError("bioread not installed. Run: pip install bioread")

    else:
        csv_path = os.path.join(input_dir, f"{subject_id}_ecg.csv")
        df       = pd.read_csv(csv_path)
        ecg      = df[config["ecg_column"]].values
        sfreq    = config["sfreq"]
        print(f"  Loaded .csv: {csv_path} | {len(ecg)/sfreq:.1f}s @ {sfreq} Hz")

    return ecg, sfreq


def load_events(subject_id, config):
    """
    Load task event trigger times.

    Expected CSV format:
        onset_sec  condition
        10.234     condition_A
        15.891     condition_B
        ...

    Returns a DataFrame or None if no events file configured.
    """
    if config["events_file"] is None:
        return None

    events_path = os.path.join(
        config["input_dir"],
        config["events_file"].replace("{subject_id}", subject_id)
    )

    if not os.path.exists(events_path):
        print(f"  Warning: events file not found: {events_path}")
        return None

    events = pd.read_csv(events_path)
    print(f"  Loaded {len(events)} events from {events_path}")
    return events


def load_respiration(subject_id, config):
    """
    Load processed respiration signal for RSA computation.
    Output of 04_respiration_processing.py.
    Returns DataFrame or None.
    """
    if not config["compute_rsa"]:
        return None

    rsp_path = os.path.join(
        config["input_dir"],
        config["rsp_file"].replace("{subject_id}", subject_id)
    )

    if not os.path.exists(rsp_path):
        print(f"  RSA: respiration file not found — RSA will be skipped.")
        return None

    rsp_df = pd.read_csv(rsp_path)
    print(f"  Loaded respiration signal for RSA.")
    return rsp_df


# ── ECG processing ─────────────────────────────────────────────────────────────

def process_ecg(ecg_signal, sfreq, rpeak_method, filter_low, filter_high):
    """
    Full ECG processing pipeline via NeuroKit2:
        1. Bandpass filter
        2. R-peak detection
        3. Signal quality index computation
        4. RR interval and instantaneous heart rate extraction

    Returns:
        signals:  DataFrame with ECG_Clean, ECG_Rate, ECG_Quality columns
        rpeaks:   dict with R-peak sample indices
        info:     processing metadata
    """
    signals, info = nk.ecg_process(
        ecg_signal,
        sampling_rate = sfreq,
        method        = rpeak_method,
    )

    n_beats = len(info["ECG_R_Peaks"])
    duration = len(ecg_signal) / sfreq
    mean_hr  = signals["ECG_Rate"].mean()
    mean_quality = signals["ECG_Quality"].mean()

    print(f"  ECG processed: {n_beats} R-peaks detected over {duration:.1f}s")
    print(f"  Mean HR: {mean_hr:.1f} bpm | Mean quality index: {mean_quality:.3f}")

    if mean_quality < 0.5:
        print(f"  WARNING: Low signal quality ({mean_quality:.3f}) — "
              f"review raw ECG before using features.")

    return signals, info


def compute_rpeaks_dataframe(signals, info, sfreq):
    """
    Build a clean DataFrame of R-peak locations and RR intervals.
    RR interval = time between consecutive R-peaks (milliseconds).
    This is the fundamental unit for all HRV calculations.
    """
    r_peaks  = info["ECG_R_Peaks"]
    rr_ms    = np.diff(r_peaks) / sfreq * 1000   # convert samples → ms

    rpeaks_df = pd.DataFrame({
        "rpeak_sample":  r_peaks[1:],              # sample index
        "rpeak_time_s":  r_peaks[1:] / sfreq,      # time in seconds
        "rr_interval_ms": rr_ms,
        "heart_rate_bpm": 60000 / rr_ms,
    })

    # Flag ectopic beats — RR intervals outside physiologically plausible range
    # (300ms = 200 bpm max, 2000ms = 30 bpm min)
    rpeaks_df["ectopic_flag"] = (
        (rpeaks_df["rr_interval_ms"] < 300) |
        (rpeaks_df["rr_interval_ms"] > 2000)
    )
    n_ectopic = rpeaks_df["ectopic_flag"].sum()
    if n_ectopic > 0:
        print(f"  Flagged {n_ectopic} potential ectopic beats — excluded from HRV.")

    return rpeaks_df


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_hrv_features(rr_intervals_ms, window_name, sfreq,
                          compute_time=True, compute_freq=False,
                          compute_nonlinear=True):
    """
    Compute HRV indices from RR intervals within a single epoch window.

    Time-domain indices (appropriate for short and long windows):
        MeanNN:  mean RR interval (ms) — inversely related to mean HR
        SDNN:    standard deviation of RR intervals — overall HRV
        RMSSD:   root mean square of successive differences — vagal tone,
                 most reliable for short windows, primary HRV metric
                 for cognitive research
        pNN50:   proportion of successive RR differences > 50ms — vagal index

    Frequency-domain indices (require ~30s+ windows, Lomb-Scargle method):
        HF power (0.15–0.4 Hz):  parasympathetic (vagal) activity
        LF power (0.04–0.15 Hz): mixed sympathetic + parasympathetic
        LF/HF ratio:             sympathovagal balance (interpret cautiously)

    Nonlinear indices (Poincaré plot):
        SD1: short-term HRV — beat-to-beat variability (correlates with RMSSD)
        SD2: long-term HRV — overall variability

    Note on frequency-domain HRV in short epochs:
        Standard guidelines (Task Force 1996, Shaffer & Ginsberg 2017) recommend
        minimum 1–2 minute windows for frequency-domain HRV. In task paradigms
        with short trials, RMSSD is the recommended primary HRV metric.
    """
    # Remove flagged ectopic beats before computing HRV
    clean_rr = rr_intervals_ms[
        (rr_intervals_ms >= 300) & (rr_intervals_ms <= 2000)
    ]

    if len(clean_rr) < 4:
        print(f"    {window_name}: insufficient RR intervals ({len(clean_rr)}) — "
              f"returning NaN for this window.")
        return {f"{window_name}_n_beats": len(clean_rr)}

    features = {f"{window_name}_n_beats": len(clean_rr)}

    if compute_time:
        hrv_time = nk.hrv_time(
            pd.DataFrame({"RRI": clean_rr}),
            sampling_rate = sfreq,
            show = False,
        )
        for col in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50"]:
            if col in hrv_time.columns:
                features[f"{window_name}_{col}"] = hrv_time[col].values[0]

    if compute_freq and len(clean_rr) >= 30:
        try:
            hrv_freq = nk.hrv_frequency(
                pd.DataFrame({"RRI": clean_rr}),
                sampling_rate = sfreq,
                show = False,
            )
            for col in ["HRV_HF", "HRV_LF", "HRV_LFHF"]:
                if col in hrv_freq.columns:
                    features[f"{window_name}_{col}"] = hrv_freq[col].values[0]
        except Exception as e:
            print(f"    {window_name}: frequency HRV failed ({e}) — skipping.")

    if compute_nonlinear:
        hrv_nl = nk.hrv_nonlinear(
            pd.DataFrame({"RRI": clean_rr}),
            sampling_rate = sfreq,
            show = False,
        )
        for col in ["HRV_SD1", "HRV_SD2"]:
            if col in hrv_nl.columns:
                features[f"{window_name}_{col}"] = hrv_nl[col].values[0]

    return features


def compute_rsa(signals, rsp_df, trial_mask, window_name, sfreq):
    """
    Respiratory Sinus Arrhythmia (RSA) — the natural coupling between
    heart rate and respiration. HR increases during inhalation and
    decreases during exhalation. RSA amplitude reflects parasympathetic
    tone and is a meaningful index of autonomic flexibility.

    Computed using the Peak-to-Trough (P2T) method:
        RSA P2T = mean HR during inhalation minus mean HR during exhalation

    Requires synchronized ECG and respiration signals.
    Returns RSA amplitude for the specified window.
    """
    try:
        rsa = nk.hrv_rsa(
            signals[trial_mask],
            rsp_df[trial_mask] if rsp_df is not None else None,
            sampling_rate = sfreq,
        )
        return {f"{window_name}_RSA_P2T": rsa.get("RSA_P2T", np.nan)}
    except Exception as e:
        print(f"    RSA computation failed ({e}) — skipping.")
        return {f"{window_name}_RSA_P2T": np.nan}


# ── Epoching and trial features ────────────────────────────────────────────────

def extract_trial_features(rpeaks_df, signals, events, rsp_df, config, sfreq):
    """
    Extract HR and HRV features for each trial, over each configured
    time window relative to event onset.

    For each trial × window combination:
        - Selects RR intervals falling within the window
        - Computes all configured HRV indices
        - Adds mean HR and HR range for the window

    Returns DataFrame with one row per trial.
    """
    if events is None:
        print("  No events file — computing HRV over full recording.")
        rr = rpeaks_df[~rpeaks_df["ectopic_flag"]]["rr_interval_ms"].values
        features = extract_hrv_features(
            rr, "full_recording", sfreq,
            compute_time      = config["compute_time_domain_hrv"],
            compute_freq      = config["compute_frequency_domain_hrv"],
            compute_nonlinear = config["compute_nonlinear_hrv"],
        )
        return pd.DataFrame([features])

    records = []

    for trial_idx, event in events.iterrows():
        onset     = event["onset_sec"]
        condition = event.get("condition", "unknown")
        row       = {"trial": trial_idx, "condition": condition, "onset_sec": onset}

        for window_name, (tmin, tmax) in config["epoch_windows"].items():
            win_start = onset + tmin
            win_end   = onset + tmax

            # Select RR intervals within this window
            mask = (
                (rpeaks_df["rpeak_time_s"] >= win_start) &
                (rpeaks_df["rpeak_time_s"] <  win_end) &
                (~rpeaks_df["ectopic_flag"])
            )
            window_rr = rpeaks_df[mask]["rr_interval_ms"].values

            # Mean HR and range for this window
            if len(window_rr) > 0:
                row[f"{window_name}_mean_hr"] = np.mean(60000 / window_rr)
                row[f"{window_name}_hr_range"] = np.ptp(60000 / window_rr)
            else:
                row[f"{window_name}_mean_hr"] = np.nan
                row[f"{window_name}_hr_range"] = np.nan

            # HRV features
            hrv_feats = extract_hrv_features(
                window_rr, window_name, sfreq,
                compute_time      = config["compute_time_domain_hrv"],
                compute_freq      = config["compute_frequency_domain_hrv"],
                compute_nonlinear = config["compute_nonlinear_hrv"],
            )
            row.update(hrv_feats)

            # RSA
            if config["compute_rsa"] and rsp_df is not None:
                sig_mask = (
                    (signals.index / sfreq >= win_start) &
                    (signals.index / sfreq <  win_end)
                )
                rsa_feats = compute_rsa(signals, rsp_df, sig_mask,
                                        window_name, sfreq)
                row.update(rsa_feats)

        records.append(row)

    return pd.DataFrame(records)


# ── Participant-level aggregation ──────────────────────────────────────────────

def aggregate_to_subject_means(trial_features):
    """
    Average per-trial HR/HRV features per condition to yield one value
    per participant per condition.

    This is the correct statistical unit for group analyses.
    Group-level t-tests, ANOVAs, and ML should always operate on
    participant means — pooling trials inflates degrees of freedom
    and violates the independence assumption.
    """
    feature_cols  = [c for c in trial_features.columns
                     if c not in ("trial", "condition", "onset_sec")]
    subject_means = trial_features.groupby("condition")[feature_cols].mean()
    subject_sem   = trial_features.groupby("condition")[feature_cols].sem()

    subject_means.columns = [f"{c}_mean" for c in subject_means.columns]
    subject_sem.columns   = [f"{c}_sem"  for c in subject_sem.columns]

    return pd.concat([subject_means, subject_sem], axis=1).reset_index()


# ── QC visualization ───────────────────────────────────────────────────────────

def plot_ecg_summary(ecg_signal, signals, rpeaks_df, subject_id, output_dir, sfreq):
    """
    Save a 3-panel QC figure:
        Panel 1: Raw ECG with R-peaks marked (first 10 seconds)
        Panel 2: Instantaneous heart rate over full recording
        Panel 3: RR interval distribution (tachogram)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"ECG Summary — {subject_id}", fontsize=13)

    # Panel 1: Raw ECG + R-peaks (first 10s)
    t          = np.arange(len(ecg_signal)) / sfreq
    t_mask     = t <= 10
    rpeak_samp = rpeaks_df["rpeak_sample"].values
    rpeak_mask = rpeak_samp < int(10 * sfreq)

    axes[0].plot(t[t_mask], ecg_signal[t_mask], color="steelblue",
                 linewidth=0.8, label="ECG")
    axes[0].scatter(rpeak_samp[rpeak_mask] / sfreq,
                    ecg_signal[rpeak_samp[rpeak_mask]],
                    color="firebrick", s=30, zorder=5, label="R-peaks")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Raw ECG — first 10 seconds")
    axes[0].legend(fontsize=9)

    # Panel 2: Instantaneous HR
    axes[1].plot(t, signals["ECG_Rate"].values, color="darkorange", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Heart rate (bpm)")
    axes[1].set_title("Instantaneous heart rate")

    # Panel 3: RR tachogram
    rr_times = rpeaks_df["rpeak_time_s"].values
    rr_vals  = rpeaks_df["rr_interval_ms"].values
    axes[2].plot(rr_times, rr_vals, color="teal", linewidth=0.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("RR interval (ms)")
    axes[2].set_title("RR interval tachogram")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{subject_id}_ecg_report.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_subject(subject_id, config):
    """
    Full HR/HRV processing pipeline for a single subject.
    """
    print(f"\n{'='*60}")
    print(f"HR/HRV Processing: {subject_id}")
    print(f"{'='*60}")

    os.makedirs(config["output_dir"], exist_ok=True)
    base  = os.path.join(config["output_dir"], subject_id)
    sfreq = config["sfreq"]

    # 1. Load data
    ecg, sfreq = load_ecg(subject_id, config)
    events     = load_events(subject_id, config)
    rsp_df     = load_respiration(subject_id, config)

    # 2. Process ECG
    signals, info = process_ecg(
        ecg, sfreq,
        config["rpeak_method"],
        config["ecg_filter_low"],
        config["ecg_filter_high"],
    )

    # 3. Build R-peaks DataFrame
    rpeaks_df = compute_rpeaks_dataframe(signals, info, sfreq)
    rpeaks_df.to_csv(f"{base}_rpeaks.csv", index=False)
    print(f"  Saved: {base}_rpeaks.csv")

    # 4. Extract trial features
    trial_features = extract_trial_features(
        rpeaks_df, signals, events, rsp_df, config, sfreq
    )
    trial_features.to_csv(f"{base}_trial_hr_features.csv", index=False)
    print(f"  Saved: {base}_trial_hr_features.csv")

    # 5. Participant-level aggregation
    subject_means = aggregate_to_subject_means(trial_features)
    subject_means.to_csv(f"{base}_subject_hr_means.csv", index=False)
    print(f"  Saved: {base}_subject_hr_means.csv")

    # 6. QC plot
    plot_ecg_summary(ecg, signals, rpeaks_df, subject_id,
                     config["output_dir"], sfreq)

    print(f"\nDone: {subject_id}")
    return trial_features, subject_means


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subject_id = "sub-01"
    process_subject(subject_id, CONFIG)
