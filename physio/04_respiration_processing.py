"""
04 — Respiration Processing
-----------------------------
Single-subject respiration signal processing for task-based cognitive research.
Designed for BIOPAC MP160 recordings via RSP100C piezoelectric belt amplifier,
sampled at 1000 Hz. Outputs per-trial respiratory features and a processed
signal for RSA computation in 03_heart_rate_processing.py.

Respiration encodes both autonomic state and cognitive load. Key features:

    Rate:       Breathing frequency (breaths/min) — increases with mental load,
                decreases with relaxation and focused attention
    Amplitude:  Tidal volume proxy — deeper breathing = higher amplitude
    Variability: Respiratory Rate Variability (RRV) — analogous to HRV,
                reflects respiratory system flexibility
    Phase:      Inhalation/exhalation ratio — stress shifts toward shorter
                exhalation relative to inhalation
    RSA coupling: Cardiac-respiratory synchrony — computed jointly with
                  03_heart_rate_processing.py

Pipeline steps:
    1.  Load raw respiration signal (.acq or .csv)
    2.  Filter and clean signal (NeuroKit2 rsp_clean)
    3.  Detect breath cycles — peaks (inhalation) and troughs (exhalation)
    4.  Compute instantaneous respiratory rate and amplitude
    5.  Align to task event triggers
    6.  Extract per-trial features over configurable time windows
    7.  Save processed signal for RSA computation (used by script 03)
    8.  Aggregate to participant means per condition

Input:
    {subject_id}_rsp.acq or {subject_id}_rsp.csv
    Optional: {subject_id}_events.csv — trigger onset times

Output:
    {subject_id}_rsp_processed.csv      — cleaned signal + cycle annotations
    {subject_id}_trial_rsp_features.csv — per-trial respiratory features
    {subject_id}_subject_rsp_means.csv  — participant means per condition
    {subject_id}_rsp_report.png         — visual QC plot

Dependencies:
    neurokit2 >= 0.2, numpy, pandas, matplotlib
    Optional: bioread (for .acq): pip install bioread
"""

import os
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
    "sfreq":       1000,    # BIOPAC MP160 RSP100C default (Hz)
    "file_format": "csv",   # "acq" or "csv"
    "rsp_column":  "RSP",   # column name in CSV

    # Respiration cleaning method
    # Options: "khodadad2018" (default), "biosppy", "hampel", "bandpass"
    "clean_method": "khodadad2018",

    # Breath detection method
    # Options: "khodadad2018", "biosppy", "scipy", "neurokit"
    "peak_method": "khodadad2018",

    # Physiologically plausible breath rate range (breaths/min)
    # Typical resting: 12-20. Task: up to ~30. Flag outside this range.
    "rate_min_bpm": 4,
    "rate_max_bpm": 35,

    # Trial epoch windows (seconds relative to event onset)
    "epoch_windows": {
        "pre_stimulus":  (-5.0, 0.0),
        "post_stimulus": (0.0,  5.0),
    },

    # Features to compute
    "compute_rate":        True,
    "compute_amplitude":   True,
    "compute_variability": True,
    "compute_phase":       True,   # inhalation/exhalation ratio

    # Events file
    "events_file": "{subject_id}_events.csv",
}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_rsp(subject_id, config):
    """
    Load raw respiration signal from BIOPAC MP160 / RSP100C.

    The RSP100C piezoelectric belt measures chest or abdominal wall
    displacement. Signal amplitude reflects relative tidal volume —
    absolute calibration to liters requires a spirometer reference,
    which is not assumed here. Features use normalized amplitude.

    Returns a 1D numpy array and sampling rate.
    """
    input_dir = config["input_dir"]

    if config["file_format"] == "acq":
        try:
            import bioread
            acq_path = os.path.join(input_dir, f"{subject_id}_rsp.acq")
            data     = bioread.read_file(acq_path)
            rsp_ch   = next(ch for ch in data.channels
                            if "RSP" in ch.name.upper() or "RESP" in ch.name.upper()
                            or "BREATH" in ch.name.upper())
            rsp    = np.array(rsp_ch.data)
            sfreq  = rsp_ch.samples_per_second
            print(f"  Loaded .acq: {len(rsp)/sfreq:.1f}s @ {sfreq} Hz")
        except ImportError:
            raise ImportError("bioread not installed. Run: pip install bioread")
    else:
        csv_path = os.path.join(input_dir, f"{subject_id}_rsp.csv")
        df       = pd.read_csv(csv_path)
        rsp      = df[config["rsp_column"]].values
        sfreq    = config["sfreq"]
        print(f"  Loaded .csv: {len(rsp)/sfreq:.1f}s @ {sfreq} Hz")

    return rsp, sfreq


def load_events(subject_id, config):
    """Load task event trigger times. See script 03 for format details."""
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
    print(f"  Loaded {len(events)} events.")
    return events


# ── Respiration processing ─────────────────────────────────────────────────────

def process_rsp(rsp_signal, sfreq, clean_method, peak_method):
    """
    Clean and analyze respiration signal via NeuroKit2.

    Steps:
        1. Clean: bandpass filter to remove noise outside breathing range
        2. Find peaks (inhalation peaks) and troughs (exhalation troughs)
        3. Compute instantaneous rate and amplitude from cycle timing

    The processed signals DataFrame includes:
        RSP_Clean:      filtered signal
        RSP_Rate:       instantaneous breathing rate (breaths/min)
        RSP_Amplitude:  peak-to-trough amplitude per cycle
        RSP_Phase:      inhalation (1) or exhalation (0) at each sample

    Returns signals DataFrame and processing info dict.
    """
    signals, info = nk.rsp_process(
        rsp_signal,
        sampling_rate = sfreq,
        method        = clean_method,
    )

    n_cycles  = len(info["RSP_Peaks"])
    duration  = len(rsp_signal) / sfreq
    mean_rate = signals["RSP_Rate"].mean()

    print(f"  Respiration processed: {n_cycles} breath cycles over {duration:.1f}s")
    print(f"  Mean respiratory rate: {mean_rate:.1f} breaths/min")

    # Flag if mean rate is outside plausible range
    if not (config["rate_min_bpm"] <= mean_rate <= config["rate_max_bpm"]):
        print(f"  WARNING: Mean rate {mean_rate:.1f} bpm outside expected range "
              f"({config['rate_min_bpm']}–{config['rate_max_bpm']}) — check signal.")

    return signals, info


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_window_rsp_features(signals, peaks_df, window_name,
                                 win_start, win_end, sfreq, config):
    """
    Extract respiratory features for a single time window.

    Features computed per window:

    Rate features:
        mean_rate:  mean breathing rate (breaths/min)
        rate_sd:    standard deviation of breathing rate — rate variability

    Amplitude features:
        mean_amplitude: mean peak-to-trough amplitude (normalized units)
        amplitude_sd:   variability of breath amplitude

    Variability (RRV — Respiratory Rate Variability):
        rmssd_breath: root mean square of successive breath-interval differences
                      analogous to cardiac RMSSD — reflects short-term variability
        sdbb:         standard deviation of breath-to-breath intervals

    Phase features:
        ie_ratio: inhalation duration / exhalation duration
                  ratio > 1 indicates relatively longer inhalation
                  ratio shifts toward shorter exhalation under stress/load
    """
    features = {}

    # Mask signals to this window
    t      = np.arange(len(signals)) / sfreq
    t_mask = (t >= win_start) & (t < win_end)
    win_signals = signals[t_mask]

    if len(win_signals) == 0:
        return {f"{window_name}_n_breaths": 0}

    # Rate
    if config["compute_rate"]:
        rate_vals = win_signals["RSP_Rate"].dropna().values
        features[f"{window_name}_mean_rate"]  = np.mean(rate_vals) if len(rate_vals) > 0 else np.nan
        features[f"{window_name}_rate_sd"]    = np.std(rate_vals)  if len(rate_vals) > 0 else np.nan

    # Amplitude
    if config["compute_amplitude"] and "RSP_Amplitude" in win_signals.columns:
        amp_vals = win_signals["RSP_Amplitude"].dropna().values
        features[f"{window_name}_mean_amplitude"] = np.mean(amp_vals) if len(amp_vals) > 0 else np.nan
        features[f"{window_name}_amplitude_sd"]   = np.std(amp_vals)  if len(amp_vals) > 0 else np.nan

    # Breath-to-breath intervals and RRV
    if config["compute_variability"] and peaks_df is not None:
        # Select peaks within this window
        win_peaks = peaks_df[
            (peaks_df["peak_time_s"] >= win_start) &
            (peaks_df["peak_time_s"] <  win_end)
        ]
        features[f"{window_name}_n_breaths"] = len(win_peaks)

        if len(win_peaks) >= 3:
            bbi = np.diff(win_peaks["peak_time_s"].values) * 1000  # ms
            features[f"{window_name}_rmssd_breath"] = np.sqrt(np.mean(np.diff(bbi)**2))
            features[f"{window_name}_sdbb"]         = np.std(bbi)
        else:
            features[f"{window_name}_rmssd_breath"] = np.nan
            features[f"{window_name}_sdbb"]         = np.nan

    # Inhalation/exhalation phase ratio
    if config["compute_phase"] and "RSP_Phase" in win_signals.columns:
        phase    = win_signals["RSP_Phase"].values
        n_inhale = np.sum(phase == 1)
        n_exhale = np.sum(phase == 0)
        features[f"{window_name}_ie_ratio"] = (n_inhale / n_exhale
                                                if n_exhale > 0 else np.nan)

    return features


def extract_trial_features(signals, peaks_df, events, config, sfreq):
    """
    Extract respiratory features for each trial over each time window.
    Returns DataFrame with one row per trial.
    """
    if events is None:
        print("  No events file — computing features over full recording.")
        features = extract_window_rsp_features(
            signals, peaks_df, "full_recording",
            0, len(signals) / sfreq, sfreq, config
        )
        return pd.DataFrame([features])

    records = []

    for trial_idx, event in events.iterrows():
        onset     = event["onset_sec"]
        condition = event.get("condition", "unknown")
        row       = {"trial": trial_idx, "condition": condition, "onset_sec": onset}

        for window_name, (tmin, tmax) in config["epoch_windows"].items():
            win_feats = extract_window_rsp_features(
                signals, peaks_df, window_name,
                onset + tmin, onset + tmax, sfreq, config
            )
            row.update(win_feats)

        records.append(row)

    return pd.DataFrame(records)


def build_peaks_dataframe(signals, info, sfreq):
    """
    Build DataFrame of breath cycle timing for RRV computation.
    """
    peak_samples = info["RSP_Peaks"]
    return pd.DataFrame({
        "peak_sample": peak_samples,
        "peak_time_s": peak_samples / sfreq,
    })


# ── Participant-level aggregation ──────────────────────────────────────────────

def aggregate_to_subject_means(trial_features):
    """
    Average per-trial features per condition.
    Participant means are the correct statistical unit — see script 03
    for detailed rationale.
    """
    feature_cols  = [c for c in trial_features.columns
                     if c not in ("trial", "condition", "onset_sec")]
    subject_means = trial_features.groupby("condition")[feature_cols].mean()
    subject_sem   = trial_features.groupby("condition")[feature_cols].sem()

    subject_means.columns = [f"{c}_mean" for c in subject_means.columns]
    subject_sem.columns   = [f"{c}_sem"  for c in subject_sem.columns]

    return pd.concat([subject_means, subject_sem], axis=1).reset_index()


# ── QC visualization ───────────────────────────────────────────────────────────

def plot_rsp_summary(rsp_signal, signals, peaks_df, subject_id, output_dir, sfreq):
    """
    3-panel QC figure:
        Panel 1: Raw vs cleaned signal with breath peaks (first 30s)
        Panel 2: Instantaneous respiratory rate over full recording
        Panel 3: Breath amplitude over full recording
    """
    t      = np.arange(len(rsp_signal)) / sfreq
    t_mask = t <= 30

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Respiration Summary — {subject_id}", fontsize=13)

    # Panel 1: Raw + cleaned + peaks (first 30s)
    axes[0].plot(t[t_mask], rsp_signal[t_mask], color="lightgray",
                 linewidth=0.8, label="Raw", alpha=0.8)
    axes[0].plot(t[t_mask], signals["RSP_Clean"].values[t_mask],
                 color="steelblue", linewidth=1.0, label="Cleaned")
    pk_mask = peaks_df["peak_time_s"] <= 30
    axes[0].scatter(peaks_df["peak_time_s"][pk_mask],
                    signals["RSP_Clean"].values[
                        peaks_df["peak_sample"][pk_mask].astype(int)
                    ],
                    color="firebrick", s=25, zorder=5, label="Inhalation peaks")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("Respiration signal — first 30 seconds")
    axes[0].legend(fontsize=9)

    # Panel 2: Respiratory rate
    axes[1].plot(t, signals["RSP_Rate"].values, color="darkorange", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Rate (breaths/min)")
    axes[1].set_title("Instantaneous respiratory rate")

    # Panel 3: Amplitude
    if "RSP_Amplitude" in signals.columns:
        axes[2].plot(t, signals["RSP_Amplitude"].values, color="teal", linewidth=0.8)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Amplitude (a.u.)")
        axes[2].set_title("Breath amplitude")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{subject_id}_rsp_report.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_subject(subject_id, config):
    """
    Full respiration processing pipeline for a single subject.
    """
    print(f"\n{'='*60}")
    print(f"Respiration Processing: {subject_id}")
    print(f"{'='*60}")

    os.makedirs(config["output_dir"], exist_ok=True)
    base  = os.path.join(config["output_dir"], subject_id)
    sfreq = config["sfreq"]

    # 1. Load
    rsp, sfreq = load_rsp(subject_id, config)
    events     = load_events(subject_id, config)

    # 2. Process
    signals, info = process_rsp(rsp, sfreq,
                                 config["clean_method"],
                                 config["peak_method"])

    # 3. Build peaks DataFrame
    peaks_df = build_peaks_dataframe(signals, info, sfreq)

    # 4. Save processed signal (used by script 03 for RSA)
    signals.to_csv(f"{base}_rsp_processed.csv", index=False)
    print(f"  Saved: {base}_rsp_processed.csv")

    # 5. Extract trial features
    trial_features = extract_trial_features(signals, peaks_df, events, config, sfreq)
    trial_features.to_csv(f"{base}_trial_rsp_features.csv", index=False)
    print(f"  Saved: {base}_trial_rsp_features.csv")

    # 6. Participant-level aggregation
    subject_means = aggregate_to_subject_means(trial_features)
    subject_means.to_csv(f"{base}_subject_rsp_means.csv", index=False)
    print(f"  Saved: {base}_subject_rsp_means.csv")

    # 7. QC plot
    plot_rsp_summary(rsp, signals, peaks_df, subject_id, config["output_dir"], sfreq)

    print(f"\nDone: {subject_id}")
    return trial_features, subject_means


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subject_id = "sub-01"
    process_subject(subject_id, CONFIG)
