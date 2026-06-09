"""
02 — EEG Feature Extraction
-----------------------------
Extracts spectral, temporal, and connectivity features from clean epoched
EEG data for a single participant. Output is a per-trial feature DataFrame
that is later aggregated across participants in 07_aggregate_features.py.

Feature sets (all configurable via CONFIG):

    A. Band power
       Per-channel power in canonical frequency bands (delta through gamma),
       computed over configurable time windows. Two windows are supported:
           - Pre-stimulus:  baseline cognitive state before event onset
           - Post-stimulus: neural response to the event
       This separation is scientifically motivated — pre-stimulus oscillatory
       state predicts cognitive performance independently of the evoked response.

    B. ERP features (optional)
       Peak amplitude and latency within defined component windows.
       Included for completeness and statistical figure generation,
       but note: ERP features should be interpreted cautiously in ML
       contexts due to sensitivity to trial count imbalance across participants.

    C. Connectivity features
       - Inter-Trial Phase Coherence (ITPC): phase consistency across trials
         per channel per frequency — captures stimulus-locked oscillatory responses
         without amplitude confounds.
       - Phase Locking Value (PLV): pairwise phase synchrony between channel pairs
         per trial — captures network-level coordination.
       - Spectral coherence: magnitude-squared coherence between channel pairs.

    D. Participant-level aggregation
       Trial-level features are averaged per condition to yield one value per
       participant per condition. This is the correct statistical unit for
       between-subjects and within-subjects inferential analyses — SEM and
       group statistics should always be computed from participant means,
       never from pooled trials.

Input:
    {subject_id}_clean_epo.fif  — output of 01_eeg_preprocessing.py

Output:
    {subject_id}_trial_features.csv     — one row per trial, all features
    {subject_id}_subject_means.csv      — one row per condition (participant mean)
    {subject_id}_itpc.npy               — ITPC array (freqs × channels × conditions)
    {subject_id}_feature_report.html    — visual QC report

Dependencies:
    mne >= 1.6, numpy, pandas, scipy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import coherence as scipy_coherence


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "input_dir":  "/path/to/processed/eeg",
    "output_dir": "/path/to/features/eeg",

    # Frequency bands (Hz) — standard cognitive neuroscience definitions
    "freq_bands": {
        "delta": (1,   4),
        "theta": (4,   8),
        "alpha": (8,  13),
        "beta":  (13, 30),
        "gamma": (30, 80),
    },

    # Time windows for band power extraction (seconds, relative to event onset)
    # Pre-stimulus window captures baseline cognitive state
    # Post-stimulus window captures the evoked/induced response
    # Set either to None to skip that window
    "power_windows": {
        "pre_stimulus":  (-0.2, 0.0),   # 200 ms before stimulus
        "post_stimulus": (0.0,  0.8),   # 800 ms after stimulus
    },

    # ERP features — component windows (seconds)
    # Define time windows for components of interest
    # These are placeholders — adjust to your paradigm and expected latencies
    "erp_windows": {
        "N100": (0.08, 0.15),
        "P200": (0.15, 0.25),
        "N200": (0.18, 0.28),
        "P300": (0.25, 0.50),
    },
    # Channels to extract ERP features from (or "all" for grand mean)
    "erp_channels": ["Fz", "Cz", "Pz"],

    # Connectivity settings
    "plv_channel_pairs": None,   # None = all pairs (slow); or list of ("Ch1","Ch2") tuples
    "coherence_channel_pairs": None,
    "itpc_freqs": np.arange(4, 50, 1),   # frequencies for ITPC (Hz)
    "itpc_n_cycles": 7,                  # Morlet wavelet cycles — higher = better freq res

    # Feature flags — set False to skip sections you don't need
    "compute_band_power":  True,
    "compute_erp":         True,
    "compute_itpc":        True,
    "compute_plv":         True,
    "compute_coherence":   False,   # computationally expensive for 64 channels

    # Sampling rate (should match output of preprocessing script)
    "sfreq": 256,
}


# ── A. Band power ──────────────────────────────────────────────────────────────

def compute_band_power(epochs, freq_bands, time_windows, sfreq):
    """
    Compute mean band power per channel per frequency band per trial,
    for each configured time window.

    Uses Welch's method (via MNE's psd_array_welch) on each epoch segment.
    Power is returned in dB (10 * log10) to approximate normality for
    downstream statistical analyses.

    Returns a DataFrame with one row per trial. Column naming convention:
        {window}_{band}_{channel}
    e.g.: pre_stimulus_alpha_Cz, post_stimulus_theta_Fz
    """
    data       = epochs.get_data()          # (n_trials, n_channels, n_times)
    times      = epochs.times
    ch_names   = epochs.ch_names
    n_trials   = data.shape[0]
    records    = []

    for trial_idx in range(n_trials):
        row = {"trial": trial_idx}

        for window_name, (tmin, tmax) in time_windows.items():
            if tmin is None or tmax is None:
                continue

            # Extract time window from this trial
            t_mask    = (times >= tmin) & (times <= tmax)
            segment   = data[trial_idx][:, t_mask]  # (n_channels, n_window_samples)

            # Welch PSD for this segment
            psds, freqs = mne.time_frequency.psd_array_welch(
                segment,
                sfreq     = sfreq,
                fmin      = 1.0,
                fmax      = 80.0,
                n_fft     = min(segment.shape[-1], int(sfreq * 0.5)),
                verbose   = False,
            )
            # psds shape: (n_channels, n_freqs)

            for band_name, (fmin, fmax) in freq_bands.items():
                freq_mask  = (freqs >= fmin) & (freqs <= fmax)
                band_power = psds[:, freq_mask].mean(axis=1)  # mean across freqs
                band_power_db = 10 * np.log10(band_power + 1e-30)  # dB, avoid log(0)

                for ch_idx, ch_name in enumerate(ch_names):
                    col = f"{window_name}_{band_name}_{ch_name}"
                    row[col] = band_power_db[ch_idx]

        records.append(row)

    return pd.DataFrame(records)


# ── B. ERP features ────────────────────────────────────────────────────────────

def compute_erp_features(epochs, erp_windows, erp_channels):
    """
    Extract peak amplitude and latency for defined ERP component windows.

    For each component window and each specified channel:
        - Peak amplitude: max absolute value within the window
        - Peak latency:   time of that peak (ms relative to stimulus)

    Note on interpretation: ERP features are appropriate for statistical
    comparisons (e.g. group differences in P300 amplitude) but require
    careful consideration of trial count balance when used in ML models.
    Unequal trial counts across participants or conditions can bias
    amplitude estimates — consider normalizing if trial counts differ.

    Returns DataFrame with one row per trial.
    """
    times    = epochs.times * 1000   # convert to ms for interpretability
    data     = epochs.get_data()
    ch_names = epochs.ch_names
    n_trials = data.shape[0]
    records  = []

    # Resolve channel indices
    ch_indices = [ch_names.index(ch) for ch in erp_channels if ch in ch_names]
    missing    = [ch for ch in erp_channels if ch not in ch_names]
    if missing:
        print(f"  ERP: channels not found and skipped: {missing}")

    for trial_idx in range(n_trials):
        row = {"trial": trial_idx}

        for comp_name, (tmin_s, tmax_s) in erp_windows.items():
            tmin_ms = tmin_s * 1000
            tmax_ms = tmax_s * 1000
            t_mask  = (times >= tmin_ms) & (times <= tmax_ms)

            for ch_idx in ch_indices:
                ch_name  = ch_names[ch_idx]
                segment  = data[trial_idx, ch_idx, t_mask]

                # Peak = largest absolute deflection in the window
                peak_idx = np.argmax(np.abs(segment))
                peak_amp = segment[peak_idx]
                peak_lat = times[t_mask][peak_idx]

                row[f"{comp_name}_{ch_name}_amp"] = peak_amp * 1e6   # convert to µV
                row[f"{comp_name}_{ch_name}_lat"] = peak_lat          # ms

        records.append(row)

    return pd.DataFrame(records)


# ── C. Connectivity features ───────────────────────────────────────────────────

def compute_itpc(epochs, freqs, n_cycles, sfreq):
    """
    Inter-Trial Phase Coherence (ITPC) — also called ITC or ERSP phase locking.

    Measures how consistently the phase of oscillatory activity aligns to the
    stimulus across trials, at each frequency and time point. High ITPC
    indicates stimulus-locked oscillatory responses that are consistent
    across repetitions.

    Unlike ERPs, ITPC is amplitude-independent — it captures phase synchrony
    regardless of whether the response is large or small. This makes it
    particularly useful when comparing across conditions or participants with
    different signal amplitudes.

    Returns:
        itpc:  array (n_channels, n_freqs, n_times) — ITPC values 0 to 1
        times: time vector
    """
    # Morlet wavelet decomposition
    power, itpc = mne.time_frequency.tfr_array_morlet(
        epochs.get_data(),
        sfreq     = sfreq,
        freqs     = freqs,
        n_cycles  = n_cycles,
        output    = "itc",
        verbose   = False,
    )
    # itpc shape: (n_channels, n_freqs, n_times)
    print(f"  ITPC computed: {len(freqs)} frequencies × "
          f"{itpc.shape[0]} channels × {itpc.shape[-1]} timepoints")
    return itpc, epochs.times


def compute_plv(epochs, channel_pairs=None):
    """
    Phase Locking Value (PLV) — pairwise phase synchrony between channels.

    For each pair of channels and each trial, PLV measures the consistency
    of the phase difference between the two signals across time points within
    the epoch. High PLV indicates the two channels are phase-synchronized,
    suggesting functional connectivity.

    PLV is computed per trial (unlike ITPC which averages across trials),
    making it suitable as a per-trial ML feature.

    If channel_pairs is None, computes all unique pairs (n_channels choose 2).
    For 64 channels this is 2016 pairs — computationally feasible but produces
    a large feature space. Consider dimensionality reduction downstream.

    Returns DataFrame with one row per trial.
    """
    data     = epochs.get_data()      # (n_trials, n_channels, n_times)
    ch_names = epochs.ch_names
    n_trials, n_ch, n_times = data.shape

    if channel_pairs is None:
        pairs = [(i, j) for i in range(n_ch) for j in range(i+1, n_ch)]
    else:
        pairs = [(ch_names.index(a), ch_names.index(b)) for a, b in channel_pairs]

    records = []

    for trial_idx in range(n_trials):
        row = {"trial": trial_idx}
        trial_data = data[trial_idx]   # (n_channels, n_times)

        # Compute analytic signal (instantaneous phase) via Hilbert transform
        from scipy.signal import hilbert
        phases = np.angle(hilbert(trial_data, axis=-1))  # (n_channels, n_times)

        for ch_i, ch_j in pairs:
            phase_diff = phases[ch_i] - phases[ch_j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            col = f"PLV_{ch_names[ch_i]}_{ch_names[ch_j]}"
            row[col] = plv

        records.append(row)

    print(f"  PLV computed: {len(pairs)} channel pairs × {n_trials} trials")
    return pd.DataFrame(records)


def compute_spectral_coherence(epochs, channel_pairs, sfreq, freq_bands):
    """
    Magnitude-squared coherence between channel pairs, averaged per frequency band.

    Coherence measures the consistency of the amplitude and phase relationship
    between two channels across the epoch time series. Unlike PLV (phase only),
    coherence captures both amplitude and phase covariation.

    Note: For 64 channels (2016 pairs) this is computationally expensive.
    Recommend specifying a targeted set of channel_pairs rather than all pairs,
    or using PLV as a lighter alternative.

    Returns DataFrame with one row per trial.
    """
    data     = epochs.get_data()
    ch_names = epochs.ch_names
    n_trials = data.shape[0]

    if channel_pairs is None:
        print("  Coherence: channel_pairs is None — skipping (too many pairs). "
              "Specify pairs in CONFIG['coherence_channel_pairs'].")
        return pd.DataFrame()

    records = []

    for trial_idx in range(n_trials):
        row = {"trial": trial_idx}
        trial_data = data[trial_idx]

        for ch_a_name, ch_b_name in channel_pairs:
            ch_a = ch_names.index(ch_a_name)
            ch_b = ch_names.index(ch_b_name)

            f, coh = scipy_coherence(trial_data[ch_a], trial_data[ch_b],
                                     fs=sfreq, nperseg=sfreq)

            for band_name, (fmin, fmax) in freq_bands.items():
                mask     = (f >= fmin) & (f <= fmax)
                mean_coh = coh[mask].mean() if mask.any() else np.nan
                col      = f"COH_{ch_a_name}_{ch_b_name}_{band_name}"
                row[col] = mean_coh

        records.append(row)

    return pd.DataFrame(records)


# ── D. Participant-level aggregation ───────────────────────────────────────────

def aggregate_to_subject_means(trial_features, epochs):
    """
    Average trial-level features per condition to yield one value per
    participant per condition.

    This is the correct statistical unit for between-subjects analyses.
    Group-level statistics (t-tests, ANOVAs, ML) should be computed from
    these participant means — never from pooled trials, which would
    artificially inflate degrees of freedom and violate independence assumptions.

    Returns a DataFrame with one row per condition.
    """
    # Attach condition labels from epochs metadata
    condition_labels = epochs.events[:, 2]
    event_id_inv     = {v: k for k, v in epochs.event_id.items()}
    conditions       = [event_id_inv.get(c, f"unknown_{c}") for c in condition_labels]

    trial_features   = trial_features.copy()
    trial_features["condition"] = conditions

    # Mean across trials within each condition
    feature_cols  = [c for c in trial_features.columns if c not in ("trial", "condition")]
    subject_means = trial_features.groupby("condition")[feature_cols].mean()
    subject_sem   = trial_features.groupby("condition")[feature_cols].sem()

    subject_means.columns = [f"{c}_mean" for c in subject_means.columns]
    subject_sem.columns   = [f"{c}_sem"  for c in subject_sem.columns]

    return pd.concat([subject_means, subject_sem], axis=1).reset_index()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def extract_features_subject(subject_id, config):
    """
    Full feature extraction pipeline for a single subject.
    Loads clean epochs from preprocessing output, computes all configured
    feature sets, and saves trial-level and participant-level outputs.
    """
    print(f"\n{'='*60}")
    print(f"Feature extraction: {subject_id}")
    print(f"{'='*60}")

    # Load clean epochs
    epochs_path = os.path.join(config["input_dir"], f"{subject_id}_clean_epo.fif")
    epochs      = mne.read_epochs(epochs_path, preload=True, verbose=False)
    print(f"  Loaded: {len(epochs)} epochs, {len(epochs.ch_names)} channels")

    feature_dfs = []

    # A. Band power
    if config["compute_band_power"]:
        print("\n[A] Band power...")
        bp_df = compute_band_power(
            epochs,
            config["freq_bands"],
            config["power_windows"],
            config["sfreq"],
        )
        feature_dfs.append(bp_df)

    # B. ERP features
    if config["compute_erp"]:
        print("\n[B] ERP features...")
        erp_df = compute_erp_features(
            epochs,
            config["erp_windows"],
            config["erp_channels"],
        )
        feature_dfs.append(erp_df)

    # C. Connectivity
    if config["compute_itpc"]:
        print("\n[C] ITPC...")
        itpc, itpc_times = compute_itpc(
            epochs,
            config["itpc_freqs"],
            config["itpc_n_cycles"],
            config["sfreq"],
        )
        # Save ITPC array separately (too large for CSV)
        os.makedirs(config["output_dir"], exist_ok=True)
        np.save(os.path.join(config["output_dir"], f"{subject_id}_itpc.npy"), itpc)
        print(f"  Saved: {subject_id}_itpc.npy")

    if config["compute_plv"]:
        print("\n[C] PLV...")
        plv_df = compute_plv(epochs, config["plv_channel_pairs"])
        feature_dfs.append(plv_df)

    if config["compute_coherence"]:
        print("\n[C] Spectral coherence...")
        coh_df = compute_spectral_coherence(
            epochs,
            config["coherence_channel_pairs"],
            config["sfreq"],
            config["freq_bands"],
        )
        if not coh_df.empty:
            feature_dfs.append(coh_df)

    # Merge all feature sets on trial index
    if feature_dfs:
        trial_features = feature_dfs[0]
        for df in feature_dfs[1:]:
            merge_cols = [c for c in df.columns if c not in trial_features.columns
                          or c == "trial"]
            trial_features = trial_features.merge(
                df[merge_cols], on="trial", how="left"
            )
    else:
        print("  No features computed — check CONFIG flags.")
        return

    # D. Participant-level aggregation
    print("\n[D] Aggregating to participant means...")
    subject_means = aggregate_to_subject_means(trial_features, epochs)

    # Save outputs
    os.makedirs(config["output_dir"], exist_ok=True)
    base = os.path.join(config["output_dir"], subject_id)

    trial_path  = f"{base}_trial_features.csv"
    means_path  = f"{base}_subject_means.csv"

    trial_features.to_csv(trial_path, index=False)
    subject_means.to_csv(means_path,  index=False)

    print(f"\n  Saved: {trial_path}")
    print(f"  Saved: {means_path}")
    print(f"  Trial features shape:   {trial_features.shape}")
    print(f"  Subject means shape:    {subject_means.shape}")
    print(f"\nDone: {subject_id}")

    return trial_features, subject_means


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    subject_id = "sub-01"
    extract_features_subject(subject_id, CONFIG)
