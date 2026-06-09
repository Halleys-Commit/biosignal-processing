"""
01 — EEG Preprocessing Pipeline
---------------------------------
Single-subject EEG preprocessing for task-based cognitive research.
Designed for BioSemi ActiveTwo 64-channel recordings (.bdf format)
sampled at 2048 Hz, following current best practices for ICA-based
artifact rejection and epoch extraction.

Pipeline steps (in order):
    1.  Load raw .bdf file and set BioSemi channel info
    2.  Set standard 10-20 electrode montage
    3.  Initial reference to CMS/DRL (BioSemi requirement before any re-ref)
    4.  Identify and interpolate bad channels
    5.  Notch filter — remove powerline noise (50 Hz + harmonics)
    6.  Broadband bandpass filter — 0.1–100 Hz (preserves full spectrum)
    7.  Downsample from 2048 Hz → 256 Hz
    8.  Create 1 Hz high-pass copy for ICA fitting (ICA requires HP >= 1 Hz)
    9.  Fit ICA on high-pass copy (extended infomax — matches ICLabel training)
    10. Auto-label components with MNE-ICALabel
    11. Exclude non-brain components (eye, muscle, heartbeat, line noise)
    12. Apply ICA weights back to broadband data (not the HP copy)
    13. Re-reference to average of all electrodes
    14. Epoch data by trigger events
    15. Baseline correction
    16. Reject epochs exceeding peak-to-peak amplitude threshold
    17. Save clean epochs and preprocessing report

Design notes:
    - Runs on ONE participant at a time. Loop this script across subjects
      using 02_aggregate_features.py or a batch wrapper.
    - All configurable parameters are in the CONFIG block below.
    - Trigger event IDs should be mapped to your paradigm in EVENT_ID.
    - Output is saved as a .fif file (MNE native format) and a summary report.
    - Bad channels are logged to a JSON sidecar for reproducibility.

Dependencies:
    mne >= 1.6
    mne-icalabel >= 0.4
    numpy, pandas, matplotlib

Install:
    pip install mne mne-icalabel
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components


# ── Configuration ──────────────────────────────────────────────────────────────
# All parameters are set here. Do not hardcode values elsewhere in the script.

CONFIG = {
    # Paths
    "data_dir":      "/path/to/raw/eeg",       # folder containing .bdf files
    "output_dir":    "/path/to/processed/eeg", # where clean epochs are saved

    # Recording parameters
    "sfreq_raw":     2048,    # BioSemi ActiveTwo native sample rate (Hz)
    "sfreq_target":  256,     # downsample target (Hz) — 256 preserves up to 128 Hz
    "n_channels":    64,      # EEG channels (excludes external/EOG channels)

    # Filtering
    "notch_freqs":   [50, 100],   # powerline + first harmonic (change to [60,120] for US)
    "bandpass_low":  0.1,         # Hz — broadband lower bound
    "bandpass_high": 100.0,       # Hz — broadband upper bound
    "ica_hp_cutoff": 1.0,         # Hz — high-pass for ICA fitting copy

    # Re-referencing
    # Options: "average" | specific channel name e.g. "TP9" | ["TP9","TP10"] for linked mastoids
    "reference":     "average",

    # ICA
    "ica_method":    "infomax",   # extended infomax — required for ICLabel compatibility
    "ica_extended":  True,
    "ica_random_state": 42,       # for reproducibility
    # ICLabel component types to EXCLUDE (everything that isn't brain signal)
    "ica_exclude_labels": ["eye blink", "eye movement", "muscle artifact",
                           "heart beat", "line noise", "channel noise"],
    "ica_confidence_threshold": 0.7,  # min confidence to auto-exclude a component

    # Epoching
    # Map your trigger codes to condition names here
    "event_id": {
        "condition_A": 1,
        "condition_B": 2,
        "condition_C": 3,
    },
    "epoch_tmin":   -0.2,    # seconds before event
    "epoch_tmax":    1.0,    # seconds after event
    "baseline":     (-0.2, 0.0),  # baseline correction window

    # Epoch rejection — peak-to-peak amplitude thresholds
    "reject_criteria": {
        "eeg": 150e-6,   # 150 µV — adjust based on your noise floor
    },

    # Bad channel detection — automated variance-based flagging
    # Channels with variance > this multiple of median are flagged for review
    "bad_channel_zscore_threshold": 3.0,
}


# ── Utility functions ──────────────────────────────────────────────────────────

def load_raw_bdf(filepath, n_eeg_channels):
    """
    Load a BioSemi .bdf file.

    BioSemi records with an active CMS/DRL reference, meaning the data
    arrives effectively unreferenced. MNE will warn about this — it is
    expected and handled at the re-referencing step.

    External channels (beyond the first n_eeg_channels) are typically
    used for EOG, ECG, or additional reference electrodes. These are
    preserved here and can be used for ICA artifact detection.
    """
    raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
    print(f"Loaded: {filepath}")
    print(f"  Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s, "
          f"Sampling rate: {raw.info['sfreq']} Hz")
    return raw


def set_biosemi_montage(raw):
    """
    Assign standard 10-20 electrode positions to the 64 BioSemi channels.
    MNE includes the BioSemi64 montage natively.

    Note: If your cap used a non-standard layout, provide a custom montage file.
    """
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="warn")
    print(f"  Montage set: BioSemi 64-channel (10-20 system)")
    return raw


def detect_bad_channels(raw, zscore_threshold):
    """
    Automated bad channel detection based on channel variance.
    Channels with variance more than `zscore_threshold` standard deviations
    above the median are flagged as potentially bad.

    This is a first pass — in a full pipeline you would visually inspect
    flagged channels and confirm before interpolation.

    Returns list of flagged channel names.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data      = raw.get_data(picks=eeg_picks)
    variances = np.var(data, axis=1)

    median_var = np.median(variances)
    mad        = np.median(np.abs(variances - median_var))
    z_scores   = (variances - median_var) / (mad + 1e-10)

    bad_idx  = np.where(np.abs(z_scores) > zscore_threshold)[0]
    bad_channels = [raw.ch_names[eeg_picks[i]] for i in bad_idx]

    if bad_channels:
        print(f"  Flagged bad channels ({len(bad_channels)}): {bad_channels}")
    else:
        print("  No bad channels detected automatically.")

    return bad_channels


def apply_notch_filter(raw, freqs, sfreq):
    """
    Notch filter at powerline frequency and harmonics.
    Use 50 Hz for Europe/UK, 60 Hz for North America.
    """
    raw.notch_filter(freqs=freqs, fir_window="hamming", verbose=False)
    print(f"  Notch filter applied: {freqs} Hz")
    return raw


def apply_bandpass_filter(raw, l_freq, h_freq):
    """
    Broadband bandpass filter.
    Lower bound of 0.1 Hz removes slow drifts while preserving
    slow cortical potentials. Upper bound of 100 Hz preserves
    gamma and high-gamma activity.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq,
               fir_window="hamming", verbose=False)
    print(f"  Bandpass filter applied: {l_freq}–{h_freq} Hz")
    return raw


def fit_ica(raw, n_components, method, extended, random_state, hp_cutoff):
    """
    Fit ICA on a 1 Hz high-pass filtered copy of the data.

    ICA requires at least 1 Hz high-pass filtering to function correctly —
    slow drifts cause ICA to produce poor decompositions. We fit on the
    HP copy but apply the resulting weights to the original broadband data,
    preserving slow cortical potentials in the cleaned output.

    Extended infomax is used because ICLabel's classifier was trained on
    components produced by this algorithm.
    """
    raw_for_ica = raw.copy().filter(l_freq=hp_cutoff, h_freq=None,
                                    fir_window="hamming", verbose=False)
    print(f"  High-pass filtered copy at {hp_cutoff} Hz for ICA fitting.")

    ica = ICA(
        n_components  = n_components,
        method        = method,
        fit_params    = {"extended": extended},
        random_state  = random_state,
        max_iter      = "auto",
        verbose       = False,
    )
    ica.fit(raw_for_ica, verbose=False)
    print(f"  ICA fitted: {ica.n_components_} components.")
    return ica


def label_and_exclude_components(raw, ica, exclude_labels, confidence_threshold):
    """
    Automatically label ICA components using MNE-ICALabel.

    ICLabel classifies each component as one of:
        brain, eye blink, eye movement, muscle artifact,
        heart beat, line noise, channel noise, other

    Components labeled as artifact types (not 'brain' or 'other') with
    confidence above the threshold are automatically excluded.

    Returns the ICA object with exclusions set, and a summary DataFrame.
    """
    labels = label_components(raw, ica, method="iclabel")

    component_labels  = labels["labels"]
    component_probs   = labels["y_pred_proba"]

    excluded    = []
    label_summary = []

    for idx, (label, probs) in enumerate(zip(component_labels, component_probs)):
        confidence = float(np.max(probs))
        label_summary.append({
            "component":  idx,
            "label":      label,
            "confidence": round(confidence, 3),
            "excluded":   False,
        })
        if label in exclude_labels and confidence >= confidence_threshold:
            excluded.append(idx)
            label_summary[-1]["excluded"] = True

    ica.exclude = excluded
    summary_df  = pd.DataFrame(label_summary)

    print(f"  ICA labels assigned. Excluding {len(excluded)} components:")
    for idx in excluded:
        row = summary_df[summary_df["component"] == idx].iloc[0]
        print(f"    Component {idx:02d}: {row['label']} "
              f"(confidence: {row['confidence']:.2f})")

    return ica, summary_df


def apply_reference(raw, reference):
    """
    Re-reference EEG to the specified reference.

    Options:
        "average"          — average of all electrodes (most common for dense arrays)
        "TP9"              — single electrode (e.g. mastoid)
        ["TP9", "TP10"]    — linked mastoids (average of two)

    BioSemi note: set_eeg_reference() requires that the data was previously
    recorded with a known reference. BioSemi data arrives as CMS/DRL —
    MNE handles this automatically when you call set_eeg_reference().
    """
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
        print("  Re-referenced: average of all electrodes")
    else:
        raw.set_eeg_reference(reference, projection=False, verbose=False)
        print(f"  Re-referenced: {reference}")
    return raw


def epoch_data(raw, event_id, tmin, tmax, baseline, reject_criteria):
    """
    Segment continuous data into epochs time-locked to trigger events.

    Events are read from the stimulus channel (STI 014 on BioSemi).
    Epochs extending beyond the recording are automatically dropped.
    Baseline correction subtracts the mean of the pre-stimulus window.

    Returns epochs object and event array.
    """
    events = mne.find_events(raw, stim_channel="Status",
                             shortest_event=1, verbose=False)
    print(f"  Found {len(events)} events total.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id        = event_id,
        tmin            = tmin,
        tmax            = tmax,
        baseline        = baseline,
        reject          = reject_criteria,
        reject_by_annotation = True,
        preload         = True,
        verbose         = False,
    )

    n_dropped = len(events) - len(epochs)
    print(f"  Epochs created: {len(epochs)} retained, {n_dropped} dropped "
          f"(artifacts or amplitude threshold).")

    return epochs, events


def save_outputs(epochs, ica_summary, bad_channels, subject_id, output_dir):
    """
    Save cleaned epochs, ICA component summary, bad channel log,
    and a preprocessing quality report.

    Output files:
        {subject_id}_clean_epo.fif       — MNE epochs file
        {subject_id}_ica_labels.csv      — per-component labels and confidence
        {subject_id}_bad_channels.json   — flagged channels log
        {subject_id}_preprocessing_report.html — visual QC report
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, subject_id)

    # Clean epochs
    epochs_path = f"{base}_clean_epo.fif"
    epochs.save(epochs_path, overwrite=True, verbose=False)
    print(f"  Saved: {epochs_path}")

    # ICA label summary
    ica_path = f"{base}_ica_labels.csv"
    ica_summary.to_csv(ica_path, index=False)
    print(f"  Saved: {ica_path}")

    # Bad channels log
    bad_path = f"{base}_bad_channels.json"
    with open(bad_path, "w") as f:
        json.dump({"subject": subject_id, "bad_channels": bad_channels}, f, indent=2)
    print(f"  Saved: {bad_path}")

    # HTML preprocessing report
    report = mne.Report(title=f"Preprocessing Report — {subject_id}", verbose=False)
    report.add_epochs(epochs, title="Clean epochs", replace=True)
    report_path = f"{base}_preprocessing_report.html"
    report.save(report_path, overwrite=True, verbose=False)
    print(f"  Saved: {report_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def preprocess_subject(bdf_filepath, subject_id, config):
    """
    Full preprocessing pipeline for a single subject.

    Parameters
    ----------
    bdf_filepath : str
        Path to the raw .bdf file for this subject.
    subject_id : str
        Unique identifier for this subject (e.g. 'sub-01').
        Used for output file naming.
    config : dict
        Pipeline configuration (see CONFIG block at top of file).
    """
    print(f"\n{'='*60}")
    print(f"Processing: {subject_id}")
    print(f"{'='*60}")

    # 1. Load raw data
    raw = load_raw_bdf(bdf_filepath, config["n_channels"])

    # 2. Set electrode montage
    raw = set_biosemi_montage(raw)

    # 3. Detect and interpolate bad channels
    bad_channels = detect_bad_channels(raw, config["bad_channel_zscore_threshold"])
    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"  Interpolated {len(bad_channels)} bad channels.")

    # 4. Notch filter (powerline noise)
    raw = apply_notch_filter(raw, config["notch_freqs"], config["sfreq_raw"])

    # 5. Broadband bandpass filter
    raw = apply_bandpass_filter(raw, config["bandpass_low"], config["bandpass_high"])

    # 6. Downsample
    raw.resample(config["sfreq_target"], npad="auto", verbose=False)
    print(f"  Downsampled: {config['sfreq_raw']} Hz → {config['sfreq_target']} Hz")

    # 7. Fit ICA on 1 Hz HP copy
    ica = fit_ica(
        raw,
        n_components  = config["n_channels"],
        method        = config["ica_method"],
        extended      = config["ica_extended"],
        random_state  = config["ica_random_state"],
        hp_cutoff     = config["ica_hp_cutoff"],
    )

    # 8. Label components and set exclusions
    ica, ica_summary = label_and_exclude_components(
        raw,
        ica,
        config["ica_exclude_labels"],
        config["ica_confidence_threshold"],
    )

    # 9. Apply ICA to broadband data
    ica.apply(raw, verbose=False)
    print(f"  ICA applied to broadband data.")

    # 10. Re-reference
    raw = apply_reference(raw, config["reference"])

    # 11. Epoch, baseline correct, reject bad epochs
    epochs, events = epoch_data(
        raw,
        config["event_id"],
        config["epoch_tmin"],
        config["epoch_tmax"],
        config["baseline"],
        config["reject_criteria"],
    )

    # 12. Save all outputs
    save_outputs(epochs, ica_summary, bad_channels,
                 subject_id, config["output_dir"])

    print(f"\nDone: {subject_id}")
    return epochs


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # To process a single subject, set these two values and run the script.
    # To process all subjects in a folder, see 07_aggregate_features.py.

    subject_id   = "sub-01"
    bdf_filename = f"{subject_id}_task.bdf"
    bdf_filepath = os.path.join(CONFIG["data_dir"], bdf_filename)

    epochs = preprocess_subject(bdf_filepath, subject_id, CONFIG)
