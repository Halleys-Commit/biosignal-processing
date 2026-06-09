# biosignal-processing

A modular, research-grade pipeline for multimodal physiological signal processing and behavioral performance decoding. Built for task-based cognitive research using wearable and laboratory biosensing equipment.

Processes EEG, heart rate, respiration, and eye tracking data from raw acquisition files through to a machine learning-ready feature dataset, with rigorous attention to signal quality, experimental design, and generalizability.

---

## Pipeline overview

```
Raw data (per participant)
    │
    ├── 01_eeg_preprocessing.py        EEG cleaning, ICA, epoching
    ├── 02_eeg_feature_extraction.py   Band power, ITPC, PLV, ERPs
    ├── 03_heart_rate_processing.py    R-peak detection, HRV, RSA
    ├── 04_respiration_processing.py   Breath cycle detection, RRV
    ├── 06_eye_tracking_processing.py  Pupil artifact rejection, fixations, saccades
    │
    ↓ (per-subject feature files)
    │
    ├── 07_aggregate_features.py       Merge all modalities across participants
    └── 08_build_ml_dataset.py         LOSO cross-validation, holdout evaluation
```

Each script runs on a single participant and saves structured output. Scripts 07–08 aggregate across all participants for group-level analysis and model evaluation.

---

## Signals and equipment

| Modality | Hardware | Sample rate | Key features extracted |
|---|---|---|---|
| EEG | BioSemi ActiveTwo (64-ch) | 2048 Hz → 256 Hz | Band power (δ θ α β γ), ITPC, PLV, ERP amplitude/latency |
| ECG / HRV | BIOPAC MP160 + ECG100C | 1000 Hz | RMSSD, SDNN, MeanNN, LF/HF power, SD1/SD2, RSA |
| Respiration | BIOPAC MP160 + RSP100C | 1000 Hz | Rate, amplitude, I/E ratio, RRV, RSA |
| Eye tracking | Tobii Pro Glasses 3 (mobile) | 50 Hz | Pupil dilation, fixation count/duration, saccade amplitude/velocity, blink rate |

---

## EEG preprocessing

Follows current best practices for ICA-based artifact rejection:

- Notch filter (50 Hz) → broadband bandpass (0.1–100 Hz) → downsample to 256 Hz
- Bad channel detection by variance threshold → spherical spline interpolation
- ICA fitted on a 1 Hz high-pass copy of the data (required for clean decomposition)
- Components labeled automatically via **MNE-ICALabel** (eye, muscle, heartbeat, line noise → excluded)
- ICA weights applied back to the original broadband signal — preserves slow cortical potentials
- Re-reference to average → epoch by trigger → baseline correct → amplitude threshold rejection

---

## Experimental design and statistical unit

All scripts are built around a within-subjects design with trial-level feature extraction. The correct statistical unit for group inference is the **participant mean** — averaging trials within a participant before computing group statistics. Scripts produce both trial-level outputs (for ML) and participant-level means (for statistical analyses), with the distinction clearly documented.

Pre-stimulus and post-stimulus feature windows are computed separately throughout, reflecting the scientific distinction between baseline cognitive state (pre-stimulus) and task-evoked response (post-stimulus).

---

## Machine learning evaluation

Behavioral performance prediction from multimodal physiological features, evaluated with a design that measures generalizability to new individuals:

**Leave-One-Subject-Out (LOSO) cross-validation**
- Train on N-1 participants → test on the held-out participant entirely
- The held-out participant is genuinely unseen during training
- Appropriate evaluation for any system intended to work on new users

**Final holdout set**
- 20% of participants reserved before any model development
- Never used for model selection or hyperparameter decisions
- Evaluated once at the end — the most honest performance estimate

**Models compared** (regression and classification):
Ridge Regression · Random Forest · XGBoost · SVM (RBF) · MLP

Performance binarization uses a within-participant median split — classifying each trial as high or low relative to that participant's own baseline, which correctly separates trial-level variability from individual differences in absolute performance level.

---

## Repository structure

```
biosignal-processing/
├── eeg/
│   ├── 01_eeg_preprocessing.py
│   └── 02_eeg_feature_extraction.py
├── physio/
│   ├── 03_heart_rate_processing.py
│   ├── 04_respiration_processing.py
│   └── 06_eye_tracking_processing.py
└── pipeline/
    ├── 07_aggregate_features.py
    └── 08_build_ml_dataset.py
```

Note: script 05 (GSR/EDA processing) is reserved for future addition.

---

## Dependencies

```bash
pip install mne mne-icalabel neurokit2 bioread scikit-learn xgboost \
            numpy pandas scipy matplotlib seaborn
```

| Library | Purpose |
|---|---|
| `mne` | EEG loading, filtering, ICA, epoching |
| `mne-icalabel` | Automated ICA component classification |
| `neurokit2` | ECG/HRV and respiration processing |
| `bioread` | BIOPAC .acq file loading |
| `scikit-learn` | ML pipelines, LOSO evaluation, preprocessing |
| `xgboost` | Gradient boosting models |
| `scipy` | Signal processing, interpolation, statistics |

---

## Usage

Each script is designed to run on a single participant. Set the paths and subject ID at the bottom of each file:

```python
if __name__ == "__main__":
    subject_id = "sub-01"
    process_subject(subject_id, CONFIG)
```

To process all participants, wrap in a loop or use the batch entry point in `07_aggregate_features.py`, which auto-discovers subject folders.

All configurable parameters (filter settings, epoch windows, thresholds, model hyperparameters) are collected in a `CONFIG` dictionary at the top of each script.

---

## Background

Pipeline developed from experience processing multimodal physiological data in applied cognitive neuroscience research, with emphasis on signal quality, reproducible preprocessing, and evaluation designs that reflect real-world deployment constraints.
