"""
06 — Eye Tracking Processing (Mobile)
---------------------------------------
Single-subject eye tracking preprocessing and feature extraction for
task-based cognitive research. Designed for Tobii Pro Glasses 3 output
(50 Hz binocular, CSV export via Tobii Pro Lab).

Mobile eye tracking introduces challenges not present in remote/screen-based
systems: head movement causes gaze signal noise, tracking loss is more
frequent, and pupil size is affected by ambient light changes and viewing
angle. This pipeline addresses each explicitly.

Pipeline steps:
    1.  Load raw Tobii Pro Glasses 3 CSV export
    2.  Parse gaze, pupil, and event channels
    3.  Validate and flag tracking loss (confidence threshold)
    4.  Pupil preprocessing:
            a. Detect and flag blinks (zero/near-zero pupil + velocity criterion)
            b. Extend blink windows to capture partial artifacts (pre/post buffer)
            c. Detect non-blink artifacts (outliers, camera loss, rapid dilation)
            d. Interpolate blink and artifact periods (cubic spline)
            e. Smooth cleaned pupil signal (Savitzky-Golay filter)
            f. Baseline correct per trial (z-score or % change from pre-stimulus)
    5.  Saccade detection (velocity-based, I-VT algorithm)
    6.  Fixation detection (minimum duration + dispersion threshold)
    7.  Blink detection (independent from pupil — duration, rate)
    8.  Align to task event triggers
    9.  Extract per-trial features:
            Pupil:    mean dilation, peak dilation, latency to peak, dilation slope
            Fixation: count, mean duration, total fixation time, dispersion
            Saccade:  count, mean amplitude, mean peak velocity, latency to first
            Blink:    count, mean duration, rate
    10. Aggregate to participant means per condition

Design notes:
    - Binocular recordings: left and right eye processed separately,
      averaged where both are valid, single eye used when one is lost.
    - Pupil size in arbitrary units (a.u.) from Tobii — not calibrated to mm.
      All pupil features use normalized/baseline-corrected values to allow
      comparison across participants and sessions.
    - Mobile recordings have higher tracking loss rates than remote systems.
      Trials with >30% data loss (configurable) are flagged and excluded.
    - Participant means are the correct statistical unit for group analyses.

Input:
    {subject_id}_eyetracking.csv   — Tobii Pro Glasses 3 CSV export
    {subject_id}_events.csv        — task trigger onset times and conditions

Output:
    {subject_id}_pupil_clean.csv         — cleaned, interpolated pupil signal
    {subject_id}_events_detected.csv     — fixation/saccade/blink event list
    {subject_id}_trial_eye_features.csv  — per-trial feature matrix
    {subject_id}_subject_eye_means.csv   — participant means per condition
    {subject_id}_eye_report.png          — visual QC plots

Dependencies:
    numpy, pandas, scipy, matplotlib
    (no specialist eye tracking library required — all algorithms implemented
    directly for transparency and portability)

Install:
    pip install numpy pandas scipy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "input_dir":  "/path/to/raw/eyetracking",
    "output_dir": "/path/to/features/eyetracking",

    # Recording parameters — Tobii Pro Glasses 3
    "sfreq":         50,     # Hz — Tobii Pro Glasses 3 default
    "eyes":          "both", # "both", "left", or "right"

    # Tobii Pro Glasses 3 CSV column names
    # Adjust if your export uses different headers
    "col_timestamp":        "Recording timestamp [ms]",
    "col_pupil_left":       "Pupil diameter left [mm]",
    "col_pupil_right":      "Pupil diameter right [mm]",
    "col_gaze_x_left":      "Gaze point left X [px]",
    "col_gaze_y_left":      "Gaze point left Y [px]",
    "col_gaze_x_right":     "Gaze point right X [px]",
    "col_gaze_y_right":     "Gaze point right Y [px]",
    "col_validity_left":    "Eye validity left",
    "col_validity_right":   "Eye validity right",

    # Tracking quality — Tobii validity codes: 0=valid, 1-4=increasingly unreliable
    # Samples with validity > this threshold are treated as lost
    "validity_threshold": 1,

    # Pupil artifact detection
    "blink_min_duration_ms":    80,    # min blink duration (ms) — shorter = likely noise
    "blink_max_duration_ms":    500,   # max blink duration (ms) — longer = likely other artifact
    "blink_buffer_pre_ms":      50,    # ms before blink onset to include in artifact window
    "blink_buffer_post_ms":     100,   # ms after blink offset (reopening artifacts persist)
    "pupil_mad_threshold":      4.0,   # median absolute deviations for outlier detection
    "pupil_velocity_threshold": 0.1,   # a.u./sample — rapid changes flagged as artifacts

    # Pupil smoothing — Savitzky-Golay filter
    "savgol_window_ms":  100,   # smoothing window (ms) — converted to samples internally
    "savgol_polyorder":  3,     # polynomial order

    # Baseline correction method
    # "zscore":     z-score relative to pre-stimulus window
    # "percent":    percent change from pre-stimulus mean
    # "subtract":   subtract pre-stimulus mean (preserves units)
    "baseline_method":   "zscore",
    "baseline_window":   (-0.5, 0.0),  # seconds relative to event onset

    # Maximum acceptable data loss per trial (proportion 0–1)
    # Trials above this threshold are flagged for exclusion
    "max_loss_proportion": 0.30,

    # Saccade detection — I-VT (velocity threshold) algorithm
    "saccade_velocity_threshold": 30,   # deg/s — standard cognitive research threshold
    "saccade_min_duration_ms":    20,   # ms
    "saccade_min_amplitude_deg":  0.5,  # degrees

    # Fixation detection — minimum duration and dispersion
    "fixation_min_duration_ms":   100,  # ms — shorter = likely microsaccade
    "fixation_max_dispersion_px": 25,   # pixels — spatial spread threshold

    # Trial epoch windows (seconds relative to event onset)
    "epoch_windows": {
        "pre_stimulus":  (-1.0, 0.0),
        "post_stimulus": (0.0,  2.0),
    },

    # Events file
    "events_file": "{subject_id}_events.csv",
}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_tobii_csv(subject_id, config):
    """
    Load Tobii Pro Glasses 3 CSV export.

    Tobii Pro Lab exports one row per sample with timestamps in milliseconds.
    Missing/invalid samples are represented as NaN or empty cells.
    Validity codes indicate tracking confidence per sample per eye.

    Returns a DataFrame with standardized column names and a time vector
    in seconds from recording start.
    """
    filepath = os.path.join(config["input_dir"], f"{subject_id}_eyetracking.csv")
    df       = pd.read_csv(filepath, sep="\t", encoding="utf-8-sig")

    # Standardize time to seconds from recording start
    df["time_s"] = (df[config["col_timestamp"]] -
                    df[config["col_timestamp"]].iloc[0]) / 1000.0

    print(f"  Loaded: {filepath}")
    print(f"  Duration: {df['time_s'].iloc[-1]:.1f}s | "
          f"{len(df)} samples @ {config['sfreq']} Hz")

    return df


def load_events(subject_id, config):
    """Load task event triggers. See scripts 03/04 for format details."""
    if config["events_file"] is None:
        return None
    path = os.path.join(
        config["input_dir"],
        config["events_file"].replace("{subject_id}", subject_id)
    )
    if not os.path.exists(path):
        print(f"  Warning: events file not found: {path}")
        return None
    events = pd.read_csv(path)
    print(f"  Loaded {len(events)} events.")
    return events


# ── Pupil preprocessing ────────────────────────────────────────────────────────

def extract_pupil_signal(df, config):
    """
    Extract and merge binocular pupil signal.

    Strategy:
        - Where both eyes are valid: use mean of left and right
        - Where one eye is valid: use that eye alone
        - Where neither is valid: mark as NaN (artifact/loss)

    Validity threshold applied per Tobii documentation:
        0 = high confidence tracking
        1 = slightly uncertain (typically acceptable)
        2-4 = progressively unreliable — treated as loss here
    """
    thresh = config["validity_threshold"]

    left_valid  = df[config["col_validity_left"]]  <= thresh
    right_valid = df[config["col_validity_right"]] <= thresh

    pupil_left  = df[config["col_pupil_left"]].copy().astype(float)
    pupil_right = df[config["col_pupil_right"]].copy().astype(float)

    # Invalid samples → NaN
    pupil_left[~left_valid]   = np.nan
    pupil_right[~right_valid] = np.nan

    # Merge: mean where both valid, single where one valid, NaN where neither
    both_valid    = left_valid  & right_valid
    only_left     = left_valid  & ~right_valid
    only_right    = right_valid & ~left_valid

    pupil = pd.Series(np.nan, index=df.index)
    pupil[both_valid]  = (pupil_left[both_valid] + pupil_right[both_valid]) / 2
    pupil[only_left]   = pupil_left[only_left]
    pupil[only_right]  = pupil_right[only_right]

    loss_pct = pupil.isna().mean() * 100
    print(f"  Pupil extracted: {loss_pct:.1f}% data loss before artifact removal")

    return pupil


def detect_blinks(pupil, time_s, config, sfreq):
    """
    Detect blinks from pupil signal using a two-stage approach:

    Stage 1 — Zero/NaN detection:
        Tobii reports 0 or NaN during confirmed tracking loss (blink or loss).
        Contiguous runs of missing data within physiologically plausible
        blink duration range are classified as blinks.

    Stage 2 — Velocity criterion:
        Rapid signal changes immediately surrounding tracking loss indicate
        the eyelid closure/reopening trajectory. These are included in the
        blink artifact window.

    Returns a boolean mask (True = blink/artifact, should be interpolated)
    and a DataFrame of individual blink events.
    """
    sfreq_ms     = 1000 / sfreq
    min_samples  = int(config["blink_min_duration_ms"] / sfreq_ms)
    max_samples  = int(config["blink_max_duration_ms"] / sfreq_ms)
    pre_samples  = int(config["blink_buffer_pre_ms"]   / sfreq_ms)
    post_samples = int(config["blink_buffer_post_ms"]  / sfreq_ms)

    is_missing = pupil.isna() | (pupil == 0)
    artifact_mask = is_missing.copy()

    blink_events = []
    in_blink     = False
    blink_start  = None

    for i, missing in enumerate(is_missing):
        if missing and not in_blink:
            in_blink    = True
            blink_start = i
        elif not missing and in_blink:
            blink_end    = i
            blink_len    = blink_end - blink_start
            in_blink     = False

            if min_samples <= blink_len <= max_samples:
                # Valid blink — extend window for buffer
                buf_start = max(0, blink_start - pre_samples)
                buf_end   = min(len(pupil), blink_end + post_samples)
                artifact_mask.iloc[buf_start:buf_end] = True

                blink_events.append({
                    "onset_s":    time_s.iloc[blink_start],
                    "offset_s":   time_s.iloc[blink_end],
                    "duration_ms": blink_len * sfreq_ms,
                })

    blinks_df = pd.DataFrame(blink_events)
    print(f"  Blinks detected: {len(blinks_df)} | "
          f"Mean duration: {blinks_df['duration_ms'].mean():.0f}ms"
          if len(blinks_df) > 0 else "  No blinks detected.")

    return artifact_mask, blinks_df


def detect_pupil_artifacts(pupil, artifact_mask, config):
    """
    Detect non-blink artifacts in the pupil signal:

    1. Statistical outliers: samples > MAD threshold deviations from
       the rolling median (robust to blink contamination).

    2. Velocity artifacts: samples with implausibly rapid dilation changes
       that cannot be physiological (camera glints, tracking jumps).

    These occur in mobile recordings due to glasses movement, ambient light
    changes, and partial obstructions — more frequent than in remote systems.

    Returns updated artifact mask.
    """
    clean_pupil = pupil.copy()
    clean_pupil[artifact_mask] = np.nan

    # Rolling median for outlier detection (window = 1 second)
    sfreq    = config["sfreq"]
    roll_med = clean_pupil.rolling(window=sfreq, center=True, min_periods=1).median()
    roll_mad = (clean_pupil - roll_med).abs().rolling(
        window=sfreq, center=True, min_periods=1).median()

    outlier_mask = (clean_pupil - roll_med).abs() > (
        config["pupil_mad_threshold"] * roll_mad
    )
    artifact_mask = artifact_mask | outlier_mask.fillna(False)

    # Velocity criterion — flag rapid changes
    velocity     = clean_pupil.diff().abs()
    vel_mask     = velocity > config["pupil_velocity_threshold"]
    artifact_mask = artifact_mask | vel_mask.fillna(False)

    total_artifact_pct = artifact_mask.mean() * 100
    print(f"  Total artifact/loss: {total_artifact_pct:.1f}% of samples")

    return artifact_mask


def interpolate_artifacts(pupil, artifact_mask, time_s):
    """
    Cubic spline interpolation over detected artifact periods.

    Uses the last valid sample before and first valid sample after each
    artifact window as interpolation anchors. This preserves the signal
    trajectory through the gap more accurately than linear interpolation,
    particularly for blink reopening dynamics.

    Gaps at the start or end of the recording cannot be interpolated
    and remain NaN.
    """
    pupil_interp = pupil.copy()
    valid        = ~artifact_mask & ~pupil.isna()
    t_valid      = time_s[valid].values
    p_valid      = pupil[valid].values

    if len(t_valid) < 4:
        print("  WARNING: Insufficient valid data for interpolation.")
        return pupil_interp

    cs = CubicSpline(t_valid, p_valid)

    # Apply interpolation only to artifact periods, not genuine long gaps
    artifact_times = time_s[artifact_mask].values
    if len(artifact_times) > 0:
        pupil_interp[artifact_mask] = cs(artifact_times)

    n_interpolated = artifact_mask.sum()
    print(f"  Interpolated {n_interpolated} artifact samples "
          f"({n_interpolated/len(pupil)*100:.1f}%)")

    return pupil_interp


def smooth_pupil(pupil_interp, config, sfreq):
    """
    Savitzky-Golay smoothing of the cleaned pupil signal.

    Preserves signal shape (peaks, slopes) better than simple moving
    average while removing high-frequency noise. Window length and
    polynomial order are configurable.

    Window must be odd — rounded up if even.
    """
    window_samples = int(config["savgol_window_ms"] / (1000 / sfreq))
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = max(window_samples, config["savgol_polyorder"] + 2)

    valid_mask   = ~pupil_interp.isna()
    pupil_smooth = pupil_interp.copy()

    if valid_mask.sum() > window_samples:
        pupil_smooth[valid_mask] = savgol_filter(
            pupil_interp[valid_mask].values,
            window_length = window_samples,
            polyorder     = config["savgol_polyorder"],
        )

    print(f"  Pupil smoothed: Savitzky-Golay "
          f"(window={config['savgol_window_ms']}ms, "
          f"order={config['savgol_polyorder']})")

    return pupil_smooth


def baseline_correct_pupil(pupil, time_s, events, config):
    """
    Baseline correct pupil signal per trial.

    Three methods available (configured via CONFIG['baseline_method']):

        zscore:  subtract pre-stimulus mean, divide by pre-stimulus SD
                 → units become standard deviations from baseline
                 → recommended for cross-participant comparisons

        percent: (pupil - baseline_mean) / baseline_mean * 100
                 → units become % change from baseline
                 → intuitive, preserves physiological meaning

        subtract: pupil - baseline_mean
                 → preserves original units (a.u.)
                 → simplest, appropriate when units are comparable

    Note: Pupil size from Tobii is in arbitrary units — zscore or percent
    correction is strongly recommended to allow comparison across
    participants, sessions, and hardware configurations.
    """
    if events is None:
        return pupil

    method = config["baseline_method"]
    bl_tmin, bl_tmax = config["baseline_window"]
    pupil_corrected  = pupil.copy()

    for _, event in events.iterrows():
        onset    = event["onset_sec"]
        bl_mask  = (time_s >= onset + bl_tmin) & (time_s < onset + bl_tmax)
        bl_vals  = pupil[bl_mask].dropna()

        if len(bl_vals) < 3:
            continue

        bl_mean = bl_vals.mean()
        bl_std  = bl_vals.std()

        # Apply correction to the full epoch
        ep_tmin = config["epoch_windows"]["pre_stimulus"][0]
        ep_tmax = config["epoch_windows"]["post_stimulus"][1]
        ep_mask = (time_s >= onset + ep_tmin) & (time_s <= onset + ep_tmax)

        if method == "zscore":
            if bl_std > 0:
                pupil_corrected[ep_mask] = (pupil[ep_mask] - bl_mean) / bl_std
        elif method == "percent":
            if bl_mean != 0:
                pupil_corrected[ep_mask] = (
                    (pupil[ep_mask] - bl_mean) / bl_mean * 100
                )
        elif method == "subtract":
            pupil_corrected[ep_mask] = pupil[ep_mask] - bl_mean

    return pupil_corrected


# ── Gaze event detection ───────────────────────────────────────────────────────

def detect_saccades(gaze_x, gaze_y, time_s, config, sfreq):
    """
    Saccade detection using the I-VT (velocity threshold) algorithm.

    Computes sample-to-sample angular velocity from gaze position.
    Contiguous samples above the velocity threshold that meet minimum
    duration and amplitude criteria are classified as saccades.

    Note: For mobile eye tracking, angular velocity is approximated from
    pixel displacement — calibration to degrees assumes fixed viewing
    distance, which varies in mobile settings. Features use pixel units
    unless viewing distance is known.

    Returns DataFrame of saccade events.
    """
    dt       = 1.0 / sfreq
    dx       = gaze_x.diff().fillna(0).values
    dy       = gaze_y.diff().fillna(0).values
    velocity = np.sqrt(dx**2 + dy**2) / dt   # pixels/second

    min_samples = int(config["saccade_min_duration_ms"] / (1000 / sfreq))
    threshold   = config["saccade_velocity_threshold"]

    above_thresh = velocity > threshold
    saccades     = []
    in_sacc      = False
    sacc_start   = None

    for i, above in enumerate(above_thresh):
        if above and not in_sacc:
            in_sacc    = True
            sacc_start = i
        elif not above and in_sacc:
            sacc_end = i
            in_sacc  = False

            if (sacc_end - sacc_start) >= min_samples:
                amplitude = np.sqrt(
                    (gaze_x.iloc[sacc_end] - gaze_x.iloc[sacc_start])**2 +
                    (gaze_y.iloc[sacc_end] - gaze_y.iloc[sacc_start])**2
                )
                if amplitude >= config["saccade_min_amplitude_deg"]:
                    saccades.append({
                        "onset_s":      time_s.iloc[sacc_start],
                        "offset_s":     time_s.iloc[sacc_end],
                        "duration_ms":  (sacc_end - sacc_start) * (1000 / sfreq),
                        "amplitude_px": amplitude,
                        "peak_velocity": velocity[sacc_start:sacc_end].max(),
                    })

    sacc_df = pd.DataFrame(saccades)
    print(f"  Saccades detected: {len(sacc_df)}")
    return sacc_df


def detect_fixations(gaze_x, gaze_y, time_s, saccade_df, config, sfreq):
    """
    Fixation detection: periods between saccades meeting minimum duration
    and maximum spatial dispersion criteria.

    Fixation = stable gaze cluster where all samples fall within
    a spatial window (dispersion threshold) for at least min_duration ms.

    Dispersion is computed as the range of x + range of y coordinates
    within the candidate fixation window (IDT algorithm).

    Returns DataFrame of fixation events.
    """
    min_samples = int(config["fixation_min_duration_ms"] / (1000 / sfreq))
    max_disp    = config["fixation_max_dispersion_px"]
    fixations   = []

    # Identify non-saccade periods
    is_saccade = pd.Series(False, index=gaze_x.index)
    if len(saccade_df) > 0:
        for _, sacc in saccade_df.iterrows():
            mask = (time_s >= sacc["onset_s"]) & (time_s <= sacc["offset_s"])
            is_saccade[mask] = True

    # Sliding window fixation detection
    i = 0
    n = len(gaze_x)

    while i < n - min_samples:
        if is_saccade.iloc[i]:
            i += 1
            continue

        # Grow window while dispersion stays within threshold
        j    = i + min_samples
        win_x = gaze_x.iloc[i:j].values
        win_y = gaze_y.iloc[i:j].values
        disp  = (win_x.max() - win_x.min()) + (win_y.max() - win_y.min())

        if disp > max_disp or j >= n:
            i += 1
            continue

        # Extend window as long as dispersion stays within threshold
        while j < n and not is_saccade.iloc[j]:
            win_x = gaze_x.iloc[i:j+1].values
            win_y = gaze_y.iloc[i:j+1].values
            if (win_x.max()-win_x.min()) + (win_y.max()-win_y.min()) > max_disp:
                break
            j += 1

        duration_ms = (j - i) * (1000 / sfreq)
        if duration_ms >= config["fixation_min_duration_ms"]:
            fixations.append({
                "onset_s":    time_s.iloc[i],
                "offset_s":   time_s.iloc[j-1],
                "duration_ms": duration_ms,
                "centroid_x":  gaze_x.iloc[i:j].mean(),
                "centroid_y":  gaze_y.iloc[i:j].mean(),
                "dispersion_px": disp,
            })
        i = j

    fix_df = pd.DataFrame(fixations)
    print(f"  Fixations detected: {len(fix_df)}")
    return fix_df


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_trial_features(pupil, time_s, saccade_df, fixation_df,
                            blink_df, events, config, sfreq):
    """
    Extract eye tracking features for each trial over each time window.

    Pupil features:
        mean_pupil:      mean dilation (baseline corrected)
        peak_pupil:      maximum dilation in window
        peak_latency_s:  time of peak dilation relative to event onset
        dilation_slope:  linear slope of pupil change (dilation velocity)
        loss_proportion: fraction of samples that are NaN/artifact

    Fixation features:
        n_fixations:     count of fixations in window
        mean_fix_duration: mean fixation duration (ms)
        total_fix_time:  total time spent fixating (ms)

    Saccade features:
        n_saccades:      count of saccades in window
        mean_saccade_amp: mean saccade amplitude (pixels)
        mean_peak_vel:   mean peak saccade velocity
        first_sacc_latency_s: latency to first saccade after event onset

    Blink features:
        n_blinks:        count of blinks in window
        mean_blink_dur:  mean blink duration (ms)
        blink_rate:      blinks per minute
    """
    if events is None:
        return pd.DataFrame()

    records = []

    for trial_idx, event in events.iterrows():
        onset     = event["onset_sec"]
        condition = event.get("condition", "unknown")
        row       = {"trial": trial_idx, "condition": condition, "onset_sec": onset}

        for window_name, (tmin, tmax) in config["epoch_windows"].items():
            win_start = onset + tmin
            win_end   = onset + tmax
            t_mask    = (time_s >= win_start) & (time_s < win_end)
            win_pupil = pupil[t_mask]
            win_time  = time_s[t_mask]
            win_dur_s = tmax - tmin

            # Data quality check
            loss_prop = win_pupil.isna().mean()
            row[f"{window_name}_loss_proportion"] = loss_prop

            if loss_prop > config["max_loss_proportion"]:
                row[f"{window_name}_flagged"] = True
            else:
                row[f"{window_name}_flagged"] = False

            # ── Pupil features ──
            valid_pupil = win_pupil.dropna()
            if len(valid_pupil) > 0:
                row[f"{window_name}_mean_pupil"]  = valid_pupil.mean()
                row[f"{window_name}_peak_pupil"]  = valid_pupil.max()

                peak_idx = valid_pupil.idxmax()
                row[f"{window_name}_peak_latency_s"] = (
                    time_s[peak_idx] - onset
                )

                # Linear slope — dilation velocity over window
                if len(valid_pupil) > 2:
                    t_rel  = win_time[valid_pupil.index] - onset
                    slope  = np.polyfit(t_rel, valid_pupil.values, 1)[0]
                    row[f"{window_name}_dilation_slope"] = slope
                else:
                    row[f"{window_name}_dilation_slope"] = np.nan
            else:
                for feat in ["mean_pupil", "peak_pupil",
                             "peak_latency_s", "dilation_slope"]:
                    row[f"{window_name}_{feat}"] = np.nan

            # ── Fixation features ──
            if len(fixation_df) > 0:
                win_fix = fixation_df[
                    (fixation_df["onset_s"] >= win_start) &
                    (fixation_df["offset_s"] <= win_end)
                ]
                row[f"{window_name}_n_fixations"]       = len(win_fix)
                row[f"{window_name}_mean_fix_duration"]  = (
                    win_fix["duration_ms"].mean() if len(win_fix) > 0 else np.nan
                )
                row[f"{window_name}_total_fix_time"]     = (
                    win_fix["duration_ms"].sum() if len(win_fix) > 0 else 0
                )
            else:
                row[f"{window_name}_n_fixations"]      = np.nan
                row[f"{window_name}_mean_fix_duration"] = np.nan
                row[f"{window_name}_total_fix_time"]    = np.nan

            # ── Saccade features ──
            if len(saccade_df) > 0:
                win_sacc = saccade_df[
                    (saccade_df["onset_s"] >= win_start) &
                    (saccade_df["offset_s"] <= win_end)
                ]
                row[f"{window_name}_n_saccades"]     = len(win_sacc)
                row[f"{window_name}_mean_sacc_amp"]  = (
                    win_sacc["amplitude_px"].mean() if len(win_sacc) > 0 else np.nan
                )
                row[f"{window_name}_mean_peak_vel"]  = (
                    win_sacc["peak_velocity"].mean() if len(win_sacc) > 0 else np.nan
                )

                # Latency to first saccade after event onset
                post_sacc = win_sacc[win_sacc["onset_s"] >= onset]
                row[f"{window_name}_first_sacc_latency_s"] = (
                    post_sacc["onset_s"].min() - onset
                    if len(post_sacc) > 0 else np.nan
                )
            else:
                for feat in ["n_saccades", "mean_sacc_amp",
                             "mean_peak_vel", "first_sacc_latency_s"]:
                    row[f"{window_name}_{feat}"] = np.nan

            # ── Blink features ──
            if len(blink_df) > 0:
                win_blinks = blink_df[
                    (blink_df["onset_s"] >= win_start) &
                    (blink_df["offset_s"] <= win_end)
                ]
                row[f"{window_name}_n_blinks"]       = len(win_blinks)
                row[f"{window_name}_mean_blink_dur"] = (
                    win_blinks["duration_ms"].mean() if len(win_blinks) > 0 else np.nan
                )
                row[f"{window_name}_blink_rate"]     = (
                    len(win_blinks) / win_dur_s * 60  # blinks per minute
                    if win_dur_s > 0 else np.nan
                )
            else:
                for feat in ["n_blinks", "mean_blink_dur", "blink_rate"]:
                    row[f"{window_name}_{feat}"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# ── Participant-level aggregation ──────────────────────────────────────────────

def aggregate_to_subject_means(trial_features):
    """
    Average per-trial eye features per condition.
    Flagged trials (high data loss) are excluded before averaging.
    Participant means are the correct statistical unit — see script 03.
    """
    # Identify flag columns and exclude flagged trials per window
    df = trial_features.copy()

    flag_cols = [c for c in df.columns if c.endswith("_flagged")]
    if flag_cols:
        # Exclude trial if ANY window is flagged
        any_flagged = df[flag_cols].any(axis=1)
        n_excluded  = any_flagged.sum()
        if n_excluded > 0:
            print(f"  Excluding {n_excluded} trials with excessive data loss.")
        df = df[~any_flagged]

    feature_cols  = [c for c in df.columns
                     if c not in ("trial", "condition", "onset_sec")
                     and not c.endswith("_flagged")]
    subject_means = df.groupby("condition")[feature_cols].mean()
    subject_sem   = df.groupby("condition")[feature_cols].sem()

    subject_means.columns = [f"{c}_mean" for c in subject_means.columns]
    subject_sem.columns   = [f"{c}_sem"  for c in subject_sem.columns]

    return pd.concat([subject_means, subject_sem], axis=1).reset_index()


# ── QC visualization ───────────────────────────────────────────────────────────

def plot_eye_summary(pupil_raw, pupil_clean, artifact_mask,
                     blink_df, subject_id, time_s, output_dir):
    """
    4-panel QC figure:
        Panel 1: Raw vs cleaned pupil (first 30s) with blink markers
        Panel 2: Artifact mask over full recording
        Panel 3: Blink duration distribution
        Panel 4: Pupil size distribution before/after cleaning
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f"Eye Tracking QC — {subject_id}", fontsize=13)
    t = time_s.values
    t_mask = t <= 30

    # Panel 1: Raw vs clean pupil
    axes[0].plot(t[t_mask], pupil_raw.values[t_mask],
                 color="lightgray", linewidth=0.8, label="Raw", alpha=0.9)
    axes[0].plot(t[t_mask], pupil_clean.values[t_mask],
                 color="steelblue", linewidth=1.0, label="Cleaned")
    if len(blink_df) > 0:
        for _, blink in blink_df[blink_df["onset_s"] <= 30].iterrows():
            axes[0].axvspan(blink["onset_s"], blink["offset_s"],
                            alpha=0.2, color="firebrick", label="_blink")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Pupil size (a.u.)")
    axes[0].set_title("Pupil signal — first 30 seconds")
    axes[0].legend(fontsize=9)

    # Panel 2: Artifact mask
    axes[1].fill_between(t, artifact_mask.values.astype(int),
                         color="firebrick", alpha=0.5, step="mid")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Artifact (1=yes)")
    axes[1].set_title(f"Artifact mask — {artifact_mask.mean()*100:.1f}% flagged")
    axes[1].set_ylim(-0.1, 1.3)

    # Panel 3: Blink duration distribution
    if len(blink_df) > 0:
        axes[2].hist(blink_df["duration_ms"], bins=20,
                     color="darkorange", edgecolor="white")
        axes[2].set_xlabel("Blink duration (ms)")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Blink duration distribution (n={len(blink_df)})")
    else:
        axes[2].text(0.5, 0.5, "No blinks detected",
                     ha="center", va="center", transform=axes[2].transAxes)

    # Panel 4: Pupil distributions
    axes[3].hist(pupil_raw.dropna().values, bins=40,
                 color="lightgray", alpha=0.7, label="Raw", density=True)
    axes[3].hist(pupil_clean.dropna().values, bins=40,
                 color="steelblue", alpha=0.7, label="Cleaned", density=True)
    axes[3].set_xlabel("Pupil size (a.u.)")
    axes[3].set_ylabel("Density")
    axes[3].set_title("Pupil size distribution")
    axes[3].legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{subject_id}_eye_report.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_subject(subject_id, config):
    """
    Full eye tracking pipeline for a single subject.
    """
    print(f"\n{'='*60}")
    print(f"Eye Tracking Processing: {subject_id}")
    print(f"{'='*60}")

    os.makedirs(config["output_dir"], exist_ok=True)
    base  = os.path.join(config["output_dir"], subject_id)
    sfreq = config["sfreq"]

    # 1. Load data
    df     = load_tobii_csv(subject_id, config)
    events = load_events(subject_id, config)
    time_s = df["time_s"]

    # 2. Extract binocular pupil
    pupil_raw = extract_pupil_signal(df, config)

    # 3. Blink detection
    artifact_mask, blink_df = detect_blinks(pupil_raw, time_s, config, sfreq)

    # 4. Additional artifact detection
    artifact_mask = detect_pupil_artifacts(pupil_raw, artifact_mask, config)

    # 5. Interpolate artifacts
    pupil_interp = interpolate_artifacts(pupil_raw, artifact_mask, time_s)

    # 6. Smooth
    pupil_smooth = smooth_pupil(pupil_interp, config, sfreq)

    # 7. Baseline correction
    pupil_clean = baseline_correct_pupil(pupil_smooth, time_s, events, config)

    # 8. Gaze event detection
    gaze_x = df[config["col_gaze_x_left"]].fillna(
              df[config["col_gaze_x_right"]])
    gaze_y = df[config["col_gaze_y_left"]].fillna(
              df[config["col_gaze_y_right"]])

    saccade_df = detect_saccades(gaze_x, gaze_y, time_s, config, sfreq)
    fixation_df = detect_fixations(gaze_x, gaze_y, time_s,
                                    saccade_df, config, sfreq)

    # 9. Save processed signals
    processed = pd.DataFrame({
        "time_s":        time_s,
        "pupil_raw":     pupil_raw,
        "pupil_clean":   pupil_clean,
        "artifact_mask": artifact_mask.astype(int),
    })
    processed.to_csv(f"{base}_pupil_clean.csv", index=False)

    events_out = pd.concat([
        saccade_df.assign(event_type="saccade"),
        fixation_df.assign(event_type="fixation"),
        blink_df.assign(event_type="blink"),
    ]).sort_values("onset_s").reset_index(drop=True)
    events_out.to_csv(f"{base}_events_detected.csv", index=False)
    print(f"  Saved: {base}_pupil_clean.csv")
    print(f"  Saved: {base}_events_detected.csv")

    # 10. Trial feature extraction
    trial_features = extract_trial_features(
        pupil_clean, time_s, saccade_df, fixation_df,
        blink_df, events, config, sfreq
    )
    trial_features.to_csv(f"{base}_trial_eye_features.csv", index=False)
    print(f"  Saved: {base}_trial_eye_features.csv")

    # 11. Participant-level aggregation
    subject_means = aggregate_to_subject_means(trial_features)
    subject_means.to_csv(f"{base}_subject_eye_means.csv", index=False)
    print(f"  Saved: {base}_subject_eye_means.csv")

    # 12. QC plot
    plot_eye_summary(pupil_raw, pupil_clean, artifact_mask,
                     blink_df, subject_id, time_s, config["output_dir"])

    print(f"\nDone: {subject_id}")
    return trial_features, subject_means


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subject_id = "sub-01"
    process_subject(subject_id, CONFIG)
