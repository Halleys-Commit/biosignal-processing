"""
07 — Aggregate Features Across Participants
--------------------------------------------
Loops all participants, loads per-modality feature files output by
scripts 01-06, and assembles a single master feature DataFrame.

Each row in the output represents one trial from one participant.
Participant ID is retained as a column to support both within-person
and cross-participant (LOSO) model evaluation in script 08.

This script is deliberately simple — it collects and aligns, it does
not transform. Feature engineering, scaling, and selection happen in
script 08 or downstream ML scripts.

Input (per participant, per modality):
    {subject_id}_trial_hr_features.csv      — from 03_heart_rate_processing.py
    {subject_id}_trial_rsp_features.csv     — from 04_respiration_processing.py
    {subject_id}_trial_eye_features.csv     — from 06_eye_tracking_processing.py
    {subject_id}_trial_features.csv         — from 02_eeg_feature_extraction.py

    Optional behavioral file:
    {subject_id}_behavior.csv               — trial-level performance scores
        Required columns: trial (int), performance_score (float)
        Optional columns: condition, reaction_time_ms, accuracy (0/1)

Output:
    master_trial_features.csv    — all participants, all modalities, all trials
    feature_summary_report.csv   — missingness and variance per feature
    participant_summary.csv      — trial counts and data quality per participant

Dependencies:
    numpy, pandas
"""

import os
import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Root directory containing per-subject feature folders
    "features_dir":  "/path/to/features",

    # Output
    "output_dir":    "/path/to/pipeline_output",

    # Subject list — either explicit list or auto-discover from folder names
    # Set to None to auto-discover all sub-* folders in features_dir
    "subject_list":  None,   # e.g. ["sub-01", "sub-02", "sub-03"]

    # Subfolder names within each subject's feature directory
    # Adjust if your folder structure differs
    "modality_subdirs": {
        "eeg":     "eeg",
        "hr":      "physio",
        "rsp":     "physio",
        "eye":     "eyetracking",
        "behavior":"behavior",
    },

    # Feature file suffixes (appended to {subject_id})
    "feature_files": {
        "eeg":      "_trial_features.csv",
        "hr":       "_trial_hr_features.csv",
        "rsp":      "_trial_rsp_features.csv",
        "eye":      "_trial_eye_features.csv",
        "behavior": "_behavior.csv",
    },

    # Columns used as merge keys — must exist in all modality files
    "merge_keys": ["trial", "condition"],

    # Maximum acceptable missing data proportion per feature (for reporting only)
    # Features above this threshold are flagged in the summary report
    # — they are NOT dropped here; that decision belongs to the ML pipeline
    "missing_flag_threshold": 0.20,

    # Minimum trials per participant to include in master dataset
    # Participants below this are excluded with a warning
    "min_trials_per_subject": 20,
}


# ── Subject discovery ──────────────────────────────────────────────────────────

def discover_subjects(config):
    """
    Return list of subject IDs to process.
    If subject_list is specified in CONFIG, use that.
    Otherwise auto-discover by finding sub-* directories in features_dir.
    """
    if config["subject_list"] is not None:
        return config["subject_list"]

    features_dir = config["features_dir"]
    subjects = sorted([
        d for d in os.listdir(features_dir)
        if os.path.isdir(os.path.join(features_dir, d))
        and d.startswith("sub-")
    ])
    print(f"Auto-discovered {len(subjects)} subjects: {subjects}")
    return subjects


# ── Per-subject loading ────────────────────────────────────────────────────────

def load_modality(subject_id, modality, config):
    """
    Load a single modality's trial feature file for one participant.
    Returns DataFrame or None if file not found.
    """
    subdir   = config["modality_subdirs"][modality]
    suffix   = config["feature_files"][modality]
    filepath = os.path.join(
        config["features_dir"], subject_id, subdir,
        f"{subject_id}{suffix}"
    )

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    return df


def merge_subject_modalities(subject_id, config):
    """
    Load and merge all modality feature files for one participant.

    Merge strategy:
        - All modalities are outer-merged on [trial, condition]
        - Missing modalities produce NaN columns for that participant
        - Participant ID is added as a column for LOSO cross-validation

    Returns a DataFrame with one row per trial, or None if no data found.
    """
    merge_keys = config["merge_keys"]
    merged     = None
    loaded     = []

    for modality in ["eeg", "hr", "rsp", "eye"]:
        df = load_modality(subject_id, modality, config)
        if df is None:
            print(f"    {modality}: not found — will be NaN for {subject_id}")
            continue

        # Drop columns already in merge_keys or that would duplicate
        # (e.g. onset_sec appears in multiple modality files)
        extra_drop = [c for c in ["onset_sec"] if c in df.columns
                      and c not in merge_keys]
        df = df.drop(columns=extra_drop, errors="ignore")

        # Prefix feature columns with modality name to avoid collisions
        rename = {
            c: f"{modality}_{c}"
            for c in df.columns
            if c not in merge_keys
        }
        df = df.rename(columns=rename)

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=merge_keys, how="outer")

        loaded.append(modality)

    if merged is None:
        print(f"  {subject_id}: no feature files found — skipping.")
        return None

    # Add behavioral performance if available
    behavior_df = load_modality(subject_id, "behavior", config)
    if behavior_df is not None:
        beh_cols = [c for c in behavior_df.columns
                    if c in merge_keys or c in [
                        "performance_score", "reaction_time_ms",
                        "accuracy", "error_rate"
                    ]]
        merged = pd.merge(merged, behavior_df[beh_cols],
                          on=merge_keys, how="left")
        loaded.append("behavior")
    else:
        print(f"    behavior: not found — performance_score will be NaN")
        merged["performance_score"] = np.nan

    merged["subject_id"] = subject_id
    print(f"  {subject_id}: {len(merged)} trials | "
          f"modalities loaded: {loaded}")

    return merged


# ── Quality reporting ──────────────────────────────────────────────────────────

def build_feature_summary(master_df, config):
    """
    Per-feature quality summary for review before running ML.

    Reports:
        - Missing data proportion across all trials
        - Near-zero variance flag (std < 0.001 of mean)
        - Flag for features exceeding missing_flag_threshold

    This report is for human review — no features are dropped here.
    Dropping decisions belong in the ML pipeline where the analyst
    can make informed choices about imputation vs exclusion.
    """
    meta_cols    = ["subject_id", "trial", "condition",
                    "onset_sec", "performance_score",
                    "reaction_time_ms", "accuracy"]
    feature_cols = [c for c in master_df.columns if c not in meta_cols]

    records = []
    for col in feature_cols:
        vals        = master_df[col]
        missing_pct = vals.isna().mean()
        valid_vals  = vals.dropna()
        std_val     = valid_vals.std() if len(valid_vals) > 1 else np.nan
        mean_val    = valid_vals.mean() if len(valid_vals) > 0 else np.nan

        near_zero_var = (
            abs(std_val) < 0.001 * abs(mean_val)
            if mean_val != 0 and not np.isnan(std_val) else False
        )

        records.append({
            "feature":          col,
            "missing_pct":      round(missing_pct * 100, 1),
            "mean":             round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
            "std":              round(std_val, 4) if not np.isnan(std_val) else np.nan,
            "flag_missing":     missing_pct > config["missing_flag_threshold"],
            "flag_low_variance": near_zero_var,
        })

    summary = pd.DataFrame(records).sort_values("missing_pct", ascending=False)

    n_flagged_missing = summary["flag_missing"].sum()
    n_flagged_var     = summary["flag_low_variance"].sum()

    print(f"\nFeature summary: {len(feature_cols)} features total")
    print(f"  Flagged for high missingness (>{config['missing_flag_threshold']*100:.0f}%): "
          f"{n_flagged_missing}")
    print(f"  Flagged for near-zero variance: {n_flagged_var}")
    print(f"  (Flagged features are NOT dropped — review before ML)")

    return summary


def build_participant_summary(master_df):
    """
    Per-participant data quality summary.
    """
    records = []
    for subj, grp in master_df.groupby("subject_id"):
        n_trials    = len(grp)
        n_conditions = grp["condition"].nunique() if "condition" in grp.columns else np.nan
        overall_missing = grp.drop(
            columns=["subject_id", "trial", "condition"], errors="ignore"
        ).isna().mean().mean() * 100

        records.append({
            "subject_id":       subj,
            "n_trials":         n_trials,
            "n_conditions":     n_conditions,
            "mean_missing_pct": round(overall_missing, 1),
        })

    return pd.DataFrame(records)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Feature Aggregation — All Participants")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    subjects = discover_subjects(CONFIG)

    all_dfs  = []
    excluded = []

    for subject_id in subjects:
        print(f"\nProcessing: {subject_id}")
        subject_df = merge_subject_modalities(subject_id, CONFIG)

        if subject_df is None:
            excluded.append(subject_id)
            continue

        if len(subject_df) < CONFIG["min_trials_per_subject"]:
            print(f"  WARNING: {subject_id} has only {len(subject_df)} trials "
                  f"(minimum: {CONFIG['min_trials_per_subject']}) — excluded.")
            excluded.append(subject_id)
            continue

        all_dfs.append(subject_df)

    if not all_dfs:
        print("\nNo valid participant data found. Check paths in CONFIG.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder: metadata columns first, then features
    meta_first = [c for c in ["subject_id", "trial", "condition",
                               "onset_sec", "performance_score",
                               "reaction_time_ms", "accuracy"]
                  if c in master_df.columns]
    other_cols = [c for c in master_df.columns if c not in meta_first]
    master_df  = master_df[meta_first + other_cols]

    print(f"\n{'='*60}")
    print(f"Master dataset: {len(master_df)} trials | "
          f"{master_df['subject_id'].nunique()} participants | "
          f"{len(other_cols)} features")
    if excluded:
        print(f"Excluded participants: {excluded}")

    # Save
    master_path = os.path.join(CONFIG["output_dir"], "master_trial_features.csv")
    master_df.to_csv(master_path, index=False)
    print(f"\nSaved: {master_path}")

    # Quality reports
    feature_summary = build_feature_summary(master_df, CONFIG)
    participant_summary = build_participant_summary(master_df)

    feature_summary.to_csv(
        os.path.join(CONFIG["output_dir"], "feature_summary_report.csv"),
        index=False
    )
    participant_summary.to_csv(
        os.path.join(CONFIG["output_dir"], "participant_summary.csv"),
        index=False
    )
    print(f"Saved: feature_summary_report.csv")
    print(f"Saved: participant_summary.csv")

    print("\nDone. Review feature_summary_report.csv before running script 08.")
    return master_df


if __name__ == "__main__":
    main()
