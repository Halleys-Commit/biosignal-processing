"""
08 — ML Dataset Construction and Model Evaluation
---------------------------------------------------
Builds a machine learning-ready dataset from the master feature file
output by 07_aggregate_features.py, and evaluates multiple models for
predicting trial-level behavioral performance from multimodal
physiological features.

Target variable:
    performance_score — continuous trial-level behavioral measure
    (e.g. reaction time, accuracy, composite score — replace placeholder
    with your paradigm's actual performance metric)

    Two prediction tasks:
        Regression:     predict raw performance_score
        Classification: predict high vs low performance
                        (binarized at within-participant median to
                        account for individual performance baselines)

Evaluation design:
    Leave-One-Subject-Out (LOSO) cross-validation:
        - Each fold trains on all trials from N-1 participants
        - Tests on the held-out participant's trials entirely
        - The held-out participant is genuinely unseen during training
        - This measures generalizability to new individuals —
          the relevant metric for any deployable cognitive monitoring system

    Final holdout set (20% of participants):
        - Reserved before any model development begins
        - Never used for model selection or hyperparameter decisions
        - Evaluated exactly once at the end to confirm generalization
        - Results on this set are the most honest performance estimate

    This design avoids the common error of pooling trials across
    participants and doing random train/test splits, which would allow
    the model to "see" a participant during training and then be tested
    on their other trials — inflating performance and producing a model
    that would fail on truly new users.

Models evaluated (regression and classification):
    - Logistic / Ridge Regression   (interpretable linear baseline)
    - Random Forest                 (nonlinear, built-in feature importance)
    - Gradient Boosting (XGBoost)   (typically best tabular performance)
    - Support Vector Machine        (RBF kernel, strong for physio features)
    - Multilayer Perceptron         (2-layer neural network)

    Note on neural networks: MLP performance is data-limited in typical
    cognitive research datasets. Included for completeness and comparison,
    but tree-based ensembles generally outperform at N < 1000 participants.

All models use sklearn Pipeline objects wrapping StandardScaler +
the model, ensuring the scaler is fitted on training data only and
never sees test data. This is non-negotiable for valid evaluation.

Input:
    master_trial_features.csv    — from 07_aggregate_features.py
    feature_summary_report.csv   — used to apply missingness filter

Output:
    loso_results_regression.csv      — per-fold and mean regression metrics
    loso_results_classification.csv  — per-fold and mean classification metrics
    holdout_results.csv              — final holdout performance (regression + classification)
    feature_importance.csv           — RF and XGBoost feature importances
    cross_modal_correlations.png     — heatmap of top feature correlations
    ml_summary_report.txt            — human-readable results summary

Dependencies:
    numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn
    pip install scikit-learn xgboost matplotlib seaborn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    balanced_accuracy_score, roc_auc_score, f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "input_dir":  "/path/to/pipeline_output",
    "output_dir": "/path/to/ml_results",

    # Target variable column name
    # Replace with your actual behavioral performance measure
    "target_col": "performance_score",

    # Final holdout — proportion of participants reserved before all modeling
    "holdout_proportion": 0.20,
    "random_seed":        42,

    # Feature preprocessing
    # Drop features with missingness above this threshold before ML
    # (separate from 07's reporting threshold — this actually drops)
    "max_missing_pct": 30.0,

    # Imputation strategy for remaining missing values
    # "median": impute with training set median (recommended)
    # "mean":   impute with training set mean
    "imputation": "median",

    # Optional PCA dimensionality reduction before modeling
    # Set to None to skip, or float (0-1) for variance explained threshold
    # e.g. 0.95 keeps components explaining 95% of variance
    "pca_variance": None,

    # Classification — binarization strategy
    # "median_split": high vs low relative to each participant's own median
    #                 (recommended — accounts for individual performance baselines)
    # "global_median": high vs low relative to group median
    "binarize_method": "median_split",

    # LOSO — minimum trials per participant to include in a fold
    "min_trials_per_fold": 10,

    # Feature importance — number of top features to report and plot
    "n_top_features": 20,

    # Cross-modal correlation — number of top features to include in heatmap
    "n_corr_features": 30,
}


# ── Model definitions ──────────────────────────────────────────────────────────

def build_regression_models(random_seed, pca_variance):
    """
    Build sklearn Pipeline objects for regression.
    Each pipeline: StandardScaler → [optional PCA] → model

    StandardScaler fitted on training data only inside each LOSO fold.
    This is enforced by the Pipeline API — the scaler never sees test data.

    XGBoost imported conditionally — falls back to GradientBoosting if
    xgboost is not installed.
    """
    steps_base = [("scaler", StandardScaler())]
    if pca_variance is not None:
        steps_base.append(("pca", PCA(n_components=pca_variance,
                                       random_state=random_seed)))

    def make_pipeline(model):
        return Pipeline(steps_base + [("model", model)])

    models = {
        "Ridge Regression": make_pipeline(
            Ridge(alpha=1.0)
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=200, max_depth=6,
                min_samples_leaf=5, random_state=random_seed, n_jobs=-1
            )
        ),
        "SVM (RBF)": make_pipeline(
            SVR(kernel="rbf", C=1.0, epsilon=0.1)
        ),
        "MLP": make_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu", max_iter=500,
                early_stopping=True, random_state=random_seed,
            )
        ),
    }

    # XGBoost — best tabular performance, handles missing values natively
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = make_pipeline(
            XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_seed, n_jobs=-1,
                eval_metric="rmse", verbosity=0,
            )
        )
    except ImportError:
        models["Gradient Boosting"] = make_pipeline(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=random_seed,
            )
        )

    return models


def build_classification_models(random_seed, pca_variance):
    """
    Build sklearn Pipeline objects for classification (high vs low performance).
    Same structure as regression models — directly comparable evaluation.
    """
    steps_base = [("scaler", StandardScaler())]
    if pca_variance is not None:
        steps_base.append(("pca", PCA(n_components=pca_variance,
                                       random_state=random_seed)))

    def make_pipeline(model):
        return Pipeline(steps_base + [("model", model)])

    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(C=1.0, max_iter=1000,
                               random_state=random_seed, n_jobs=-1)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=6,
                min_samples_leaf=5, random_state=random_seed,
                class_weight="balanced", n_jobs=-1,
            )
        ),
        "SVM (RBF)": make_pipeline(
            SVC(kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=random_seed)
        ),
        "MLP": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu", max_iter=500,
                early_stopping=True, random_state=random_seed,
            )
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_seed, n_jobs=-1,
                eval_metric="logloss", verbosity=0,
                use_label_encoder=False,
            )
        )
    except ImportError:
        models["Gradient Boosting"] = make_pipeline(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=random_seed,
            )
        )

    return models


# ── Data preparation ───────────────────────────────────────────────────────────

def load_and_prepare(config):
    """
    Load master feature file, apply quality filters, and prepare
    feature matrix X, continuous target y_reg, binary target y_clf,
    and participant group labels.

    Steps:
        1. Drop features with excessive missingness
        2. Drop near-constant features (std < 1e-6)
        3. Impute remaining missing values with training-set statistics
           (imputation parameters stored for application to test sets)
        4. Binarize performance score for classification
        5. Return X, y_reg, y_clf, groups, feature_names
    """
    master_path = os.path.join(config["input_dir"], "master_trial_features.csv")
    df          = pd.read_csv(master_path)

    print(f"Loaded: {len(df)} trials | "
          f"{df['subject_id'].nunique()} participants")

    target_col = config["target_col"]
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns[:10])}..."
        )

    # Drop trials with missing target
    df = df.dropna(subset=[target_col])
    print(f"After dropping missing targets: {len(df)} trials")

    # Identify feature columns
    non_feature_cols = ["subject_id", "trial", "condition", "onset_sec",
                        "performance_score", "reaction_time_ms",
                        "accuracy", "error_rate"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Drop high-missingness features
    missing_pcts = df[feature_cols].isna().mean() * 100
    keep_features = missing_pcts[
        missing_pcts <= config["max_missing_pct"]
    ].index.tolist()
    dropped = len(feature_cols) - len(keep_features)
    if dropped > 0:
        print(f"Dropped {dropped} features exceeding "
              f"{config['max_missing_pct']}% missingness threshold")
    feature_cols = keep_features

    # Drop near-constant features
    stds = df[feature_cols].std()
    keep_features = stds[stds > 1e-6].index.tolist()
    dropped_var = len(feature_cols) - len(keep_features)
    if dropped_var > 0:
        print(f"Dropped {dropped_var} near-constant features")
    feature_cols = keep_features

    print(f"Final feature count: {len(feature_cols)}")

    X      = df[feature_cols].values.astype(float)
    y_reg  = df[target_col].values.astype(float)
    groups = df["subject_id"].values

    # Binarize performance for classification
    y_clf = binarize_performance(y_reg, groups, config["binarize_method"])

    return X, y_reg, y_clf, groups, feature_cols, df


def binarize_performance(y, groups, method):
    """
    Convert continuous performance score to binary high/low labels.

    median_split (recommended):
        Each participant's trials are split at their own median.
        High performance = above own median (label 1)
        Low performance  = below own median (label 0)

        This accounts for the fact that participants differ in absolute
        performance level — a fast participant's slow trials may still
        be faster than a slow participant's fast trials. Splitting within
        participant asks "was this a good trial for this person?"
        which is the scientifically meaningful question.

    global_median:
        Split at the group-level median. Simpler but conflates
        individual differences with trial-level variability.
    """
    y_clf = np.zeros(len(y), dtype=int)

    if method == "median_split":
        for subj in np.unique(groups):
            mask   = groups == subj
            median = np.median(y[mask])
            y_clf[mask] = (y[mask] >= median).astype(int)
        print(f"  Binarized: within-participant median split | "
              f"class balance: {y_clf.mean()*100:.1f}% high performance")
    else:
        global_median = np.median(y)
        y_clf = (y >= global_median).astype(int)
        print(f"  Binarized: global median split | "
              f"class balance: {y_clf.mean()*100:.1f}% high performance")

    return y_clf


def impute_with_train_stats(X_train, X_test, method="median"):
    """
    Impute missing values using training set statistics only.
    Applying training statistics to the test set prevents data leakage
    from test distribution into the preprocessing step.
    """
    if method == "median":
        stats = np.nanmedian(X_train, axis=0)
    else:
        stats = np.nanmean(X_train, axis=0)

    # Replace NaN stats (all-missing features) with 0
    stats = np.where(np.isnan(stats), 0, stats)

    X_train_imp = np.where(np.isnan(X_train),
                            np.tile(stats, (X_train.shape[0], 1)), X_train)
    X_test_imp  = np.where(np.isnan(X_test),
                            np.tile(stats, (X_test.shape[0], 1)), X_test)

    return X_train_imp, X_test_imp


# ── Holdout split ──────────────────────────────────────────────────────────────

def split_holdout(X, y_reg, y_clf, groups, config):
    """
    Reserve a proportion of participants as a final holdout set.

    Crucially: the split is at the PARTICIPANT level, not trial level.
    All trials from a held-out participant are withheld together —
    they cannot appear in the LOSO training folds.

    The holdout set is set aside and not touched until the very end.
    """
    splitter = GroupShuffleSplit(
        n_splits      = 1,
        test_size     = config["holdout_proportion"],
        random_state  = config["random_seed"],
    )
    dev_idx, holdout_idx = next(splitter.split(X, y_reg, groups))

    X_dev,      X_hold      = X[dev_idx],      X[holdout_idx]
    y_reg_dev,  y_reg_hold  = y_reg[dev_idx],  y_reg[holdout_idx]
    y_clf_dev,  y_clf_hold  = y_clf[dev_idx],  y_clf[holdout_idx]
    groups_dev, groups_hold = groups[dev_idx], groups[holdout_idx]

    n_hold_subj = len(np.unique(groups_hold))
    n_dev_subj  = len(np.unique(groups_dev))
    print(f"\nHoldout split:")
    print(f"  Development set: {len(X_dev)} trials, {n_dev_subj} participants")
    print(f"  Holdout set:     {len(X_hold)} trials, {n_hold_subj} participants "
          f"(UNTOUCHED until final evaluation)")

    return (X_dev, y_reg_dev, y_clf_dev, groups_dev,
            X_hold, y_reg_hold, y_clf_hold, groups_hold)


# ── LOSO cross-validation ──────────────────────────────────────────────────────

def run_loso_regression(X, y, groups, models, config):
    """
    Leave-One-Subject-Out cross-validation for regression.

    For each participant in the development set:
        - Train all models on remaining participants' trials
        - Predict on held-out participant's trials
        - Evaluate: R², MAE, RMSE

    Returns a DataFrame of per-fold results and mean ± SD across folds.

    Note: R² per fold can be negative when a model performs worse than
    simply predicting the mean — this is meaningful information, not an error.
    Negative R² on individual participants is common and does not invalidate
    the approach; the mean across folds is the summary metric.
    """
    subjects    = np.unique(groups)
    all_results = {name: [] for name in models}

    print(f"\nLOSO Regression — {len(subjects)} folds")

    for subj in subjects:
        train_mask = groups != subj
        test_mask  = groups == subj

        if test_mask.sum() < config["min_trials_per_fold"]:
            print(f"  Skipping {subj}: only {test_mask.sum()} test trials")
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        X_train, X_test = impute_with_train_stats(
            X_train, X_test, config["imputation"]
        )

        for name, pipeline in models.items():
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                all_results[name].append({
                    "subject":  subj,
                    "model":    name,
                    "r2":       r2_score(y_test, y_pred),
                    "mae":      mean_absolute_error(y_test, y_pred),
                    "rmse":     np.sqrt(mean_squared_error(y_test, y_pred)),
                    "n_test":   len(y_test),
                })
            except Exception as e:
                print(f"  {name} failed on {subj}: {e}")

    # Compile results
    records = []
    for name, folds in all_results.items():
        if not folds:
            continue
        fold_df = pd.DataFrame(folds)
        records.append({
            "model":    name,
            "mean_r2":  fold_df["r2"].mean(),
            "std_r2":   fold_df["r2"].std(),
            "mean_mae": fold_df["mae"].mean(),
            "std_mae":  fold_df["mae"].std(),
            "mean_rmse":fold_df["rmse"].mean(),
            "std_rmse": fold_df["rmse"].std(),
            "n_folds":  len(fold_df),
        })

    results_df = pd.DataFrame(records).sort_values("mean_r2", ascending=False)

    print("\nLOSO Regression Results (mean ± SD across participants):")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<25} "
              f"R²={row['mean_r2']:+.3f}±{row['std_r2']:.3f}  "
              f"MAE={row['mean_mae']:.3f}±{row['std_mae']:.3f}")

    return results_df


def run_loso_classification(X, y, groups, models, config):
    """
    Leave-One-Subject-Out cross-validation for classification.

    Metrics:
        Balanced accuracy: mean recall per class — appropriate for
                           potentially imbalanced high/low splits
        AUC-ROC:           area under ROC curve — threshold-independent
        F1 (macro):        harmonic mean of precision and recall

    Chance level for balanced accuracy = 0.50.
    """
    subjects    = np.unique(groups)
    all_results = {name: [] for name in models}

    print(f"\nLOSO Classification — {len(subjects)} folds")

    for subj in subjects:
        train_mask = groups != subj
        test_mask  = groups == subj

        if test_mask.sum() < config["min_trials_per_fold"]:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f"  Skipping {subj}: only one class in test set")
            continue

        X_train, X_test = impute_with_train_stats(
            X_train, X_test, config["imputation"]
        )

        for name, pipeline in models.items():
            try:
                pipeline.fit(X_train, y_train)
                y_pred      = pipeline.predict(X_test)
                y_pred_prob = (pipeline.predict_proba(X_test)[:, 1]
                               if hasattr(pipeline.named_steps["model"],
                                          "predict_proba")
                               else y_pred.astype(float))

                all_results[name].append({
                    "subject":           subj,
                    "model":             name,
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "auc_roc":           roc_auc_score(y_test, y_pred_prob),
                    "f1_macro":          f1_score(y_test, y_pred,
                                                   average="macro",
                                                   zero_division=0),
                    "n_test": len(y_test),
                })
            except Exception as e:
                print(f"  {name} failed on {subj}: {e}")

    records = []
    for name, folds in all_results.items():
        if not folds:
            continue
        fold_df = pd.DataFrame(folds)
        records.append({
            "model":                name,
            "mean_balanced_acc":    fold_df["balanced_accuracy"].mean(),
            "std_balanced_acc":     fold_df["balanced_accuracy"].std(),
            "mean_auc_roc":         fold_df["auc_roc"].mean(),
            "std_auc_roc":          fold_df["auc_roc"].std(),
            "mean_f1_macro":        fold_df["f1_macro"].mean(),
            "std_f1_macro":         fold_df["f1_macro"].std(),
            "n_folds":              len(fold_df),
        })

    results_df = pd.DataFrame(records).sort_values(
        "mean_balanced_acc", ascending=False
    )

    print("\nLOSO Classification Results (chance = 0.50):")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<25} "
              f"BalAcc={row['mean_balanced_acc']:.3f}±{row['std_balanced_acc']:.3f}  "
              f"AUC={row['mean_auc_roc']:.3f}±{row['std_auc_roc']:.3f}")

    return results_df


# ── Final holdout evaluation ───────────────────────────────────────────────────

def evaluate_holdout(X_dev, y_reg_dev, y_clf_dev, groups_dev,
                     X_hold, y_reg_hold, y_clf_hold,
                     reg_models, clf_models, config):
    """
    Train best models on full development set, evaluate on holdout.

    This is the single honest performance estimate — run once,
    reported as-is. The holdout was not used for any previous decision.

    Trains each model on ALL development set trials, then predicts on
    the held-out participants' trials. No hyperparameter tuning here.
    """
    print("\n" + "="*60)
    print("FINAL HOLDOUT EVALUATION")
    print("="*60)
    print("(These participants were never seen during model development)")

    X_dev_imp,  X_hold_imp = impute_with_train_stats(
        X_dev, X_hold, config["imputation"]
    )

    records = []

    # Regression
    for name, pipeline in reg_models.items():
        try:
            pipeline.fit(X_dev_imp, y_reg_dev)
            y_pred = pipeline.predict(X_hold_imp)
            records.append({
                "model":  name, "task": "regression",
                "r2":     r2_score(y_reg_hold, y_pred),
                "mae":    mean_absolute_error(y_reg_hold, y_pred),
                "rmse":   np.sqrt(mean_squared_error(y_reg_hold, y_pred)),
            })
            print(f"  [Regression]  {name:<25} "
                  f"R²={records[-1]['r2']:+.3f}  "
                  f"MAE={records[-1]['mae']:.3f}")
        except Exception as e:
            print(f"  {name} holdout failed: {e}")

    # Classification
    for name, pipeline in clf_models.items():
        try:
            pipeline.fit(X_dev_imp, y_clf_dev)
            y_pred = pipeline.predict(X_hold_imp)
            y_prob = (pipeline.predict_proba(X_hold_imp)[:, 1]
                      if hasattr(pipeline.named_steps["model"], "predict_proba")
                      else y_pred.astype(float))
            records.append({
                "model":             name, "task": "classification",
                "balanced_accuracy": balanced_accuracy_score(y_clf_hold, y_pred),
                "auc_roc":           roc_auc_score(y_clf_hold, y_prob),
                "f1_macro":          f1_score(y_clf_hold, y_pred,
                                               average="macro", zero_division=0),
            })
            print(f"  [Classification] {name:<23} "
                  f"BalAcc={records[-1]['balanced_accuracy']:.3f}  "
                  f"AUC={records[-1]['auc_roc']:.3f}")
        except Exception as e:
            print(f"  {name} holdout failed: {e}")

    return pd.DataFrame(records)


# ── Feature importance ─────────────────────────────────────────────────────────

def extract_feature_importance(X_dev, y_reg_dev, groups_dev,
                                feature_names, config):
    """
    Extract feature importances from tree-based models trained on
    full development set.

    Random Forest: mean decrease in impurity (Gini importance)
    XGBoost:       gain-based importance

    Both are normalized to sum to 1 for comparability.
    Top N features reported and saved.
    """
    X_imp, _ = impute_with_train_stats(X_dev, X_dev, config["imputation"])

    importances = {}

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=config["random_seed"], n_jobs=-1
    )
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    rf.fit(X_scaled, y_reg_dev)
    importances["Random Forest"] = rf.feature_importances_

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=config["random_seed"], verbosity=0, n_jobs=-1
        )
        xgb.fit(X_scaled, y_reg_dev)
        importances["XGBoost"] = xgb.feature_importances_
    except ImportError:
        pass

    # Build importance DataFrame
    imp_df = pd.DataFrame(importances, index=feature_names)
    imp_df["mean_importance"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean_importance", ascending=False)

    top_n = imp_df.head(config["n_top_features"])

    print(f"\nTop {config['n_top_features']} features by mean importance:")
    for feat, row in top_n.iterrows():
        modality = feat.split("_")[0] if "_" in feat else "unknown"
        print(f"  {feat:<50} {row['mean_importance']:.4f} [{modality}]")

    return imp_df


# ── Cross-modal correlation heatmap ───────────────────────────────────────────

def plot_cross_modal_correlations(X_dev, feature_names, y_reg_dev,
                                   imp_df, config, output_dir):
    """
    Correlation heatmap of top features across modalities.

    Shows how features from different physiological signals co-vary,
    which is scientifically meaningful — physiological systems are
    coupled, and this coupling is part of what the ML models exploit.

    Also includes the target variable (performance_score) to show
    which features correlate most with behavioral outcomes.
    """
    X_imp, _ = impute_with_train_stats(X_dev, X_dev, config["imputation"])

    top_features = imp_df.head(config["n_corr_features"]).index.tolist()
    feat_indices = [list(feature_names).index(f) for f in top_features
                    if f in feature_names]

    if not feat_indices:
        print("  Skipping correlation heatmap — no matching features.")
        return

    X_top  = X_imp[:, feat_indices]
    df_top = pd.DataFrame(X_top, columns=top_features)
    df_top["performance_score"] = y_reg_dev

    corr = df_top.corr()

    # Color-code feature labels by modality
    modality_colors = {
        "eeg":  "#4C72B0",
        "hr":   "#DD8452",
        "rsp":  "#55A868",
        "eye":  "#C44E52",
        "performance": "#8172B3",
    }

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True,
        linewidths=0.3, cbar_kws={"shrink": 0.6},
        annot=False,
    )
    ax.set_title("Cross-modal feature correlations — top features + performance",
                 fontsize=12, pad=12)

    # Color tick labels by modality
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        text     = label.get_text()
        modality = text.split("_")[0] if "_" in text else "other"
        color    = modality_colors.get(modality, "#333333")
        label.set_color(color)
        label.set_fontsize(8)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Legend for modality colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=m.upper())
        for m, c in modality_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.18, 1.02), title="Modality", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cross_modal_correlations.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


# ── Summary report ─────────────────────────────────────────────────────────────

def write_summary_report(loso_reg, loso_clf, holdout_df,
                          n_subjects, n_trials, n_features, output_dir):
    """
    Write a human-readable plain text summary of all results.
    """
    path = os.path.join(output_dir, "ml_summary_report.txt")

    with open(path, "w") as f:
        f.write("MULTIMODAL PHYSIOLOGICAL SIGNAL DECODING\n")
        f.write("Predicting Behavioral Performance from EEG + Physio Features\n")
        f.write("="*65 + "\n\n")
        f.write(f"Dataset: {n_trials} trials | {n_subjects} participants | "
                f"{n_features} features\n")
        f.write(f"Evaluation: Leave-One-Subject-Out (LOSO) cross-validation\n")
        f.write(f"Target: performance_score (trial-level behavioral measure)\n\n")

        f.write("LOSO REGRESSION (predict continuous performance score)\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Model':<25} {'R² (mean±SD)':<20} {'MAE (mean±SD)'}\n")
        for _, row in loso_reg.iterrows():
            f.write(f"{row['model']:<25} "
                    f"{row['mean_r2']:+.3f}±{row['std_r2']:.3f}        "
                    f"{row['mean_mae']:.3f}±{row['std_mae']:.3f}\n")

        f.write("\nLOSO CLASSIFICATION (high vs low performance, chance=0.50)\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Model':<25} {'BalAcc (mean±SD)':<22} {'AUC (mean±SD)'}\n")
        for _, row in loso_clf.iterrows():
            f.write(f"{row['model']:<25} "
                    f"{row['mean_balanced_acc']:.3f}±{row['std_balanced_acc']:.3f}          "
                    f"{row['mean_auc_roc']:.3f}±{row['std_auc_roc']:.3f}\n")

        f.write("\nFINAL HOLDOUT EVALUATION (unseen participants)\n")
        f.write("-"*50 + "\n")
        for _, row in holdout_df.iterrows():
            if row["task"] == "regression":
                f.write(f"{row['model']:<25} [regression]     "
                        f"R²={row['r2']:+.3f}  MAE={row['mae']:.3f}\n")
            else:
                f.write(f"{row['model']:<25} [classification] "
                        f"BalAcc={row['balanced_accuracy']:.3f}  "
                        f"AUC={row['auc_roc']:.3f}\n")

    print(f"Saved: {path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ML Dataset Construction and Model Evaluation")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Load and prepare data
    X, y_reg, y_clf, groups, feature_names, df = load_and_prepare(CONFIG)

    # 2. Split holdout — set aside before any modeling
    (X_dev, y_reg_dev, y_clf_dev, groups_dev,
     X_hold, y_reg_hold, y_clf_hold, groups_hold) = split_holdout(
        X, y_reg, y_clf, groups, CONFIG
    )

    # 3. Build model sets
    reg_models = build_regression_models(
        CONFIG["random_seed"], CONFIG["pca_variance"]
    )
    clf_models = build_classification_models(
        CONFIG["random_seed"], CONFIG["pca_variance"]
    )

    # 4. LOSO cross-validation on development set
    loso_reg = run_loso_regression(
        X_dev, y_reg_dev, groups_dev, reg_models, CONFIG
    )
    loso_clf = run_loso_classification(
        X_dev, y_clf_dev, groups_dev, clf_models, CONFIG
    )

    # 5. Final holdout evaluation
    holdout_df = evaluate_holdout(
        X_dev, y_reg_dev, y_clf_dev, groups_dev,
        X_hold, y_reg_hold, y_clf_hold,
        reg_models, clf_models, CONFIG
    )

    # 6. Feature importance
    imp_df = extract_feature_importance(
        X_dev, y_reg_dev, groups_dev, feature_names, CONFIG
    )

    # 7. Cross-modal correlation heatmap
    plot_cross_modal_correlations(
        X_dev, feature_names, y_reg_dev, imp_df, CONFIG, CONFIG["output_dir"]
    )

    # 8. Save all results
    out = CONFIG["output_dir"]
    loso_reg.to_csv(os.path.join(out, "loso_results_regression.csv"),    index=False)
    loso_clf.to_csv(os.path.join(out, "loso_results_classification.csv"), index=False)
    holdout_df.to_csv(os.path.join(out, "holdout_results.csv"),           index=False)
    imp_df.to_csv(os.path.join(out, "feature_importance.csv"))

    write_summary_report(
        loso_reg, loso_clf, holdout_df,
        n_subjects = len(np.unique(groups)),
        n_trials   = len(X),
        n_features = len(feature_names),
        output_dir = out,
    )

    print("\nAll done. Results saved to:", out)


if __name__ == "__main__":
    main()
