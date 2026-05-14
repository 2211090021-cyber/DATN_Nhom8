"""
=============================================================================
 Cardiovascular Complication Prediction — Model Training
=============================================================================
 Dataset  : data/processed/X_train_final.csv  (SMOTE + Scaled)
            data/processed/X_test_final.csv
            data/processed/y_train_final.csv
            data/processed/y_test_final.csv
 EDA Ref  : outputs/statistical_summary.csv   (feature selection guide)
 Models   : Logistic Regression, Random Forest, XGBoost
 Outputs  : models/          -> best model artifacts (.pkl) + feature list
             outputs/model_results/  -> metrics, ROC curves, Confusion Matrix
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import argparse
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    brier_score_loss, classification_report,
)
from sklearn.calibration     import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans"})


# =============================================================================
# 0.  PATHS & DIRECTORIES
# =============================================================================
parser = argparse.ArgumentParser(description="Cardiovascular Complication Prediction — Model Training")
parser.add_argument("--processed_dir", type=str, default=os.path.join("data", "processed"), help="Directory containing preprocessed data")
parser.add_argument("--stat_csv", type=str, default=os.path.join("outputs", "statistical_summary.csv"), help="EDA statistical summary CSV")
parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
parser.add_argument("--result_dir", type=str, default=os.path.join("outputs", "model_results"), help="Directory to save evaluation results")
args = parser.parse_args()

PROCESSED_DIR = args.processed_dir
STAT_CSV      = args.stat_csv
MODEL_DIR     = args.model_dir
RESULT_DIR    = args.result_dir

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Visual style ──────────────────────────────────────────────────────────
BG       = "#FAFBFD"
GRID_CLR = "#E4E8ED"
PALETTE  = {"Logistic Regression": "#4A90D9",
            "Random Forest":       "#27AE60",
            "XGBoost":             "#E67E22"}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.grid(color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C0C4CA")
    if title:  ax.set_title(title, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def save_fig(fig, name):
    path = os.path.join(RESULT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {path}")


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 1 — LOADING PREPROCESSED DATA")
print("=" * 70)

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_final.csv"))
X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_final.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train_final.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_final.csv")).squeeze()

print(f"  X_train : {X_train.shape}  |  y_train distribution: {dict(y_train.value_counts())}")
print(f"  X_test  : {X_test.shape}   |  y_test  distribution: {dict(y_test.value_counts())}")


# =============================================================================
# 2.  FEATURE SELECTION  (based on EDA statistical summary)
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 2 — FEATURE SELECTION (EDA-guided)")
print("=" * 70)

# ── 2a. Drop non-significant features (p >= 0.05) from EDA ───────────────
# These features showed no statistically significant difference between the
# complication vs no-complication groups in Section 3 of EDA.
stat_df = pd.read_csv(STAT_CSV)

# Collect non-significant original column names
nonsig_cols = stat_df.loc[~stat_df["Significant"], "OriginalCol"].tolist()
# Tang_huyet_ap is zero-variance (all patients = 1), force-add if not already
if "Tang_huyet_ap" not in nonsig_cols:
    nonsig_cols.append("Tang_huyet_ap")

# Only drop columns that actually exist in the dataset
cols_to_drop = [c for c in nonsig_cols if c in X_train.columns]

print(f"\n  Dropping {len(cols_to_drop)} non-significant / zero-variance features:")
for c in cols_to_drop:
    row = stat_df[stat_df["OriginalCol"] == c]
    pval = f"p={row['p-value'].values[0]:.4f}" if not row.empty else "zero-variance"
    print(f"    [-] {c:<25}  ({pval})")

X_train_sel = X_train.drop(columns=cols_to_drop)
X_test_sel  = X_test.drop(columns=cols_to_drop)

print(f"\n  Features before selection : {X_train.shape[1]}")
print(f"  Features after  selection : {X_train_sel.shape[1]}")

# ── 2b. Persist selected feature list for downstream use (Streamlit app) ──
selected_features = X_train_sel.columns.tolist()
feature_list_path = os.path.join(MODEL_DIR, "selected_features.json")
with open(feature_list_path, "w", encoding="utf-8") as f:
    json.dump(selected_features, f, indent=2)
print(f"\n  Selected features saved -> {feature_list_path}")
print(f"  Features: {selected_features}")


# =============================================================================
# 3.  MODEL DEFINITIONS + HYPERPARAMETER GRIDS
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 3 — MODEL DEFINITIONS & HYPERPARAMETER GRIDS")
print("=" * 70)

# Stratified K-Fold: ensures class ratio is preserved in every fold.
# Critical for imbalanced medical data (only ~5% complication cases).
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Primary optimization metric: F1-Score
# Medical context: We want a model that detects high-risk patients (Recall) 
# but doesn't produce too many false alarms (Precision). F1 balances both.
SCORING = "f1"

# ── Model 1: Logistic Regression ─────────────────────────────────────────
# penalty='l2'   : Ridge regularization prevents overfitting
# solver         : 'liblinear' supports L2, efficient for small-to-medium data
# class_weight   : 'balanced' auto-adjusts weights inversely to class freq
lr_model  = LogisticRegression(penalty="l2", solver="liblinear",
                               class_weight="balanced", max_iter=1000, random_state=42)
lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

# ── Model 2: Random Forest ────────────────────────────────────────────────
# class_weight='balanced_subsample': rebalances each bootstrap sample
rf_model  = RandomForestClassifier(class_weight="balanced_subsample",
                                   random_state=42, n_jobs=-1)
rf_params = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "max_features":    ["sqrt", "log2", 0.5], # Adjusted for standard tabular data
}

# ── Model 3: XGBoost ─────────────────────────────────────────────────────
# scale_pos_weight: ratio of negative to positive samples — compensates for
# class imbalance inside XGBoost's tree-building objective
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
spw   = round(n_neg / n_pos, 2)
print(f"\n  XGBoost scale_pos_weight = {n_neg}/{n_pos} = {spw}")

xgb_model  = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                            use_label_encoder=False, random_state=42, n_jobs=-1)
xgb_params = {
    "learning_rate":    [0.01, 0.05, 0.1], # Lower learning rates for harder problem
    "max_depth":        [3, 5, 7],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8], # Restored to higher values since dominant features are removed
    "gamma":            [0, 0.1, 1], # Reduced regularization to allow learning weak signals
}

MODELS = {
    "Logistic Regression": (lr_model,  lr_params),
    "Random Forest":       (rf_model,  rf_params),
    "XGBoost":             (xgb_model, xgb_params),
}


# =============================================================================
# 4.  TRAINING WITH GridSearchCV (5-Fold Stratified CV)
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 4 — GRIDSEARCHCV TRAINING (5-Fold Stratified CV)")
print("=" * 70)

best_estimators = {}   # Store best estimator per model name
cv_results_log  = []   # Store CV training summary per model

for model_name, (base_model, param_grid) in MODELS.items():
    print(f"\n  [ {model_name} ]")
    print(f"    Grid size : {np.prod([len(v) for v in param_grid.values()])} combinations")

    gs = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = CV,
        scoring    = SCORING,          # Optimize for F1-Score
        refit      = True,             # Refit on full train set with best params
        n_jobs     = -1,
        verbose    = 0,
    )
    gs.fit(X_train_sel, y_train)

    best_estimators[model_name] = gs.best_estimator_
    best_recall_cv = gs.best_score_

    print(f"    Best params       : {gs.best_params_}")
    print(f"    Best CV Recall    : {best_recall_cv:.4f}")

    cv_results_log.append({
        "Model":           model_name,
        "Best_Params":     str(gs.best_params_),
        "CV_Recall_Mean":  round(best_recall_cv, 4),
    })

    # ── Save best model to models/ directory ─────────────────────────────
    # Persist using joblib (more efficient than pickle for sklearn estimators)
    safe_name   = model_name.lower().replace(" ", "_")
    model_path  = os.path.join(MODEL_DIR, f"best_{safe_name}.pkl")
    joblib.dump(gs.best_estimator_, model_path)
    print(f"    Saved model       : {model_path}")


# =============================================================================
# 5.  EVALUATION ON TEST SET
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 5 — EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

metrics_rows = []
roc_data     = {}   # For ROC curve plot

for model_name, estimator in best_estimators.items():
    y_pred      = estimator.predict(X_test_sel)
    y_prob      = estimator.predict_proba(X_test_sel)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    brier   = brier_score_loss(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_data[model_name] = (fpr, tpr, auc)

    print(f"\n  [ {model_name} ]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}  <-- PRIMARY METRIC")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    Brier     : {brier:.4f}")
    print(f"\n    Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Complication', 'Complication'])}")

    metrics_rows.append({
        "Model":     model_name,
        "Accuracy":  round(acc,   4),
        "Precision": round(prec,  4),
        "Recall":    round(rec,   4),
        "F1-Score":  round(f1,    4),
        "ROC-AUC":   round(auc,   4),
        "Brier":     round(brier, 4),
    })

# ── Save metrics comparison table ────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_rows).sort_values(
    by=["F1-Score", "ROC-AUC", "Recall"], 
    ascending=[False, False, False]
)
metrics_csv = os.path.join(RESULT_DIR, "model_comparison.csv")
metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
print(f"\n  [Saved] Metrics table -> {metrics_csv}")
print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Brier':>10}")
print("  " + "-" * 85)
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']:<22} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1-Score']:>10.4f} {row['ROC-AUC']:>10.4f} {row['Brier']:>10.4f}")


# =============================================================================
# 6.  BEST MODEL SELECTION & ADDITIONAL SAVE
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 6 — BEST MODEL SELECTION")
print("=" * 70)

# Primary criterion: highest F1-Score. Tie-break: AUC then Recall
best_row   = metrics_df.iloc[0]
best_name  = best_row["Model"]
best_model = best_estimators[best_name]

print(f"\n  Best model overall  : {best_name}")
print(f"  Recall              : {best_row['Recall']:.4f}")
print(f"  AUC                 : {best_row['ROC-AUC']:.4f}")
print(f"  F1-Score            : {best_row['F1-Score']:.4f}")

# Save best model with a clear "champion" filename for the Streamlit app
champion_path = os.path.join(MODEL_DIR, "champion_model.pkl")
joblib.dump(best_model, champion_path)
print(f"\n  Champion model saved -> {champion_path}")

# Save champion model name for app reference
meta = {"champion": best_name, "metrics": best_row.to_dict()}
with open(os.path.join(MODEL_DIR, "champion_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)


# =============================================================================
# 7.  VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 7 — GENERATING EVALUATION PLOTS")
print("=" * 70)

# ── 7a. Confusion Matrices (one per model) ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig.suptitle("Confusion Matrix — All Models (Test Set)",
             fontsize=14, fontweight="bold")

for ax, (model_name, estimator) in zip(axes, best_estimators.items()):
    y_pred = estimator.predict(X_test_sel)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Comp.", "Comp."],
                yticklabels=["No Comp.", "Comp."],
                linewidths=0.5, linecolor="#DDD",
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(model_name, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig.tight_layout()
save_fig(fig, "01_confusion_matrices.png")

# ── 7b. ROC Curves (all models on same plot) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1, label="Random")

for model_name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, color=PALETTE[model_name], linewidth=2,
            label=f"{model_name}  (AUC = {auc:.3f})")

style_ax(ax, title="ROC Curves — Test Set",
         xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
save_fig(fig, "02_roc_curves.png")

# ── 7c. Calibration Curves + Brier Scores ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1, label="Perfectly calibrated")

for model_name, estimator in best_estimators.items():
    y_prob  = estimator.predict_proba(X_test_sel)[:, 1]
    brier   = brier_score_loss(y_test, y_prob)
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
            color=PALETTE[model_name],
            label=f"{model_name}  (Brier = {brier:.4f})")

style_ax(ax, title="Calibration Curves — Test Set",
         xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")
ax.legend(fontsize=10)
fig.tight_layout()
save_fig(fig, "03_calibration_curves.png")

# ── 7d. Metric comparison bar chart ──────────────────────────────────────
metric_cols  = ["Recall", "ROC-AUC", "F1-Score", "Precision", "Accuracy"]
x            = np.arange(len(metric_cols))
bar_width    = 0.25
model_names  = metrics_df["Model"].tolist()

fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
for i, mn in enumerate(model_names):
    row    = metrics_df[metrics_df["Model"] == mn].iloc[0]
    values = [row[m] for m in metric_cols]
    bars   = ax.bar(x + i * bar_width, values, bar_width,
                    color=PALETTE[mn], edgecolor="white",
                    linewidth=0.7, label=mn, alpha=0.9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
style_ax(ax, title="Model Comparison — Key Metrics (Test Set)", ylabel="Score")
# Annotate Recall bar as primary metric
ax.axvline(x=-0.15, color="#D94A4A", linestyle=":", linewidth=1.5, alpha=0.6)
fig.tight_layout()
save_fig(fig, "04_metric_comparison.png")

# ── 7e. Random Forest — Top feature importances ───────────────────────────
if "Random Forest" in best_estimators:
    rf_best   = best_estimators["Random Forest"]
    importances = pd.Series(rf_best.feature_importances_, index=selected_features)
    importances = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    ax.barh(importances.index, importances.values,
            color=PALETTE["Random Forest"], edgecolor="white", alpha=0.85)
    style_ax(ax, title="Random Forest — Top 15 Feature Importances",
             xlabel="Importance Score")
    fig.tight_layout()
    save_fig(fig, "05_rf_feature_importance.png")

# ── 7f. XGBoost — Top feature importances ────────────────────────────────
if "XGBoost" in best_estimators:
    xgb_best    = best_estimators["XGBoost"]
    importances = pd.Series(xgb_best.feature_importances_, index=selected_features)
    importances = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    ax.barh(importances.index, importances.values,
            color=PALETTE["XGBoost"], edgecolor="white", alpha=0.85)
    style_ax(ax, title="XGBoost — Top 15 Feature Importances",
             xlabel="Importance Score")
    fig.tight_layout()
    save_fig(fig, "06_xgb_feature_importance.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  Models trained           : {', '.join(MODELS.keys())}")
print(f"  Features used            : {len(selected_features)}")
print(f"  Champion model           : {best_name}")
print(f"  Champion Recall          : {best_row['Recall']:.4f}")
print(f"  Champion AUC             : {best_row['ROC-AUC']:.4f}")
print(f"\n  Saved models:")
for mname in MODELS:
    safe = mname.lower().replace(" ", "_")
    print(f"    -> models/best_{safe}.pkl")
print(f"    -> {champion_path}  (champion)")
print(f"    -> {feature_list_path}  (selected features)")
print(f"\n  Evaluation plots  : {os.path.abspath(RESULT_DIR)}/")
print(f"  Metrics CSV       : {metrics_csv}")
print("=" * 70 + "\n")
