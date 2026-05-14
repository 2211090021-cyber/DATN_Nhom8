"""
=============================================================================
 Cardiovascular Complication Prediction — Explainable AI (SHAP)
=============================================================================
 This script loads the Champion XGBoost model and generates SHAP explanations
 for both Global feature importance and Local case studies.
 Outputs are saved as high-resolution images for the project report.
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# =============================================================================
# 1. SETUP PATHS & UTILS
# =============================================================================
PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = os.path.join("models")
SHAP_OUT_DIR  = os.path.join("outputs", "shap_plots")

os.makedirs(SHAP_OUT_DIR, exist_ok=True)

def save_shap_fig(filename, width=10, height=6):
    """
    Utility function to save SHAP plots properly without cutting off text.
    SHAP automatically draws to the current matplotlib figure.
    """
    fig = plt.gcf()
    # Apply sizing
    fig.set_size_inches(width, height)
    path = os.path.join(SHAP_OUT_DIR, filename)
    # Save with tight bounding box
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 2. LOAD DATA & MODEL
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 1 — LOADING DATA AND CHAMPION MODEL")
print("=" * 70)

# Load the test set (represents unseen real-world patients)
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_final.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_final.csv")).squeeze()

# Load the selected features used during training
with open(os.path.join(MODEL_DIR, "selected_features.json"), "r", encoding="utf-8") as f:
    selected_features = json.load(f)

# Filter columns
X_test_sel = X_test[selected_features]

# Load Best XGBoost model for SHAP Analysis (XGBoost has best F1/AUC and natively supports TreeExplainer)
model_path = os.path.join(MODEL_DIR, "best_xgboost.pkl")
model = joblib.load(model_path)

print(f"  Model Loaded   : {model_path}")
print(f"  Test Data Shape: {X_test_sel.shape}")


# =============================================================================
# 3. INITIALIZE SHAP EXPLAINER
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 2 — CALCULATING SHAP VALUES (TreeExplainer)")
print("=" * 70)

# TreeExplainer is strictly for tree-based models (XGBoost, Random Forest).
# It's highly optimized and mathematically exact.
explainer = shap.TreeExplainer(model)

# For modern SHAP waterfall plots, we need the Explanation object
shap_explanation = explainer(X_test_sel)

# For older summary/dependence plots, we need just the raw values array.
# XGBoost binary classification returns a single array of log-odds.
shap_values_arr = shap_explanation.values

print("  SHAP computation completed successfully.")


# =============================================================================
# 4. GLOBAL EXPLANATIONS (For Population-level Report)
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 3 — GENERATING GLOBAL EXPLANATION PLOTS")
print("=" * 70)

# ── 4a. Summary Plot (Bar) - Feature Magnitude ───────────────────────────
# Shows the average absolute impact of each feature (regardless of direction)
shap.summary_plot(shap_values_arr, X_test_sel, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global Magnitude)", fontweight="bold")
save_shap_fig("01_shap_summary_bar.png")

# ── 4b. Summary Plot (Beeswarm) - Impact Direction ───────────────────────
# Shows how high/low values of a feature push the prediction (+/-)
shap.summary_plot(shap_values_arr, X_test_sel, show=False)
plt.title("SHAP Beeswarm Plot (Impact Direction)", fontweight="bold")
save_shap_fig("02_shap_summary_beeswarm.png")

# ── 4c. Dependence Plots for Top 2 Features ──────────────────────────────
# Identify the top 2 most important features globally
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1]
top_features = [selected_features[i] for i in top_idx[:2]]

print(f"  Top 2 features identified for Dependence Plot: {top_features}")

for feat in top_features:
    # Dependence plot shows the threshold effect of a specific feature
    shap.dependence_plot(feat, shap_values_arr, X_test_sel, show=False)
    plt.title(f"SHAP Dependence Plot: {feat}", fontweight="bold")
    save_shap_fig(f"03_shap_dependence_{feat}.png")


# =============================================================================
# 5. LOCAL EXPLANATIONS (Case Studies for Doctors)
# =============================================================================
print("\n" + "=" * 70)
print("  STEP 4 — GENERATING LOCAL EXPLANATION PLOTS (CASE STUDIES)")
print("=" * 70)

# Predict probabilities to select clear examples
probs = model.predict_proba(X_test_sel)[:, 1]

# ── Case 1: True Positive (High Risk correctly predicted) ────────────────
tp_indices = np.where((probs > 0.8) & (y_test == 1))[0]
if len(tp_indices) > 0:
    idx = tp_indices[0]
    print(f"  -> Case High Risk (Idx: {idx}): Actual=1, Pred_Prob={probs[idx]:.2%}")
    # Waterfall plot decomposes the specific prediction
    shap.plots.waterfall(shap_explanation[idx], show=False)
    # plt.title(f"Case Study: High Risk Prediction", fontweight="bold", y=1.05)
    save_shap_fig("04_case_study_high_risk.png", width=10, height=7)

# ── Case 2: True Negative (Low Risk correctly predicted) ─────────────────
tn_indices = np.where((probs < 0.1) & (y_test == 0))[0]
if len(tn_indices) > 0:
    idx = tn_indices[0]
    print(f"  -> Case Low Risk  (Idx: {idx}): Actual=0, Pred_Prob={probs[idx]:.2%}")
    shap.plots.waterfall(shap_explanation[idx], show=False)
    save_shap_fig("05_case_study_low_risk.png", width=10, height=7)

# ── Case 3: False Negative (Actual High Risk but model missed it) ────────
# This is crucial in medicine: analyzing WHY the model failed
fn_indices = np.where((probs < 0.3) & (y_test == 1))[0]
if len(fn_indices) > 0:
    idx = fn_indices[0]
    print(f"  -> Case False Neg (Idx: {idx}): Actual=1, Pred_Prob={probs[idx]:.2%} (Model Missed!)")
    shap.plots.waterfall(shap_explanation[idx], show=False)
    save_shap_fig("06_case_study_false_negative.png", width=10, height=7)

print("\n" + "=" * 70)
print("  SHAP ANALYSIS COMPLETE. Check outputs/shap_plots/")
print("=" * 70 + "\n")
