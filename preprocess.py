"""
=============================================================================
 Cardiovascular Complication Prediction — Clinical Preprocessing Pipeline
=============================================================================
 Dataset  : docs/YCDL_Features_Mapped.xlsx
 Target   : 'Target'  (0 = no complication, 1 = complication)
 Output   : data/processed/
               ├── X_train_final.csv
               ├── y_train_final.csv
               ├── X_test_final.csv
               └── y_test_final.csv
=============================================================================
"""

import os
import argparse
import warnings
import logging
import joblib

# ── Third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# 0.  CONSTANTS / CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser(description="Preprocess clinical features for ML.")
parser.add_argument("--input", type=str, default="data/preprocess/YCDL_Features_Mapped.xlsx", help="Input file path")
parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
args = parser.parse_args()

EXCEL_PATH   = args.input
OUTPUT_DIR   = args.output_dir
MODEL_DIR    = args.model_dir
TARGET_COL   = "Target"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MISSING_THRESHOLD = 0.60          # drop columns with > 60 % missing
N_CV_FOLDS   = 5
IQR_MULT     = 1.5

# Columns that are identifiers / administrative — not predictive features
# Drop these before modelling so they never pollute the feature set.
ID_COLUMNS = [
    "TenBenhNhan",          # patient name
    "SoVaoVien",            # admission number
    "NamSinh",              # birth year (Age is derived from this)
    "MaICD Bn nội trú",     # ICD code – inpatient (free text, 70.6 % missing)
    "MaICD BN ngoại trú",   # ICD code – outpatient (free text)
]

# Binary (0/1) medical-history flags — already integer, but tracked here for
# explicit LabelEncoder pass so the pipeline stays self-documenting.
BINARY_COLS = [
    "GioiTinh",        # Gender: 'Nam'/'Nữ'  → will be label-encoded
    "dai_thao_duong",  # diabetes mellitus
    "rl_lipid_mau",    # dyslipidaemia
    "suy_than_man",    # chronic kidney disease
    "benh_mach_vanh",  # coronary artery disease
    "Tang_huyet_ap",   # hypertension
]

# Continuous laboratory indicators — IQR capping + StandardScaler
CONTINUOUS_COLS = [
    "HBa1C_mean", "HBa1C_max", "HBa1C_min", "HBa1C_std", "HBa1C_last",
    "LDL_mean",   "LDL_max",   "LDL_min",   "LDL_std",   "LDL_last",
    "HDL_mean",   "HDL_max",   "HDL_min",   "HDL_std",   "HDL_last",
    "Triglycerid_mean", "Triglycerid_max", "Triglycerid_min",
    "Triglycerid_std",  "Triglycerid_last",
    "Age",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# 1.  LOAD DATA
# =============================================================================

log.info("Loading dataset …")
df_raw = pd.read_excel(EXCEL_PATH)
log.info(f"  Raw shape: {df_raw.shape}  (rows × cols)")
log.info(f"  Target distribution:\n{df_raw[TARGET_COL].value_counts().to_string()}")


# =============================================================================
# 2.  DROP ADMINISTRATIVE / ID COLUMNS
# =============================================================================

log.info("Dropping administrative / identifier columns …")
df = df_raw.drop(columns=[c for c in ID_COLUMNS if c in df_raw.columns])
log.info(f"  Shape after ID-column removal: {df.shape}")


# =============================================================================
# 3.  SEPARATE FEATURES AND TARGET
# =============================================================================

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# =============================================================================
# 4.  STRATIFIED TRAIN / TEST SPLIT  (80 / 20, stratify on y)
# =============================================================================

log.info("Stratified 80/20 train/test split …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
log.info(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
log.info(
    f"  Train class ratio → "
    f"0: {(y_train == 0).sum()}  |  1: {(y_train == 1).sum()}"
)
log.info(
    f"  Test  class ratio → "
    f"0: {(y_test == 0).sum()}   |  1: {(y_test == 1).sum()}"
)


# =============================================================================
# 5.  MISSING-DATA HANDLING
# =============================================================================

# ── 5a. Drop columns with > 60 % missing (fitted on training subset only) ──
log.info(f"Assessing missing ratio (threshold = {MISSING_THRESHOLD*100:.0f} %) …")
train_miss_ratio = X_train.isnull().mean()
cols_to_drop = train_miss_ratio[train_miss_ratio > MISSING_THRESHOLD].index.tolist()

if cols_to_drop:
    log.info(f"  Dropping {len(cols_to_drop)} column(s): {cols_to_drop}")
else:
    log.info("  No columns exceed the missing threshold — none dropped.")

X_train = X_train.drop(columns=cols_to_drop)
X_test  = X_test.drop(columns=cols_to_drop)

# Update column lists after possible drops
remaining_continuous = [c for c in CONTINUOUS_COLS if c in X_train.columns]
remaining_binary     = [c for c in BINARY_COLS     if c in X_train.columns]

# ── 5b. Label-encode the Gender column BEFORE imputation ──────────────────
#        (SimpleImputer works on numerics; encoding first makes this seamless)
gender_col = "GioiTinh"
label_encoders: dict[str, LabelEncoder] = {}

if gender_col in X_train.columns:
    log.info(f"  Label-encoding '{gender_col}' …")
    le = LabelEncoder()
    # Fit only on non-null training values
    train_non_null_mask = X_train[gender_col].notna()
    le.fit(X_train.loc[train_non_null_mask, gender_col])
    label_encoders[gender_col] = le

    # Transform (handle NaN separately — keep as NaN for imputer)
    def _safe_le_transform(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        result = series.copy().astype(object)
        mask = series.notna()
        result[mask] = encoder.transform(series[mask])
        return pd.to_numeric(result, errors="coerce")

    X_train[gender_col] = _safe_le_transform(X_train[gender_col], le)
    X_test[gender_col]  = _safe_le_transform(X_test[gender_col],  le)

# ── 5c. Numerical imputation with Median (fit on train only) ──────────────
log.info("  Numerical imputation (Median) …")
num_imputer = SimpleImputer(strategy="median")
num_imputer.fit(X_train[remaining_continuous])

X_train[remaining_continuous] = num_imputer.transform(X_train[remaining_continuous])
X_test[remaining_continuous]  = num_imputer.transform(X_test[remaining_continuous])

# ── 5d. Categorical / binary imputation with Mode (fit on train only) ─────
#        The medical-history flags are already integers (0/1) with no missing,
#        but we run the imputer defensively for pipeline robustness.
log.info("  Categorical/binary imputation (Mode) …")
cat_imputer = SimpleImputer(strategy="most_frequent")
cat_imputer.fit(X_train[remaining_binary])

X_train[remaining_binary] = cat_imputer.transform(X_train[remaining_binary])
X_test[remaining_binary]  = cat_imputer.transform(X_test[remaining_binary])

log.info(
    f"  Post-imputation missing — train: {X_train.isnull().sum().sum()}  "
    f"| test: {X_test.isnull().sum().sum()}"
)


# =============================================================================
# 6.  OUTLIER HANDLING — IQR CAPPING (Winsorization)
#     Calculated on the TRAINING set; applied to both train and test.
# =============================================================================

log.info("IQR capping (Winsorization) on continuous laboratory indicators …")

iqr_bounds: dict[str, tuple[float, float]] = {}

for col in remaining_continuous:
    q1  = X_train[col].quantile(0.25)
    q3  = X_train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULT * iqr
    upper = q3 + IQR_MULT * iqr
    iqr_bounds[col] = (lower, upper)

    X_train[col] = X_train[col].clip(lower=lower, upper=upper)
    X_test[col]  = X_test[col].clip(lower=lower, upper=upper)

log.info(f"  IQR capping applied to {len(remaining_continuous)} feature(s).")


# =============================================================================
# 7.  FEATURE ENCODING
#     • One-Hot Encoding  — non-ordered categoricals (none in this dataset
#       after dropping ICD-code text columns, but structure is kept for
#       extensibility / future drug-group columns).
#     • Label Encoding    — binary columns (GioiTinh already done above).
# =============================================================================

log.info("Feature encoding …")

# Identify any remaining object columns (would be OHE candidates)
remaining_object_cols = X_train.select_dtypes(include="object").columns.tolist()

if remaining_object_cols:
    log.info(f"  One-Hot Encoding for: {remaining_object_cols}")
    X_train = pd.get_dummies(X_train, columns=remaining_object_cols, drop_first=False)
    X_test  = pd.get_dummies(X_test,  columns=remaining_object_cols, drop_first=False)
    # Align columns (test may miss categories unseen in test split)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
else:
    log.info("  No additional non-ordered categorical columns to one-hot encode.")

# Binary integer columns — already 0/1; no further action needed beyond
# the label-encoding of GioiTinh (already performed in step 5b).
log.info(f"  Binary columns retained as-is (0/1): {remaining_binary}")


# =============================================================================
# 8.  STANDARD SCALING  (fit on train, transform both)
# =============================================================================

log.info("StandardScaler on continuous features …")
scaler = StandardScaler()
scaler.fit(X_train[remaining_continuous])

X_train[remaining_continuous] = scaler.transform(X_train[remaining_continuous])
X_test[remaining_continuous]  = scaler.transform(X_test[remaining_continuous])

log.info(
    f"  After scaling — train mean ≈ "
    f"{X_train[remaining_continuous].mean().mean():.4f}  "
    f"std ≈ {X_train[remaining_continuous].std().mean():.4f}"
)


# =============================================================================
# 9.  CLASS IMBALANCE — SMOTE  (Training set ONLY)
# =============================================================================

log.info("Applying SMOTE to training set …")
log.info(
    f"  Pre-SMOTE class counts → "
    f"0: {(y_train == 0).sum()}  |  1: {(y_train == 1).sum()}"
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

log.info(
    f"  Post-SMOTE class counts → "
    f"0: {(y_train_smote == 0).sum()}  |  1: {(y_train_smote == 1).sum()}"
)
log.info(f"  Post-SMOTE X_train shape: {X_train_smote.shape}")


# =============================================================================
# 10. SMOTE + STRATIFIED K-FOLD VALIDATION (diagnostic confirmation)
#     This confirms that SMOTE + stratification are compatible.
#     The actual fold loop can be re-used during model training later.
# =============================================================================

log.info(
    f"Stratified {N_CV_FOLDS}-fold cross-validation structure (SMOTE inside fold) …"
)

skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_class_counts = []

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_fold_tr, y_fold_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # SMOTE applied inside each fold to avoid data leakage
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_fold_tr, y_fold_tr)

    fold_class_counts.append(
        {
            "fold": fold_idx,
            "train_0": int((y_res == 0).sum()),
            "train_1": int((y_res == 1).sum()),
            "val_0":   int((y_fold_val == 0).sum()),
            "val_1":   int((y_fold_val == 1).sum()),
        }
    )

cv_summary = pd.DataFrame(fold_class_counts)
log.info(f"  Cross-validation summary:\n{cv_summary.to_string(index=False)}")


# =============================================================================
# 11. SAVE PROCESSED DATA & PREPROCESSING ARTIFACTS
# =============================================================================

log.info("Saving preprocessing artifacts to models/ directory …")
pipeline_artifacts = {
    "label_encoders": label_encoders,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "iqr_bounds": iqr_bounds,
    "scaler": scaler,
    "remaining_continuous": remaining_continuous,
    "remaining_binary": remaining_binary
}
joblib.dump(pipeline_artifacts, os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))

log.info(f"Saving processed files to '{OUTPUT_DIR}' …")

# Convert SMOTE output arrays back to DataFrames with proper column names
X_train_final = pd.DataFrame(X_train_smote, columns=X_train.columns)
y_train_final = pd.Series(y_train_smote, name=TARGET_COL)

X_test_final  = X_test.reset_index(drop=True)
y_test_final  = y_test.reset_index(drop=True)

# Define file paths
files = {
    "X_train_final.csv": X_train_final,
    "y_train_final.csv": y_train_final.to_frame(),
    "X_test_final.csv":  X_test_final,
    "y_test_final.csv":  y_test_final.to_frame(),
}

for fname, data in files.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    data.to_csv(fpath, index=False, encoding="utf-8-sig")
    log.info(f"  ✔  {fpath}  →  shape {data.shape}")

log.info("=" * 60)
log.info("Preprocessing complete.  Summary:")
log.info(f"  X_train_final : {X_train_final.shape}")
log.info(f"  y_train_final : {y_train_final.shape}")
log.info(f"  X_test_final  : {X_test_final.shape}")
log.info(f"  y_test_final  : {y_test_final.shape}")
log.info(
    f"  Features kept : {list(X_train_final.columns)}"
)
log.info("=" * 60)
