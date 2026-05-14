# Preprocessing Pipeline Overview — Cardiovascular Complication Prediction

This document summarizes the results of the clinical preprocessing pipeline applied to the medical dataset `docs/YCDL_Features_Mapped.xlsx`.

## 1. Data Summary
| Phase | Rows | Columns | Description |
| :--- | :--- | :--- | :--- |
| **Raw Dataset** | 12,355 | 36 | Initial medical dataset from Excel |
| **Filtered Dataset** | 12,355 | 31 | After dropping administrative/ID columns |
| **Final Training (SMOTE)** | 18,754 | 30 | Post-SMOTE, capped, and scaled training set |
| **Final Testing** | 2,471 | 30 | Processed test features (scaling/encoding only) |

## 2. Class Distribution (Target: 'Target')
| Split | Class 0 (None) | Class 1 (Complication) | Balance Ratio (0:1) |
| :--- | :--- | :--- | :--- |
| **Original** | 11,721 | 634 | 18.5 : 1 |
| **Train (Initial)** | 9,377 | 507 | 18.5 : 1 |
| **Train (Final - SMOTE)** | 9,377 | 9,377 | **1 : 1** |
| **Test** | 2,344 | 127 | 18.5 : 1 |

## 3. Preprocessing Methodology
The pipeline follows strict clinical research standards to ensure reproducibility and prevent data leakage:

*   **Stratified Split**: 80% Training / 20% Testing split with stratification on the Target to maintain class proportions.
*   **Missing Data**: 
    *   Columns with > 60% missing values were discarded.
    *   **Numerical Imputation**: Missing values in continuous laboratory indicators were filled with the **Median**.
    *   **Categorical Imputation**: Binary variables and categorical features were filled with the **Mode**.
*   **Outlier Handling (Winsorization)**: Applied IQR capping ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$) to 21 continuous laboratory indicators to minimize the impact of extreme values without deleting clinical records.
*   **Feature Encoding**:
    *   **Label Encoding**: Applied to binary variables like `GioiTinh`.
    *   **One-Hot Encoding**: Structure maintained for non-ordered categorical variables (drug groups).
*   **Normalization**: `StandardScaler` was applied to all continuous variables (e.g., HbA1c, LDL-C, Age) to achieve a mean of 0 and standard deviation of 1.
*   **SMOTE (Synthetic Minority Oversampling Technique)**: Applied to the training set only to balance classes. Validation was confirmed using 5-fold stratified cross-validation with SMOTE applied within each fold.

## 4. Processed Features List
The final feature set (30 columns) includes:
- **Demographics**: `Age`, `GioiTinh`
- **Lab Indicators (mean, max, min, std, last)**: `HBa1C`, `LDL`, `HDL`, `Triglycerid`
- **Medical History**: `dai_thao_duong`, `rl_lipid_mau`, `suy_than_man`, `benh_mach_vanh`, `suy_tim`, `dot_quy`, `rung_nhi`, `Tang_huyet_ap`

## 5. Output Artifacts (`data/processed/`)
The final processed datasets are exported as CSV files for model training:
- `X_train_final.csv`: (18754, 30) - Features after SMOTE, capping, and scaling.
- `y_train_final.csv`: (18754, 1) - Target labels after SMOTE.
- `X_test_final.csv`: (2471, 30) - Processed test features.
- `y_test_final.csv`: (2471, 1) - Test target labels.
