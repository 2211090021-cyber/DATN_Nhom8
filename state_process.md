# Preprocessing Pipeline: Step-by-Step State Tracking

This document breaks down the clinical preprocessing pipeline used in `preprocess.py`. It explains each step and tracks how a sample patient's data changes as it moves through the pipeline.

## Initial Sample Data
Let's imagine a raw row from the Excel file representing a single patient. 
*Note: Some columns are omitted for brevity.*

```python
raw_sample = {
    "TenBenhNhan": "Nguyen Van A",  # Patient Name
    "SoVaoVien": 123456,            # Admission Number
    "GioiTinh": "Nam",              # Gender (Male)
    "Age": 65,                      # Age
    "HBa1C_mean": np.nan,           # Missing lab value
    "LDL_max": 8.5,                 # Extremely high lab value (Outlier)
    "dai_thao_duong": 1,            # Diabetes history (1 = Yes)
    "Target": 1                     # Target label (Complication)
}
```

---

### Step 2: Drop Administrative / ID Columns
**What it does:** Removes columns that identify the patient or hold free text. These are purely administrative and hold no predictive clinical value (and would cause data leakage).
**Code Snippet:** `df = df_raw.drop(columns=ID_COLUMNS)`

**State Tracker:**
```diff
- "TenBenhNhan": "Nguyen Van A"
- "SoVaoVien": 123456
  "GioiTinh": "Nam"
  "Age": 65
  "HBa1C_mean": np.nan
  "LDL_max": 8.5
  "dai_thao_duong": 1
  "Target": 1
```

---

### Step 3 & 4: Separate Features/Target and Train/Test Split
**What it does:** Detaches the `Target` column into its own variable `y`, leaving clinical features in `X`. Then splits the data into 80% Training and 20% Testing, ensuring the proportion of `Target` (0 vs 1) remains the same in both sets (`stratify=y`).

**State Tracker:**
```python
# y_train gets the target:
y_train_sample = 1

# X_train gets the features:
X_train_sample = {
    "GioiTinh": "Nam",
    "Age": 65,
    "HBa1C_mean": np.nan,
    "LDL_max": 8.5,
    "dai_thao_duong": 1
}
```

---

### Step 5: Missing-Data Handling
**What it does:** 
1. **Drop (>60%):** Drops any column with more than 60% missing data.
2. **Label Encode BEFORE Imputation:** Converts strings to numbers so the imputer can work. `LabelEncoder` fits on non-null training data. `"Nam"` -> `0`, `"Nữ"` -> `1`.
3. **Numerical Imputation:** Uses the **Median** of the training set to fill missing continuous variables (like `HBa1C_mean`). Median is robust to skewed clinical data.
4. **Categorical Imputation:** Uses the **Mode** (most frequent) to fill missing binary flags.

**State Tracker:**
```diff
- "GioiTinh": "Nam"
+ "GioiTinh": 0           # Label Encoded (Nam -> 0)

- "HBa1C_mean": np.nan
+ "HBa1C_mean": 6.2       # Imputed with training Median

  "Age": 65               # No missing values
  "LDL_max": 8.5
  "dai_thao_duong": 1     # No missing values, confirmed by Mode imputer
```

---

### Step 6: Outlier Handling (IQR Capping / Winsorization)
**What it does:** Clinical lab tests can have extreme outliers. Instead of deleting these records (which loses valuable data), we cap them to a calculated boundary using the Interquartile Range ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$).
*Assumption for this example: The calculated upper bound for `LDL_max` across the training set is `5.2`.*

**Code Snippet:** `X_train[col] = X_train[col].clip(lower=lower, upper=upper)`

**State Tracker:**
```diff
  "GioiTinh": 0
  "Age": 65
  "HBa1C_mean": 6.2
- "LDL_max": 8.5
+ "LDL_max": 5.2          # Capped to the IQR upper boundary
  "dai_thao_duong": 1
```

---

### Step 8: Standard Scaling
**What it does:** Machine learning models perform better when features are on the same scale. `StandardScaler` shifts the data to have a mean of 0 and a standard deviation of 1.
*Assumption for this example: Age mean is 50, std is 10. LDL_max mean is 3.2, std is 1.0.*

**Code Snippet:** `X_train = scaler.transform(X_train)`

**State Tracker:**
```diff
  "GioiTinh": 0           # Label encoded flags are NOT scaled
- "Age": 65
+ "Age": 1.50             # (65 - 50) / 10 = 1.5
- "HBa1C_mean": 6.2
+ "HBa1C_mean": 0.15      # Scaled
- "LDL_max": 5.2
+ "LDL_max": 2.00         # (5.2 - 3.2) / 1.0 = 2.0
  "dai_thao_duong": 1
```

---

### Step 9: Handling Class Imbalance (SMOTE)
**What it does:** The dataset has far more cases of `Target=0` (no complication) than `Target=1` (complication). SMOTE generates synthetic clinical profiles based on the nearest neighbors of the existing minority class. **This is only applied to the Training set.**

**State Tracker:**
The original sample is kept exactly as is. However, if our sample patient was in the minority class (Target=1), SMOTE might generate a new "synthetic" patient next to them:

```python
# Original sample (Target = 1)
patient_A = {"GioiTinh": 0, "Age": 1.5, "HBa1C_mean": 0.15, "LDL_max": 2.0, "dai_thao_duong": 1}

# ... SMOTE finds another real patient with Complication (Target=1)
# patient_B = {"GioiTinh": 0, "Age": 1.3, "HBa1C_mean": 0.25, "LDL_max": 1.8, "dai_thao_duong": 1}

# SMOTE creates a new synthetic row by blending them:
synthetic_sample = {
    "GioiTinh": 0,          
    "Age": 1.40,            # Blended age
    "HBa1C_mean": 0.20,     # Blended HBa1C
    "LDL_max": 1.90,        # Blended LDL
    "dai_thao_duong": 1
}
# Both original and synthetic samples are added to X_train_final.csv with Target=1
```

---

### Final Output Structure
After passing through the entire pipeline, the data is entirely numerical, free of missing values, resistant to extreme outliers, perfectly scaled, and the training set is balanced 1:1. 

Ready for model ingestion via `data/processed/X_train_final.csv`.
