import pandas as pd
import json

df = pd.read_excel('docs/YCDL_raw_clean.xlsx')
res = {
    "total_rows": len(df),
    "unique_patients": df['SoVaoVien'].nunique(),
    "nan_ldl": int(df['LDL'].isna().sum()),
    "nan_hdl": int(df['HDL'].isna().sum()),
    "nan_tg": int(df['Triglycerid'].isna().sum()),
}

with open('scratch/db_check.json', 'w') as f:
    json.dump(res, f, indent=2)
