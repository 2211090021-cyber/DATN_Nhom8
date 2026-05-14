import pandas as pd
import re

# Đọc dữ liệu
df = pd.read_excel('docs/YCDL_Raw.xlsx')

# Tính toán Age
if 'NamSinh' in df.columns:
    df['Age'] = 2026 - pd.to_numeric(df['NamSinh'], errors='coerce')

# Kết hợp mã ICD
df['_all_icd'] = (
    df.get('MaICD Bn nội trú', pd.Series(dtype=str)).fillna('').astype(str) + ' ' + 
    df.get('MaICD BN ngoại trú', pd.Series(dtype=str)).fillna('').astype(str)
)

# Ánh xạ đặc trưng ICD dựa trên selected_features.json
icd_feature_map = {
    'dai_thao_duong': r'\b(E10|E11|E12|E13|E14)(\.[0-9]+)?\b',
    'rl_lipid_mau':   r'\bE78(\.[0-9]+)?\b',
    'suy_than_man':   r'\bN18(\.[0-9]+)?\b',
}

for feature_name, pattern in icd_feature_map.items():
    df[feature_name] = (
        df['_all_icd']
        .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        .astype('int64')
    )

# Giữ lại các cột quan trọng và các đặc trưng (loại bỏ HBa1C, các mã ICD gốc và NamSinh)
columns_to_keep = [
    'TenBenhNhan', 'SoVaoVien', 'NgayThucHien', 
    'GioiTinh', 'Age', 
    'LDL', 'HDL', 'Triglycerid', 
    'dai_thao_duong', 'rl_lipid_mau', 'suy_than_man',
    'Target'
]

# Chỉ lấy những cột thực sự có trong dataframe
final_cols = [col for col in columns_to_keep if col in df.columns]

df_clean = df[final_cols]

# Lưu ra file mới
output_path = 'docs/YCDL_raw_clean.xlsx'
df_clean.to_excel(output_path, index=False)
print(f"Đã làm sạch dữ liệu và lưu vào {output_path}")
