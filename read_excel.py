import pandas as pd
import numpy as np

file_path = 'docs/YCDL.xlsx'
sheets = ['HBa1C', 'LDL', 'HDL', 'Triglycerid']

dfs = []
col_mapping = {
    'MaICD Nội Trú': 'MaICD Bn nội trú',
    'MaICD Nội trú': 'MaICD Bn nội trú',
    'MaICD Ngoại Trú': 'MaICD BN ngoại trú',
    'MaICD Ngoại trú': 'MaICD BN ngoại trú',
    # Giả định file excel có cột ngày. Hãy đổi tên nếu file gốc của bạn khác
    'NgayThucHien': 'NgayThucHien' 
}

# 1. Định nghĩa các cột cơ bản
base_cols = ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh', 'NgayThucHien', 'MaICD Bn nội trú', 'MaICD BN ngoại trú']

for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    df = df.rename(columns=col_mapping)
    
    # Fill NaN cho các cột cơ bản nếu thiếu
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    if 'KetQua' in df.columns:
        # Ép kiểu dữ liệu về số (float)
        df[sheet] = pd.to_numeric(df['KetQua'], errors='coerce')
    else:
        df[sheet] = np.nan
        
    cols_to_keep = base_cols + [sheet]
    df = df[cols_to_keep]
    dfs.append(df)

# 2. Nối dữ liệu và Sắp xếp
df_concat = pd.concat(dfs, ignore_index=True)
df_concat = df_concat.sort_values(by=['TenBenhNhan', 'SoVaoVien', 'NgayThucHien'])

# 3. Khai báo các hàm tổng hợp (Feature Aggregation)
group_cols = ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh']
numeric_agg = ['mean', 'max', 'min', 'std', 'last']

def join_unique_icd(x):
    valid_icds = x.dropna().astype(str).str.strip().unique()
    return ', '.join(valid_icds) if len(valid_icds) > 0 else None

agg_funcs = {
    'MaICD Bn nội trú': join_unique_icd,
    'MaICD BN ngoại trú': join_unique_icd,
    'HBa1C': numeric_agg,
    'LDL': numeric_agg,
    'HDL': numeric_agg,
    'Triglycerid': numeric_agg
}

df_grouped = df_concat.groupby(group_cols, dropna=False).agg(agg_funcs)

# 4. Làm phẳng cấu trúc cột MultiIndex
new_columns = []
for col in df_grouped.columns:
    col_name = col[0]
    func_name = col[1]
    
    if func_name in numeric_agg:
        new_columns.append(f"{col_name}_{func_name}")
    else:
        new_columns.append(col_name)

df_grouped.columns = new_columns
df_final = df_grouped.reset_index()

# 5. XỬ LÝ LỖI LOGIC NÂNG CAO: Chuẩn hóa cột _std
# Chỉ điền 0 cho std khi bệnh nhân có xét nghiệm (mean != NaN) nhưng chỉ đo 1 lần (std == NaN)
for sheet in sheets:
    mean_col = f"{sheet}_mean"
    std_col = f"{sheet}_std"
    
    if mean_col in df_final.columns and std_col in df_final.columns:
        # mask = Bệnh nhân CÓ thực hiện xét nghiệm VÀ giá trị std bị rỗng
        mask = df_final[mean_col].notnull() & df_final[std_col].isnull()
        df_final.loc[mask, std_col] = 0.0

# 6. Hoàn thiện và Xuất file
df_final = df_final.sort_values(by=['TenBenhNhan']).reset_index(drop=True)


# Create target for data
target_codes = ['I21', 'I22', 'I64', 'I50']

def create_target(row):
    # Nối tất cả mã ICD thành một chuỗi duy nhất để kiểm tra
    full_icd = str(row['MaICD Bn nội trú']) + " " + str(row['MaICD BN ngoại trú'])
    if any(code in full_icd for code in target_codes):
        return 1 # Có biến chứng
    return 0 # Không có biến chứng

df_final['Target'] = df_final.apply(create_target, axis=1)

# Chuyển đổi toàn bộ NaN còn lại (các bệnh nhân không xét nghiệm) thành None để xuất ra Excel
df_final = df_final.replace({np.nan: None})

output_path = 'docs/YCDL_Total.xlsx'
with pd.ExcelWriter(output_path) as writer:
    df_final.to_excel(writer, sheet_name='Total', index=False)

print('Data successfully exported to', output_path)