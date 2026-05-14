import pandas as pd
import numpy as np
import re
import argparse
import os

def apply_iqr_filter(df, column):
    """
    Áp dụng IQR để lọc các giá trị ngoại lai (do lỗi nhập liệu) trên dữ liệu gốc.
    Các giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR] sẽ được thay bằng NaN
    để không ảnh hưởng đến bước tính mean, min, max sau này.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Số lượng ngoại lai bị phát hiện
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if outliers > 0:
        print(f"  [IQR] Tìm thấy {outliers} giá trị ngoại lai ở cột {column}, tiến hành gán thành NaN.")
        
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
    return df

def process_data(input_path, output_path):
    print(f"Đang đọc dữ liệu gốc từ: {input_path}")
    sheets = ['HBa1C', 'LDL', 'HDL', 'Triglycerid']
    
    col_mapping = {
        'MaICD Nội Trú': 'MaICD Bn nội trú',
        'MaICD Nội trú': 'MaICD Bn nội trú',
        'MaICD Ngoại Trú': 'MaICD BN ngoại trú',
        'MaICD Ngoại trú': 'MaICD BN ngoại trú',
        'NgayThucHien': 'NgayThucHien' 
    }
    
    base_cols = ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh', 'NgayThucHien', 'MaICD Bn nội trú', 'MaICD BN ngoại trú']
    
    # ---------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU THÔ VÀ LỌC NGOẠI LAI VỚI IQR
    # ---------------------------------------------------------
    print("\n--- BƯỚC 1: ĐỌC VÀ LỌC NGOẠI LAI (IQR) TRÊN DỮ LIỆU THÔ ---")
    dfs = []
    for sheet in sheets:
        try:
            df = pd.read_excel(input_path, sheet_name=sheet)
        except Exception as e:
            print(f"Lỗi khi đọc sheet {sheet}: {e}")
            continue
            
        df = df.rename(columns=col_mapping)
        
        for col in base_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        if 'NgayThucHien' in df.columns:
            df['NgayThucHien'] = pd.to_datetime(df['NgayThucHien'], errors='coerce')

        if 'KetQua' in df.columns:
            df[sheet] = pd.to_numeric(df['KetQua'], errors='coerce')
            print(f"Đang xử lý sheet: {sheet}")
            df = apply_iqr_filter(df, sheet)
        else:
            df[sheet] = np.nan
            
        dfs.append(df[base_cols + [sheet]])

    df_concat = pd.concat(dfs, ignore_index=True)

    # Lọc khung thời gian
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2026-02-28')
    df_concat = df_concat.dropna(subset=['NgayThucHien'])
    df_concat = df_concat[(df_concat['NgayThucHien'] >= start_date) & (df_concat['NgayThucHien'] <= end_date)]
    df_concat = df_concat.sort_values(by=['TenBenhNhan', 'SoVaoVien', 'NgayThucHien'])

    # ---------------------------------------------------------
    # BƯỚC 2: TỔNG HỢP DỮ LIỆU (FEATURE AGGREGATION)
    # ---------------------------------------------------------
    print("\n--- BƯỚC 2: TỔNG HỢP DỮ LIỆU TỪNG BỆNH NHÂN (GROUPBY) ---")
    group_cols = ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh']
    numeric_agg = ['mean', 'max', 'min', 'std', 'last']

    def join_unique_icd(x):
        valid_icds = x.dropna().astype(str).str.strip().unique()
        return ', '.join(valid_icds) if len(valid_icds) > 0 else None

    agg_funcs = {
        'MaICD Bn nội trú': join_unique_icd,
        'MaICD BN ngoại trú': join_unique_icd,
        'NgayThucHien': ['min', 'max'],
        'HBa1C': numeric_agg,
        'LDL': numeric_agg,
        'HDL': numeric_agg,
        'Triglycerid': numeric_agg
    }

    df_grouped = df_concat.groupby(group_cols, dropna=False).agg(agg_funcs)

    # Làm phẳng (Flatten) MultiIndex column
    df_grouped.columns = [f"{c[0]}_{c[1]}" if c[1] in numeric_agg + ['min', 'max'] else c[0] for c in df_grouped.columns]
    df_final = df_grouped.reset_index()

    # ---------------------------------------------------------
    # BƯỚC 3: BỘ LỌC ĐỐI TƯỢNG (INCLUSION/EXCLUSION)
    # ---------------------------------------------------------
    print("\n--- BƯỚC 3: LỌC ĐỐI TƯỢNG NGHIÊN CỨU ---")
    # Lọc tuổi >= 18
    df_final['Age'] = 2026 - pd.to_numeric(df_final['NamSinh'], errors='coerce')
    df_final = df_final[df_final['Age'] >= 18]

    # Lọc theo dõi >= 12 tháng (365 ngày)
    df_final['FollowUp_Days'] = (df_final['NgayThucHien_max'] - df_final['NgayThucHien_min']).dt.days
    df_final = df_final[df_final['FollowUp_Days'] >= 365]

    # Lọc mã ICD (Có THA, Không Ung Thư)
    def validate_diagnosis(row):
        full_icd = str(row['MaICD Bn nội trú']) + " " + str(row['MaICD BN ngoại trú'])
        is_htn = bool(re.search(r'I1[0-5]', full_icd))
        is_malignant = bool(re.search(r'C\d{2}', full_icd))
        return is_htn and not is_malignant

    df_final = df_final[df_final.apply(validate_diagnosis, axis=1)]

    # ---------------------------------------------------------
    # BƯỚC 4: TẠO NHÃN (TARGET) & CHUẨN HOÁ STD
    # ---------------------------------------------------------
    print("\n--- BƯỚC 4: TẠO NHÃN VÀ TRÍCH XUẤT ĐẶC TRƯNG TỪ ICD ---")
    # Chuẩn hóa giá trị std = NaN thành 0 nếu bệnh nhân chỉ đo 1 lần
    for s in sheets:
        mean_col, std_col = f"{s}_mean", f"{s}_std"
        if mean_col in df_final.columns and std_col in df_final.columns:
            mask = df_final[mean_col].notnull() & df_final[std_col].isnull()
            df_final.loc[mask, std_col] = 0.0

    target_codes = ['I21', 'I22', 'I64', 'I50']
    def define_target(row):
        full_icd = str(row['MaICD Bn nội trú']) + " " + str(row['MaICD BN ngoại trú'])
        return 1 if any(code in full_icd for code in target_codes) else 0

    df_final['Target'] = df_final.apply(define_target, axis=1)

    # ---------------------------------------------------------
    # BƯỚC 5: TRÍCH XUẤT ĐẶC TRƯNG BỆNH LÝ TỪ ICD
    # ---------------------------------------------------------
    df_final['_all_icd'] = (
        df_final['MaICD Bn nội trú'].fillna('').astype(str) + ' ' + 
        df_final['MaICD BN ngoại trú'].fillna('').astype(str)
    )

    icd_feature_map = {
        'dai_thao_duong': r'\b(E10|E11|E12|E13|E14)(\.[0-9]+)?\b',
        'rl_lipid_mau':   r'\bE78(\.[0-9]+)?\b',
        'suy_than_man':   r'\bN18(\.[0-9]+)?\b',
        'benh_mach_vanh': r'\bI25\.0\b',
        'Tang_huyet_ap':  r'\b(I10|I11|I12|I13|I14|I15)(\.[0-9]+)?\b',
    }

    for feature_name, pattern in icd_feature_map.items():
        df_final[feature_name] = (
            df_final['_all_icd']
            .str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            .astype('int64')
        )

    # Dọn dẹp
    df_final.drop(columns=['_all_icd', 'NgayThucHien_min', 'NgayThucHien_max', 'FollowUp_Days'], inplace=True, errors='ignore')
    df_final = df_final.sort_values(by=['TenBenhNhan']).replace({np.nan: None})

    # Lưu kết quả
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='Total', index=False)
        
    print(f"\n✅ Hoàn tất! Dữ liệu đã xử lý lưu tại: {output_path}")
    print(f"Tổng số bệnh nhân hợp lệ: {len(df_final)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Làm sạch dữ liệu EMR thô, lọc IQR, và tổng hợp đặc trưng")
    parser.add_argument("--input", type=str, default="docs/YCDL.xlsx", help="Đường dẫn đến file dữ liệu thô (ví dụ: docs/YCDL.xlsx)")
    parser.add_argument("--output", type=str, default="data/preprocess/YCDL_Features_Mapped.xlsx", help="Đường dẫn file kết quả (ví dụ: data/preprocess/YCDL_Features_Mapped.xlsx)")
    args = parser.parse_args()
    
    process_data(args.input, args.output)
