"""
=============================================================================
 merge_excel.py — Gộp dữ liệu thô từ nhiều sheet thành 1 file Excel
=============================================================================
 Logic: Giữ nguyên từng lần khám (không tính mean/max/std...).
 Output: Mỗi dòng = 1 lần khám của 1 bệnh nhân, với đầy đủ cột xét nghiệm.

 Cấu trúc output:
   TenBenhNhan | SoVaoVien | NamSinh | GioiTinh | NgayThucHien
   | MaICD Bn nội trú | MaICD BN ngoại trú
   | HBa1C | LDL | HDL | Triglycerid | Target

 Fix so với v1:
   - Chuẩn hoá NgayThucHien → datetime ngay ở STEP 1 (tránh type mismatch)
   - Dùng pd.concat + groupby thay vì outer merge lặp (an toàn với ~2M dòng)
=============================================================================
"""
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
FILE_PATH   = 'docs/YCDL.xlsx'
OUTPUT_PATH = 'docs/YCDL_Raw.xlsx'
SHEETS      = ['HBa1C', 'LDL', 'HDL', 'Triglycerid']

COL_MAPPING = {
    'MaICD Nội Trú'  : 'MaICD Bn nội trú',
    'MaICD Nội trú'  : 'MaICD Bn nội trú',
    'MaICD Ngoại Trú': 'MaICD BN ngoại trú',
    'MaICD Ngoại trú': 'MaICD BN ngoại trú',
}

VISIT_KEY = ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh', 'NgayThucHien']
ICD_COLS  = ['MaICD Bn nội trú', 'MaICD BN ngoại trú']
BASE_COLS = VISIT_KEY + ICD_COLS

TARGET_ICD = ['I21', 'I22', 'I64', 'I50']

# Số bệnh nhân tối đa lấy từ mỗi sheet (None = lấy tất cả)
N_PATIENTS_PER_SHEET = 30

# =============================================================================
# STEP 1 — ĐỌC TỪNG SHEET, CHUẨN HOÁ KIỂU DỮ LIỆU
# =============================================================================
print("=" * 60)
print("  STEP 1 — Đọc và chuẩn hoá từng sheet")
print("=" * 60)

all_dfs = []   # list of long-format DataFrames, mỗi phần tử là 1 sheet

for sheet in SHEETS:
    print(f"  Đang đọc sheet [{sheet}]...", flush=True)
    df = pd.read_excel(FILE_PATH, sheet_name=sheet)
    df = df.rename(columns=COL_MAPPING)

    # Đảm bảo tất cả BASE_COLS tồn tại
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # ── FIX: Chuẩn hoá NgayThucHien → datetime64 ngay tại đây ──
    df['NgayThucHien'] = pd.to_datetime(df['NgayThucHien'], errors='coerce')

    # Lấy giá trị xét nghiệm
    if 'KetQua' in df.columns:
        df[sheet] = pd.to_numeric(df['KetQua'], errors='coerce')
    else:
        df[sheet] = np.nan

    # Giữ các cột cần thiết (BASE + metric value)
    df = df[BASE_COLS + [sheet]].copy()

    # ── Lọc N_PATIENTS_PER_SHEET bệnh nhân (50% Target=1, 50% Target=0) ──
    if N_PATIENTS_PER_SHEET is not None:
        # Tính target nháp để phân loại bệnh nhân trong sheet này
        def tmp_target(row):
            s = str(row.get('MaICD Bn nội trú', '')) + " " + str(row.get('MaICD BN ngoại trú', ''))
            return 1 if any(code in s for code in TARGET_ICD) else 0
            
        df_tmp = df[['TenBenhNhan', 'MaICD Bn nội trú', 'MaICD BN ngoại trú']].copy()
        df_tmp['tgt'] = df_tmp.apply(tmp_target, axis=1)
        # Bệnh nhân có ít nhất 1 dòng mang target = 1 thì xếp vào nhóm nguy cơ (1)
        pt_target = df_tmp.groupby('TenBenhNhan')['tgt'].max()
        
        pos_pts = pt_target[pt_target == 1].index.tolist()
        neg_pts = pt_target[pt_target == 0].index.tolist()
        
        n_half = N_PATIENTS_PER_SHEET // 2
        
        # Cố gắng lấy 50% là nhóm có nguy cơ
        sel_pos = pos_pts[:n_half]
        # Lấy nhóm không nguy cơ để bù vào cho đủ số lượng tổng
        sel_neg = neg_pts[:(N_PATIENTS_PER_SHEET - len(sel_pos))]
        
        # Nếu nhóm không nguy cơ không đủ bù, lấy thêm nhóm có nguy cơ nếu còn
        if len(sel_pos) + len(sel_neg) < N_PATIENTS_PER_SHEET:
            sel_pos.extend(pos_pts[len(sel_pos) : N_PATIENTS_PER_SHEET - len(sel_neg)])
            
        patient_ids = sel_pos + sel_neg
        df = df[df['TenBenhNhan'].isin(patient_ids)].copy()

    print(f"    → {len(df):,} dòng  |  "
          f"{df['TenBenhNhan'].nunique():,} bệnh nhân  |  "
          f"có giá trị: {df[sheet].notna().sum():,}")
    all_dfs.append(df)

# =============================================================================
# STEP 2 — CONCAT TẤT CẢ SHEET (long format) + COLLAPSE THEO VISIT_KEY
#
#  Thay vì outer merge lặp (nguy cơ bộ nhớ bùng nổ), dùng:
#    1. pd.concat  → long DataFrame với nhiều NaN
#    2. groupby(VISIT_KEY).agg(first non-null)  → 1 dòng per (BN, ngày)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2 — Concat & collapse theo visit key")
print("=" * 60)

print("  Đang concat...", flush=True)
df_long = pd.concat(all_dfs, ignore_index=True)
print(f"  Long DataFrame: {len(df_long):,} dòng")

# Hàm lấy giá trị không-NaN đầu tiên trong nhóm
def first_valid(series):
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else np.nan

# Hàm gộp ICD duy nhất
def join_unique_icd(series):
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals.str.len() > 0]
    unique = vals.unique()
    return ', '.join(unique) if len(unique) > 0 else np.nan

print("  Đang groupby + collapse (có thể mất vài phút)...", flush=True)

agg_funcs = {
    **{icd: join_unique_icd for icd in ICD_COLS},
    **{sheet: first_valid    for sheet in SHEETS},
}

# NaN trong VISIT_KEY (đặc biệt NamSinh, GioiTinh) được giữ nguyên
df_visit = (
    df_long
    .groupby(VISIT_KEY, dropna=False)
    .agg(agg_funcs)
    .reset_index()
)

print(f"  Sau collapse: {len(df_visit):,} dòng (lần khám duy nhất)")
print(f"  Số bệnh nhân duy nhất: {df_visit['TenBenhNhan'].nunique():,}")

# =============================================================================
# STEP 3 — GỘP MÃ ICD THEO BỆNH NHÂN (mỗi BN chỉ có 1 bộ ICD)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3 — Gộp mã ICD theo bệnh nhân")
print("=" * 60)

PATIENT_KEY = ['TenBenhNhan', 'SoVaoVien']

icd_per_patient = (
    df_visit
    .groupby(PATIENT_KEY, dropna=False)[ICD_COLS]
    .agg(join_unique_icd)
    .reset_index()
    .rename(columns={
        'MaICD Bn nội trú'  : '_icd_noi',
        'MaICD BN ngoại trú': '_icd_ngoai',
    })
)

# Thay cột ICD per-visit bằng ICD per-patient
df_visit.drop(columns=ICD_COLS, inplace=True)
df_visit = df_visit.merge(icd_per_patient, on=PATIENT_KEY, how='left')
df_visit.rename(columns={
    '_icd_noi'  : 'MaICD Bn nội trú',
    '_icd_ngoai': 'MaICD BN ngoại trú',
}, inplace=True)

# =============================================================================
# STEP 4 — TẠO CỘT TARGET
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4 — Tạo cột Target")
print("=" * 60)

def create_target(row):
    full_icd = (str(row.get('MaICD Bn nội trú', '')) + " " +
                str(row.get('MaICD BN ngoại trú', '')))
    return 1 if any(code in full_icd for code in TARGET_ICD) else 0

df_visit['Target'] = df_visit.apply(create_target, axis=1)

n_pos = (df_visit.groupby('TenBenhNhan')['Target'].max() == 1).sum()
print(f"  Bệnh nhân có biến chứng (Target=1): {n_pos:,}")

# =============================================================================
# STEP 5 — SẮP XẾP & HOÀN THIỆN
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5 — Sắp xếp và chuẩn bị xuất")
print("=" * 60)

df_visit = df_visit.sort_values(
    by=['TenBenhNhan', 'SoVaoVien', 'NgayThucHien']
).reset_index(drop=True)

# Sắp xếp thứ tự cột
final_cols = (
    ['TenBenhNhan', 'SoVaoVien', 'NamSinh', 'GioiTinh', 'NgayThucHien',
     'MaICD Bn nội trú', 'MaICD BN ngoại trú']
    + SHEETS
    + ['Target']
)
final_cols = [c for c in final_cols if c in df_visit.columns]
df_visit = df_visit[final_cols]

print(f"\n  Tổng số dòng (lần khám)     : {len(df_visit):,}")
print(f"  Tổng số bệnh nhân           : {df_visit['TenBenhNhan'].nunique():,}")
print(f"  Khoảng thời gian            : "
      f"{df_visit['NgayThucHien'].min().date()} → "
      f"{df_visit['NgayThucHien'].max().date()}")
print(f"\n  Số dòng có giá trị mỗi cột:")
for col in SHEETS:
    if col in df_visit.columns:
        n = df_visit[col].notna().sum()
        print(f"    {col:15s}: {n:,} / {len(df_visit):,}")

# =============================================================================
# STEP 6 — XUẤT FILE
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 6 — Xuất file Excel")
print("=" * 60)

print(f"  Đang ghi ra {OUTPUT_PATH}...", flush=True)
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    df_visit.to_excel(writer, sheet_name='Raw_Visits', index=False)

print(f"\n  ✅ Xuất thành công: {OUTPUT_PATH}")
print("=" * 60 + "\n")
