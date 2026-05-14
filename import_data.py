import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "clinic.db")

def import_excel_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Đọc dữ liệu
    df = pd.read_excel('docs/YCDL_raw_clean.xlsx')
    
    # Mục 1: Lọc bỏ rows chứa NaN trong LDL, HDL, Triglycerid
    df = df.dropna(subset=['LDL', 'HDL', 'Triglycerid'])
    
    # Đảm bảo datetime
    if 'NgayThucHien' in df.columns:
        df['NgayThucHien'] = pd.to_datetime(df['NgayThucHien'], errors='coerce')
    
    # Mục 2: Xử lý và insert Patients (Tránh trùng lặp UNIQUE constraint)
    patients_df = df.drop_duplicates(subset=['SoVaoVien']).copy()
    
    # Mục 3: Tính birth_year
    patients_df['birth_year'] = 2026 - patients_df['Age']
    
    # (Mục 4: Drop target column) - Dữ liệu Target sẽ không được insert vào db
    # (Mục 5: Bỏ suy_tim, dot_quy, rung_nhi) - Schema đã được update và không có các cột này
    
    for _, row in patients_df.iterrows():
        patient_code = str(row['SoVaoVien'])
        name = str(row['TenBenhNhan'])
        birth_year = int(row['birth_year']) if pd.notnull(row['birth_year']) else 1900
        gender_str = str(row['GioiTinh']).lower().strip()
        gender = 1 if gender_str == 'nam' else (0 if gender_str == 'nữ' else 1)
        dtd = int(row['dai_thao_duong']) if pd.notnull(row['dai_thao_duong']) else 0
        rlm = int(row['rl_lipid_mau']) if pd.notnull(row['rl_lipid_mau']) else 0
        stm = int(row['suy_than_man']) if pd.notnull(row['suy_than_man']) else 0
        
        c.execute("""
            INSERT OR IGNORE INTO patients 
            (patient_code, name, birth_year, gender, dai_thao_duong, rl_lipid_mau, suy_than_man)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_code, name, birth_year, gender, dtd, rlm, stm))
        
    # Tạo mapping từ patient_code -> patient_id
    c.execute("SELECT patient_code, id FROM patients")
    patient_map = {row[0]: row[1] for row in c.fetchall()}
    
    # Xử lý và insert Visits
    visits_data = []
    for _, row in df.iterrows():
        patient_code = str(row['SoVaoVien'])
        patient_id = patient_map.get(patient_code)
        
        if not patient_id:
            continue
            
        visit_date = row['NgayThucHien'].strftime('%Y-%m-%d') if pd.notnull(row['NgayThucHien']) else '1970-01-01'
        ldl = float(row['LDL'])
        hdl = float(row['HDL'])
        tg = float(row['Triglycerid'])
        
        visits_data.append((patient_id, visit_date, ldl, hdl, tg))
        
    c.executemany("""
        INSERT INTO visits (patient_id, visit_date, ldl, hdl, triglycerid)
        VALUES (?, ?, ?, ?, ?)
    """, visits_data)
    
    conn.commit()
    conn.close()
    
    print(f"Da import thanh cong {len(patients_df)} benh nhan va {len(visits_data)} luot kham vao co so du lieu.")

if __name__ == "__main__":
    print("Bat dau nap du lieu tu file Excel vao Database...")
    import_excel_data()
