"""
=============================================================================
 Initialize SQLite Database + Seed Demo Patients
=============================================================================
"""
import os
import sqlite3

DB_PATH = os.path.join("data", "clinic.db")


def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            birth_year INTEGER NOT NULL,
            gender INTEGER NOT NULL DEFAULT 1,
            dai_thao_duong INTEGER DEFAULT 0,
            rl_lipid_mau INTEGER DEFAULT 0,
            suy_than_man INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            visit_date DATE NOT NULL,
            ldl REAL NOT NULL,
            hdl REAL NOT NULL,
            triglycerid REAL NOT NULL,
            prediction_prob REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    conn.commit()
    return conn


def seed_demo_data(conn):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM patients")
    if c.fetchone()[0] > 0:
        print("  Demo data already exists. Skipping seed.")
        return

    # ── Patient 1: High risk (elderly male, heart failure + AF) ───────────
    c.execute("""
        INSERT INTO patients (patient_code, name, birth_year, gender,
                              dai_thao_duong)
        VALUES ('BN001', 'Nguyen Van An', 1945, 1, 1)
    """)
    p1 = c.lastrowid
    c.executemany("""
        INSERT INTO visits (patient_id, visit_date, ldl, hdl, triglycerid)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (p1, '2023-03-15', 4.2, 0.90, 2.8),
        (p1, '2023-09-20', 4.5, 0.85, 3.1),
        (p1, '2024-03-10', 3.8, 0.95, 2.5),
        (p1, '2024-09-15', 4.1, 0.88, 2.9),
    ])

    # ── Patient 2: Low risk (middle-aged female, dyslipidaemia only) ──────
    c.execute("""
        INSERT INTO patients (patient_code, name, birth_year, gender,
                              rl_lipid_mau)
        VALUES ('BN002', 'Tran Thi Binh', 1970, 0, 1)
    """)
    p2 = c.lastrowid
    c.executemany("""
        INSERT INTO visits (patient_id, visit_date, ldl, hdl, triglycerid)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (p2, '2023-06-10', 2.1, 1.40, 1.2),
        (p2, '2024-01-15', 2.3, 1.35, 1.4),
        (p2, '2024-07-20', 2.0, 1.50, 1.1),
    ])

    # ── Patient 3: Moderate risk (elderly male, diabetes + CKD) ───────────
    c.execute("""
        INSERT INTO patients (patient_code, name, birth_year, gender,
                              dai_thao_duong, suy_than_man)
        VALUES ('BN003', 'Le Hoang Cuong', 1950, 1, 1, 1)
    """)
    p3 = c.lastrowid
    c.executemany("""
        INSERT INTO visits (patient_id, visit_date, ldl, hdl, triglycerid)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (p3, '2023-04-01', 3.0, 1.10, 2.0),
        (p3, '2023-10-15', 3.2, 1.05, 2.3),
        (p3, '2024-04-20', 2.8, 1.15, 1.8),
    ])

    conn.commit()
    print("  Demo data seeded successfully!")


if __name__ == "__main__":
    print("Initializing database...")
    conn = init_database()
    seed_demo_data(conn)
    conn.close()
    print(f"  Database created at: {os.path.abspath(DB_PATH)}")
