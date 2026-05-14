"""
=============================================================================
 Streamlit App — Cardiovascular Complication Risk Prediction
=============================================================================
"""
import os, json, sqlite3
from datetime import date
import joblib, numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt, shap, streamlit as st

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

st.set_page_config(page_title="Dự đoán Biến chứng Tim mạch", page_icon="❤️",
                   layout="wide", initial_sidebar_state="expanded")

# ── 1. LOAD ML ARTIFACTS ─────────────────────────────────────────────────
MODEL_DIR = "models"
DB_PATH = os.path.join("data", "clinic.db")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "champion_model.pkl"))
    pipeline = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
    with open(os.path.join(MODEL_DIR, "selected_features.json"), "r") as f:
        features = json.load(f)
    exp = shap.TreeExplainer(model)
    smap = {}
    for i, col in enumerate(pipeline["remaining_continuous"]):
        smap[col] = (pipeline["scaler"].mean_[i], pipeline["scaler"].scale_[i])
    return model, pipeline, features, exp, smap

model, pipeline, SELECTED_FEATURES, explainer, scaler_map = load_artifacts()
CONTINUOUS_SELECTED = [f for f in SELECTED_FEATURES if f in scaler_map]
LABEL_VI = {"GioiTinh":"Giới tính","dai_thao_duong":"Đái tháo đường",
    "rl_lipid_mau":"Rối loạn lipid máu","suy_than_man":"Suy thận mạn",
    "benh_mach_vanh":"Bệnh mạch vành"}

# ── 2. DATABASE HELPERS ──────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def fetch_patient_by_code(code):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE patient_code=?", (code,)).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM patients LIMIT 0").description]
    conn.close()
    return dict(zip(cols, row)) if row else None

def fetch_patient(pid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM patients LIMIT 0").description]
    conn.close()
    return dict(zip(cols, row)) if row else None

def fetch_visits(pid):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM visits WHERE patient_id=? ORDER BY visit_date", (pid,)).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM visits LIMIT 0").description]
    conn.close()
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

def insert_patient(code, birth_year, gender, comorbidities: dict):
    conn = get_conn()
    conn.execute("""INSERT INTO patients (patient_code,name,birth_year,gender,
        dai_thao_duong,rl_lipid_mau,suy_than_man)
        VALUES (?,?,?,?,?,?,?)""",
        (code, "", birth_year, gender,
         comorbidities.get("dai_thao_duong",0), comorbidities.get("rl_lipid_mau",0),
         comorbidities.get("suy_than_man",0)))
    conn.commit(); pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close(); return pid

def update_patient_comorbidities(pid, comorbidities: dict):
    conn = get_conn()
    conn.execute("""UPDATE patients SET dai_thao_duong=?,rl_lipid_mau=?,suy_than_man=?,
        updated_at=CURRENT_TIMESTAMP WHERE id=?""",
        (comorbidities["dai_thao_duong"], comorbidities["rl_lipid_mau"],
         comorbidities["suy_than_man"], pid))
    conn.commit(); conn.close()

def insert_visit(pid, visit_date, ldl, hdl, trig, prob=None):
    conn = get_conn()
    conn.execute("INSERT INTO visits (patient_id,visit_date,ldl,hdl,triglycerid,prediction_prob) VALUES (?,?,?,?,?,?)",
        (pid, visit_date, ldl, hdl, trig, prob))
    conn.commit(); conn.close()

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────
def compute_features(patient, visits_df, new_ldl, new_hdl, new_trig, visit_year):
    all_ldl = list(visits_df["ldl"])+[new_ldl] if not visits_df.empty else [new_ldl]
    all_hdl = list(visits_df["hdl"])+[new_hdl] if not visits_df.empty else [new_hdl]
    all_trig = list(visits_df["triglycerid"])+[new_trig] if not visits_df.empty else [new_trig]
    def agg(vals):
        arr = np.array(vals, dtype=float)
        return {"mean":arr.mean(),"max":arr.max(),"min":arr.min(),
                "std":arr.std(ddof=1) if len(arr)>1 else 0.0,"last":arr[-1]}
    l,h,t = agg(all_ldl), agg(all_hdl), agg(all_trig)
    gender_str = "Nam" if patient["gender"] == 1 else "Nữ"
    encoded_gender = pipeline["label_encoders"]["GioiTinh"].transform([gender_str])[0]
    
    feat = {"GioiTinh": encoded_gender,
        "LDL_mean":l["mean"],"LDL_max":l["max"],"LDL_min":l["min"],"LDL_std":l["std"],"LDL_last":l["last"],
        "HDL_mean":h["mean"],"HDL_max":h["max"],"HDL_min":h["min"],"HDL_std":h["std"],"HDL_last":h["last"],
        "Triglycerid_mean":t["mean"],"Triglycerid_max":t["max"],"Triglycerid_min":t["min"],"Triglycerid_std":t["std"],"Triglycerid_last":t["last"],
        "Age": 2026 - patient["birth_year"],
        "dai_thao_duong":patient.get("dai_thao_duong", 0),"rl_lipid_mau":patient.get("rl_lipid_mau", 0),
        "suy_than_man":patient.get("suy_than_man", 0)}
    return pd.DataFrame([feat], columns=SELECTED_FEATURES)

# ── 4. PREPROCESSING ─────────────────────────────────────────────────────
def preprocess(df_raw):
    df = df_raw.copy(); iqr_bounds = pipeline["iqr_bounds"]
    for col in CONTINUOUS_SELECTED:
        if col in iqr_bounds:
            lo, hi = iqr_bounds[col]; df[col] = df[col].clip(lo, hi)
        if col in scaler_map:
            mean, scale = scaler_map[col]; df[col] = (df[col]-mean)/scale
    return df

# ── 5. PREDICT + SHAP ────────────────────────────────────────────────────
def predict_risk(df_proc):
    return float(model.predict_proba(df_proc)[:,1][0])

def shap_waterfall(df_proc):
    sv = explainer(df_proc)
    fig, _ = plt.subplots(figsize=(8,6))
    shap.plots.waterfall(sv[0], show=False)
    fig = plt.gcf(); fig.set_size_inches(9,6); plt.tight_layout()
    return fig

# ── 6. RENDER PREDICTION RESULTS (reusable) ──────────────────────────────
def render_results(prob, df_proc, pid):
    pct = prob * 100
    st.markdown("---")
    res_col, shap_col = st.columns([1, 2])
    with res_col:
        if pct >= 70:
            css, icon, text = "risk-high", "🚨", "NGUY CƠ CAO"
        elif pct >= 30:
            css, icon, text = "risk-medium", "⚠️", "CẦN THEO DÕI"
        else:
            css, icon, text = "risk-low", "✅", "NGUY CƠ THẤP"
        st.markdown(f"""<div class="{css}">
            <div class="risk-label">Xác suất biến chứng tim mạch</div>
            <div class="risk-value">{pct:.1f}%</div>
            <div class="risk-text">{icon} {text}</div></div>""", unsafe_allow_html=True)
        st.markdown("")
        if st.button("💾 Lưu hồ sơ", use_container_width=True):
            v = st.session_state["last_visit"]
            if pid is None:
                pat_info = st.session_state.get("new_patient_info")
                try:
                    pid = insert_patient(pat_info["code"], pat_info["birth_year"], pat_info["gender"], pat_info["comor"])
                except sqlite3.IntegrityError:
                    st.error("Mã bệnh nhân đã tồn tại!")
                    st.stop()
            insert_visit(pid, str(v["date"]), v["ldl"], v["hdl"], v["trig"], prob)
            st.success("Đã lưu thành công!")
            for k in ["last_prob","last_df_proc","last_visit","last_pid","new_patient_info"]:
                st.session_state.pop(k, None)
            st.rerun()
    with shap_col:
        st.markdown("#### 🧠 Giải thích Mô hình (SHAP)")
        st.caption("Biểu đồ cho thấy yếu tố nào đẩy nguy cơ lên (đỏ) hoặc kéo xuống (xanh).")
        try:
            fig = shap_waterfall(df_proc)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        except Exception as e:
            st.warning(f"Không thể hiển thị SHAP: {e}")

# ── 7. CUSTOM CSS ─────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #1a3a6b 0%, #2d6cdf 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; }
.main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
.main-header p  { color: #ccdcf5; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
.card { background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 1rem; border: 1px solid #eef1f6; }
.card h3 { margin-top: 0; color: #1a3a6b; font-size: 1.1rem; }
.risk-high   { background: linear-gradient(135deg, #ff4d4f, #cf1322);
    color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
.risk-medium { background: linear-gradient(135deg, #faad14, #d48806);
    color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
.risk-low    { background: linear-gradient(135deg, #52c41a, #237804);
    color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
.risk-label  { font-size: 1rem; opacity: 0.9; margin-bottom: 0.3rem; }
.risk-value  { font-size: 2.8rem; font-weight: 700; }
.risk-text   { font-size: 1.1rem; font-weight: 600; margin-top: 0.3rem; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-box { flex: 1; background: #f7f9fc; border-radius: 10px; padding: 0.8rem 1rem;
    border: 1px solid #e8ecf2; text-align: center; }
.metric-box .label { font-size: 0.75rem; color: #8c8c8c; }
.metric-box .value { font-size: 1.3rem; font-weight: 600; color: #1a3a6b; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #f0f4fa 0%, #fff 100%); }
</style>""", unsafe_allow_html=True)

# ── 8. HEADER ─────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
    <h1>❤️ Hệ thống Dự đoán Nguy cơ Biến chứng Tim mạch</h1>
    <p>Ứng dụng Machine Learning (XGBoost) kết hợp Explainable AI (SHAP) — Đồ án tốt nghiệp</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 9.  SIDEBAR — Search by Patient Code
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🔍 Tìm kiếm Bệnh nhân")
search_code = st.sidebar.text_input("Nhập Mã Bệnh nhân", placeholder="VD: BN001",
                                     key="search_code")
btn_search = st.sidebar.button("🔎 Tìm kiếm", use_container_width=True)

if btn_search and search_code.strip():
    new_search = search_code.strip()
    if st.session_state.get("searched_code") != new_search:
        # Xóa kết quả dự đoán của bệnh nhân cũ khi tìm bệnh nhân mới
        for k in ["last_prob", "last_df_proc", "last_visit", "last_pid", "new_patient_info"]:
            st.session_state.pop(k, None)
    st.session_state["searched_code"] = new_search

searched_code = st.session_state.get("searched_code", "")

# ── Landing page (no search yet) ─────────────────────────────────────────
if not searched_code:
    st.info("👈 Nhập **Mã Bệnh nhân** ở thanh bên trái và ấn **Tìm kiếm** để bắt đầu.")
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#aaa;font-size:.8rem;'>"
        "© 2026 — Đồ án Ứng dụng ML trong Dự đoán Nguy cơ Biến chứng Tim mạch</div>",
        unsafe_allow_html=True)
    st.stop()

patient = fetch_patient_by_code(searched_code)

# ══════════════════════════════════════════════════════════════════════════
# 10.  PATIENT NOT FOUND → Create New + First Visit
# ══════════════════════════════════════════════════════════════════════════
if patient is None:
    st.warning(f"Không tìm thấy bệnh nhân có mã **{searched_code}**. "
               "Vui lòng tạo hồ sơ mới bên dưới.")

    with st.form("new_patient_full"):
        st.markdown("### 📝 Tạo Hồ sơ Bệnh nhân Mới")

        # ── Row 1: Administrative info ────────────────────────────────────
        st.markdown("**Thông tin hành chính**")
        c1, c2 = st.columns(2)
        new_byear  = c1.number_input("Năm sinh *", 1920, 2010, 1960)
        new_gender = c2.selectbox("Giới tính", ["Nam", "Nữ"])

        # ── Row 2: Comorbidities ──────────────────────────────────────────
        st.markdown("**Tiền sử bệnh nền**")
        cc1, cc2, cc3 = st.columns(3)
        cb_dtd = cc1.checkbox("Đái tháo đường")
        cb_rl  = cc2.checkbox("Rối loạn lipid máu")
        cb_stm = cc3.checkbox("Suy thận mạn")

        # ── Row 3: First visit lab values ─────────────────────────────────
        st.markdown("**Kết quả xét nghiệm lần khám đầu tiên**")
        v1, v2, v3, v4 = st.columns(4)
        fv_date = v1.date_input("Ngày khám", date.today(), key="fv_date")
        fv_ldl  = v2.number_input("LDL (mmol/L)", 0.1, 15.0, 2.5, 0.1, key="fv_ldl")
        fv_hdl  = v3.number_input("HDL (mmol/L)", 0.1, 5.0, 1.2, 0.05, key="fv_hdl")
        fv_trig = v4.number_input("Triglycerid (mmol/L)", 0.1, 30.0, 1.5, 0.1, key="fv_trig")

        submitted = st.form_submit_button("🔍 Phân tích Nguy cơ",
                                          type="primary", use_container_width=True)
        if submitted:
            comor = {"dai_thao_duong":int(cb_dtd),"rl_lipid_mau":int(cb_rl),
                     "suy_than_man":int(cb_stm)}
            # Build patient dict for prediction
            pat = {"gender":1 if new_gender=="Nam" else 0,
                   "birth_year":new_byear, **comor}
            df_raw = compute_features(pat, pd.DataFrame(), fv_ldl, fv_hdl, fv_trig, fv_date.year)
            df_proc = preprocess(df_raw)
            prob = predict_risk(df_proc)
            
            st.session_state["last_prob"] = prob
            st.session_state["last_df_proc"] = df_proc
            st.session_state["last_visit"] = {"date":fv_date,"ldl":fv_ldl,"hdl":fv_hdl,"trig":fv_trig}
            st.session_state["last_pid"] = None
            st.session_state["new_patient_info"] = {
                "code": searched_code,
                "birth_year": new_byear,
                "gender": 1 if new_gender=="Nam" else 0,
                "comor": comor
            }

    # Show results if just predicted
    if "last_prob" in st.session_state and "last_pid" in st.session_state:
        render_results(st.session_state["last_prob"],
                       st.session_state["last_df_proc"],
                       st.session_state["last_pid"])
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# 11.  PATIENT FOUND → Existing patient flow
# ══════════════════════════════════════════════════════════════════════════
visits = fetch_visits(patient["id"])
age_now = date.today().year - patient["birth_year"]

# ── Patient Profile Card ─────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f"### 📋 Hồ sơ Bệnh nhân: {patient['patient_code']}")
col_info, col_comor = st.columns(2)
with col_info:
    st.markdown(f"""<div class="metric-row">
      <div class="metric-box"><div class="label">Tuổi</div><div class="value">{age_now}</div></div>
      <div class="metric-box"><div class="label">Giới tính</div><div class="value">{'Nam' if patient['gender']==1 else 'Nữ'}</div></div>
      <div class="metric-box"><div class="label">Số lần khám</div><div class="value">{len(visits)}</div></div>
    </div>""", unsafe_allow_html=True)
with col_comor:
    cn = [LABEL_VI.get(k,k) for k in ["dai_thao_duong","rl_lipid_mau","suy_than_man"] if patient.get(k,0)==1]
    st.markdown("**Tiền sử bệnh:** " + (" · ".join([f"🔴 {c}" for c in cn]) if cn else "🟢 Không có bệnh nền"))
st.markdown('</div>', unsafe_allow_html=True)

# ── Visit History ────────────────────────────────────────────────────────
if not visits.empty:
    st.markdown('<div class="card"><h3>📊 Lịch sử Xét nghiệm</h3>', unsafe_allow_html=True)
    ddf = visits[["visit_date","ldl","hdl","triglycerid","prediction_prob"]].copy()
    ddf.columns = ["Ngày khám","LDL (mmol/L)","HDL (mmol/L)","Triglycerid (mmol/L)","Nguy cơ (%)"]
    ddf["Nguy cơ (%)"] = ddf["Nguy cơ (%)"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    st.dataframe(ddf, use_container_width=True, hide_index=True)
    cdf = visits[["visit_date","ldl","hdl","triglycerid"]].copy()
    cdf["visit_date"] = pd.to_datetime(cdf["visit_date"])
    cdf = cdf.set_index("visit_date"); cdf.columns = ["LDL","HDL","Triglycerid"]
    st.line_chart(cdf, height=220)
    st.markdown('</div>', unsafe_allow_html=True)

# ── New Visit Input ──────────────────────────────────────────────────────
st.markdown('<div class="card"><h3>🩺 Nhập Kết quả Khám hôm nay</h3>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
v_date = c1.date_input("Ngày khám", date.today())
v_ldl  = c2.number_input("LDL (mmol/L)", 0.1, 15.0, 2.5, 0.1)
v_hdl  = c3.number_input("HDL (mmol/L)", 0.1, 5.0, 1.2, 0.05)
v_trig = c4.number_input("Triglycerid (mmol/L)", 0.1, 30.0, 1.5, 0.1)

with st.expander("🔄 Cập nhật tiền sử bệnh (nếu có thay đổi)"):
    uc = st.columns(3)
    up_dtd = uc[0].checkbox("Đái tháo đường", value=bool(patient.get("dai_thao_duong", 0)), key="up_dtd")
    up_rl  = uc[1].checkbox("Rối loạn lipid máu", value=bool(patient.get("rl_lipid_mau", 0)), key="up_rl")
    up_stm = uc[2].checkbox("Suy thận mạn", value=bool(patient.get("suy_than_man", 0)), key="up_stm")
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────────────────
if st.button("🔍 Phân tích Nguy cơ", type="primary", use_container_width=True):
    comor = {"dai_thao_duong":int(up_dtd),"rl_lipid_mau":int(up_rl),
             "suy_than_man":int(up_stm)}
    update_patient_comorbidities(patient["id"], comor)
    pat = fetch_patient(patient["id"]); pat.update(comor)
    df_raw = compute_features(pat, visits, v_ldl, v_hdl, v_trig, v_date.year)
    df_proc = preprocess(df_raw)
    prob = predict_risk(df_proc)
    st.session_state["last_prob"] = prob
    st.session_state["last_df_proc"] = df_proc
    st.session_state["last_visit"] = {"date":v_date,"ldl":v_ldl,"hdl":v_hdl,"trig":v_trig}
    st.session_state["last_pid"] = patient["id"]

# ── Display Results ──────────────────────────────────────────────────────
if "last_prob" in st.session_state and st.session_state.get("last_pid") == patient["id"]:
    render_results(st.session_state["last_prob"],
                   st.session_state["last_df_proc"], patient["id"])

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#aaa;font-size:.8rem;'>"
    "© 2026 — Đồ án Ứng dụng ML trong Dự đoán Nguy cơ Biến chứng Tim mạch</div>",
    unsafe_allow_html=True)
