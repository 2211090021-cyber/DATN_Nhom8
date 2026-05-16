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

def compute_shap(df_proc):
    return explainer(df_proc)

def shap_waterfall(sv):
    fig, _ = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(sv[0], show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 6); plt.tight_layout()
    return fig

# ── 5b. CLINICAL COMMENTARY ──────────────────────────────────────────────
def _ldl_comment(v):
    if v > 4.12:   return ("🔴", f"**LDL rất cao** ({v:.1f} mmol/L > 4.1): tăng tích tụ cholesterol trong động mạch, nguy cơ xơ vữa cao")
    if v > 3.35:   return ("🟠", f"**LDL cao** ({v:.1f} mmol/L > 3.4): cần xem xét thuốc hạ lipid và kiểm soát chế độ ăn")
    if v > 2.6:    return ("🟡", f"**LDL ranh giới** ({v:.1f} mmol/L > 2.6): cần theo dõi định kỳ, điều chỉnh chế độ ăn")
    return             ("🟢", f"**LDL kiểm soát tốt** ({v:.1f} mmol/L): trong ngưỡng an toàn")

def _hdl_comment(v):
    if v < 1.0:    return ("🔴", f"**HDL thấp** ({v:.2f} mmol/L < 1.0): khả năng bảo vệ tim mạch suy giảm nghiêm trọng")
    if v < 1.3:    return ("🟠", f"**HDL chưa đủ** ({v:.2f} mmol/L): mức lý tưởng > 1.55 mmol/L, nên tăng cường vận động")
    if v < 1.55:   return ("🟡", f"**HDL chấp nhận được** ({v:.2f} mmol/L): cần duy trì lối sống lành mạnh")
    return             ("🟢", f"**HDL bảo vệ tốt** ({v:.2f} mmol/L ≥ 1.55): cholesterol tốt ở mức lý tưởng")

def _trig_comment(v):
    if v > 5.65:   return ("🔴", f"**Triglycerid rất cao** ({v:.1f} mmol/L > 5.65): nguy cơ viêm tụy cấp và bệnh tim nghiêm trọng")
    if v > 2.26:   return ("🟠", f"**Triglycerid cao** ({v:.1f} mmol/L > 2.26): cần hạn chế đường, rượu và chất béo")
    if v > 1.7:    return ("🟡", f"**Triglycerid ranh giới** ({v:.1f} mmol/L > 1.7): nên kiểm soát chế độ ăn uống")
    return             ("🟢", f"**Triglycerid bình thường** ({v:.1f} mmol/L): trong ngưỡng an toàn")

def _age_comment(v):
    v = int(v)
    if v >= 75:    return ("🔴", f"**Tuổi rất cao** ({v} tuổi ≥ 75): nhóm nguy cơ tim mạch rất cao theo độ tuổi")
    if v >= 65:    return ("🟠", f"**Cao tuổi** ({v} tuổi ≥ 65): yếu tố nguy cơ không thể thay đổi, cần kiểm soát các yếu tố khác")
    if v >= 45:    return ("🟡", f"**Trung niên** ({v} tuổi ≥ 45): nguy cơ bắt đầu tích lũy, cần tầm soát định kỳ")
    return             ("🟢", f"**Tuổi** ({v} tuổi): yếu tố tuổi chưa phải nguy cơ chính")

_COMOR_META = {
    "dai_thao_duong": ("🔴", "Đái tháo đường", "làm tổn thương mạch máu và thần kinh, tăng biến chứng tim mạch"),
    "rl_lipid_mau":   ("🟠", "Rối loạn lipid máu", "ảnh hưởng trực tiếp đến chuyển hóa mỡ và xơ vữa mạch"),
    "suy_than_man":   ("🔴", "Suy thận mạn", "thận suy làm tăng huyết áp và gánh nặng tim mạch"),
}

def generate_commentary(df_raw, sv, last_visit=None):
    """Trả về list (icon, text) theo mức độ đóng góp SHAP, dựa trên ngưỡng lâm sàng."""
    raw = df_raw.iloc[0].to_dict()
    shap_dict = dict(zip(sv[0].feature_names, sv[0].values))

    # Gộp các feature cùng nhóm, lấy feature có |SHAP| lớn nhất đại diện
    GROUP_PREFIXES = [("LDL", "LDL"), ("HDL", "HDL"), ("Triglycerid", "Triglycerid")]
    SCALAR_FEATS  = ["Age", "dai_thao_duong", "rl_lipid_mau", "suy_than_man"]

    group_best = {}  # group_name -> (feature_name, shap_val)
    for prefix, gname in GROUP_PREFIXES:
        candidates = [(f, v) for f, v in shap_dict.items() if f.startswith(prefix)]
        if candidates:
            best = max(candidates, key=lambda x: abs(x[1]))
            group_best[gname] = best
    for feat in SCALAR_FEATS:
        if feat in shap_dict:
            group_best[feat] = (feat, shap_dict[feat])

    # Sắp xếp nhóm theo |SHAP| giảm dần
    ranked = sorted(group_best.items(), key=lambda x: abs(x[1][1]), reverse=True)

    lines = []
    for gname, (_, sval) in ranked:
        if gname == "LDL":
            v = last_visit["ldl"] if last_visit else raw.get("LDL_last", raw.get("LDL_mean"))
            if v is not None: lines.append(_ldl_comment(float(v)))
        elif gname == "HDL":
            v = last_visit["hdl"] if last_visit else raw.get("HDL_mean")
            if v is not None: lines.append(_hdl_comment(float(v)))
        elif gname == "Triglycerid":
            v = last_visit["trig"] if last_visit else raw.get("Triglycerid_last", raw.get("Triglycerid_mean"))
            if v is not None: lines.append(_trig_comment(float(v)))
        elif gname == "Age":
            v = raw.get("Age")
            if v is not None: lines.append(_age_comment(float(v)))
        elif gname in _COMOR_META:
            if raw.get(gname, 0) == 1:
                ico, label, detail = _COMOR_META[gname]
                lines.append((ico, f"**{label}**: {detail}"))
            # Nếu bệnh nền = 0, không nhận xét
    return lines[:5]

# ── 6. RENDER PREDICTION RESULTS (reusable) ──────────────────────────────
def render_results(prob, df_raw, df_proc, pid):
    pct = prob * 100
    st.markdown("---")

    # Tính SHAP ngầm để ranking — không hiển thị chart
    sv = None
    try:
        sv = compute_shap(df_proc)
    except Exception:
        pass

    res_col, comment_col = st.columns([1, 1.6])

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
            for k in ["last_prob", "last_df_raw", "last_df_proc", "last_visit", "last_pid", "new_patient_info"]:
                st.session_state.pop(k, None)
            st.rerun()

    with comment_col:
        st.markdown("#### 🩺 Nhận xét Lâm sàng")
        st.caption("Các chỉ số ảnh hưởng nhiều nhất đến kết quả, so với ngưỡng lâm sàng chuẩn.")
        if sv is not None:
            comments = generate_commentary(df_raw, sv, st.session_state.get("last_visit"))
            if comments:
                for ico, msg in comments:
                    st.markdown(f"{ico} &nbsp; {msg}")
            else:
                st.info("Tất cả chỉ số trong ngưỡng bình thường.")
        else:
            st.info("Không thể tạo nhận xét.")

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
        for k in ["last_prob", "last_df_raw", "last_df_proc", "last_visit", "last_pid", "new_patient_info"]:
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
            st.session_state["last_df_raw"] = df_raw
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
                       st.session_state["last_df_raw"],
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
    st.session_state["last_df_raw"] = df_raw
    st.session_state["last_df_proc"] = df_proc
    st.session_state["last_visit"] = {"date":v_date,"ldl":v_ldl,"hdl":v_hdl,"trig":v_trig}
    st.session_state["last_pid"] = patient["id"]

# ── Display Results ──────────────────────────────────────────────────────
if "last_prob" in st.session_state and st.session_state.get("last_pid") == patient["id"]:
    render_results(st.session_state["last_prob"],
                   st.session_state["last_df_raw"],
                   st.session_state["last_df_proc"], patient["id"])

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#aaa;font-size:.8rem;'>"
    "© 2026 — Đồ án Ứng dụng ML trong Dự đoán Nguy cơ Biến chứng Tim mạch</div>",
    unsafe_allow_html=True)