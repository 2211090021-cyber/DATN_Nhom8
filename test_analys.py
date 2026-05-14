"""
=============================================================================
 test_analys.py — Phân Tích Bệnh Nhân Nguy Cơ Cao / Thấp (SHAP + Champion Model)
=============================================================================
 Script load champion_model.pkl, chạy trên toàn bộ test set, sau đó:
   1. Chọn top-10 bệnh nhân nguy cơ CAO nhất (True Positive ưu tiên)
   2. Chọn top-10 bệnh nhân nguy cơ THẤP nhất (True Negative ưu tiên)
   3. Phân tích SHAP waterfall cho từng ca
   4. Tổng hợp kết luận lâm sàng → in ra console + lưu file báo cáo
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "champion_model.pkl")
FEAT_PATH     = os.path.join(BASE_DIR, "models", "selected_features.json")
X_TEST_PATH   = os.path.join(BASE_DIR, "data", "processed", "X_test_final.csv")
Y_TEST_PATH   = os.path.join(BASE_DIR, "data", "processed", "y_test_final.csv")
OUT_DIR       = os.path.join(BASE_DIR, "outputs", "patient_analysis")
REPORT_PATH   = os.path.join(OUT_DIR, "patient_risk_report.txt")

os.makedirs(OUT_DIR, exist_ok=True)

N_PATIENTS    = 10       # số bệnh nhân mỗi nhóm
HIGH_THRESH   = 0.5      # ngưỡng xác suất -> nguy cơ cao
LOW_THRESH    = 0.5      # ngưỡng xác suất -> nguy cơ thấp

# Mapping tên đặc trưng cho dễ đọc
FEAT_LABELS = {
    "Age":              "Tuổi",
    "GioiTinh":         "Giới tính (0=Nữ, 1=Nam)",
    "LDL_mean":         "LDL trung bình (mỡ xấu)",
    "LDL_max":          "LDL cao nhất",
    "LDL_min":          "LDL thấp nhất",
    "LDL_std":          "LDL độ biến động",
    "LDL_last":         "LDL lần đo cuối",
    "HDL_mean":         "HDL trung bình (mỡ tốt)",
    "HDL_min":          "HDL thấp nhất",
    "HDL_std":          "HDL độ biến động",
    "Triglycerid_mean": "Triglycerid trung bình",
    "Triglycerid_max":  "Triglycerid cao nhất",
    "Triglycerid_min":  "Triglycerid thấp nhất",
    "Triglycerid_last": "Triglycerid lần đo cuối",
    "dai_thao_duong":   "Đái tháo đường (0/1)",
    "rl_lipid_mau":     "Rối loạn lipid máu (0/1)",
    "suy_than_man":     "Suy thận mãn (0/1)",
}

def banner(text):
    w = 72
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)

# =============================================================================
# STEP 1 — LOAD
# =============================================================================
banner("STEP 1 — LOADING MODEL & DATA")

with open(FEAT_PATH, "r", encoding="utf-8") as f:
    selected_features = json.load(f)

model  = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

X_sel  = X_test[selected_features]

print(f"  Model        : {MODEL_PATH}")
print(f"  Test samples : {len(X_sel)}")
print(f"  Features used: {selected_features}")

# =============================================================================
# STEP 2 — PREDICT
# =============================================================================
banner("STEP 2 — GENERATING PREDICTIONS")

probs   = model.predict_proba(X_sel)[:, 1]
preds   = (probs >= HIGH_THRESH).astype(int)

acc  = (preds == y_test.values).mean()
print(f"  Accuracy (threshold={HIGH_THRESH}): {acc:.2%}")
print(f"  Positive rate in test set: {y_test.mean():.2%}")

# =============================================================================
# STEP 3 — SHAP VALUES
# =============================================================================
banner("STEP 3 — COMPUTING SHAP VALUES")

explainer       = shap.TreeExplainer(model)
shap_explanation = explainer(X_sel)
shap_values      = shap_explanation.values   # shape (n, n_features)

print("  SHAP computation done.")

# =============================================================================
# STEP 4 — SELECT PATIENTS
# =============================================================================
banner("STEP 4 — SELECTING TOP-10 HIGH RISK & TOP-10 LOW RISK PATIENTS")

df_result = pd.DataFrame({
    "idx":       range(len(X_sel)),
    "prob":      probs,
    "pred":      preds,
    "actual":    y_test.values,
})

# --- HIGH RISK: ưu tiên True Positive, sau đó lấy xác suất cao nhất ---
tp_mask  = (df_result["prob"] >= HIGH_THRESH) & (df_result["actual"] == 1)
fp_mask  = (df_result["prob"] >= HIGH_THRESH) & (df_result["actual"] == 0)

tp_sorted = df_result[tp_mask].sort_values("prob", ascending=False)
fp_sorted = df_result[fp_mask].sort_values("prob", ascending=False)
high_candidates = pd.concat([tp_sorted, fp_sorted])
high_10 = high_candidates.head(N_PATIENTS)

# --- LOW RISK: ưu tiên True Negative, sau đó lấy xác suất thấp nhất ---
tn_mask  = (df_result["prob"] < LOW_THRESH) & (df_result["actual"] == 0)
fn_mask  = (df_result["prob"] < LOW_THRESH) & (df_result["actual"] == 1)

tn_sorted = df_result[tn_mask].sort_values("prob", ascending=True)
fn_sorted = df_result[fn_mask].sort_values("prob", ascending=True)
low_candidates = pd.concat([tn_sorted, fn_sorted])
low_10 = low_candidates.head(N_PATIENTS)

print(f"  HIGH RISK patients selected : {len(high_10)}")
print(f"  LOW  RISK patients selected : {len(low_10)}")


# =============================================================================
# HELPER — PHÂN TÍCH LÂM SÀNG TỪ SHAP
# =============================================================================
# Các feature nhị phân (không z-score)
BINARY_FEATS = {"GioiTinh", "dai_thao_duong", "rl_lipid_mau", "suy_than_man", "benh_mach_vanh", "Tang_huyet_ap"}

# Nhãn giá trị nhị phân
BINARY_VALUE_LABELS = {
    "GioiTinh":      {0: "Nữ",                   1: "Nam"},
    "dai_thao_duong":{0: "Không có đái tháo đường", 1: "Có đái tháo đường"},
    "rl_lipid_mau":  {0: "Không có rối loạn lipid", 1: "Có rối loạn lipid máu"},
    "suy_than_man":  {0: "Không có suy thận mãn",   1: "Có suy thận mãn"},
    "benh_mach_vanh":{0: "Không có bệnh mạch vành", 1: "Có bệnh mạch vành"},
    "Tang_huyet_ap": {0: "Không tăng HA",          1: "Có tăng huyết áp"},
}

# Nhãn mô tả z-score theo ngưỡng
def zscore_label(z):
    """Chuyển z-score -> mô tả mức độ."""
    if   z >  1.5: return "rất cao"
    elif z >  0.5: return "cao"
    elif z >  0.0: return "hơi cao hơn trung bình"
    elif z > -0.5: return "hơi thấp hơn trung bình"
    elif z > -1.5: return "thấp"
    else:          return "rất thấp"

# Các feature mà giá trị CAO là TỐT (bảo vệ) — dùng để đảo nhãn mô tả
HIGH_IS_GOOD = {"HDL_mean", "HDL_max", "HDL_min", "HDL_last"}

def describe_feature_value(feat, fv):
    """
    Trả về chuỗi mô tả lâm sàng cụ thể cho giá trị đặc trưng.
    fv là giá trị z-score (continuous) hoặc 0/1 (binary).
    """
    if feat in BINARY_FEATS:
        key = 1 if round(fv) >= 1 else 0
        return BINARY_VALUE_LABELS.get(feat, {}).get(key, f"={int(round(fv))}")

    level = zscore_label(fv)
    # Tên gốc tường minh hơn
    short = feat.replace("_mean"," TB").replace("_max"," cao nhất") \
                .replace("_min"," thấp nhất").replace("_std"," biến động") \
                .replace("_last"," lần đo cuối")
    if feat == "Age":
        if   fv >  1.5: return "tuổi rất cao (người già)"
        elif fv >  0.5: return "tuổi cao"
        elif fv > -0.5: return "tuổi trung niên"
        elif fv > -1.5: return "tuổi trẻ"
        else:           return "tuổi rất trẻ"
    return f"{level}"


def clinical_reason(shap_row, feat_names, X_row, threshold=0.05):
    """
    Trả về 2 list: (yếu tố TĂNG nguy cơ, yếu tố BẢO VỆ/giảm nguy cơ).
    Mỗi entry có diễn giải lâm sàng cụ thể.
    """
    increase, decrease = [], []
    for feat, sv, fv in zip(feat_names, shap_row, X_row):
        base_label = FEAT_LABELS.get(feat, feat)
        if abs(sv) < threshold:
            continue
        val_desc = describe_feature_value(feat, fv)
        # Xây dựng chuỗi giải thích
        entry = f"{base_label}: {val_desc}  (tác động SHAP={sv:+.3f})"
        # Dùng SHAP để sort sau
        if sv > 0:
            increase.append((sv, entry))
        else:
            decrease.append((sv, entry))
    increase.sort(key=lambda x: x[0], reverse=True)
    decrease.sort(key=lambda x: x[0])
    return [e for _, e in increase], [e for _, e in decrease]


def summarize_patient(row_info, shap_row, feat_names, X_row, group):
    """Tạo chuỗi tóm tắt lâm sàng một bệnh nhân với giải thích cụ thể."""
    idx    = int(row_info["idx"])
    prob   = row_info["prob"]
    actual = int(row_info["actual"])
    tag    = "TP" if (group == "HIGH" and actual == 1) else \
             "FP" if (group == "HIGH" and actual == 0) else \
             "TN" if (group == "LOW"  and actual == 0) else "FN"

    # Chú thích tag
    tag_note = {
        "TP": "Dự đoán ĐÚNG — bệnh nhân thực sự có biến chứng",
        "FP": "Dự đoán SAI — mô hình báo cao nhưng thực tế không biến chứng",
        "TN": "Dự đoán ĐÚNG — bệnh nhân thực sự không biến chứng",
        "FN": "Dự đoán SAI (bỏ sót) — bệnh nhân thực sự CÓ biến chứng!",
    }[tag]

    inc, dec = clinical_reason(shap_row, feat_names, X_row)
    lines = []
    lines.append(f"  Bệnh nhân #{idx:04d}  |  Xác suất={prob:.1%}  |  "
                 f"Thực tế={'CÓ BIẾN CHỨNG' if actual == 1 else 'KHÔNG BC'}  |  [{tag}]")
    lines.append(f"  → {tag_note}")
    if inc:
        lines.append("    ▲ Yếu tố TĂNG nguy cơ (đẩy xác suất lên):")
        for e in inc:
            lines.append(f"      + {e}")
    if dec:
        lines.append("    ▼ Yếu tố BẢO VỆ (kéo xác suất xuống):")
        for e in dec:
            lines.append(f"      - {e}")
    return "\n".join(lines)


# =============================================================================
# STEP 5 — WATERFALL PLOTS
# =============================================================================
banner("STEP 5 — SAVING WATERFALL PLOTS")

def save_waterfall(idx, group, rank, tag):
    shap.plots.waterfall(shap_explanation[idx], show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 7)
    prob_val = probs[idx]
    actual   = y_test.values[idx]
    plt.suptitle(
        f"[{group}] Bệnh Nhân #{idx} — Rank {rank+1}  |  "
        f"P(nguy cơ)={prob_val:.1%}  |  Thực tế={'CÓ BC' if actual==1 else 'KHÔNG BC'}  [{tag}]",
        fontsize=9, y=1.01
    )
    fname = f"{group.lower()}_risk_rank{rank+1:02d}_idx{idx:04d}_{tag}.png"
    path  = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    [Saved] {path}")

print("\n  — HIGH RISK patients —")
for rank, (_, row) in enumerate(high_10.iterrows()):
    idx = int(row["idx"])
    tag = "TP" if row["actual"] == 1 else "FP"
    save_waterfall(idx, "HIGH", rank, tag)

print("\n  — LOW RISK patients —")
for rank, (_, row) in enumerate(low_10.iterrows()):
    idx = int(row["idx"])
    tag = "TN" if row["actual"] == 0 else "FN"
    save_waterfall(idx, "LOW", rank, tag)


# =============================================================================
# STEP 6 — BÁO CÁO PHÂN TÍCH
# =============================================================================
banner("STEP 6 — GENERATING CLINICAL ANALYSIS REPORT")

X_sel_arr = X_sel.values
report_lines = []

report_lines.append("=" * 72)
report_lines.append("  BÁO CÁO PHÂN TÍCH NGUY CƠ TIM MẠCH — CHAMPION MODEL (XGBoost)")
report_lines.append("  Dự án: Dự đoán biến chứng tim mạch ở bệnh nhân tăng huyết áp")
report_lines.append("=" * 72)
report_lines.append(f"\nTổng số bệnh nhân trong test set : {len(X_sel)}")
report_lines.append(f"Tỷ lệ thực sự CÓ biến chứng     : {y_test.mean():.2%}")
report_lines.append(f"Độ chính xác mô hình             : {acc:.2%}")
report_lines.append(f"Features sử dụng                 : {', '.join(selected_features)}")

# ── HIGH RISK ──
report_lines.append("\n" + "─" * 72)
report_lines.append("  NHÓM NGUY CƠ CAO — TOP 10 BỆNH NHÂN")
report_lines.append("─" * 72)
report_lines.append("  (Chọn theo xác suất dự đoán cao nhất; ưu tiên True Positive)\n")

for rank, (_, row) in enumerate(high_10.iterrows()):
    idx = int(row["idx"])
    tag = "TP" if row["actual"] == 1 else "FP"
    report_lines.append(f"{'─'*40}  [{rank+1}/10]")
    report_lines.append(
        summarize_patient(row, shap_values[idx], selected_features, X_sel_arr[idx], "HIGH")
    )
    report_lines.append("")

# ── LOW RISK ──
report_lines.append("\n" + "─" * 72)
report_lines.append("  NHÓM NGUY CƠ THẤP — TOP 10 BỆNH NHÂN")
report_lines.append("─" * 72)
report_lines.append("  (Chọn theo xác suất dự đoán thấp nhất; ưu tiên True Negative)\n")

for rank, (_, row) in enumerate(low_10.iterrows()):
    idx = int(row["idx"])
    tag = "TN" if row["actual"] == 0 else "FN"
    report_lines.append(f"{'─'*40}  [{rank+1}/10]")
    report_lines.append(
        summarize_patient(row, shap_values[idx], selected_features, X_sel_arr[idx], "LOW")
    )
    report_lines.append("")

# ── KẾT LUẬN TỔNG QUAN ──
report_lines.append("\n" + "=" * 72)
report_lines.append("  KẾT LUẬN LÂM SÀNG")
report_lines.append("=" * 72)

# Tính SHAP trung bình cho từng nhóm
high_idxs = high_10["idx"].astype(int).tolist()
low_idxs  = low_10["idx"].astype(int).tolist()

high_shap_mean = shap_values[high_idxs].mean(axis=0)
low_shap_mean  = shap_values[low_idxs].mean(axis=0)

report_lines.append("""
A. KHI NÀO BỆNH NHÂN CÓ NGUY CƠ CAO?
   Mô hình đánh giá nguy cơ CAO khi bệnh nhân hội tụ nhiều yếu tố sau:
""")
for feat, sv in sorted(zip(selected_features, high_shap_mean),
                        key=lambda x: x[1], reverse=True):
    if sv > 0.02:
        label = FEAT_LABELS.get(feat, feat)
        report_lines.append(f"   ➤ {label:40s}  (SHAP trung bình: {sv:+.4f})")

report_lines.append("""
   → Tổng kết: Bệnh nhân nguy cơ CAO thường là người CAO TUỔI, có LDL cao
     (đặc biệt LDL_max và LDL_mean), có bệnh đái tháo đường kèm theo, và/hoặc
     suy thận mãn. Triglycerid cao cũng là một yếu tố đẩy nguy cơ lên.
     HDL thấp (mỡ tốt thấp) làm mất đi tác dụng bảo vệ tim mạch.
""")

report_lines.append("""
B. KHI NÀO BỆNH NHÂN CÓ NGUY CƠ THẤP?
   Mô hình đánh giá nguy cơ THẤP khi bệnh nhân có đặc điểm sau:
""")
for feat, sv in sorted(zip(selected_features, low_shap_mean),
                        key=lambda x: x[1]):
    if sv < -0.02:
        label = FEAT_LABELS.get(feat, feat)
        report_lines.append(f"   ✓ {label:40s}  (SHAP trung bình: {sv:+.4f})")

report_lines.append("""
   → Tổng kết: Bệnh nhân nguy cơ THẤP thường là người TRẺ TUỔI, không có
     đái tháo đường hay suy thận mãn, có LDL được kiểm soát tốt (thấp),
     và HDL ở mức cao (mỡ tốt bảo vệ tim mạch). Triglycerid ổn định thấp.
""")

report_lines.append("""
C. GHI CHÚ VỀ FALSE NEGATIVE (Mô hình bỏ sót):
   Một số bệnh nhân THỰC TẾ CÓ biến chứng nhưng mô hình đánh giá nguy cơ thấp.
   Đây là "góc khuất" của dữ liệu — bệnh nhân này thường:
     - Còn trẻ và có xét nghiệm máu bình thường
     - Nguyên nhân biến chứng có thể do yếu tố KHÔNG được ghi nhận:
       hút thuốc lá, rượu bia, căng thẳng mãn tính, yếu tố di truyền.
   → Đề xuất: Thu thập thêm các biến lối sống để cải thiện recall.
""")

report_lines.append("=" * 72)
report_lines.append("  END OF REPORT")
report_lines.append("=" * 72 + "\n")

report_text = "\n".join(report_lines)

# In ra console
print(report_text)

# Lưu file
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n  [Báo cáo đã lưu] {REPORT_PATH}")
print(f"  [Waterfall plots] {OUT_DIR}/")
banner("PHÂN TÍCH HOÀN TẤT")
