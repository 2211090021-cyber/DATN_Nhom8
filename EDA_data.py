"""
=============================================================================
 Cardiovascular Complication Prediction — Exploratory Data Analysis (EDA)
=============================================================================
 Dataset  : docs/YCDL_Features_Mapped-dropped.xlsx
 Target   : 'Target'  (0 = No Complication, 1 = Complication)
 Output   : outputs/eda_plots/
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# ── Paths ──────────────────────────────────────────────────────────────────
EXCEL_PATH = os.path.join("docs", "YCDL_Features_Mapped-dropped.xlsx")
PLOT_DIR   = os.path.join("outputs", "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Medical journal style ──────────────────────────────────────────────────
PALETTE    = {"No Complication": "#4C9BE8", "Complication": "#E05C5C"}
HUE_ORDER  = ["No Complication", "Complication"]
GRID_COLOR = "#E8ECF0"
BG_COLOR   = "#FAFBFC"

def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCED0")
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# =============================================================================
# 0.  LOAD DATA
# =============================================================================
print("\n" + "="*65)
print("  LOADING DATASET")
print("="*65)

df = pd.read_excel(EXCEL_PATH)
print(f"  Shape: {df.shape}  (rows × cols)")

TARGET_COL = "Target"
df["_label"] = df[TARGET_COL].map({0: "No Complication", 1: "Complication"})

# ── Column groups ──────────────────────────────────────────────────────────
CONTINUOUS_COLS = [
    "HBa1C_mean", "HBa1C_max", "HBa1C_min", "HBa1C_std", "HBa1C_last",
    "LDL_mean",   "LDL_max",   "LDL_min",   "LDL_std",   "LDL_last",
    "HDL_mean",   "HDL_max",   "HDL_min",   "HDL_std",   "HDL_last",
    "Triglycerid_mean", "Triglycerid_max", "Triglycerid_min",
    "Triglycerid_std",  "Triglycerid_last",
    "Age",
]
CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]

BINARY_COLS = [
    "GioiTinh", "dai_thao_duong", "rl_lipid_mau", "suy_than_man",
    "benh_mach_vanh", "suy_tim", "dot_quy", "rung_nhi", "Tang_huyet_ap",
]
BINARY_COLS = [c for c in BINARY_COLS if c in df.columns]

LABEL_MAP = {
    "GioiTinh":       "Gender",
    "dai_thao_duong": "Diabetes",
    "rl_lipid_mau":   "Dyslipidaemia",
    "suy_than_man":   "Chronic Kidney Dis.",
    "benh_mach_vanh": "Coronary Artery Dis.",
    "suy_tim":        "Heart Failure",
    "dot_quy":        "Stroke",
    "rung_nhi":       "Atrial Fibrillation",
    "Tang_huyet_ap":  "Hypertension",
}

KEY_BIOMARKERS = ["HBa1C_mean", "LDL_mean", "HDL_mean", "Triglycerid_mean", "Age"]
KEY_BIOMARKERS = [c for c in KEY_BIOMARKERS if c in df.columns]


# =============================================================================
# 1.  DATA OVERVIEW & DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*65)
print("  SECTION 1 — DATA OVERVIEW & DESCRIPTIVE STATISTICS")
print("="*65)

# ── 1a. Numerical summary ──────────────────────────────────────────────────
num_stats = df[CONTINUOUS_COLS].agg(["mean", "median", "std", "min", "max"])
iqr_vals  = df[CONTINUOUS_COLS].quantile(0.75) - df[CONTINUOUS_COLS].quantile(0.25)
num_stats.loc["IQR"] = iqr_vals
num_stats = num_stats.T.round(4)
print("\n  Numerical Summary (Continuous Variables):")
print(num_stats.to_string())

# ── 1b. Categorical / binary frequency tables ──────────────────────────────
print("\n  Categorical Frequency Tables:")
for col in BINARY_COLS:
    counts = df[col].value_counts(dropna=False)
    pct    = (counts / len(df) * 100).round(2)
    tbl    = pd.DataFrame({"Count": counts, "Pct (%)": pct})
    label  = LABEL_MAP.get(col, col)
    print(f"\n  [{label}]")
    safe_str = tbl.to_string().encode('ascii', errors='replace').decode('ascii')
    print(safe_str)

# ── 1c. Missing value heatmap ──────────────────────────────────────────────
miss_pct = df[CONTINUOUS_COLS + BINARY_COLS].isnull().mean() * 100
miss_df  = miss_pct.reset_index()
miss_df.columns = ["Feature", "Missing (%)"]
miss_matrix = miss_df.set_index("Feature").T  # 1 × n for heatmap

fig, ax = plt.subplots(figsize=(16, 2.5), facecolor=BG_COLOR)
sns.heatmap(
    miss_matrix, annot=True, fmt=".1f", cmap="Reds",
    linewidths=0.5, linecolor="#DDD",
    cbar_kws={"label": "Missing (%)", "shrink": 0.6},
    ax=ax, annot_kws={"size": 7},
)
ax.set_title("Missing Value Heatmap (% per Feature)", fontsize=12, fontweight="bold", pad=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
fig.tight_layout()
save(fig, "01_missing_value_heatmap.png")


# =============================================================================
# 2.  UNIVARIATE ANALYSIS
# =============================================================================
print("\n" + "="*65)
print("  SECTION 2 — UNIVARIATE ANALYSIS")
print("="*65)

# ── 2a. Histograms + KDE for key biomarkers ────────────────────────────────
n_bio = len(KEY_BIOMARKERS)
fig, axes = plt.subplots(1, n_bio, figsize=(4 * n_bio, 4), facecolor=BG_COLOR)
fig.suptitle("Distribution of Key Biomarkers (Histogram + KDE)", fontsize=13, fontweight="bold", y=1.01)

for ax, col in zip(axes, KEY_BIOMARKERS):
    data_clean = df[col].dropna()
    ax.hist(data_clean, bins=40, color="#4C9BE8", alpha=0.55, density=True, edgecolor="white", linewidth=0.3)
    data_clean.plot.kde(ax=ax, color="#1A3A6B", linewidth=2)
    # Normality test
    if len(data_clean) >= 8:
        _, p_sw = stats.shapiro(data_clean.sample(min(5000, len(data_clean)), random_state=42))
        norm_txt = f"Shapiro p={'<0.001' if p_sw < 0.001 else f'{p_sw:.3f}'}"
        ax.text(0.97, 0.96, norm_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#444", bbox=dict(fc="white", alpha=0.7, ec="none", pad=2))
    apply_style(ax, title=col.replace("_", " "), xlabel="Value", ylabel="Density")

fig.tight_layout()
save(fig, "02_biomarker_histograms_kde.png")

# ── 2b. Boxplots for all continuous variables ──────────────────────────────
n_cols_cont = len(CONTINUOUS_COLS)
n_cols_plot = 7
n_rows_plot = int(np.ceil(n_cols_cont / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                          figsize=(n_cols_plot * 2.2, n_rows_plot * 3.5),
                          facecolor=BG_COLOR)
axes_flat = axes.flatten() if n_rows_plot > 1 else axes
fig.suptitle("Boxplots — Continuous Variables (Post-Capping)", fontsize=13, fontweight="bold", y=1.01)

for ax, col in zip(axes_flat, CONTINUOUS_COLS):
    data_clean = df[col].dropna()
    bp = ax.boxplot(data_clean, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor="#4C9BE8", color="#1A3A6B", alpha=0.75),
                    medianprops=dict(color="#E05C5C", linewidth=2),
                    whiskerprops=dict(color="#555", linestyle="--"),
                    capprops=dict(color="#555"),
                    flierprops=dict(marker="o", color="#E05C5C", alpha=0.3, markersize=3))
    apply_style(ax, title=col.replace("_", " "), ylabel="Value")
    ax.set_xticks([])

for ax in axes_flat[n_cols_cont:]:
    ax.set_visible(False)

fig.tight_layout()
save(fig, "03_boxplots_continuous.png")

# ── 2c. Target class balance ───────────────────────────────────────────────
target_counts = df["_label"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=BG_COLOR)
fig.suptitle("Target Variable Distribution", fontsize=13, fontweight="bold")

colors_list = [PALETTE["No Complication"], PALETTE["Complication"]]
bars = axes[0].bar(target_counts.index, target_counts.values, color=colors_list, edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
apply_style(axes[0], title="Absolute Count", xlabel="Class", ylabel="Count")

wedge_colors = colors_list
axes[1].pie(target_counts.values, labels=target_counts.index,
            colors=wedge_colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10))
axes[1].set_title("Proportion", fontsize=11, fontweight="bold")

fig.tight_layout()
save(fig, "04_target_balance.png")


# =============================================================================
# 3.  BIVARIATE ANALYSIS
# =============================================================================
print("\n" + "="*65)
print("  SECTION 3 — BIVARIATE ANALYSIS")
print("="*65)

# ── 3a. Grouped boxplots: numerical vs. target ─────────────────────────────
n_bp = len(CONTINUOUS_COLS)
n_cols_biv = 7
n_rows_biv = int(np.ceil(n_bp / n_cols_biv))
fig, axes = plt.subplots(n_rows_biv, n_cols_biv,
                          figsize=(n_cols_biv * 2.5, n_rows_biv * 4),
                          facecolor=BG_COLOR)
axes_flat = axes.flatten() if n_rows_biv > 1 else axes
fig.suptitle("Lab Indicators by Complication Status (Grouped Boxplots)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, col in zip(axes_flat, CONTINUOUS_COLS):
    plot_df = df[["_label", col]].dropna()
    order   = HUE_ORDER
    palette = [PALETTE[k] for k in order if k in plot_df["_label"].unique()]
    present = [k for k in order if k in plot_df["_label"].unique()]
    sns.boxplot(data=plot_df, x="_label", y=col, order=present,
                palette=dict(zip(present, palette)),
                width=0.55, linewidth=1.0, fliersize=2, ax=ax)
    apply_style(ax, title=col.replace("_", " "), ylabel="Value")
    ax.set_xlabel("")
    ax.set_xticklabels(["No Comp.", "Comp."], fontsize=7)

for ax in axes_flat[n_bp:]:
    ax.set_visible(False)

fig.tight_layout()
save(fig, "05_bivariate_grouped_boxplots.png")

# ── 3b. Stacked bar charts: categorical vs. target ─────────────────────────
n_cat = len(BINARY_COLS)
n_cols_cat = 3
n_rows_cat = int(np.ceil(n_cat / n_cols_cat))
fig, axes = plt.subplots(n_rows_cat, n_cols_cat,
                          figsize=(n_cols_cat * 5, n_rows_cat * 4),
                          facecolor=BG_COLOR)
axes_flat = axes.flatten()
fig.suptitle("Complication Rate by Categorical Feature (Stacked Bar %)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, col in zip(axes_flat, BINARY_COLS):
    label = LABEL_MAP.get(col, col)
    ct    = pd.crosstab(df[col], df["_label"], normalize="index") * 100
    # ensure both columns present
    for cls in HUE_ORDER:
        if cls not in ct.columns:
            ct[cls] = 0.0
    ct = ct[HUE_ORDER]
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[PALETTE[k] for k in HUE_ORDER],
            edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)", fontsize=8)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    apply_style(ax)

for ax in axes_flat[n_cat:]:
    ax.set_visible(False)

fig.tight_layout()
save(fig, "06_categorical_stacked_bar.png")


# =============================================================================
# 4.  STATISTICAL HYPOTHESIS TESTING
# =============================================================================
print("\n" + "="*65)
print("  SECTION 4 — STATISTICAL HYPOTHESIS TESTING")
print("="*65)

group0 = df[df[TARGET_COL] == 0]
group1 = df[df[TARGET_COL] == 1]

stat_results = []

# ── 4a. Numerical: Shapiro → t-test or Mann-Whitney ───────────────────────
print("\n  Numerical Feature Tests:")
print(f"  {'Feature':<22} {'Test':<15} {'Statistic':>10} {'p-value':>10} {'Sig?':>6}")
print("  " + "-"*67)

for col in CONTINUOUS_COLS:
    g0 = group0[col].dropna()
    g1 = group1[col].dropna()
    if len(g0) < 8 or len(g1) < 8:
        continue

    # Shapiro on a sample (max 5000)
    s0 = g0.sample(min(5000, len(g0)), random_state=42)
    s1 = g1.sample(min(5000, len(g1)), random_state=42)
    _, p0 = stats.shapiro(s0)
    _, p1 = stats.shapiro(s1)
    normal = (p0 > 0.05) and (p1 > 0.05)

    if normal:
        stat_val, p_val = stats.ttest_ind(g0, g1, equal_var=False)
        test_name = "Welch t-test"
    else:
        stat_val, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        test_name = "Mann-Whitney U"

    sig = "✔" if p_val < 0.05 else "✗"
    print(f"  {col:<22} {test_name:<15} {stat_val:>10.3f} {p_val:>10.4f} {sig:>6}")
    stat_results.append({"feature": col, "type": "numerical",
                          "test": test_name, "statistic": stat_val,
                          "p_value": p_val, "significant": p_val < 0.05})

# ── 4b. Categorical: Chi-square ────────────────────────────────────────────
print("\n  Categorical Feature Tests (Chi-square):")
print(f"  {'Feature':<25} {'Chi2':>10} {'p-value':>10} {'Sig?':>6}")
print("  " + "-"*55)

for col in BINARY_COLS:
    ct = pd.crosstab(df[col], df[TARGET_COL])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    sig = "✔" if p_val < 0.05 else "✗"
    label = LABEL_MAP.get(col, col)
    print(f"  {label:<25} {chi2:>10.3f} {p_val:>10.4f} {sig:>6}")
    stat_results.append({"feature": label, "type": "categorical",
                          "test": "Chi-square", "statistic": chi2,
                          "p_value": p_val, "significant": p_val < 0.05})

stat_df = pd.DataFrame(stat_results)

# ── 4c. Significance plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
sig_plot = stat_df.copy()
sig_plot["-log10(p)"] = -np.log10(sig_plot["p_value"].clip(lower=1e-300))
sig_plot = sig_plot.sort_values("-log10(p)", ascending=True)

colors_bar = ["#E05C5C" if s else "#9BB8D4" for s in sig_plot["significant"]]
ax.barh(sig_plot["feature"], sig_plot["-log10(p)"], color=colors_bar, edgecolor="white", linewidth=0.5)
ax.axvline(-np.log10(0.05), color="#333", linestyle="--", linewidth=1.2, label="p = 0.05 threshold")
ax.set_xlabel("−log₁₀(p-value)", fontsize=10)
apply_style(ax, title="Statistical Significance: −log₁₀(p-value) per Feature")
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, "07_statistical_significance.png")


# =============================================================================
# 5.  CORRELATION & MULTICOLLINEARITY
# =============================================================================
print("\n" + "="*65)
print("  SECTION 5 — CORRELATION & MULTICOLLINEARITY")
print("="*65)

# ── 5a. Pearson Correlation Heatmap ───────────────────────────────────────
corr_df   = df[CONTINUOUS_COLS].dropna(how="all")
corr_mat  = corr_df.corr(method="pearson")

mask = np.triu(np.ones_like(corr_mat, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG_COLOR)
sns.heatmap(
    corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-1, vmax=1, linewidths=0.4, linecolor="#DDD",
    annot_kws={"size": 6.5}, cbar_kws={"shrink": 0.7},
    ax=ax,
)
ax.set_title("Pearson Correlation Heatmap — Continuous Features",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)
fig.tight_layout()
save(fig, "08_correlation_heatmap.png")

# ── 5b. High-correlation pairs (|r| > 0.85) ───────────────────────────────
HIGH_CORR_THRESHOLD = 0.85
high_corr_pairs = []
cols = corr_mat.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr_mat.iloc[i, j]
        if abs(r) > HIGH_CORR_THRESHOLD:
            high_corr_pairs.append({
                "Feature A": cols[i],
                "Feature B": cols[j],
                "Pearson r": round(r, 4),
            })

hc_df = pd.DataFrame(high_corr_pairs).sort_values("Pearson r", ascending=False)

print(f"\n  High-Correlation Pairs (|r| > {HIGH_CORR_THRESHOLD}):")
if hc_df.empty:
    print("  None found.")
else:
    print(hc_df.to_string(index=False))


# =============================================================================
# 6.  SUMMARY REPORT
# =============================================================================
print("\n" + "="*65)
print("  SUMMARY REPORT")
print("="*65)

sig_features = stat_df[stat_df["significant"]]["feature"].tolist()
print(f"\n  Statistically Significant Features (p < 0.05)  [{len(sig_features)}]:")
for f in sig_features:
    print(f"    • {f}")

print(f"\n  Top Correlated Feature Pairs (|r| > {HIGH_CORR_THRESHOLD})  [{len(hc_df)}]:")
if hc_df.empty:
    print("    None — no multicollinearity concerns.")
else:
    for _, row in hc_df.iterrows():
        print(f"    • {row['Feature A']}  ↔  {row['Feature B']}  (r = {row['Pearson r']})")

print(f"\n  All plots saved to: {os.path.abspath(PLOT_DIR)}")
print("="*65 + "\n")
