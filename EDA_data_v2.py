"""
=============================================================================
 Cardiovascular Complication — Exploratory Data Analysis (v2 Refactored)
=============================================================================
 Dataset  : docs/YCDL_Features_Mapped-dropped.xlsx  (raw, pre-scaling)
 Target   : 'Target'  (0 = No Complication, 1 = Complication)
 Outputs  : outputs/eda_plots_v2/   (all figures)
             outputs/statistical_summary.csv
             outputs/descriptive_stats.csv
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

# ── Paths ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Exploratory Data Analysis (v2)")
parser.add_argument("--input", type=str, default="data/preprocess/YCDL_Features_Mapped.xlsx", help="Input file path")
parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
args = parser.parse_args()

EXCEL_PATH = args.input
PLOT_DIR   = os.path.join(args.output_dir, "eda_plots_v2")
STAT_CSV   = os.path.join(args.output_dir, "statistical_summary.csv")
DESC_CSV   = os.path.join(args.output_dir, "descriptive_stats.csv")
CAT_DIR    = os.path.join(args.output_dir, "categories")
CONT_DIR   = os.path.join(PLOT_DIR, "continuous_distribution")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(CONT_DIR, exist_ok=True)

# ── Style palette ──────────────────────────────────────────────────────────
PAL = {"No Complication": "#4A90D9", "Complication": "#D94A4A"}
HUE_ORDER = ["No Complication", "Complication"]
BG      = "#FAFBFD"
GRID_CLR = "#E4E8ED"
ACCENT   = "#2C3E50"

def style(ax, title="", xlabel="", ylabel=""):
    """Apply clean journal-grade style."""
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C0C4CA")
    if title:  ax.set_title(title,  fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {path}")


# =============================================================================
# 0.  LOAD & CONFIGURE
# =============================================================================
print("\n" + "=" * 70)
print("  LOADING DATASET")
print("=" * 70)

df = pd.read_excel(EXCEL_PATH)
print(f"  Shape : {df.shape}  ({df.shape[0]:,} patients x {df.shape[1]} features)")

TARGET = "Target"
df["_label"] = df[TARGET].map({0: "No Complication", 1: "Complication"})

# ── Encode GioiTinh (object -> numeric for stats) ─────────────────────────
if df["GioiTinh"].dtype == object:
    df["GioiTinh"] = df["GioiTinh"].replace({"Nam": 1, "Nữ": 0, "Nu": 0})
    df["GioiTinh"] = pd.to_numeric(df["GioiTinh"], errors="coerce").fillna(0).astype(int)
    print("  GioiTinh encoded: Nam=1, Nữ/Nu=0")

# ── Column classification ──────────────────────────────────────────────────
CONTINUOUS_COLS = [
    "HBa1C_mean", "HBa1C_max", "HBa1C_min", "HBa1C_std", "HBa1C_last",
    "LDL_mean",   "LDL_max",   "LDL_min",   "LDL_std",   "LDL_last",
    "HDL_mean",   "HDL_max",   "HDL_min",   "HDL_std",   "HDL_last",
    "Triglycerid_mean", "Triglycerid_max", "Triglycerid_min",
    "Triglycerid_std",  "Triglycerid_last",
    "Age",
]
CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c in df.columns]

CATEGORICAL_COLS = [
    "GioiTinh", "dai_thao_duong", "rl_lipid_mau", "suy_than_man",
    "benh_mach_vanh", "Tang_huyet_ap",
]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

LABEL_MAP = {
    "GioiTinh":        "Giới tính",
    "dai_thao_duong":  "Đái tháo đường",
    "rl_lipid_mau":    "Rối loạn lipid máu",
    "suy_than_man":    "Suy thận mạn",
    "benh_mach_vanh":  "Bệnh mạch vành",
    "Tang_huyet_ap":   "Tăng huyết áp",
}
FEAT_LABEL = {c: LABEL_MAP.get(c, c.replace("_", " ")) for c in CONTINUOUS_COLS + CATEGORICAL_COLS}

# ── Flag zero-variance columns ────────────────────────────────────────────
zero_var = [c for c in CATEGORICAL_COLS if df[c].nunique() <= 1]
if zero_var:
    print(f"  [!] Zero-variance (will exclude): {zero_var}")
CATEGORICAL_ACTIVE = [c for c in CATEGORICAL_COLS if c not in zero_var]


# =============================================================================
# 1.  DESCRIPTIVE STATISTICS  (grouped by Target)
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 1 — DESCRIPTIVE STATISTICS (by Complication Group)")
print("=" * 70)

g0 = df[df[TARGET] == 0]
g1 = df[df[TARGET] == 1]

# ── 1a  Continuous summary by group ───────────────────────────────────────
def summarize_group(data, cols):
    d = data[cols].describe().T[["mean", "std", "min", "max", "50%"]]
    d.columns = ["Mean", "Std", "Min", "Max", "Median"]
    q1 = data[cols].quantile(0.25)
    q3 = data[cols].quantile(0.75)
    d["IQR"] = (q3 - q1).values
    return d[["Mean", "Median", "Std", "Min", "Max", "IQR"]].round(4)

desc_all = summarize_group(df, CONTINUOUS_COLS)
desc_g0  = summarize_group(g0, CONTINUOUS_COLS)
desc_g1  = summarize_group(g1, CONTINUOUS_COLS)

# Combine into multi-level table
desc_combined = pd.concat(
    {"All": desc_all, "No Complication": desc_g0, "Complication": desc_g1},
    axis=1
)
desc_combined.to_csv(DESC_CSV, encoding="utf-8-sig")
print(f"\n  Descriptive stats saved -> {DESC_CSV}")

print("\n  --- No Complication Group (n={:,}) ---".format(len(g0)))
print(desc_g0.to_string())
print("\n  --- Complication Group (n={:,}) ---".format(len(g1)))
print(desc_g1.to_string())

# ── 1b  Categorical frequency by group ────────────────────────────────────
print("\n  --- Categorical Frequencies ---")
for col in CATEGORICAL_ACTIVE:
    label = LABEL_MAP.get(col, col)
    ct = pd.crosstab(df[col], df["_label"], margins=True)
    pct = pd.crosstab(df[col], df["_label"], normalize="columns").round(4) * 100
    
    print(f"\n  [{label}]")
    safe = ct.to_string().encode("ascii", errors="replace").decode("ascii")
    print(f"  {safe}")
    
    cat_summary = pd.concat({"Count": ct, "Percentage (%)": pct}, axis=1)
    
    safe_filename = col.replace(" ", "_") + "_stats.csv"
    csv_path = os.path.join(CAT_DIR, safe_filename)
    cat_summary.to_csv(csv_path, encoding="utf-8-sig")

print(f"\n  [Saved] Categorical stats -> {CAT_DIR}/")


# =============================================================================
# 2.  VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 2 — VISUALIZATION")
print("=" * 70)

# ── 2a  Target class balance ──────────────────────────────────────────────
target_counts = df["_label"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=BG)
fig.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")

colors_list = [PAL["No Complication"], PAL["Complication"]]
bars = axes[0].bar(target_counts.index, target_counts.values,
                   color=colors_list, edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
style(axes[0], title="Absolute Count", xlabel="Class", ylabel="Count")

axes[1].pie(target_counts.values, labels=target_counts.index,
            colors=colors_list, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10))
axes[1].set_title("Proportion", fontsize=11, fontweight="bold")
fig.tight_layout()
save_fig(fig, "01_target_balance.png")

# ── 2b  Histogram + Boxplot per continuous variable ───────────────────────
print(f"  -> Plotting {len(CONTINUOUS_COLS)} continuous distributions to continuous_distribution/")

for col in CONTINUOUS_COLS:
    data = df[col].dropna()
    
    fig = plt.figure(figsize=(6, 5), facecolor=BG)
    gs_inner = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax_hist = fig.add_subplot(gs_inner[0])
    ax_box  = fig.add_subplot(gs_inner[1], sharex=ax_hist)

    ax_hist.hist(data, bins=40, color=PAL["No Complication"], alpha=0.55,
                 density=True, edgecolor="white", linewidth=0.3)
    data.plot.kde(ax=ax_hist, color="#1A3A6B", linewidth=1.8)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title(f"Distribution & Outliers: {FEAT_LABEL.get(col, col)}", fontsize=11, fontweight="bold", pad=8)
    ax_hist.tick_params(labelbottom=False)
    style(ax_hist)

    ax_box.boxplot(
        data, vert=False, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor=PAL["No Complication"], alpha=0.70, color="#1A3A6B"),
        medianprops=dict(color=PAL["Complication"], linewidth=2),
        whiskerprops=dict(color="#555", linestyle="--"),
        capprops=dict(color="#555"),
        flierprops=dict(marker="o", color=PAL["Complication"], alpha=0.35, markersize=3),
    )
    ax_box.set_xlabel(FEAT_LABEL.get(col, col), fontsize=10)
    ax_box.set_yticks([])
    style(ax_box)

    fig.tight_layout()
    safe_filename = col.replace(" ", "_") + "_dist.png"
    save_fig(fig, os.path.join("continuous_distribution", safe_filename))

# ── 2c  Grouped Boxplots: continuous by Target ───────────────────────────
n_cont = len(CONTINUOUS_COLS)
n_cols_biv = 4
n_rows_biv = int(np.ceil(n_cont / n_cols_biv))
fig, axes = plt.subplots(n_rows_biv, n_cols_biv,
                          figsize=(n_cols_biv * 4.5, n_rows_biv * 4.5), facecolor=BG)
axes_flat = axes.flatten()
fig.suptitle("Lab Indicators by Complication Status (Grouped Boxplots)",
             fontsize=14, fontweight="bold", y=1.01)

for ax, col in zip(axes_flat, CONTINUOUS_COLS):
    plot_df = df[["_label", col]].dropna()
    sns.boxplot(data=plot_df, x="_label", y=col, order=HUE_ORDER,
                palette=PAL, width=0.55, linewidth=1.0, fliersize=2, ax=ax)
    style(ax, title=FEAT_LABEL.get(col, col), ylabel="Value")
    ax.set_xlabel("")
    ax.set_xticklabels(["No Comp.", "Comp."], fontsize=9)

for ax in axes_flat[n_cont:]:
    ax.set_visible(False)
fig.tight_layout()
save_fig(fig, "03_grouped_boxplots.png")

# ── 2d  Bar charts — complication proportion per categorical variable ────
n_cat = len(CATEGORICAL_ACTIVE)
cols_cat = 3
rows_cat = int(np.ceil(n_cat / cols_cat))
fig, axes = plt.subplots(rows_cat, cols_cat, figsize=(cols_cat * 5.5, rows_cat * 4.5), facecolor=BG)
axes_flat = axes.flatten() if rows_cat > 1 else (axes if n_cat > 1 else [axes])
fig.suptitle("Complication Rate by Categorical Feature",
             fontsize=14, fontweight="bold", y=1.01)

for ax, col in zip(axes_flat, CATEGORICAL_ACTIVE):
    label = LABEL_MAP.get(col, col)
    ct    = pd.crosstab(df[col], df["_label"], normalize="index") * 100
    for cls in HUE_ORDER:
        if cls not in ct.columns:
            ct[cls] = 0.0
    ct = ct[HUE_ORDER]

    x = np.arange(len(ct))
    w = 0.38
    bars0 = ax.bar(x - w/2, ct["No Complication"], w, color=PAL["No Complication"],
                   edgecolor="white", linewidth=0.8, label="No Complication")
    bars1 = ax.bar(x + w/2, ct["Complication"],    w, color=PAL["Complication"],
                   edgecolor="white", linewidth=0.8, label="Complication")

    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in ct.index], fontsize=9)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=7, loc="upper right")
    style(ax, title=label, ylabel="Proportion (%)")

for ax in axes_flat[n_cat:]:
    ax.set_visible(False)
fig.tight_layout()
save_fig(fig, "04_categorical_complication_bars.png")

# ── 2e  Pearson Correlation Heatmap ──────────────────────────────────────
corr = df[CONTINUOUS_COLS].corr(method="pearson")
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.4, linecolor="#DDD",
    annot_kws={"size": 6.5}, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    ax=ax,
)
ax.set_title("Pearson Correlation Heatmap — Continuous Features",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
fig.tight_layout()
save_fig(fig, "05_correlation_heatmap.png")


# =============================================================================
# 3.  STATISTICAL HYPOTHESIS TESTING
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 3 — STATISTICAL HYPOTHESIS TESTING")
print("=" * 70)

results = []

# ── 3a  Continuous: Shapiro-Wilk -> t-test / Mann-Whitney U ─────────────
print(f"\n  {'Feature':<22} {'Normal?':>8} {'Test':<16} {'Statistic':>12} {'p-value':>12} {'Sig':>5}")
print("  " + "-" * 78)

for col in CONTINUOUS_COLS:
    d0 = g0[col].dropna()
    d1 = g1[col].dropna()
    if len(d0) < 8 or len(d1) < 8:
        continue

    s0 = d0.sample(min(5000, len(d0)), random_state=42)
    s1 = d1.sample(min(5000, len(d1)), random_state=42)
    _, p_shap0 = stats.shapiro(s0)
    _, p_shap1 = stats.shapiro(s1)
    is_normal  = (p_shap0 > 0.05) and (p_shap1 > 0.05)

    if is_normal:
        stat_val, p_val = stats.ttest_ind(d0, d1, equal_var=False)
        test_name = "Welch t-test"
    else:
        stat_val, p_val = stats.mannwhitneyu(d0, d1, alternative="two-sided")
        test_name = "Mann-Whitney U"

    sig = "Yes" if p_val < 0.05 else "No"
    print(f"  {col:<22} {'Yes' if is_normal else 'No':>8} {test_name:<16} {stat_val:>12.3f} {p_val:>12.6f} {sig:>5}")

    results.append({
        "Feature":       FEAT_LABEL.get(col, col),
        "OriginalCol":   col,
        "Type":          "Continuous",
        "Normal":        is_normal,
        "Test":          test_name,
        "Statistic":     round(stat_val, 4),
        "p-value":       round(p_val, 6),
        "Significant":   p_val < 0.05,
        "Mean_Group0":   round(d0.mean(), 4),
        "Mean_Group1":   round(d1.mean(), 4),
        "Median_Group0": round(d0.median(), 4),
        "Median_Group1": round(d1.median(), 4),
    })

# ── 3b  Categorical: Chi-square ──────────────────────────────────────────
print(f"\n  {'Feature':<26} {'Test':<12} {'Chi2':>12} {'p-value':>12} {'Sig':>5}")
print("  " + "-" * 70)

for col in CATEGORICAL_ACTIVE:
    label = LABEL_MAP.get(col, col)
    ct    = pd.crosstab(df[col], df[TARGET])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    sig = "Yes" if p_val < 0.05 else "No"
    print(f"  {label:<26} {'Chi-square':<12} {chi2:>12.3f} {p_val:>12.6f} {sig:>5}")

    results.append({
        "Feature":       label,
        "OriginalCol":   col,
        "Type":          "Categorical",
        "Normal":        "N/A",
        "Test":          "Chi-square",
        "Statistic":     round(chi2, 4),
        "p-value":       round(p_val, 6),
        "Significant":   p_val < 0.05,
        "Mean_Group0":   "N/A",
        "Mean_Group1":   "N/A",
        "Median_Group0": "N/A",
        "Median_Group1": "N/A",
    })

# ── Save summary CSV ─────────────────────────────────────────────────────
stat_df = pd.DataFrame(results)
stat_df.to_csv(STAT_CSV, index=False, encoding="utf-8-sig")
print(f"\n  [Saved] Statistical summary -> {STAT_CSV}")

# ── Significance bar plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
plot_df = stat_df.copy()
plot_df["-log10(p)"] = -np.log10(plot_df["p-value"].clip(lower=1e-300))
plot_df = plot_df.sort_values("-log10(p)", ascending=True)
bar_colors = [PAL["Complication"] if s else "#AABDD2" for s in plot_df["Significant"]]

ax.barh(plot_df["Feature"], plot_df["-log10(p)"],
        color=bar_colors, edgecolor="white", linewidth=0.5, height=0.65)
ax.axvline(-np.log10(0.05), color=ACCENT, linestyle="--", linewidth=1.2, label="p = 0.05")
ax.legend(fontsize=9)
style(ax, title="Statistical Significance — All Features", xlabel="-log10(p-value)")
fig.tight_layout()
save_fig(fig, "06_significance_barplot.png")


# =============================================================================
# 4.  MULTICOLLINEARITY CHECK  (threshold |r| > 0.9 per context.docx)
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 4 — MULTICOLLINEARITY CHECK")
print("=" * 70)

HIGH_R = 0.9  # Threshold from context.docx section 3.4
pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) > HIGH_R:
            pairs.append((cols[i], cols[j], round(r, 4)))

pairs.sort(key=lambda x: -abs(x[2]))

if pairs:
    print(f"\n  Feature pairs with |r| > {HIGH_R}:")
    print(f"  {'Feature A':<22} {'Feature B':<22} {'r':>8}")
    print("  " + "-" * 55)
    for a, b, r in pairs:
        print(f"  {a:<22} {b:<22} {r:>8.4f}")
    print(f"\n  [!] {len(pairs)} pair(s) flagged — consider retaining only representative")
    print("      aggregates (e.g., *_mean or *_last) per biomarker group.")
else:
    print("  No pairs exceed the threshold — multicollinearity is low.")


# =============================================================================
# 5.  FEATURE SELECTION RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("  SECTION 5 — FEATURE SELECTION RECOMMENDATIONS")
print("=" * 70)

# ── Non-significant features (p >= 0.05) ─────────────────────────────────
nonsig = stat_df[~stat_df["Significant"]]
print(f"\n  Features to consider removing (p >= 0.05):")
if nonsig.empty:
    print("    None — all features are statistically significant.")
else:
    for _, row in nonsig.iterrows():
        print(f"    [-] {row['Feature']:<26}  (p = {row['p-value']:.4f})")

# ── Zero-variance features ───────────────────────────────────────────────
if zero_var:
    print(f"\n  Zero-variance features (must remove):")
    for c in zero_var:
        lbl = LABEL_MAP.get(c, c)
        print(f"    [-] {lbl} ({c}) — all values identical, no discriminative power")

# ── Top significant features ─────────────────────────────────────────────
sig_only = stat_df[stat_df["Significant"]].copy()
top10 = sig_only.nsmallest(10, "p-value")
print(f"\n  Top 10 Most Significant Features:")
print(f"  {'#':<4} {'Feature':<26} {'Test':<16} {'p-value':>12}")
print("  " + "-" * 62)
for rank, (_, row) in enumerate(top10.iterrows(), start=1):
    print(f"  {rank:<4} {row['Feature']:<26} {row['Test']:<16} {row['p-value']:>12.6f}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
sig_count = stat_df["Significant"].sum()
tot = len(stat_df)
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  Total features tested             : {tot}")
print(f"  Statistically significant (p<0.05): {sig_count}  ({sig_count/tot*100:.0f} %)")
print(f"  High-correlation pairs (|r|>{HIGH_R})  : {len(pairs)}")
print(f"  Zero-variance features            : {len(zero_var)}")
print(f"  Plots saved to                    : {os.path.abspath(PLOT_DIR)}")
print(f"  Continuous dist directory         : {os.path.abspath(CONT_DIR)}")
print(f"  Statistical summary CSV           : {os.path.abspath(STAT_CSV)}")
print(f"  Descriptive stats CSV             : {os.path.abspath(DESC_CSV)}")
print(f"  Categorical stats directory       : {os.path.abspath(CAT_DIR)}")
print("=" * 70 + "\n")
