# Auto-exported from 05_gnomad_figure7.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% Cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from google.colab import drive
drive.mount("/content/drive")

OUT_DIR = "/content/drive/MyDrive/gnomad_pbm_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figure7_panels")
os.makedirs(FIG_DIR, exist_ok=True)

variant_effect_summary = pd.read_csv(
    os.path.join(OUT_DIR, "variant_effect_summary.csv")
)

variant_pdz_delta = pd.read_csv(
    os.path.join(OUT_DIR, "variant_pdz_delta_long.csv")
)

pos_order = ["P-5", "P-4", "P-3", "P-2", "P-1", "P0"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close()


# ============================================================
# Figure 7B: variant count by PBM position
# ============================================================

pos_count = (
    variant_effect_summary
    .groupby("PBM_position")["variant_id"]
    .nunique()
    .reindex(pos_order)
    .reset_index()
    .rename(columns={"variant_id": "n_variants"})
)

plt.figure(figsize=(3.2, 2.6))
plt.bar(
    pos_count["PBM_position"],
    pos_count["n_variants"],
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("PBM position")
plt.ylabel("Number of variants")
plt.title("PBM-altering variants by position")
savefig("Figure7B_variant_count_by_position")


# ============================================================
# Figure 7C: max |ΔpKd| by PBM position
# ============================================================

plot_df = variant_effect_summary.copy()
plot_df["PBM_position"] = pd.Categorical(
    plot_df["PBM_position"],
    categories=pos_order,
    ordered=True,
)

data = [
    plot_df.loc[plot_df["PBM_position"] == pos, "max_abs_delta_pKd"].dropna()
    for pos in pos_order
]

plt.figure(figsize=(3.6, 2.8))
box = plt.boxplot(
    data,
    labels=pos_order,
    showfliers=False,
    patch_artist=True,
    widths=0.55,
)

for patch in box["boxes"]:
    patch.set_facecolor("#DCEBFA")
    patch.set_edgecolor("black")
    patch.set_linewidth(0.7)
    patch.set_alpha(0.55)
    patch.set_zorder(1)

for whisker in box["whiskers"]:
    whisker.set_zorder(1)

for cap in box["caps"]:
    cap.set_zorder(1)

for median in box["medians"]:
    median.set_color("black")
    median.set_linewidth(1.0)
    median.set_zorder(2)

rng = np.random.default_rng(1)
for i, vals in enumerate(data, start=1):
    x = rng.normal(i, 0.055, size=len(vals))
    plt.scatter(
        x,
        vals,
        s=12,
        alpha=0.70,
        color="#2F5D8C",
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

plt.xlabel("PBM position")
plt.ylabel("Max |ΔpKd| across PDZ panel")
plt.title("Predicted variant effect by position")
savefig("Figure7C_delta_by_position")


# ============================================================
# Optional Figure 7D: AF class vs max |ΔpKd|
# ============================================================

def classify_af(af):
    if pd.isna(af):
        return "unknown"
    if af < 1e-4:
        return "<1e-4"
    if af < 1e-3:
        return "1e-4-1e-3"
    if af < 1e-2:
        return "1e-3-1e-2"
    return ">=1e-2"


if "joint_af" in variant_effect_summary.columns:
    af_df = variant_effect_summary.copy()
    af_df["AF_class"] = af_df["joint_af"].apply(classify_af)

    af_order = ["<1e-4", "1e-4-1e-3", "1e-3-1e-2", ">=1e-2"]
    af_df = af_df[af_df["AF_class"].isin(af_order)].copy()
    af_df["AF_class"] = pd.Categorical(
        af_df["AF_class"],
        categories=af_order,
        ordered=True,
    )

    af_data = [
        af_df.loc[af_df["AF_class"] == cls, "max_abs_delta_pKd"].dropna()
        for cls in af_order
    ]

    plt.figure(figsize=(3.8, 2.8))
    box = plt.boxplot(
        af_data,
        labels=af_order,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )

    for patch in box["boxes"]:
        patch.set_facecolor("#E8E0F8")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
        patch.set_alpha(0.55)
        patch.set_zorder(1)

    for whisker in box["whiskers"]:
        whisker.set_zorder(1)

    for cap in box["caps"]:
        cap.set_zorder(1)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)
        median.set_zorder(2)

    rng = np.random.default_rng(2)
    for i, vals in enumerate(af_data, start=1):
        x = rng.normal(i, 0.055, size=len(vals))
        plt.scatter(
            x,
            vals,
            s=12,
            alpha=0.70,
            color="#6F4CA2",
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )

    plt.xlabel("gnomAD joint allele frequency")
    plt.ylabel("Max |ΔpKd| across PDZ panel")
    plt.title("Predicted effect by allele frequency")
    plt.xticks(rotation=25, ha="right")
    savefig("Figure7D_delta_by_AF_class")


# ============================================================
# Figure 7E: top variant x PDZ ΔpKd heatmap
# ============================================================

pdz_cmap = LinearSegmentedColormap.from_list(
    "pdz_redblue",
    ["#2B6CB0", "#F7F7F7", "#C94C4C"],
    N=256
)

TOP_VARIANTS = 15
TOP_PDZ_PER_VARIANT = 8

top_variants = (
    variant_effect_summary
    .sort_values("max_abs_delta_pKd", ascending=False)
    .head(TOP_VARIANTS)
    .copy()
)

top_variant_ids = top_variants["variant_id"].tolist()

sub = variant_pdz_delta[
    variant_pdz_delta["variant_id"].isin(top_variant_ids)
].copy()

top_pairs = []
for vid, g in sub.groupby("variant_id"):
    top_pairs.append(
        g.sort_values("abs_delta_pKd", ascending=False)
        .head(TOP_PDZ_PER_VARIANT)
    )

top_pair_df = pd.concat(top_pairs, ignore_index=True)

top_pdz_labels = (
    top_pair_df
    .groupby("pdz_label")["abs_delta_pKd"]
    .max()
    .sort_values(ascending=False)
    .index
    .tolist()
)

sub["variant_label"] = (
    sub["query_gene"].astype(str)
    + "\n"
    + sub["WT_PBM6"].astype(str)
    + "->"
    + sub["MUT_PBM6"].astype(str)
)

top_variants["variant_label"] = (
    top_variants["query_gene"].astype(str)
    + "\n"
    + top_variants["WT_PBM6"].astype(str)
    + "->"
    + top_variants["MUT_PBM6"].astype(str)
)

variant_order = top_variants["variant_label"].tolist()

heat = (
    sub[sub["pdz_label"].isin(top_pdz_labels)]
    .pivot_table(
        index="variant_label",
        columns="pdz_label",
        values="delta_pKd",
        aggfunc="mean",
    )
    .reindex(index=variant_order, columns=top_pdz_labels)
)

vmax = np.nanmax(np.abs(heat.values))

plt.figure(figsize=(max(5.5, 0.28 * heat.shape[1]), 0.32 * heat.shape[0] + 1.3))
im = plt.imshow(
    heat.values,
    aspect="auto",
    cmap=pdz_cmap,
    vmin=-vmax,
    vmax=vmax,
    interpolation="nearest",
)

cbar = plt.colorbar(im, fraction=0.025, pad=0.02)
cbar.set_label("ΔpKd = MUT - WT")

plt.xticks(
    np.arange(heat.shape[1]),
    heat.columns,
    rotation=90,
    fontsize=7,
)
plt.yticks(
    np.arange(heat.shape[0]),
    heat.index,
    fontsize=7,
)

plt.xlabel("Affected PDZ domains")
plt.ylabel("PBM variants")
plt.title("Top PBM variants remodel predicted PDZ-binding profiles")
savefig("Figure7E_top_variant_pdz_heatmap")


# ============================================================
# Figure 7F: representative WT vs MUT case study
# ============================================================

top_case = (
    variant_effect_summary
    .sort_values("max_abs_delta_pKd", ascending=False)
    .iloc[0]
)

case_vid = top_case["variant_id"]

case_df = variant_pdz_delta[
    variant_pdz_delta["variant_id"] == case_vid
].copy()

case_df = (
    case_df
    .sort_values("abs_delta_pKd", ascending=False)
    .head(20)
    .sort_values("delta_pKd")
)

y = np.arange(case_df.shape[0])

plt.figure(figsize=(4.2, 4.8))

for i, row in enumerate(case_df.itertuples()):
    plt.plot(
        [row.WT_pKd_mean, row.MUT_pKd_mean],
        [i, i],
        color="#9CA3AF",
        linewidth=1.0,
        zorder=1,
    )

plt.scatter(
    case_df["WT_pKd_mean"],
    y,
    label="WT PBM",
    s=28,
    color="#4C78A8",
    zorder=2,
)

plt.scatter(
    case_df["MUT_pKd_mean"],
    y,
    label="MUT PBM",
    s=28,
    color="#D14B4B",
    zorder=3,
)

plt.yticks(y, case_df["pdz_label"], fontsize=7)
plt.xlabel("Predicted pKd")
plt.ylabel("PDZ domain")
plt.title(
    f"{top_case['query_gene']} {top_case['WT_PBM6']}->{top_case['MUT_PBM6']}"
)
plt.legend(frameon=False, fontsize=8)
savefig("Figure7F_case_study_WT_MUT_pKd")


print(f"Done. Figure panels saved to: {FIG_DIR}")
