# Auto-exported from 06_performance_graphs.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% Cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

# =========================
# File paths
# =========================
baseline_path = Path("/content/drive/MyDrive/PDZ_DL/splits/results/baselines/baseline_summary.csv")
cnn_path = Path("/content/drive/MyDrive/PDZ_DL/results/dl_cnn_concat/dl_cnn_concat_summary.csv")
inter_path = Path("/content/drive/MyDrive/PDZ_DL/results/dl_interaction_map/interaction_map_summary.csv")
mj_path = Path("/content/drive/MyDrive/PDZ_DL/results/dl_interaction_map_mj/interaction_map_mj_summary.csv")

# =========================
# Helper to load pandas multiindex summary csv
# =========================
def load_summary_csv(path):
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    df = df.reset_index()

    # flatten columns
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                new_cols.append(c[0])
            else:
                new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c)
    df.columns = new_cols

    return df

baseline_df = load_summary_csv(baseline_path)
cnn_df = load_summary_csv(cnn_path)
inter_df = load_summary_csv(inter_path)
mj_df = load_summary_csv(mj_path)

print("baseline_df")
display(baseline_df.head())

print("cnn_df")
display(cnn_df.head())

print("inter_df")
display(inter_df.head())

print("mj_df")
display(mj_df.head())

# %% Cell 1
# =========================
# Keep only the models for main Figure 4
# =========================
baseline_keep = ["rf_pbm_only", "rf_pdz_only", "rf_pdz_pbm_concat"]

baseline_df = baseline_df[baseline_df["model"].isin(baseline_keep)].copy()
cnn_df = cnn_df.copy()
inter_df = inter_df.copy()
mj_df = mj_df.copy()

# =========================
# Rename models for paper
# =========================
model_name_map = {
    "rf_pbm_only": "RF, PBM-only",
    "rf_pdz_only": "RF, PDZ-only",
    "rf_pdz_pbm_concat": "RF, PDZ+PBM",
    "cnn_concat": "CNN-concat",
    "interaction_map": "Interaction-map",
    "interaction_map_mj": "Interaction-map + MJ",
}

for df in [baseline_df, cnn_df, inter_df, mj_df]:
    df["model"] = df["model"].map(model_name_map)

# =========================
# Rename settings
# =========================
setting_name_map = {
    "random": "Random",
    "pbm_heldout": "PBM-heldout",
    "pdz_heldout": "PDZ-heldout",
}

for df in [baseline_df, cnn_df, inter_df, mj_df]:
    df["setting"] = df["setting"].map(setting_name_map)

# combine
plot_df = pd.concat([baseline_df, cnn_df, inter_df, mj_df], axis=0, ignore_index=True)

# order
model_order = [
    "RF, PBM-only",
    "RF, PDZ-only",
    "RF, PDZ+PBM",
    "CNN-concat",
    "Interaction-map",
    "Interaction-map + MJ",
]

setting_order = ["Random", "PBM-heldout", "PDZ-heldout"]

plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
plot_df["setting"] = pd.Categorical(plot_df["setting"], categories=setting_order, ordered=True)

plot_df = plot_df.sort_values(["setting", "model"]).reset_index(drop=True)

display(plot_df[[
    "setting", "model",
    "pearson_mean", "pearson_std",
    "rmse_mean", "rmse_std"
]])

# %% Cell 2
# =========================
# Figure 4: Pearson and RMSE, publication-style
# =========================
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

x = np.arange(len(setting_order))
bar_width = 0.13

# Optional: cleaner model labels if needed
model_order = [
    "RF, PBM-only",
    "RF, PDZ-only",
    "RF, PDZ+PBM",
    "CNN-concat",
    "Interaction-map",
    "Interaction-map + MJ",
]

# Manually define colors for consistency
colors = {
    "RF, PBM-only": "#9ecae1",
    "RF, PDZ-only": "#6baed6",
    "RF, PDZ+PBM": "#2171b5",
    "CNN-concat": "#fdae6b",
    "Interaction-map": "#fd8d3c",
    "Interaction-map + MJ": "#e6550d",
}

# =========================
# Panel A: Pearson
# =========================
ax = axes[0]

for i, model in enumerate(model_order):
    sub = plot_df[plot_df["model"] == model].sort_values("setting")
    xpos = x + (i - (len(model_order) - 1) / 2) * bar_width

    ax.bar(
        xpos,
        sub["pearson_mean"].values,
        width=bar_width,
        yerr=sub["pearson_std"].values,
        capsize=2.5,
        label=model,
        color=colors[model],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 1, "capthick": 1}
    )

ax.set_xticks(x)
ax.set_xticklabels(setting_order, fontsize=10)
ax.set_ylabel("Pearson correlation", fontsize=11)
ax.set_title("Pearson correlation", fontsize=12, pad=10)
ax.set_ylim(0, 0.9)

ax.text(
    -0.12, 1.08, "A",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Panel B: RMSE
# =========================
ax = axes[1]

for i, model in enumerate(model_order):
    sub = plot_df[plot_df["model"] == model].sort_values("setting")
    xpos = x + (i - (len(model_order) - 1) / 2) * bar_width

    ax.bar(
        xpos,
        sub["rmse_mean"].values,
        width=bar_width,
        yerr=sub["rmse_std"].values,
        capsize=2.5,
        label=model,
        color=colors[model],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 1, "capthick": 1}
    )

ax.set_xticks(x)
ax.set_xticklabels(setting_order, fontsize=10)
ax.set_ylabel("RMSE", fontsize=11)
ax.set_title("RMSE", fontsize=12, pad=10)
ax.set_ylim(0, 0.60)

ax.text(
    -0.12, 1.08, "B",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Shared legend
# =========================
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# =========================
# Save
# =========================
outdir = Path("/content/drive/MyDrive/PDZ_DL/figures")
outdir.mkdir(exist_ok=True)

plt.savefig(outdir / "Figure4_model_comparison_publication.png", bbox_inches="tight", dpi=600)
plt.savefig(outdir / "Figure4_model_comparison_publication.svg", bbox_inches="tight")

plt.show()

# %% Cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

# =========================
# Load processed pair-level dataset
# =========================
data_path = Path("/content/drive/MyDrive/PDZ_DL/all_data_pair_aggregated.csv")

df = pd.read_csv(data_path)

print(df.shape)
print(df.columns)
display(df.head())

# %% Cell 4
# =========================
# Column names
# =========================
pdz_col = "pdz_sequence"

# 你的原始 peptide 是 10 aa，可能叫 prey_sequence 或 pbm_sequence_10aa
pbm10_col = "pbm_sequence_10aa"   # 如果不对，改成你的真实列名
pkd_col = "pKd_label"         # 如果不对，改成 pKd 或其他真实列名
censor_col = "is_censored_label"

# 从 10 aa peptide 取 C-terminal 6 aa
df["pbm_sequence_6aa"] = df[pbm10_col].astype(str).str[-6:]
pbm6_col = "pbm_sequence_6aa"

n_pairs = df.shape[0]
n_pdz = df[pdz_col].nunique()
n_pbm10 = df[pbm10_col].nunique()
n_pbm6 = df[pbm6_col].nunique()

n_censored = int(df[censor_col].astype(bool).sum())
n_uncensored = int((~df[censor_col].astype(bool)).sum())

print("Unique PDZ-PBM pairs:", n_pairs)
print("Unique PDZ domains:", n_pdz)
print("Unique PBM10 peptides:", n_pbm10)
print("Unique PBM6 motifs:", n_pbm6)
print("Censored:", n_censored)
print("Uncensored:", n_uncensored)
composition = pd.Series({
    "PDZ domains": n_pdz,
    "PBM10 peptides": n_pbm10,
    "PBM6 motifs": n_pbm6,
    "PDZ–PBM pairs": n_pairs
})

# %% Cell 5
# =========================
# Figure 2
# =========================
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)

# -------------------------
# A: Full pKd distribution
# -------------------------
ax = axes[0, 0]

ax.hist(
    df[pkd_col].dropna(),
    bins=40,
    edgecolor="black",
    linewidth=0.4
)

ax.set_xlabel("pKd")
ax.set_ylabel("Number of PDZ–PBM pairs")
ax.set_title("Full pKd distribution", pad=8)

ax.text(
    -0.16, 1.08, "A",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# -------------------------
# B: Uncensored pKd distribution
# -------------------------
ax = axes[0, 1]

uncensored_df = df[df[censor_col]==0].copy()

ax.hist(
    uncensored_df[pkd_col].dropna(),
    bins=30,
    edgecolor="black",
    linewidth=0.4
)

ax.set_xlabel("pKd")
ax.set_ylabel("Number of uncensored pairs")
ax.set_title("Uncensored pKd distribution", pad=8)

ax.text(
    -0.16, 1.08, "B",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# -------------------------
# C: Censored vs uncensored
# -------------------------
ax = axes[1, 0]

censor_counts = pd.Series({
    "Uncensored": n_uncensored,
    "Censored": n_censored
})

xpos = np.arange(len(censor_counts))

ax.bar(
    xpos,
    censor_counts.values,
    edgecolor="black",
    linewidth=0.5
)

ax.set_xticks(xpos)
ax.set_xticklabels(censor_counts.index)

for i, v in enumerate(censor_counts.values):
    pct = v / n_pairs * 100
    ax.text(
        i,
        v + max(censor_counts.values) * 0.03,
        f"{v:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_ylabel("Number of pairs")
ax.set_title("Censored measurements", pad=8)

ax.text(
    -0.16, 1.08, "C",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# -------------------------
# D: Dataset composition
# -------------------------
ax = axes[1, 1]

composition = pd.Series({
    "PDZ domains": n_pdz,
    "PBM10 peptides": n_pbm10,
    "PBM6 motifs": n_pbm6,
    "PDZ–PBM pairs": n_pairs
})

xpos = np.arange(len(composition))

ax.bar(
    xpos,
    composition.values,
    edgecolor="black",
    linewidth=0.5
)

ax.set_xticks(xpos)
ax.set_xticklabels(composition.index, rotation=25, ha="right")

for i, v in enumerate(composition.values):
    ax.text(
        i,
        v * 1.08,
        f"{v:,}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_yscale("log")
ax.set_ylabel("Count, log scale")
ax.set_title("Dataset composition", pad=8)

ax.text(
    -0.16, 1.08, "D",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()

# =========================
# Save
# =========================
outdir = Path("/content/drive/MyDrive/PDZ_DL/figures")
outdir.mkdir(exist_ok=True)

plt.savefig(outdir / "Figure2_dataset_characterization.png", bbox_inches="tight", dpi=600)
plt.savefig(outdir / "Figure2_dataset_characterization.svg", bbox_inches="tight")

plt.show()
