
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from _utils import read_csv, first_existing_column, save_current_figure, add_panel_label, TABLE_DIR, ensure_output_dirs

parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 5: gnomAD-P3Bind PBM variant effects.")
parser.add_argument("--variants", default="pbm_variant_effects.csv", help="Variant summary table under data/processed/.")
parser.add_argument("--matrix", default="variant_pdz_delta_matrix.csv", help="Wide top-variant by PDZ ΔpKd matrix.")
parser.add_argument("--representative", default="representative_variant_profile.csv", help="Representative WT/MUT pKd profile table.")
parser.add_argument("--output", default="fig5_gnomad_variants.pdf")
args = parser.parse_args()

ensure_output_dirs()
var = read_csv(args.variants)
pos_col = first_existing_column(var, ["pbm_position", "position"])
eff_col = first_existing_column(var, ["max_abs_delta_pKd", "max_abs_delta_pkd", "max_abs_delta", "effect_size"])
af_col = first_existing_column(var, ["af_bin", "allele_frequency_bin", "joint_af_bin"], required=False)
var_id_col = first_existing_column(var, ["variant_id", "variant", "variant_label"], required=False)

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.2, 1.0])

axA = fig.add_subplot(gs[0, :])
axA.text(0.5, 0.5, "gnomAD-P3Bind variant-effect workflow\nvariants → PBM6 WT/MUT construction → P3Bind prediction → variant-by-PDZ effect matrix", ha="center", va="center", fontsize=11)
axA.set_axis_off()
add_panel_label(axA, "A")

axB = fig.add_subplot(gs[1, 0])
counts = var[pos_col].astype(str).value_counts().sort_index()
axB.bar(counts.index, counts.values)
axB.set_xlabel("PBM position")
axB.set_ylabel("Number of variants")
axB.set_title("PBM variant distribution by position")
add_panel_label(axB, "B")

axC = fig.add_subplot(gs[1, 1])
positions = list(counts.index)
data = [var.loc[var[pos_col].astype(str)==p, eff_col].dropna().to_numpy() for p in positions]
axC.boxplot(data, labels=positions, showfliers=False)
for i, arr in enumerate(data, start=1):
    if len(arr):
        axC.scatter(np.random.default_rng(1).normal(i, 0.04, len(arr)), arr, s=7, alpha=0.35)
axC.set_xlabel("PBM position")
axC.set_ylabel("Max |ΔpKd| across PDZ panel")
axC.set_title("Predicted variant effect by position")
add_panel_label(axC, "C")

axD = fig.add_subplot(gs[2, 0])
try:
    mat = read_csv(args.matrix)
    if var_id_col and var_id_col in mat.columns:
        ylabels = mat[var_id_col].astype(str).tolist()
        values = mat.drop(columns=[var_id_col])
    elif "variant_id" in mat.columns:
        ylabels = mat["variant_id"].astype(str).tolist()
        values = mat.drop(columns=["variant_id"])
    else:
        ylabels = [str(i) for i in range(len(mat))]
        values = mat
    im = axD.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    axD.set_xlabel("Affected PDZ domains")
    axD.set_ylabel("PBM variants")
    axD.set_title("Top PBM variants remodel predicted PDZ-binding profiles")
    axD.set_yticks(np.arange(len(ylabels))); axD.set_yticklabels(ylabels, fontsize=6)
    axD.set_xticks([])
    plt.colorbar(im, ax=axD, fraction=0.046, pad=0.04).set_label("ΔpKd = MUT - WT")
except Exception as e:
    axD.text(0.5, 0.5, f"variant_pdz_delta_matrix.csv not available\n{e}", ha="center", va="center")
    axD.set_axis_off()
add_panel_label(axD, "D")

axE = fig.add_subplot(gs[2, 1])
try:
    rep = read_csv(args.representative)
    pdz_col = first_existing_column(rep, ["pdz_id", "pdz"])
    wt_col = first_existing_column(rep, ["wt_pKd", "wt_pKd_mean", "wildtype_pKd"])
    mut_col = first_existing_column(rep, ["mut_pKd", "mut_pKd_mean", "mutant_pKd"])
    top = rep.head(15).iloc[::-1]
    y = np.arange(len(top))
    axE.hlines(y, top[wt_col], top[mut_col], linewidth=1)
    axE.scatter(top[wt_col], y, label="WT PBM", s=18)
    axE.scatter(top[mut_col], y, label="MUT PBM", s=18)
    axE.set_yticks(y); axE.set_yticklabels(top[pdz_col], fontsize=6)
    axE.set_xlabel("Predicted pKd")
    axE.set_ylabel("PDZ domain")
    axE.set_title("Representative high-impact variant")
    axE.legend(frameon=False, fontsize=8)
except Exception as e:
    if af_col is not None:
        groups = list(dict.fromkeys(var[af_col].astype(str).tolist()))
        data = [var.loc[var[af_col].astype(str)==g, eff_col].dropna().to_numpy() for g in groups]
        axE.boxplot(data, labels=groups, showfliers=False)
        axE.set_xlabel("gnomAD joint allele-frequency bin")
        axE.set_ylabel("Max |ΔpKd| across PDZ panel")
        axE.set_title("Predicted effect by allele frequency")
    else:
        axE.text(0.5, 0.5, f"representative/AF data not available\n{e}", ha="center", va="center")
        axE.set_axis_off()
add_panel_label(axE, "E/F")

# Save top variants table for supplement.
top_cols = [c for c in [var_id_col, pos_col, eff_col, af_col] if c is not None]
var.sort_values(eff_col, ascending=False).head(50)[top_cols].to_csv(TABLE_DIR / "fig5_top_pbm_variants.csv", index=False)
save_current_figure(args.output)
