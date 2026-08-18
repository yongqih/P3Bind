
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from _utils import read_csv, resolve_processed_csv, first_existing_column, PBM_POSITIONS, AA_ORDER, save_current_figure, add_panel_label, plot_heatmap

parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 4: P3Bind motif preference landscape.")
parser.add_argument("--enrichment", default="motif_enrichment_matrix.csv", help="Long table with position, amino_acid, log2_enrichment; optional pdz_id.")
parser.add_argument("--importance", default="motif_position_importance.csv", help="Position-level importance table.")
parser.add_argument("--cluster", default="motif_cluster_matrix.csv", help="Optional PDZ-by-feature enrichment matrix.")
parser.add_argument("--representatives", default="representative_motif_profiles.csv", help="Optional representative PDZ profile long table.")
parser.add_argument("--output", default="fig4_motif_landscape.pdf")
args = parser.parse_args()

enrich = read_csv(args.enrichment)
pos_col = first_existing_column(enrich, ["position", "pbm_position"])
aa_col = first_existing_column(enrich, ["amino_acid", "aa", "residue"])
val_col = first_existing_column(enrich, ["log2_enrichment", "enrichment", "value"])
enrich[pos_col] = enrich[pos_col].astype(str).replace({"-5":"P-5","-4":"P-4","-3":"P-3","-2":"P-2","-1":"P-1","0":"P0"})
avg = enrich.pivot_table(index=aa_col, columns=pos_col, values=val_col, aggfunc="mean")
avg = avg.reindex([aa for aa in AA_ORDER if aa in avg.index]).reindex(columns=[p for p in PBM_POSITIONS if p in avg.columns])

imp = read_csv(args.importance)
imp_pos = first_existing_column(imp, ["position", "pbm_position"])
imp_val = first_existing_column(imp, ["mean_abs_log2_enrichment", "importance", "value"])
imp[imp_pos] = imp[imp_pos].astype(str).replace({"-5":"P-5","-4":"P-4","-3":"P-3","-2":"P-2","-1":"P-1","0":"P0"})
available_positions = set(imp[imp_pos])
imp = imp.set_index(imp_pos).reindex([p for p in PBM_POSITIONS if p in available_positions]).reset_index()

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3])
axA = fig.add_subplot(gs[0, 0])
plot_heatmap(axA, avg.to_numpy(dtype=float), list(avg.columns), list(avg.index), "Average PBM enrichment", "log2 enrichment")
axA.set_xlabel("PBM position"); axA.set_ylabel("Amino acid")
add_panel_label(axA, "A")

axB = fig.add_subplot(gs[0, 1])
yerr = None
if {"ci_low", "ci_high"}.issubset(imp.columns):
    yerr = np.vstack([imp[imp_val] - imp["ci_low"], imp["ci_high"] - imp[imp_val]])
axB.bar(imp[imp_pos], imp[imp_val], yerr=yerr, capsize=3)
axB.set_xlabel("PBM position"); axB.set_ylabel("Mean absolute log2 enrichment")
axB.set_title("Position-level PBM importance")
add_panel_label(axB, "B")

axC = fig.add_subplot(gs[1:, 0])
try:
    cluster = read_csv(args.cluster)
    if "pdz_id" in cluster.columns:
        row_labels = cluster["pdz_id"].astype(str).tolist()
        mat = cluster.drop(columns=["pdz_id"])
    else:
        row_labels = [str(i) for i in range(len(cluster))]
        mat = cluster
    im = axC.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    axC.set_xlabel("Position-amino acid enrichment features")
    axC.set_ylabel("PDZ domains")
    axC.set_title("PDZ-specific enrichment profiles")
    axC.set_yticks([])
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04).set_label("log2 enrichment")
except Exception as e:
    axC.text(0.5, 0.5, f"motif_cluster_matrix.csv not available\n{e}", ha="center", va="center")
    axC.set_axis_off()
add_panel_label(axC, "C")

axD = fig.add_subplot(gs[1:, 1])
try:
    reps = read_csv(args.representatives)
    pdz_col = first_existing_column(reps, ["pdz_id", "pdz"])
    rpos = first_existing_column(reps, ["position", "pbm_position"])
    raa = first_existing_column(reps, ["amino_acid", "aa", "residue"])
    rval = first_existing_column(reps, ["log2_enrichment", "enrichment", "value"])
    reps[rpos] = reps[rpos].astype(str).replace({"-5":"P-5","-4":"P-4","-3":"P-3","-2":"P-2","-1":"P-1","0":"P0"})
    # Display the first representative profile by default.
    first_pdz = reps[pdz_col].dropna().astype(str).iloc[0]
    sub = reps[reps[pdz_col].astype(str) == first_pdz]
    mat = sub.pivot_table(index=raa, columns=rpos, values=rval, aggfunc="mean")
    mat = mat.reindex([aa for aa in AA_ORDER if aa in mat.index]).reindex(columns=[p for p in PBM_POSITIONS if p in mat.columns])
    plot_heatmap(axD, mat.to_numpy(dtype=float), list(mat.columns), list(mat.index), f"Representative profile: {first_pdz}", "log2 enrichment")
    axD.set_xlabel("PBM position"); axD.set_ylabel("Amino acid")
except Exception as e:
    axD.text(0.5, 0.5, f"representative_motif_profiles.csv not available\n{e}", ha="center", va="center")
    axD.set_axis_off()
add_panel_label(axD, "D")

save_current_figure(args.output)
