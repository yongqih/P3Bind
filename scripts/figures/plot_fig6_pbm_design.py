
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from _utils import read_csv, save_current_figure, add_panel_label, TABLE_DIR, ensure_output_dirs

parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 6: PDZ-specific PBM candidate design.")
parser.add_argument("--candidates", default="design_candidates.csv", help="Design candidate table under data/processed/.")
parser.add_argument("--trajectory", default="optimization_trajectory.csv", help="Optional simulated annealing trajectory table.")
parser.add_argument("--top-n", type=int, default=20)
parser.add_argument("--output", default="fig6_pbm_design.pdf")
args = parser.parse_args()

ensure_output_dirs()
df = read_csv(args.candidates)
required = {"pbm6", "target_pKd_mean", "background_pKd_mean"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"design_candidates.csv missing required columns: {missing}")
score_col = "specificity_score" if "specificity_score" in df.columns else None
if score_col is None:
    df["specificity_score"] = df["target_pKd_mean"] - df["background_pKd_mean"]
    score_col = "specificity_score"
df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
top = df.head(args.top_n).copy()
top.to_csv(TABLE_DIR / "fig6_top_design_candidates.csv", index=False)

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
axA = fig.add_subplot(gs[0, 0])
axA.text(0.5, 0.5, "P3Bind-design workflow schematic\n(see manuscript Figure 6A)", ha="center", va="center", fontsize=11)
axA.set_axis_off()
add_panel_label(axA, "A")

axB = fig.add_subplot(gs[0, 1])
axB.scatter(df["background_pKd_mean"], df["target_pKd_mean"], s=8, alpha=0.35)
axB.scatter(top["background_pKd_mean"], top["target_pKd_mean"], s=28, alpha=0.9, label=f"Top {args.top_n}")
lo = min(df["background_pKd_mean"].min(), df["target_pKd_mean"].min())
hi = max(df["background_pKd_mean"].max(), df["target_pKd_mean"].max())
axB.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
axB.set_xlabel("Mean background predicted pKd")
axB.set_ylabel("Target predicted pKd")
axB.set_title("PDZ-specific PBM candidate ranking")
axB.legend(frameon=False)
add_panel_label(axB, "B")

axC = fig.add_subplot(gs[1, 0])
try:
    traj = read_csv(args.trajectory)
    step_col = "step" if "step" in traj.columns else traj.columns[0]
    val_col = "best_specificity_score" if "best_specificity_score" in traj.columns else "specificity_score"
    axC.plot(traj[step_col], traj[val_col])
    axC.set_xlabel("Optimization step")
    axC.set_ylabel("Best specificity score")
    axC.set_title("Simulated annealing optimization trajectory")
except Exception as e:
    axC.text(0.5, 0.5, f"optimization_trajectory.csv not available\n{e}", ha="center", va="center")
    axC.set_axis_off()
add_panel_label(axC, "C")

axD = fig.add_subplot(gs[1, 1])
display_cols = [c for c in ["pbm6", "target_pKd_mean", "background_pKd_mean", "specificity_score"] if c in top.columns]
table_df = top[display_cols].head(10).copy()
for col in table_df.columns:
    if table_df[col].dtype.kind in "fc":
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")
axD.axis("off")
table = axD.table(cellText=table_df.values, colLabels=table_df.columns, loc="center")
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.25)
axD.set_title("Top-ranked PBM6 candidates")
add_panel_label(axD, "D")

save_current_figure(args.output)
