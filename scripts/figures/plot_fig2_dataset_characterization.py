
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from _utils import read_csv, first_existing_column, save_current_figure, add_panel_label

parser = argparse.ArgumentParser(description="Reproduce Supplementary Figure S1: dataset characterization.")
parser.add_argument("--input", default="pair_level_dataset.csv", help="Pair-level processed dataset under data/processed/.")
parser.add_argument("--output", default="supp_fig_s1_dataset_characterization.pdf")
args = parser.parse_args()

df = read_csv(args.input)
pkd_col = first_existing_column(df, ["pKd", "pkd", "target_pKd", "target_pKd_mean", "pKd_median", "y"])
cens_col = first_existing_column(df, ["censored", "is_censored", "censored_pair", "is_censored_pair"], required=False)
pdz_col = first_existing_column(df, ["pdz_sequence", "PDZ_sequence", "pdz_seq"], required=False)
pbm10_col = first_existing_column(df, ["pbm10", "PBM10", "peptide_sequence", "pbm_sequence", "prey_sequence"], required=False)
pbm6_col = first_existing_column(df, ["pbm6", "PBM6", "c_terminal_pbm6"], required=False)

if cens_col is not None:
    cens = df[cens_col].astype(str).str.lower().isin(["1", "true", "yes", "censored"])
else:
    cens = np.repeat(False, len(df))

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
ax = axes[0, 0]
ax.hist(df[pkd_col].dropna(), bins=40)
ax.set_xlabel("pKd")
ax.set_ylabel("Number of PDZ-PBM pairs")
ax.set_title("Full pKd distribution")
add_panel_label(ax, "A")

ax = axes[0, 1]
uncensored_values = df.loc[~cens, pkd_col].dropna()
ax.hist(uncensored_values, bins=35)
ax.set_xlabel("pKd")
ax.set_ylabel("Number of uncensored pairs")
ax.set_title("Uncensored pKd distribution")
add_panel_label(ax, "B")

ax = axes[1, 0]
n_uncens = int((~cens).sum())
n_cens = int(cens.sum())
bars = ax.bar(["Uncensored", "Censored"], [n_uncens, n_cens])
total = max(1, n_uncens + n_cens)
for b, n in zip(bars, [n_uncens, n_cens]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{n:,}\n({100*n/total:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of pairs")
ax.set_title("Censored measurement status")
add_panel_label(ax, "C")

ax = axes[1, 1]
labels, values = [], []
if pdz_col: labels.append("PDZ domains"); values.append(df[pdz_col].nunique())
if pbm10_col: labels.append("PBM10 peptides"); values.append(df[pbm10_col].nunique())
if pbm6_col: labels.append("PBM6 motifs"); values.append(df[pbm6_col].nunique())
labels.append("PDZ-PBM pairs"); values.append(len(df))
bars = ax.bar(labels, values)
ax.set_yscale("log")
ax.set_ylabel("Count")
ax.set_title("Data composition")
ax.tick_params(axis="x", rotation=35)
for b, n in zip(bars, values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(n):,}", ha="center", va="bottom", fontsize=9)
add_panel_label(ax, "D")

save_current_figure(args.output)
