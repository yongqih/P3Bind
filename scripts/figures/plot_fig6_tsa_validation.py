from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _utils import add_panel_label, first_existing_column, read_csv, save_current_figure


parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 6: TSA validation.")
parser.add_argument("--summary", default="tsa_validation_summary.csv")
parser.add_argument("--output", default="fig6_tsa_validation.pdf")
args = parser.parse_args()

data = read_csv(args.summary)
candidate = first_existing_column(data, ["candidate", "name", "peptide"])
predicted = first_existing_column(data, ["predicted_pKd", "predicted_pKd_mean", "pKd"])
response = first_existing_column(data, ["delta_Tm", "TSA_response", "experimental_response"])
response_sd = first_existing_column(data, ["delta_Tm_sd", "TSA_response_sd", "experimental_sd"], required=False)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(len(data))
ax_a.bar(x, data[predicted])
ax_a.set_xticks(x, data[candidate], rotation=45, ha="right")
ax_a.set_ylabel("Predicted pKd")
ax_a.set_title("P3Bind-ranked DLG1-PDZ3 candidates")
add_panel_label(ax_a, "A")

ax_b.errorbar(data[predicted], data[response], yerr=data[response_sd] if response_sd else None,
              fmt="o", capsize=3)
for row in data.itertuples(index=False):
    ax_b.annotate(str(getattr(row, candidate)), (getattr(row, predicted), getattr(row, response)),
                  xytext=(4, 4), textcoords="offset points", fontsize=8)
ax_b.set_xlabel("Predicted pKd")
ax_b.set_ylabel(response.replace("_", " "))
ax_b.set_title("Thermal-shift validation")
add_panel_label(ax_b, "B")
save_current_figure(args.output)
