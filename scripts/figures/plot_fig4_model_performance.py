
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from _utils import read_csv, grouped_metric_bars, save_current_figure, add_panel_label

parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 4: baseline and neural model performance.")
parser.add_argument("--input", default="benchmark_results.csv", help="Processed performance CSV under data/processed/.")
parser.add_argument("--output", default="fig4_model_performance.pdf")
args = parser.parse_args()

df = read_csv(args.input)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
grouped_metric_bars(axes[0], df, metric="pearson", ylabel="Pearson correlation", title="Pearson correlation")
add_panel_label(axes[0], "A")
grouped_metric_bars(axes[1], df, metric="rmse", ylabel="RMSE", title="RMSE")
add_panel_label(axes[1], "B")
save_current_figure(args.output)
