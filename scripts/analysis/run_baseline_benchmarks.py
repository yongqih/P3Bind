from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from p3bind.benchmarks import regression_metrics, run_baseline_benchmarks
from p3bind.core import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run mean, Ridge, and random-forest baselines.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/raw/all_data_pair_aggregated.csv")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/tables/benchmarks")
    parser.add_argument("--settings", nargs="+", default=["random", "pbm_heldout", "pdz_heldout"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = run_baseline_benchmarks(
        args.dataset, args.split_dir, settings=args.settings, folds=args.folds
    )
    metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    summary = metrics.groupby(["setting", "model"])[["rmse", "mae", "r2", "pearson", "spearman"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "baseline_summary.csv")
    uncensored_rows = []
    for keys, group in predictions.loc[predictions.is_censored == 0].groupby(
        ["setting", "fold", "model"]
    ):
        row = regression_metrics(group.y_true, group.y_pred)
        row.update(zip(["setting", "fold", "model"], keys))
        row["n_test_uncensored"] = len(group)
        uncensored_rows.append(row)
    uncensored = pd.DataFrame(uncensored_rows)
    uncensored.to_csv(args.output_dir / "baseline_metrics_uncensored_only.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
