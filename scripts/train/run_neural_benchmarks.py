from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from p3bind.benchmarks import NEURAL_MODELS, regression_metrics, train_neural_benchmark
from p3bind.core import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run the main-text neural benchmark models.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/raw/all_data_pair_aggregated.csv")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/training/benchmarks")
    parser.add_argument("--settings", nargs="+", default=["random", "pbm_heldout", "pdz_heldout"])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--models", nargs="+", choices=sorted(NEURAL_MODELS), default=list(NEURAL_MODELS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, predictions = [], []
    for setting in args.settings:
        for fold in args.folds:
            for model_name in args.models:
                print(f"training setting={setting} fold={fold} model={model_name}")
                row, pred = train_neural_benchmark(
                    args.dataset, args.split_dir / f"{setting}_split.csv", args.output_dir / "checkpoints",
                    setting=setting, fold=fold, model_name=model_name, epochs=args.epochs,
                    batch_size=args.batch_size, device=args.device,
                )
                metrics.append(row)
                predictions.append(pred)
    pd.DataFrame(metrics).to_csv(args.output_dir / "neural_metrics.csv", index=False)
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(args.output_dir / "neural_predictions.csv", index=False)
    uncensored = []
    for keys, group in prediction_table.loc[prediction_table.is_censored == 0].groupby(
        ["setting", "fold", "model"]
    ):
        row = regression_metrics(group.y_true, group.y_pred)
        row.update(zip(["setting", "fold", "model"], keys))
        row["n_test_uncensored"] = len(group)
        uncensored.append(row)
    pd.DataFrame(uncensored).to_csv(args.output_dir / "neural_metrics_uncensored_only.csv", index=False)


if __name__ == "__main__":
    main()
