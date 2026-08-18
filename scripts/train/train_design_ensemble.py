from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from p3bind.core import REPO_ROOT
from p3bind.training import train_design_fold


def main():
    parser = argparse.ArgumentParser(description="Train the five-model P3Bind design ensemble.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/raw/all_data_pair_aggregated.csv")
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "data/splits/random_split.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/training/design_models")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N (default: auto)")
    args = parser.parse_args()

    rows = []
    for fold in args.folds:
        result = train_design_fold(
            args.dataset, args.split, fold, args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        )
        rows.append(vars(result))
        print(f"fold={fold} best_epoch={result.best_epoch} val_rmse={result.best_val_rmse:.4f}")
    summary = pd.DataFrame(rows)
    summary["checkpoint"] = summary["checkpoint"].astype(str)
    summary.to_csv(args.output_dir / "training_summary.csv", index=False)


if __name__ == "__main__":
    main()
