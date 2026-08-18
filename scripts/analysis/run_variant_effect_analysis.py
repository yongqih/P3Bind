from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from p3bind.core import REPO_ROOT
from p3bind.variants import score_variant_effects, summarize_variant_effects


def main():
    parser = argparse.ArgumentParser(description="Score cleaned natural PBM variants across the PDZ panel.")
    parser.add_argument("--variants-csv", type=Path, required=True, help="CSV containing WT_PBM6 and MUT_PBM6")
    parser.add_argument("--background-csv", type=Path, default=REPO_ROOT / "data/processed/background_pdz.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/tables/variants")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(args.variants_csv)
    long = score_variant_effects(
        variants, background_csv=args.background_csv, checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
    )
    summary = summarize_variant_effects(long)
    long.to_csv(args.output_dir / "variant_pdz_effects_long.csv", index=False)
    summary.to_csv(args.output_dir / "variant_effect_summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
