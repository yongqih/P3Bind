from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from p3bind.core import REPO_ROOT, load_background_pdzs, load_models
from p3bind.motif import (PBM_POSITIONS, compute_enrichment, generate_random_pbm6_library,
                          score_pbm6_library, summarize_prediction_distribution)


def main():
    parser = argparse.ArgumentParser(description="Generate the main-text PDZ motif-preference landscape.")
    parser.add_argument("--background-csv", type=Path, default=REPO_ROOT / "data/processed/background_pdz.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/tables/motif_landscape")
    parser.add_argument("--library-size", type=int, default=100000)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--limit-pdz", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_background_pdzs(args.background_csv)
    if args.limit_pdz is not None:
        panel = panel.head(args.limit_pdz)
    models, _ = load_models(args.checkpoint_dir, device=args.device)
    library = generate_random_pbm6_library(args.library_size, args.seed)
    enrichments, summaries = [], []
    for row in panel.itertuples(index=False):
        print(f"scoring {row.pdz_id}")
        scored = score_pbm6_library(row.pdz_sequence, library, models, args.batch_size)
        enrichment = compute_enrichment(scored, top_fraction=args.top_fraction)
        enrichment.insert(0, "pdz_id", row.pdz_id)
        enrichments.append(enrichment)
        summary = summarize_prediction_distribution(scored, args.top_fraction)
        summary["pdz_id"] = row.pdz_id
        summaries.append(summary)
    long = pd.concat(enrichments, ignore_index=True)
    long.to_csv(args.output_dir / "motif_enrichment_long.csv", index=False)
    matrix = long.pivot_table(index="pdz_id", columns=["position", "amino_acid"], values="log2_enrichment")
    matrix = matrix.reindex(columns=pd.MultiIndex.from_product([PBM_POSITIONS, list("ACDEFGHIKLMNPQRSTVWY")]))
    matrix.to_csv(args.output_dir / "motif_cluster_matrix.csv")
    importance = long.assign(abs_log2_enrichment=long.log2_enrichment.abs()).groupby(
        ["pdz_id", "position"], as_index=False
    ).abs_log2_enrichment.mean()
    importance.rename(columns={"abs_log2_enrichment": "position_importance"}).to_csv(
        args.output_dir / "motif_position_importance.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(args.output_dir / "pdz_prediction_distribution.csv", index=False)


if __name__ == "__main__":
    main()
