from __future__ import annotations

import argparse
from pathlib import Path

from p3bind.core import REPO_ROOT
from p3bind.design import optimize_pbm6


def main():
    parser = argparse.ArgumentParser(description="Run specificity-aware PBM6 design.")
    parser.add_argument("--target-pdz-seq", required=True)
    parser.add_argument("--background-csv", type=Path, default=REPO_ROOT / "data/processed/background_pdz.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/tables/design")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-initializations", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--local-refinement-rounds", type=int, default=2)
    parser.add_argument("--local-refinement-top-n", type=int, default=20)
    parser.add_argument("--manual-candidate", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, trajectory, all_candidates = optimize_pbm6(
        args.target_pdz_seq, background_csv=args.background_csv, checkpoint_dir=args.checkpoint_dir,
        alpha=args.alpha, random_initializations=args.random_initializations, steps=args.steps,
        top_k=args.top_k, local_refinement_rounds=args.local_refinement_rounds,
        local_refinement_top_n=args.local_refinement_top_n,
        manual_candidates=args.manual_candidate, seed=args.seed,
    )
    candidates.to_csv(args.output_dir / "design_candidates.csv", index=False)
    trajectory.to_csv(args.output_dir / "optimization_trajectory.csv", index=False)
    all_candidates.to_csv(args.output_dir / "all_candidates_reranked.csv", index=False)
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
