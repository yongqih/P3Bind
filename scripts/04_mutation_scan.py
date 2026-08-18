import argparse
from pathlib import Path
from p3bind.core import REPO_ROOT, mutation_scan

parser = argparse.ArgumentParser(description="Run a PBM6 single-mutant scan for a target PDZ.")
parser.add_argument("--target-pdz-seq", required=True)
parser.add_argument("--pbm", required=True, help="PBM peptide; C-terminal 6 aa will be used")
parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (defaults to the repository release checkpoints)")
parser.add_argument("--background-csv", default=None, help="Optional background_pdz.csv. If provided, specificity deltas are calculated.")
parser.add_argument("--output-csv", default=REPO_ROOT / "results" / "predictions" / "mutation_scan.csv")
args = parser.parse_args()

out = mutation_scan(
    target_pdz_sequence=args.target_pdz_seq,
    pbm_or_peptide=args.pbm,
    checkpoint_dir=args.checkpoint_dir,
    background_csv=args.background_csv,
)
output = Path(args.output_csv)
output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output, index=False)
print(out.head(20).to_string(index=False))
print(f"Saved mutation scan: {output}")
