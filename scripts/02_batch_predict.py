import argparse
from p3bind.core import REPO_ROOT, batch_predict

parser = argparse.ArgumentParser(description="Batch predict pKd for PDZ-PBM pairs from CSV.")
parser.add_argument("--input-csv", default=REPO_ROOT / "examples" / "example_pairs.csv")
parser.add_argument("--output-csv", default=REPO_ROOT / "results" / "predictions" / "example_predictions.csv")
parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (defaults to the repository release checkpoints)")
args = parser.parse_args()

out = batch_predict(args.input_csv, args.output_csv, checkpoint_dir=args.checkpoint_dir)
print(out.head().to_string(index=False))
print(f"Saved: {args.output_csv}")
