import argparse
from p3bind.core import batch_predict

parser = argparse.ArgumentParser(description="Batch predict pKd for PDZ-PBM pairs from CSV.")
parser.add_argument("--input-csv", default="examples/example_pairs.csv")
parser.add_argument("--output-csv", default="results/predictions/example_predictions.csv")
parser.add_argument("--checkpoint-dir", default="checkpoints/design_models")
args = parser.parse_args()

out = batch_predict(args.input_csv, args.output_csv, checkpoint_dir=args.checkpoint_dir)
print(out.head().to_string(index=False))
print(f"Saved: {args.output_csv}")
