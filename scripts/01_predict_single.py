import argparse
import pandas as pd
from p3bind.core import predict_pair

parser = argparse.ArgumentParser(description="Predict pKd for one PDZ-PBM pair.")
parser.add_argument("--pdz-seq", required=True, help="PDZ domain amino-acid sequence")
parser.add_argument("--pbm", required=True, help="PBM peptide; C-terminal 6 aa will be used")
parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (defaults to the repository release checkpoints)")
args = parser.parse_args()

res = predict_pair(args.pdz_seq, args.pbm, checkpoint_dir=args.checkpoint_dir)
print(pd.Series(res).to_string())
