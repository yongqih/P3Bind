import argparse
import pandas as pd
from p3bind.core import specificity_profile

parser = argparse.ArgumentParser(description="Predict target affinity and off-target/background PDZ profile.")
parser.add_argument("--target-pdz-seq", required=True)
parser.add_argument("--pbm", required=True)
parser.add_argument("--background-csv", default="data/processed/background_pdz.csv")
parser.add_argument("--checkpoint-dir", default="checkpoints/design_models")
parser.add_argument("--summary-csv", default="results/predictions/specificity_summary.csv")
parser.add_argument("--profile-csv", default="results/predictions/offtarget_profile.csv")
args = parser.parse_args()

summary, profile = specificity_profile(
    target_pdz_sequence=args.target_pdz_seq,
    pbm_or_peptide=args.pbm,
    background_csv=args.background_csv,
    checkpoint_dir=args.checkpoint_dir,
)
summary_df = pd.DataFrame([summary])
summary_df.to_csv(args.summary_csv, index=False)
profile.to_csv(args.profile_csv, index=False)
print(summary_df.to_string(index=False))
print("Top predicted off-target PDZs:")
print(profile.head(10)[["pdz_id", "background_pKd_mean", "background_pKd_std"]].to_string(index=False))
