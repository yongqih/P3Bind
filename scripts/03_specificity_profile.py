import argparse
from pathlib import Path
import pandas as pd
from p3bind.core import DEFAULT_BACKGROUND_CSV, REPO_ROOT, specificity_profile

parser = argparse.ArgumentParser(description="Predict target affinity and off-target/background PDZ profile.")
parser.add_argument("--target-pdz-seq", required=True)
parser.add_argument("--pbm", required=True)
parser.add_argument("--background-csv", default=DEFAULT_BACKGROUND_CSV)
parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (defaults to the repository release checkpoints)")
parser.add_argument("--summary-csv", default=REPO_ROOT / "results" / "predictions" / "specificity_summary.csv")
parser.add_argument("--profile-csv", default=REPO_ROOT / "results" / "predictions" / "offtarget_profile.csv")
args = parser.parse_args()

summary, profile = specificity_profile(
    target_pdz_sequence=args.target_pdz_seq,
    pbm_or_peptide=args.pbm,
    background_csv=args.background_csv,
    checkpoint_dir=args.checkpoint_dir,
)
summary_df = pd.DataFrame([summary])
summary_path = Path(args.summary_csv)
profile_path = Path(args.profile_csv)
summary_path.parent.mkdir(parents=True, exist_ok=True)
profile_path.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(summary_path, index=False)
profile.to_csv(profile_path, index=False)
print(summary_df.to_string(index=False))
print("Top predicted off-target PDZs:")
print(profile.head(10)[["pdz_id", "background_pKd_mean", "background_pKd_std"]].to_string(index=False))
