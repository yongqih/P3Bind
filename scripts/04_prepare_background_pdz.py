import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Prepare background_pdz.csv from a PDZ table.")
parser.add_argument("--input-csv", required=True)
parser.add_argument("--output-csv", default="data/processed/background_pdz.csv")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
if "pdz_sequence" not in df.columns:
    raise ValueError("Input must contain pdz_sequence")
if "pdz_id" not in df.columns:
    if "pdz" in df.columns:
        df["pdz_id"] = df["pdz"].astype(str)
    else:
        df["pdz_id"] = [f"PDZ_{i+1}" for i in range(len(df))]
# If both pdz and pdz_id exist, make IDs unique/readable.
if "pdz" in df.columns:
    df["pdz_id"] = df["pdz"].astype(str) + "_" + df["pdz_id"].astype(str)
out = df[["pdz_id", "pdz_sequence"]].drop_duplicates()
out.to_csv(args.output_csv, index=False)
print(f"Saved {len(out)} background PDZs to {args.output_csv}")
