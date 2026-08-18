import argparse
from pathlib import Path
import pandas as pd
from p3bind.core import DEFAULT_BACKGROUND_CSV

parser = argparse.ArgumentParser(description="Prepare background_pdz.csv from a PDZ table.")
parser.add_argument("--input-csv", required=True)
parser.add_argument("--output-csv", default=DEFAULT_BACKGROUND_CSV)
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
if "pdz_sequence" not in df.columns:
    raise ValueError("Input must contain pdz_sequence")
df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.strip().str.upper()
metadata = [column for column in ["pdz_gene", "pdz_sequence", "pdz_uniprot", "pdz_site"] if column in df]
out = df[metadata].dropna(subset=["pdz_sequence"]).drop_duplicates("pdz_sequence").reset_index(drop=True)
alphabet = set("ACDEFGHIKLMNPQRSTVWY")
out = out.loc[out.pdz_sequence.map(lambda seq: len(seq) >= 40 and set(seq) <= alphabet)].reset_index(drop=True)
out["pdz_id"] = [f"PDZ_{index:03d}" for index in range(len(out))]
if {"pdz_gene", "pdz_site"}.issubset(out.columns):
    out["pdz_label"] = out["pdz_gene"].astype(str) + "_" + out["pdz_site"].astype(str)
out = out[["pdz_id", "pdz_sequence"] + [
    column for column in ["pdz_gene", "pdz_uniprot", "pdz_site", "pdz_label"] if column in out
]]
if not out["pdz_id"].is_unique or not out["pdz_sequence"].is_unique:
    raise RuntimeError("Failed to create a unique background PDZ panel.")
output = Path(args.output_csv)
output.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output, index=False)
print(f"Saved {len(out)} background PDZs to {output}")
