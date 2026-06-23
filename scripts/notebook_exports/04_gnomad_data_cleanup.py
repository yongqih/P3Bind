# Auto-exported from 04_gnomad_data_cleanup.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% Cell 0
import os
import math
import random
import argparse
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

# %% Cell 1
DATA_PATH = Path("/content/drive/MyDrive/PDZ_DL/all_data_pair_aggregated.csv")
df = pd.read_csv(DATA_PATH)
df.head(10)

# %% Cell 2
profiled_pbm_df = (
    df[["pbm_sequence_10aa", "pbm_gene", "pbm_uniprot"]]
    .drop_duplicates()
)
profiled_pbm_df

# %% Cell 3
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

OUT_DIR = "/content/drive/MyDrive/gnomad_pbm_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Load PBM table =====
pbm_meta = profiled_pbm_df.copy()

pbm_meta["pbm_sequence_10aa"] = pbm_meta["pbm_sequence_10aa"].astype(str).str.upper()
pbm_meta["pbm_gene"] = pbm_meta["pbm_gene"].astype(str).str.strip()
pbm_meta["pbm_uniprot"] = pbm_meta["pbm_uniprot"].astype(str).str.strip()
pbm_meta["WT_PBM6"] = pbm_meta["pbm_sequence_10aa"].str[-6:]

# remove obvious non-human / viral entries
nonhuman_pattern = "WNV|VIRUS|NS5|DENV|ZIKV|HCV|HIV|EBV|CMV|SARS|INFLUENZA|HPV"
pbm_meta = pbm_meta[
    ~pbm_meta["pbm_gene"].str.contains(nonhuman_pattern, case=False, na=False)
].copy()

pbm_meta = pbm_meta.drop_duplicates(
    subset=["pbm_gene", "pbm_uniprot", "pbm_sequence_10aa"]
).reset_index(drop=True)

gene_list = sorted(pbm_meta["pbm_gene"].dropna().unique())

print("PBM entries:", pbm_meta.shape)
print("Unique genes:", len(gene_list))
display(pbm_meta.head())

# %% Cell 4
GNOMAD_API = "https://gnomad.broadinstitute.org/api"

QUERY_GNOMAD_GENE_VARIANTS_SIMPLE = """
query GeneVariants($geneSymbol: String!, $datasetId: DatasetId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    symbol
    gene_id
    canonical_transcript_id
    variants(dataset: $datasetId) {
      variant_id
      chrom
      pos
      ref
      alt
      consequence
      exome {
        ac
        an
        filters
      }
      genome {
        ac
        an
        filters
      }
      joint {
        ac
        an
        filters
      }
    }
  }
}
"""

def query_gnomad_gene_variants_simple(
    gene_symbol,
    dataset_id="gnomad_r4",
    sleep_sec=4,
    max_retries=3
):
    variables = {
        "geneSymbol": gene_symbol,
        "datasetId": dataset_id
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(
                GNOMAD_API,
                json={"query": QUERY_GNOMAD_GENE_VARIANTS_SIMPLE, "variables": variables},
                timeout=120
            )

            if r.status_code == 200:
                data = r.json()
                if "errors" in data:
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                time.sleep(sleep_sec)
                return data["data"]["gene"]

            # rate-limit or transient errors
            if r.status_code in [429, 500, 502, 503, 504]:
                wait = sleep_sec * (attempt + 2)
                print(f"{gene_symbol}: HTTP {r.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue

            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait = sleep_sec * (attempt + 2)
            print(f"{gene_symbol}: {e}, retrying in {wait}s")
            time.sleep(wait)

    return None


def flatten_gnomad_simple(gene_data):
    rows = []

    if gene_data is None:
        return rows

    gene_symbol = gene_data.get("symbol")
    gene_id = gene_data.get("gene_id")
    canonical_tx = gene_data.get("canonical_transcript_id")

    for v in gene_data.get("variants", []):
        exome = v.get("exome") or {}
        genome = v.get("genome") or {}
        joint = v.get("joint") or {}

        rows.append({
            "query_gene": gene_symbol,
            "gene_id": gene_id,
            "canonical_transcript_id": canonical_tx,

            "variant_id": v.get("variant_id"),
            "chrom": v.get("chrom"),
            "pos": v.get("pos"),
            "ref": v.get("ref"),
            "alt": v.get("alt"),
            "gnomad_consequence": v.get("consequence"),

            "exome_ac": exome.get("ac"),
            "exome_an": exome.get("an"),
            "exome_filters": ";".join(exome.get("filters") or []),

            "genome_ac": genome.get("ac"),
            "genome_an": genome.get("an"),
            "genome_filters": ";".join(genome.get("filters") or []),

            "joint_ac": joint.get("ac"),
            "joint_an": joint.get("an"),
            "joint_filters": ";".join(joint.get("filters") or []),
        })

    return rows

# %% Cell 5
def fetch_gnomad_for_genes(
    gene_list,
    dataset_id="gnomad_r4",
    sleep_sec=4,
    cache_csv=os.path.join(OUT_DIR, "gnomad_pbm_genes_raw.csv"),
    failed_csv=os.path.join(OUT_DIR, "gnomad_pbm_genes_failed.csv")
):
    if os.path.exists(cache_csv):
        print("Loading cached gnomAD result:", cache_csv)
        return pd.read_csv(cache_csv), pd.read_csv(failed_csv) if os.path.exists(failed_csv) else pd.DataFrame()

    all_rows = []
    failed = []

    for gene in tqdm(gene_list, desc="Querying gnomAD genes"):
        try:
            gene_data = query_gnomad_gene_variants_simple(
                gene,
                dataset_id=dataset_id,
                sleep_sec=sleep_sec
            )
            rows = flatten_gnomad_simple(gene_data)
            all_rows.extend(rows)
            print(gene, "variants:", len(rows))
        except Exception as e:
            print("FAILED:", gene, e)
            failed.append({"gene": gene, "error": str(e)})

    gnomad_raw = pd.DataFrame(all_rows)
    failed_df = pd.DataFrame(failed)

    gnomad_raw.to_csv(cache_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)

    return gnomad_raw, failed_df


# Full run:
gnomad_raw, failed_df = fetch_gnomad_for_genes(gene_list, sleep_sec=4)

print("gnomAD raw:", gnomad_raw.shape)
print("Failed:", failed_df.shape)
display(gnomad_raw.head())

# %% Cell 6
for prefix in ["joint", "exome", "genome"]:
    gnomad_raw[f"{prefix}_ac"] = pd.to_numeric(gnomad_raw[f"{prefix}_ac"], errors="coerce")
    gnomad_raw[f"{prefix}_an"] = pd.to_numeric(gnomad_raw[f"{prefix}_an"], errors="coerce")
    gnomad_raw[f"{prefix}_af"] = gnomad_raw[f"{prefix}_ac"] / gnomad_raw[f"{prefix}_an"]

missense_gnomad = gnomad_raw[
    gnomad_raw["gnomad_consequence"].astype(str).str.contains("missense", case=False, na=False)
].copy()

# Optional: keep SNV only first; missense PBM variant analysis can start from SNVs.
missense_gnomad_snv = missense_gnomad[
    (missense_gnomad["ref"].astype(str).str.len() == 1) &
    (missense_gnomad["alt"].astype(str).str.len() == 1)
].copy()

print("missense_gnomAD:", missense_gnomad.shape)
print("missense SNV:", missense_gnomad_snv.shape)
display(missense_gnomad_snv.head())

# %% Cell 7
import time
import random
import requests

VEP_URL = (
    "https://rest.ensembl.org/vep/homo_sapiens/region"
    "?pick=1&canonical=1&mane=1&protein=1&symbol=1"
)

def query_vep_region_batch(vep_lines, sleep_sec=2, max_retries=8, max_items_per_request=100):
    """
    Safe VEP query:
    - Automatically splits large input into <= max_items_per_request
    - Handles 429 rate limit
    - Uses pick=1 and no hgvs to reduce response size
    """
    all_results = []
    vep_lines = list(vep_lines)

    chunks = [
        vep_lines[i:i + max_items_per_request]
        for i in range(0, len(vep_lines), max_items_per_request)
    ]

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    for chunk_idx, chunk in enumerate(chunks):
        payload = {"variants": chunk}

        for attempt in range(max_retries):
            r = requests.post(
                VEP_URL,
                headers=headers,
                json=payload,
                timeout=180
            )

            if r.ok:
                all_results.extend(r.json())
                time.sleep(sleep_sec)
                break

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    wait = int(retry_after) + 5
                else:
                    wait = min(90, 10 * (attempt + 1)) + random.uniform(0, 5)

                print(f"VEP 429 rate limit. Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue

            if r.status_code in [500, 502, 503, 504]:
                wait = min(90, 10 * (attempt + 1)) + random.uniform(0, 5)
                print(f"VEP HTTP {r.status_code}. Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue

            # 如果还是 message too large，进一步拆小
            if r.status_code == 400 and "message too large" in r.text.lower():
                raise RuntimeError(
                    f"VEP message too large even after splitting. "
                    f"chunk size={len(chunk)}. Try max_items_per_request=50. "
                    f"Error: {r.text[:500]}"
                )

            raise RuntimeError(f"VEP HTTP {r.status_code}: {r.text[:1000]}")

        else:
            raise RuntimeError("VEP failed after max retries.")

    return all_results

# %% Cell 8
import os
import time
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def chunk_list(x, size):
    return [x[i:i + size] for i in range(0, len(x), size)]


def run_one_vep_batch(batch):
    try:
        vep_json = query_vep_region_batch(batch, sleep_sec=0)
        rows = flatten_vep_response(vep_json)
        return rows, None
    except Exception as e:
        return [], str(e)


def run_vep_parallel_incremental(
    variant_df,
    batch_size=500,
    max_workers=6,
    cache_csv=os.path.join(OUT_DIR, "gnomad_missense_vep_parallel.csv"),
    failed_csv=os.path.join(OUT_DIR, "gnomad_missense_vep_parallel_failed.csv")
):
    df = variant_df.copy()
    df["vep_input"] = df.apply(make_vep_vcf_line, axis=1)

    all_inputs = df["vep_input"].drop_duplicates().tolist()

    old = pd.DataFrame()
    done_inputs = set()

    if os.path.exists(cache_csv):
        old = pd.read_csv(cache_csv)
        if "vep_input" in old.columns:
            done_inputs = set(old["vep_input"].dropna().unique())
        print("Loaded existing cache:", old.shape)
        print("Already done:", len(done_inputs))

    remaining = [x for x in all_inputs if x not in done_inputs]
    batches = chunk_list(remaining, batch_size)

    print("Total unique variants:", len(all_inputs))
    print("Remaining variants:", len(remaining))
    print("Number of batches:", len(batches))
    print("Batch size:", batch_size)
    print("Workers:", max_workers)

    failed_batches = []
    new_rows_buffer = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(run_one_vep_batch, batch): (i, batch)
            for i, batch in enumerate(batches)
        }

        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="VEP parallel"):
            batch_idx, batch = future_to_batch[future]
            rows, err = future.result()

            if err is not None:
                print(f"FAILED batch {batch_idx}: {err[:300]}")
                failed_batches.append({
                    "batch_index": batch_idx,
                    "n_variants": len(batch),
                    "first_input": batch[0] if len(batch) else None,
                    "error": err
                })
            else:
                new_rows_buffer.extend(rows)

            if len(new_rows_buffer) >= 5000:
                new_df = pd.DataFrame(new_rows_buffer)
                old = pd.concat([old, new_df], ignore_index=True)
                old = old.drop_duplicates()
                old.to_csv(cache_csv, index=False)
                new_rows_buffer = []

            if len(failed_batches) > 0:
                pd.DataFrame(failed_batches).to_csv(failed_csv, index=False)

    if len(new_rows_buffer) > 0:
        new_df = pd.DataFrame(new_rows_buffer)
        old = pd.concat([old, new_df], ignore_index=True)
        old = old.drop_duplicates()
        old.to_csv(cache_csv, index=False)

    final = pd.read_csv(cache_csv) if os.path.exists(cache_csv) else pd.DataFrame()
    failed = pd.read_csv(failed_csv) if os.path.exists(failed_csv) else pd.DataFrame()

    print("Final VEP rows:", final.shape)
    print("Failed batches:", failed.shape)

    return final, failed

# %% Cell 9
vep_raw_parallel, failed_vep = run_vep_parallel_incremental(
    missense_gnomad_snv,
    batch_size=500,
    max_workers=6,
    cache_csv=os.path.join(OUT_DIR, "gnomad_missense_vep_parallel_pick1.csv"),
    failed_csv=os.path.join(OUT_DIR, "gnomad_missense_vep_parallel_pick1_failed.csv")
)

# %% Cell 10
vep = pd.read_csv(
    os.path.join(OUT_DIR, "gnomad_missense_vep_parallel_pick1.csv"),
    low_memory=False
)

vep_missense = vep[
    vep["consequence_terms"].astype(str).str.contains("missense_variant", na=False)
].copy()

vep_missense = vep_missense.dropna(subset=["amino_acids", "protein_start"]).copy()

# amino_acids: e.g. V/F
aa_split = vep_missense["amino_acids"].astype(str).str.split("/", expand=True)
vep_missense["ref_aa"] = aa_split[0]
vep_missense["alt_aa"] = aa_split[1]

vep_missense = vep_missense[
    vep_missense["ref_aa"].str.len().eq(1) &
    vep_missense["alt_aa"].str.len().eq(1)
].copy()

vep_missense["protein_start"] = pd.to_numeric(
    vep_missense["protein_start"], errors="coerce"
)
vep_missense = vep_missense.dropna(subset=["protein_start"]).copy()
vep_missense["protein_start"] = vep_missense["protein_start"].astype(int)

print("VEP missense usable:", vep_missense.shape)

display(vep_missense[[
    "gene_symbol", "transcript_id", "protein_id",
    "consequence_terms", "amino_acids",
    "ref_aa", "alt_aa", "protein_start"
]].head())

# %% Cell 11
def make_vep_vcf_line(row):
    """
    Convert a gnomAD variant row to Ensembl VEP region POST VCF-like input.

    Required columns:
    chrom, pos, variant_id, ref, alt

    Output format:
    chrom pos id ref alt qual filter info

    Example:
    9 69145707 9-69145707-A-G A G . . .
    """
    chrom = str(row["chrom"]).replace("chr", "")
    pos = int(row["pos"])
    var_id = str(row["variant_id"])
    ref = str(row["ref"])
    alt = str(row["alt"])

    return f"{chrom} {pos} {var_id} {ref} {alt} . . ."

gnomad_raw = pd.read_csv(os.path.join(OUT_DIR, "gnomad_pbm_genes_raw.csv"), low_memory=False)

gnomad = missense_gnomad_snv.copy()

gnomad["vep_input"] = gnomad.apply(make_vep_vcf_line, axis=1)

for prefix in ["joint", "exome", "genome"]:
    gnomad[f"{prefix}_ac"] = pd.to_numeric(gnomad[f"{prefix}_ac"], errors="coerce")
    gnomad[f"{prefix}_an"] = pd.to_numeric(gnomad[f"{prefix}_an"], errors="coerce")
    gnomad[f"{prefix}_af"] = gnomad[f"{prefix}_ac"] / gnomad[f"{prefix}_an"]

merged = gnomad.merge(
    vep_missense,
    on="vep_input",
    how="inner",
    suffixes=("_gnomad", "_vep")
)

merged = merged[
    merged["query_gene"].astype(str).str.upper() ==
    merged["gene_symbol"].astype(str).str.upper()
].copy()

print("Merged gnomAD + VEP:", merged.shape)

display(merged[[
    "query_gene", "variant_id", "joint_af",
    "gene_symbol", "amino_acids", "ref_aa", "alt_aa", "protein_start"
]].head())

# %% Cell 12
def fetch_uniprot_sequence(accession, sleep_sec=0.2, max_retries=3):
    accession = str(accession).strip()
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

    for attempt in range(max_retries):
        r = requests.get(url, timeout=60)

        if r.ok:
            text = r.text.strip()
            if not text.startswith(">"):
                return None

            lines = text.splitlines()
            seq = "".join([x.strip() for x in lines if not x.startswith(">")])
            time.sleep(sleep_sec)
            return seq

        if r.status_code in [429, 500, 502, 503, 504]:
            time.sleep(2 * (attempt + 1))
            continue

        return None

    return None


def build_uniprot_sequence_table(pbm_meta, cache_csv=os.path.join(OUT_DIR, "pbm_uniprot_sequences.csv")):
    if os.path.exists(cache_csv):
        print("Loading cached UniProt sequences:", cache_csv)
        return pd.read_csv(cache_csv)

    accessions = sorted(pbm_meta["pbm_uniprot"].dropna().unique())

    rows = []
    for acc in tqdm(accessions, desc="Fetching UniProt sequences"):
        seq = fetch_uniprot_sequence(acc)
        rows.append({
            "pbm_uniprot": acc,
            "uniprot_seq": seq,
            "protein_length": len(seq) if isinstance(seq, str) else np.nan,
            "uniprot_last10": seq[-10:] if isinstance(seq, str) and len(seq) >= 10 else np.nan,
            "uniprot_last6": seq[-6:] if isinstance(seq, str) and len(seq) >= 6 else np.nan,
        })

    seq_df = pd.DataFrame(rows)
    seq_df.to_csv(cache_csv, index=False)
    return seq_df


seq_df = build_uniprot_sequence_table(pbm_meta)

display(seq_df.head())
print("Sequences fetched:", seq_df["uniprot_seq"].notna().sum(), "/", seq_df.shape[0])

# %% Cell 13
pbm_meta2 = pbm_meta.copy()
pbm_meta2["pbm_sequence_10aa"] = pbm_meta2["pbm_sequence_10aa"].astype(str).str.upper()
pbm_meta2["WT_PBM6"] = pbm_meta2["pbm_sequence_10aa"].str[-6:]

pbm_meta2 = pbm_meta2.merge(
    seq_df,
    on="pbm_uniprot",
    how="left"
)

# 只保留 UniProt canonical sequence 末端 10 aa 和训练集 PBM peptide 对得上的
pbm_meta2["cterm10_match"] = (
    pbm_meta2["uniprot_last10"].astype(str).str.upper() ==
    pbm_meta2["pbm_sequence_10aa"].astype(str).str.upper()
)

print("PBM rows:", pbm_meta2.shape)
print("C-terminal 10 aa match:", pbm_meta2["cterm10_match"].sum())

display(pbm_meta2[[
    "pbm_gene", "pbm_uniprot", "pbm_sequence_10aa",
    "uniprot_last10", "cterm10_match"
]].head(20))

# %% Cell 14
candidate = merged.merge(
    pbm_meta2[[
        "pbm_gene", "pbm_uniprot", "pbm_sequence_10aa",
        "WT_PBM6", "uniprot_seq", "protein_length",
        "uniprot_last10", "uniprot_last6", "cterm10_match"
    ]],
    left_on="query_gene",
    right_on="pbm_gene",
    how="inner"
)

candidate = candidate[candidate["cterm10_match"]].copy()

candidate["protein_length"] = pd.to_numeric(candidate["protein_length"], errors="coerce")
candidate = candidate.dropna(subset=["protein_length"]).copy()
candidate["protein_length"] = candidate["protein_length"].astype(int)

candidate["pbm6_start_pos"] = candidate["protein_length"] - 5
candidate["in_last6"] = candidate["protein_start"] >= candidate["pbm6_start_pos"]

candidate = candidate[candidate["in_last6"]].copy()

# PBM index: 0..5
candidate["PBM6_index"] = candidate["protein_start"] - candidate["pbm6_start_pos"]

# PBM position label
candidate["PBM_position"] = candidate["PBM6_index"].map({
    0: "P-5",
    1: "P-4",
    2: "P-3",
    3: "P-2",
    4: "P-1",
    5: "P0",
})

candidate["pbm_ref_aa"] = candidate.apply(
    lambda r: r["WT_PBM6"][int(r["PBM6_index"])]
    if pd.notna(r["PBM6_index"]) and 0 <= int(r["PBM6_index"]) < 6
    else np.nan,
    axis=1
)

candidate["ref_match_pbm"] = candidate["pbm_ref_aa"] == candidate["ref_aa"]

candidate = candidate[candidate["ref_match_pbm"]].copy()

print("True PBM6 variants:", candidate.shape)

display(candidate[[
    "query_gene", "pbm_uniprot", "pbm_sequence_10aa", "WT_PBM6",
    "variant_id", "joint_af",
    "amino_acids", "ref_aa", "alt_aa",
    "protein_start", "protein_length",
    "PBM_position", "pbm_ref_aa"
]].head(20))

# %% Cell 15
def make_mut_pbm6(row):
    wt = list(row["WT_PBM6"])
    idx = int(row["PBM6_index"])
    wt[idx] = row["alt_aa"]
    return "".join(wt)

candidate["MUT_PBM6"] = candidate.apply(make_mut_pbm6, axis=1)

AA1_TO_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter"
}

def make_hgvsp_like(row):
    ref3 = AA1_TO_AA3.get(row["ref_aa"])
    alt3 = AA1_TO_AA3.get(row["alt_aa"])
    if ref3 is None or alt3 is None:
        return np.nan
    return f"p.{ref3}{int(row['protein_start'])}{alt3}"

candidate["hgvsp_like"] = candidate.apply(make_hgvsp_like, axis=1)

final_cols = [
    "query_gene", "pbm_uniprot", "pbm_sequence_10aa",
    "WT_PBM6", "MUT_PBM6",
    "variant_id", "chrom", "pos", "ref", "alt",
    "joint_ac", "joint_an", "joint_af",
    "exome_ac", "exome_an", "exome_af",
    "genome_ac", "genome_an", "genome_af",
    "amino_acids", "hgvsp_like",
    "ref_aa", "alt_aa",
    "protein_start", "protein_length",
    "PBM_position", "PBM6_index",
    "transcript_id", "protein_id",
    "gene_symbol", "consequence_terms"
]

pbm6_variant_table = candidate[final_cols].drop_duplicates().copy()

out_path = os.path.join(OUT_DIR, "pbm6_gnomad_variants_clean.csv")
pbm6_variant_table.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Final PBM6 variant table:", pbm6_variant_table.shape)

display(pbm6_variant_table.head(30))

# %% Cell 16
print("Unique genes with PBM6 variants:", pbm6_variant_table["query_gene"].nunique())
print("Unique WT PBM6:", pbm6_variant_table["WT_PBM6"].nunique())
print("Unique MUT PBM6:", pbm6_variant_table["MUT_PBM6"].nunique())

print("\nVariant count by PBM position:")
display(
    pbm6_variant_table["PBM_position"]
    .value_counts()
    .reindex(["P-5", "P-4", "P-3", "P-2", "P-1", "P0"])
    .reset_index()
    .rename(columns={"index": "PBM_position", "PBM_position": "n_variants"})
)

print("\nTop variants:")
display(
    pbm6_variant_table.sort_values("joint_af", ascending=False)[[
        "query_gene", "WT_PBM6", "MUT_PBM6",
        "PBM_position", "hgvsp_like", "joint_af", "amino_acids"
    ]].head(20)
)
