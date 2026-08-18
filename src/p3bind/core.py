from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from .model import (
    AA_ORDER,
    DEVICE,
    InteractionAwareModel,
    encode_single_pdz,
    encode_pdz_batch,
    encode_pbm6,
    load_design_ensemble_models,
    predict_pKd_batch,
    predict_pKd_ensemble,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "design_models"
DEFAULT_BACKGROUND_CSV = REPO_ROOT / "data" / "processed" / "background_pdz.csv"


def validate_pdz_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    if len(seq) < 30:
        raise ValueError("PDZ sequence appears too short; please provide a PDZ domain amino-acid sequence.")
    invalid = set(seq) - set(AA_ORDER)
    if invalid:
        raise ValueError(f"PDZ sequence contains invalid amino acids: {sorted(invalid)}")
    return seq


def validate_pbm6(seq: str) -> str:
    pbm6 = str(seq).strip().upper()[-6:]
    if len(pbm6) != 6:
        raise ValueError("PBM input must contain at least 6 amino acids; the C-terminal 6 aa are used.")
    invalid = set(pbm6) - set(AA_ORDER)
    if invalid:
        raise ValueError(f"PBM sequence contains invalid amino acids: {sorted(invalid)}")
    return pbm6


def load_background_pdzs(path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DEFAULT_BACKGROUND_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Background PDZ file not found: {path}. Expected columns: pdz_id,pdz_sequence."
        )
    df = pd.read_csv(path)
    if "pdz_sequence" not in df.columns:
        raise ValueError("background_pdz.csv must contain a 'pdz_sequence' column.")
    df = df.copy()
    df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["pdz_sequence"]).reset_index(drop=True)

    invalid_rows = [
        i for i, seq in enumerate(df["pdz_sequence"])
        if len(seq) < 30 or set(seq) - set(AA_ORDER)
    ]
    if invalid_rows:
        raise ValueError(f"Invalid PDZ sequences in background table at rows: {invalid_rows[:10]}")

    if "pdz_id" not in df.columns:
        if "pdz" in df.columns:
            df["pdz_id"] = df["pdz"].astype(str)
        else:
            df["pdz_id"] = [f"PDZ_{i+1:03d}" for i in range(len(df))]
    else:
        df["pdz_id"] = df["pdz_id"].astype(str)

    if df["pdz_id"].duplicated().any() and "pdz" in df.columns:
        df["pdz_id"] = df["pdz"].astype(str) + "_" + df["pdz_id"]
    if df["pdz_id"].duplicated().any():
        counts = df.groupby("pdz_id").cumcount()
        duplicate_group = df["pdz_id"].duplicated(keep=False)
        df.loc[duplicate_group, "pdz_id"] = (
            df.loc[duplicate_group, "pdz_id"]
            + "_"
            + (counts.loc[duplicate_group] + 1).astype(str)
        )

    metadata = [
        column for column in ["pdz_id", "pdz_sequence", "pdz_gene", "pdz_uniprot", "pdz_site", "pdz_label"]
        if column in df
    ]
    return df[metadata].reset_index(drop=True)


def load_models(
    checkpoint_dir: Optional[str | Path] = None,
    device: str | torch.device | None = None,
):
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    device = DEVICE if device is None else torch.device(device)
    models, model_files = load_design_ensemble_models(
        model_class=InteractionAwareModel,
        model_dir=checkpoint_dir,
        device=device,
    )
    return models, model_files


def predict_pair(pdz_sequence: str, pbm_or_peptide: str, checkpoint_dir: Optional[str | Path] = None, models=None):
    pdz_sequence = validate_pdz_sequence(pdz_sequence)
    pbm6 = validate_pbm6(pbm_or_peptide)
    if models is None:
        models, _ = load_models(checkpoint_dir)
    return predict_pKd_ensemble(pdz_sequence=pdz_sequence, pbm_or_peptide=pbm6, models=models)


def batch_predict(input_csv: str | Path, output_csv: str | Path, checkpoint_dir: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "pdz_sequence" not in df.columns:
        raise ValueError("Input CSV is missing required column: pdz_sequence")
    pbm_column = next(
        (column for column in ["pbm6", "pbm_sequence_10aa", "peptide"] if column in df.columns),
        None,
    )
    if pbm_column is None:
        raise ValueError("Input CSV must contain one of: pbm6, pbm_sequence_10aa, peptide")
    models, _ = load_models(checkpoint_dir)
    rows = []
    for i, row in df.iterrows():
        pair_id = row.get("pair_id", i)
        try:
            res = predict_pair(row["pdz_sequence"], row[pbm_column], models=models)
            rows.append({"pair_id": pair_id, **res, "status": "ok", "error": ""})
        except Exception as e:
            rows.append({"pair_id": pair_id, "pbm6_used": row.get(pbm_column, ""), "status": "failed", "error": str(e)})
    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def specificity_profile(
    target_pdz_sequence: str,
    pbm_or_peptide: str,
    background_csv: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    alpha: float = 1.0,
    models=None,
    batch_size: int = 512,
) -> tuple[dict, pd.DataFrame]:
    """Predict target affinity and background/off-target PDZ profile."""
    target_pdz_sequence = validate_pdz_sequence(target_pdz_sequence)
    pbm6 = validate_pbm6(pbm_or_peptide)
    background = load_background_pdzs(background_csv)
    if models is None:
        models, _ = load_models(checkpoint_dir)

    target_res = predict_pKd_ensemble(target_pdz_sequence, pbm6, models=models)
    background = background[background["pdz_sequence"] != target_pdz_sequence].reset_index(drop=True)
    means, stds = predict_pKd_batch(
        background["pdz_sequence"].tolist(),
        [pbm6] * len(background),
        models=models,
        batch_size=batch_size,
    )
    bg = background.copy()
    bg["pbm6"] = pbm6
    bg["background_pKd_mean"] = means
    bg["background_pKd_std"] = stds
    bg = bg.sort_values("background_pKd_mean", ascending=False).reset_index(drop=True)
    summary = {
        "pbm6": pbm6,
        "target_pKd_mean": target_res["predicted_pKd_mean"],
        "target_pKd_std": target_res["predicted_pKd_std"],
        "background_pKd_mean": float(bg["background_pKd_mean"].mean()) if len(bg) else np.nan,
        "background_pKd_std": float(bg["background_pKd_mean"].std(ddof=1)) if len(bg) > 1 else 0.0,
        "max_background_pKd": float(bg["background_pKd_mean"].max()) if len(bg) else np.nan,
        "top5_background_pKd_mean": float(bg.head(5)["background_pKd_mean"].mean()) if len(bg) else np.nan,
        "specificity_score": float(target_res["predicted_pKd_mean"] - alpha * bg["background_pKd_mean"].mean()) if len(bg) else np.nan,
        "target_minus_max_background": float(target_res["predicted_pKd_mean"] - bg["background_pKd_mean"].max()) if len(bg) else np.nan,
        "n_background_pdzs": int(len(bg)),
        "n_models": int(target_res["n_models"]),
    }
    return summary, bg


def single_mutant_pbm6_sequences(pbm_or_peptide: str) -> list[dict]:
    """Generate all 6 x 19 single-amino-acid mutants for a PBM6 sequence."""
    pbm6 = validate_pbm6(pbm_or_peptide)
    rows = []
    for pos in range(6):
        original = pbm6[pos]
        for aa in AA_ORDER:
            if aa == original:
                continue
            mutant = pbm6[:pos] + aa + pbm6[pos + 1:]
            rows.append({
                "position": pos + 1,
                "original_aa": original,
                "mutant_aa": aa,
                "mutant_pbm6": mutant,
            })
    return rows


def mutation_scan(
    target_pdz_sequence: str,
    pbm_or_peptide: str,
    checkpoint_dir: Optional[str | Path] = None,
    background_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Run a PBM6 single-mutant scan.

    If background_csv is provided, the output includes delta_specificity. This is slower because
    every mutant is also scored against the background PDZ set.
    """
    target_pdz_sequence = validate_pdz_sequence(target_pdz_sequence)
    pbm6 = validate_pbm6(pbm_or_peptide)
    models, _ = load_models(checkpoint_dir)

    baseline = predict_pair(target_pdz_sequence, pbm6, models=models)
    baseline_pkd = baseline["predicted_pKd_mean"]

    baseline_specificity = None
    if background_csv is not None:
        baseline_summary, _ = specificity_profile(
            target_pdz_sequence=target_pdz_sequence,
            pbm_or_peptide=pbm6,
            background_csv=background_csv,
            checkpoint_dir=checkpoint_dir,
            models=models,
        )
        baseline_specificity = baseline_summary["specificity_score"]

    out_rows = []
    for row in single_mutant_pbm6_sequences(pbm6):
        mutant = row["mutant_pbm6"]
        pred = predict_pair(target_pdz_sequence, mutant, models=models)
        result = {
            "pbm6_wt": pbm6,
            **row,
            "target_pKd_mean": pred["predicted_pKd_mean"],
            "target_pKd_std": pred["predicted_pKd_std"],
            "wt_target_pKd_mean": baseline_pkd,
            "delta_pKd": pred["predicted_pKd_mean"] - baseline_pkd,
        }
        if background_csv is not None:
            summary, _ = specificity_profile(
                target_pdz_sequence=target_pdz_sequence,
                pbm_or_peptide=mutant,
                background_csv=background_csv,
                checkpoint_dir=checkpoint_dir,
                models=models,
            )
            result.update({
                "specificity_score": summary["specificity_score"],
                "wt_specificity_score": baseline_specificity,
                "delta_specificity": summary["specificity_score"] - baseline_specificity,
                "background_pKd_mean": summary["background_pKd_mean"],
                "max_background_pKd": summary["max_background_pKd"],
            })
        out_rows.append(result)

    return pd.DataFrame(out_rows)
