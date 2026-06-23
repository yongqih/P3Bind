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
    if "pdz_id" not in df.columns:
        if "pdz" in df.columns:
            df["pdz_id"] = df["pdz"].astype(str)
        else:
            df["pdz_id"] = [f"PDZ_{i+1}" for i in range(len(df))]
    df = df[["pdz_id", "pdz_sequence"]].copy()
    df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.upper()
    return df.drop_duplicates().reset_index(drop=True)


def load_models(checkpoint_dir: Optional[str | Path] = None, device: torch.device = DEVICE):
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
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
    required = {"pdz_sequence", "pbm6"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    models, _ = load_models(checkpoint_dir)
    rows = []
    for i, row in df.iterrows():
        pair_id = row.get("pair_id", i)
        try:
            res = predict_pair(row["pdz_sequence"], row["pbm6"], models=models)
            rows.append({"pair_id": pair_id, **res, "status": "ok", "error": ""})
        except Exception as e:
            rows.append({"pair_id": pair_id, "pbm6_used": row.get("pbm6", ""), "status": "failed", "error": str(e)})
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
) -> tuple[dict, pd.DataFrame]:
    """Predict target affinity and background/off-target PDZ profile."""
    target_pdz_sequence = validate_pdz_sequence(target_pdz_sequence)
    pbm6 = validate_pbm6(pbm_or_peptide)
    background = load_background_pdzs(background_csv)
    models, _ = load_models(checkpoint_dir)

    target_res = predict_pKd_ensemble(target_pdz_sequence, pbm6, models=models)
    bg_rows = []
    for _, row in background.iterrows():
        seq = row["pdz_sequence"]
        if seq == target_pdz_sequence:
            continue
        res = predict_pKd_ensemble(seq, pbm6, models=models)
        bg_rows.append({
            "pdz_id": row["pdz_id"],
            "pdz_sequence": seq,
            "pbm6": pbm6,
            "background_pKd_mean": res["predicted_pKd_mean"],
            "background_pKd_std": res["predicted_pKd_std"],
        })
    bg = pd.DataFrame(bg_rows).sort_values("background_pKd_mean", ascending=False).reset_index(drop=True)
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
