"""Natural PBM variant-effect scoring against a PDZ panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core import load_background_pdzs, load_models, validate_pbm6
from .model import predict_pKd_batch


def score_variant_effects(
    variants: pd.DataFrame,
    *,
    background_csv: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    models=None,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Create the manuscript's long variant-by-PDZ ΔpKd table."""
    required = {"WT_PBM6", "MUT_PBM6"}
    missing = required - set(variants.columns)
    if missing:
        raise ValueError(f"Variant table is missing columns: {sorted(missing)}")
    variants = variants.copy()
    variants["WT_PBM6"] = variants["WT_PBM6"].map(validate_pbm6)
    variants["MUT_PBM6"] = variants["MUT_PBM6"].map(validate_pbm6)
    if models is None:
        models, _ = load_models(checkpoint_dir)
    panel = load_background_pdzs(background_csv)

    motifs = sorted(set(variants["WT_PBM6"]) | set(variants["MUT_PBM6"]))
    motif_rows = []
    for motif in motifs:
        means, stds = predict_pKd_batch(
            panel["pdz_sequence"].tolist(),
            [motif] * len(panel),
            models=models,
            batch_size=batch_size,
        )
        block = panel.copy()
        block["pbm6"] = motif
        block["pKd_mean"] = means
        block["pKd_std"] = stds
        motif_rows.append(block)
    predictions = pd.concat(motif_rows, ignore_index=True)

    panel_columns = [column for column in panel.columns if column != "pdz_sequence"]
    wt = predictions.rename(
        columns={"pbm6": "WT_PBM6", "pKd_mean": "WT_pKd_mean", "pKd_std": "WT_pKd_std"}
    )[["WT_PBM6", "pdz_sequence", *panel_columns, "WT_pKd_mean", "WT_pKd_std"]]
    mutant = predictions.rename(
        columns={"pbm6": "MUT_PBM6", "pKd_mean": "MUT_pKd_mean", "pKd_std": "MUT_pKd_std"}
    )[["MUT_PBM6", "pdz_id", "MUT_pKd_mean", "MUT_pKd_std"]]
    out = variants.merge(wt, on="WT_PBM6", how="left", validate="many_to_many")
    out = out.merge(mutant, on=["MUT_PBM6", "pdz_id"], how="left", validate="many_to_one")
    out["delta_pKd"] = out["MUT_pKd_mean"] - out["WT_pKd_mean"]
    out["abs_delta_pKd"] = out["delta_pKd"].abs()
    out["effect_direction"] = np.select(
        [out["delta_pKd"] >= 0.5, out["delta_pKd"] <= -0.5],
        ["predicted_gain", "predicted_loss"],
        default="small_or_neutral",
    )
    return out


def summarize_variant_effects(long_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize maximum and thresholded PDZ effects per unique variant."""
    required = {"pdz_id", "delta_pKd", "abs_delta_pKd", "WT_PBM6", "MUT_PBM6"}
    missing = required - set(long_table.columns)
    if missing:
        raise ValueError(f"Long variant table is missing columns: {sorted(missing)}")
    identity_candidates = [
        "query_gene", "pbm_uniprot", "variant_id", "pbm_sequence_10aa", "WT_PBM6", "MUT_PBM6",
        "PBM_position", "PBM6_index", "protein_start", "protein_length", "ref_aa", "alt_aa",
        "aa_change_1letter", "hgvsp_like", "joint_af", "exome_af", "genome_af",
    ]
    identity = [column for column in identity_candidates if column in long_table.columns]
    rows = []
    for keys, group in long_table.groupby(identity, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(identity, keys))
        strongest = group.loc[group["abs_delta_pKd"].idxmax()]
        top_gain = group.loc[group["delta_pKd"].idxmax()]
        top_loss = group.loc[group["delta_pKd"].idxmin()]
        record.update(
            {
                "n_PDZ_scored": int(group["pdz_id"].nunique()),
                "max_abs_delta_pKd": float(strongest["abs_delta_pKd"]),
                "top_abs_delta_pKd": float(strongest["delta_pKd"]),
                "top_abs_PDZ": strongest["pdz_id"],
                "top_abs_WT_pKd": float(strongest["WT_pKd_mean"]),
                "top_abs_MUT_pKd": float(strongest["MUT_pKd_mean"]),
                "max_gain_delta_pKd": float(top_gain["delta_pKd"]),
                "top_gain_PDZ": top_gain["pdz_id"],
                "top_gain_WT_pKd": float(top_gain["WT_pKd_mean"]),
                "top_gain_MUT_pKd": float(top_gain["MUT_pKd_mean"]),
                "max_loss_delta_pKd": float(top_loss["delta_pKd"]),
                "top_loss_PDZ": top_loss["pdz_id"],
                "top_loss_WT_pKd": float(top_loss["WT_pKd_mean"]),
                "top_loss_MUT_pKd": float(top_loss["MUT_pKd_mean"]),
                "mean_abs_delta_pKd": float(group["abs_delta_pKd"].mean()),
                "median_abs_delta_pKd": float(group["abs_delta_pKd"].median()),
                "n_PDZ_abs_delta_ge_0.25": int((group["abs_delta_pKd"] >= 0.25).sum()),
                "n_PDZ_abs_delta_ge_0.5": int((group["abs_delta_pKd"] >= 0.5).sum()),
                "n_PDZ_abs_delta_ge_1.0": int((group["abs_delta_pKd"] >= 1.0).sum()),
                "n_PDZ_gain_ge_0.5": int((group["delta_pKd"] >= 0.5).sum()),
                "n_PDZ_loss_le_minus_0.5": int((group["delta_pKd"] <= -0.5).sum()),
            }
        )
        rows.append(record)
    out = pd.DataFrame(rows).sort_values("max_abs_delta_pKd", ascending=False).reset_index(drop=True)
    out["predicted_variant_effect"] = np.select(
        [out.max_abs_delta_pKd >= 1.0, out.max_abs_delta_pKd >= 0.5],
        ["strong_perturbation", "moderate_perturbation"],
        default="small_or_neutral",
    )
    if "joint_af" in out:
        out["joint_af_class"] = pd.cut(
            out.joint_af,
            bins=[-np.inf, 1e-4, 1e-3, 1e-2, np.inf],
            labels=["ultra_rare_AF<1e-4", "rare_1e-4_to_1e-3",
                    "low_frequency_1e-3_to_1e-2", "common_AF>=1e-2"],
            right=False,
        ).astype(object).fillna("unknown")
    return out
