"""PBM preference-landscape analysis from the design ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import validate_pdz_sequence
from .model import AA_ORDER, predict_pKd_batch


PBM_POSITIONS = ["P-5", "P-4", "P-3", "P-2", "P-1", "P0"]


def generate_random_pbm6_library(
    size: int,
    seed: int = 20260608,
    terminal_allowed: str | None = None,
) -> list[str]:
    if size < 1:
        raise ValueError("size must be at least 1")
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list(AA_ORDER))
    sequences = set()
    while len(sequences) < size:
        sequence = "".join(rng.choice(alphabet, size=6))
        if terminal_allowed is not None:
            sequence = sequence[:5] + str(rng.choice(list(terminal_allowed)))
        sequences.add(sequence)
    return sorted(sequences)


def score_pbm6_library(
    pdz_sequence: str,
    pbm6_library: list[str],
    models,
    batch_size: int = 1024,
) -> pd.DataFrame:
    pdz_sequence = validate_pdz_sequence(pdz_sequence)
    means, stds = predict_pKd_batch(
        [pdz_sequence] * len(pbm6_library),
        pbm6_library,
        models=models,
        batch_size=batch_size,
    )
    return pd.DataFrame(
        {"pbm6": pbm6_library, "predicted_pKd_mean": means, "predicted_pKd_std": stds}
    )


def compute_enrichment(
    scored: pd.DataFrame,
    *,
    top_fraction: float = 0.01,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Compute per-position log2 enrichment among top predicted binders."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive.")
    required = {"pbm6", "predicted_pKd_mean"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Scored library is missing columns: {sorted(missing)}")
    n_top = max(1, int(len(scored) * top_fraction))
    top = scored.nlargest(n_top, "predicted_pKd_mean")
    rows = []
    for position, label in enumerate(PBM_POSITIONS):
        background_counts = scored["pbm6"].str[position].value_counts()
        top_counts = top["pbm6"].str[position].value_counts()
        for aa in AA_ORDER:
            background_frequency = (background_counts.get(aa, 0) + pseudocount) / (
                len(scored) + pseudocount * len(AA_ORDER)
            )
            top_frequency = (top_counts.get(aa, 0) + pseudocount) / (
                len(top) + pseudocount * len(AA_ORDER)
            )
            rows.append(
                {
                    "position": label,
                    "amino_acid": aa,
                    "log2_enrichment": float(np.log2(top_frequency / background_frequency)),
                    "top_count": int(top_counts.get(aa, 0)),
                    "background_count": int(background_counts.get(aa, 0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction_distribution(scored: pd.DataFrame, top_fraction: float = 0.01) -> dict:
    values = scored["predicted_pKd_mean"].to_numpy(dtype=float)
    n_top = max(1, int(len(values) * top_fraction))
    top = np.sort(values)[::-1][:n_top]
    return {
        "library_n": int(len(values)),
        "pKd_mean": float(np.mean(values)),
        "pKd_std": float(np.std(values, ddof=1)),
        "pKd_median": float(np.median(values)),
        "pKd_q75": float(np.quantile(values, 0.75)),
        "pKd_q90": float(np.quantile(values, 0.90)),
        "pKd_q95": float(np.quantile(values, 0.95)),
        "pKd_q99": float(np.quantile(values, 0.99)),
        "pKd_max": float(np.max(values)),
        "top_fraction": float(top_fraction),
        "top_n": n_top,
        "top_mean_pKd": float(np.mean(top)),
        "top_min_pKd": float(np.min(top)),
        "selectivity_gap_q99_minus_median": float(np.quantile(values, 0.99) - np.median(values)),
        "selectivity_gap_max_minus_median": float(np.max(values) - np.median(values)),
        "promiscuity_fraction_pKd_ge_4": float(np.mean(values >= 4.0)),
        "promiscuity_fraction_pKd_ge_5": float(np.mean(values >= 5.0)),
        "promiscuity_fraction_pKd_ge_6": float(np.mean(values >= 6.0)),
    }
