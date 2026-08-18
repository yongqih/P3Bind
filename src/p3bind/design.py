"""Specificity-aware PBM6 candidate design."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import load_background_pdzs, load_models, validate_pbm6, validate_pdz_sequence
from .model import AA_ORDER, encode_pdz_batch, encode_single_pdz, score_sequence_with_ensemble


TERMINAL_ALLOWED = "LIVFC"


def optimize_pbm6(
    target_pdz_sequence: str,
    *,
    background_csv: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    alpha: float = 1.0,
    random_initializations: int = 2000,
    steps: int = 12000,
    initial_temperature: float = 1.3,
    final_temperature: float = 0.01,
    top_k: int = 50,
    local_refinement_rounds: int = 2,
    local_refinement_top_n: int = 20,
    manual_candidates: list[str] | None = None,
    seed: int = 42,
    models=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the manuscript's random-init plus simulated-annealing PBM6 search."""
    target_pdz_sequence = validate_pdz_sequence(target_pdz_sequence)
    if random_initializations < 1:
        raise ValueError("random_initializations must be at least 1")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if models is None:
        models, _ = load_models(checkpoint_dir)
    device = next(models[0].parameters()).device
    background = load_background_pdzs(background_csv)
    background = background.loc[background["pdz_sequence"] != target_pdz_sequence]
    if background.empty:
        raise ValueError("At least one non-target background PDZ sequence is required.")

    target_t = encode_single_pdz(target_pdz_sequence, device=device)
    background_t = encode_pdz_batch(background["pdz_sequence"].tolist(), device=device)
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache: dict[str, dict] = {}

    def score(seq: str) -> dict:
        seq = validate_pbm6(seq)
        if seq not in cache:
            cache[seq] = score_sequence_with_ensemble(
                seq, target_t, background_t, models, alpha=alpha
            )
        return cache[seq]

    def random_seq():
        return "".join(rng.choices(AA_ORDER, k=5)) + rng.choice(TERMINAL_ALLOWED)

    initial_candidates = sorted({random_seq() for _ in range(random_initializations)})
    visited_records = []
    for sequence in initial_candidates:
        record = score(sequence).copy()
        record["source"] = "random_init"
        visited_records.append(record)
    current = max(cache, key=lambda seq: cache[seq]["specificity_score"])
    current_score = cache[current]["specificity_score"]
    trajectory = []

    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        temperature = initial_temperature * (
            final_temperature / initial_temperature
        ) ** fraction
        position = rng.randrange(6)
        allowed = TERMINAL_ALLOWED if position == 5 else AA_ORDER
        replacement = rng.choice([aa for aa in allowed if aa != current[position]])
        candidate = current[:position] + replacement + current[position + 1 :]
        candidate_score = score(candidate)["specificity_score"]
        delta = candidate_score - current_score
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
            current = candidate
            current_score = candidate_score
        record = score(candidate).copy()
        record["source"] = "simulated_annealing"
        visited_records.append(record)
        if step % 10 == 0 or step == steps - 1:
            best = max(cache, key=lambda seq: cache[seq]["specificity_score"])
            trajectory.append(
                {
                    "step": step,
                    "temperature": temperature,
                    "current_pbm6": current,
                    "current_specificity_score": current_score,
                    "best_pbm6": best,
                    "best_specificity_score": cache[best]["specificity_score"],
                }
            )

    for sequence in manual_candidates or []:
        record = score(sequence).copy()
        record["source"] = "manual_candidate"
        visited_records.append(record)

    seeds = sorted(cache, key=lambda seq: cache[seq]["specificity_score"], reverse=True)[
        :local_refinement_top_n
    ]
    if local_refinement_rounds < 0:
        raise ValueError("local_refinement_rounds must be non-negative")
    for _ in range(local_refinement_rounds):
        next_seeds = list(seeds)
        for seq in seeds:
            for position in range(6):
                allowed = TERMINAL_ALLOWED if position == 5 else AA_ORDER
                for aa in allowed:
                    if aa != seq[position]:
                        next_seeds.append(seq[:position] + aa + seq[position + 1 :])
        before_best = max(cache[seq]["specificity_score"] for seq in cache)
        for seq in sorted(set(next_seeds) - set(cache)):
            record = score(seq).copy()
            record["source"] = "local_refinement"
            visited_records.append(record)
        seeds = sorted(cache, key=lambda seq: cache[seq]["specificity_score"], reverse=True)[
            :local_refinement_top_n
        ]
        after_best = cache[seeds[0]]["specificity_score"]
        if after_best - before_best < 1e-4:
            break

    candidates = pd.DataFrame(visited_records).sort_values(
        "specificity_score", ascending=False
    ).drop_duplicates("pbm6").reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates.head(top_k).copy(), pd.DataFrame(trajectory), candidates
