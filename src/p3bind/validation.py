"""Release-data and split validation used by local/PyCharm workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import AA_ORDER


REQUIRED_DATA_COLUMNS = {
    "pdz_sequence",
    "pbm_sequence_10aa",
    "pKd",
    "is_censored",
}
REQUIRED_SPLIT_COLUMNS = {"pair_id", "split", "fold", "setting"}
EXPECTED_SPLITS = {"train", "val", "test"}


def load_pair_dataset(path: str | Path) -> pd.DataFrame:
    """Load the aggregated pair table and add stable pair/PBM6 columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pair-level dataset not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_DATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pair-level dataset is missing columns: {sorted(missing)}")
    df = df.reset_index(drop=True).copy()
    df["pair_id"] = df.index.astype(int)
    df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.strip().str.upper()
    df["pbm_sequence_10aa"] = df["pbm_sequence_10aa"].astype(str).str.strip().str.upper()
    df["pbm6"] = df["pbm_sequence_10aa"].str[-6:]
    df["pKd"] = pd.to_numeric(df["pKd"], errors="raise")
    df["is_censored"] = pd.to_numeric(df["is_censored"], errors="raise").astype(int)
    if "pKd_label" in df:
        df["pKd_label"] = pd.to_numeric(df["pKd_label"], errors="raise")
    if "is_censored_label" in df:
        df["is_censored_label"] = pd.to_numeric(
            df["is_censored_label"], errors="raise"
        ).astype(int)
    return df


def validate_pair_dataset(path: str | Path) -> dict:
    """Validate the manuscript's aggregated PDZ-PBM pair table."""
    df = load_pair_dataset(path)
    alphabet = set(AA_ORDER)
    invalid_pdz = ~df["pdz_sequence"].map(lambda seq: bool(seq) and set(seq) <= alphabet)
    invalid_pbm = ~df["pbm6"].map(lambda seq: len(seq) == 6 and set(seq) <= alphabet)
    if invalid_pdz.any():
        raise ValueError(f"Invalid PDZ sequences at pair IDs: {df.loc[invalid_pdz, 'pair_id'].head(10).tolist()}")
    if invalid_pbm.any():
        raise ValueError(f"Invalid PBM6 sequences at pair IDs: {df.loc[invalid_pbm, 'pair_id'].head(10).tolist()}")
    censor_column = "is_censored_label" if "is_censored_label" in df else "is_censored"
    if not set(df[censor_column].unique()) <= {0, 1}:
        raise ValueError(f"{censor_column} must contain only 0/1 values.")
    # Rows are aggregated by the assayed 10-aa peptide. Distinct 10-aa
    # peptides may share the same terminal PBM6 and must remain separate.
    if df.duplicated(["pdz_sequence", "pbm_sequence_10aa"]).any():
        raise ValueError("Aggregated dataset contains duplicate PDZ-sequence/10-aa-peptide pairs.")

    return {
        "n_pairs": int(len(df)),
        "n_pdz_sequences": int(df["pdz_sequence"].nunique()),
        "n_pbm10_sequences": int(df["pbm_sequence_10aa"].nunique()),
        "n_pbm6_sequences": int(df["pbm6"].nunique()),
        "n_uncensored": int((df[censor_column] == 0).sum()),
        "n_censored": int((df[censor_column] == 1).sum()),
    }


def validate_split_file(
    dataset_path: str | Path,
    split_path: str | Path,
    expected_setting: str | None = None,
) -> dict:
    """Validate five-fold assignments and held-out group isolation."""
    data = load_pair_dataset(dataset_path)
    split_path = Path(split_path)
    split = pd.read_csv(split_path)
    missing = REQUIRED_SPLIT_COLUMNS - set(split.columns)
    if missing:
        raise ValueError(f"{split_path.name} is missing columns: {sorted(missing)}")
    if split.duplicated(["pair_id", "fold"]).any():
        raise ValueError(f"{split_path.name} contains duplicate pair_id/fold assignments.")
    if set(split["split"].astype(str)) != EXPECTED_SPLITS:
        raise ValueError(f"{split_path.name} must contain train, val, and test assignments.")

    settings = set(split["setting"].astype(str))
    if len(settings) != 1:
        raise ValueError(f"{split_path.name} contains multiple setting labels: {sorted(settings)}")
    setting = next(iter(settings))
    if expected_setting is not None and setting != expected_setting:
        raise ValueError(f"Expected setting {expected_setting!r}, found {setting!r}.")

    data_ids = set(data["pair_id"])
    fold_reports = []
    group_column = {"pbm_heldout": "pbm6", "pdz_heldout": "pdz_sequence"}.get(setting)
    for fold in sorted(split["fold"].unique()):
        fold_split = split.loc[split["fold"] == fold].copy()
        if set(fold_split["pair_id"]) != data_ids:
            missing_ids = sorted(data_ids - set(fold_split["pair_id"]))[:10]
            extra_ids = sorted(set(fold_split["pair_id"]) - data_ids)[:10]
            raise ValueError(
                f"{split_path.name} fold {fold} does not cover the dataset exactly; "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        joined = fold_split.merge(
            data[["pair_id", "pbm6", "pdz_sequence"]],
            on="pair_id",
            how="left",
            validate="one_to_one",
        )
        if group_column is not None:
            test_groups = set(joined.loc[joined["split"] == "test", group_column])
            development_groups = set(joined.loc[joined["split"] != "test", group_column])
            overlap = test_groups & development_groups
            if overlap:
                raise ValueError(
                    f"{split_path.name} fold {fold} leaks held-out {group_column} groups: "
                    f"{sorted(overlap)[:10]}"
                )
        counts = joined["split"].value_counts().to_dict()
        fold_reports.append({"fold": int(fold), **{k: int(counts[k]) for k in sorted(counts)}})

    return {
        "setting": setting,
        "n_rows": int(len(split)),
        "n_folds": int(split["fold"].nunique()),
        "folds": fold_reports,
    }
