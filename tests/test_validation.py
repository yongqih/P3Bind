from pathlib import Path

from p3bind.core import REPO_ROOT, load_background_pdzs
from p3bind.validation import validate_pair_dataset, validate_split_file


DATASET = REPO_ROOT / "data" / "raw" / "all_data_pair_aggregated.csv"


def test_release_dataset_matches_manuscript_counts():
    report = validate_pair_dataset(DATASET)
    assert report == {
        "n_pairs": 45595,
        "n_pdz_sequences": 260,
        "n_pbm10_sequences": 435,
        "n_pbm6_sequences": 424,
        "n_uncensored": 13487,
        "n_censored": 32108,
    }


def test_release_splits_have_no_heldout_group_leakage():
    for setting in ["random", "pbm_heldout", "pdz_heldout"]:
        report = validate_split_file(
            DATASET,
            REPO_ROOT / "data" / "splits" / f"{setting}_split.csv",
            expected_setting=setting,
        )
        assert report["n_folds"] == 5
        assert report["n_rows"] == 45595 * 5


def test_release_background_panel_has_260_unique_sequences_and_ids():
    panel = load_background_pdzs(REPO_ROOT / "data/processed/background_pdz.csv")
    assert len(panel) == 260
    assert panel["pdz_sequence"].is_unique
    assert panel["pdz_id"].is_unique
