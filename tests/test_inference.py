from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import torch

from p3bind.core import (
    load_background_pdzs,
    load_models,
    predict_pair,
    validate_pbm6,
    validate_pdz_sequence,
)
from p3bind.model import predict_pKd_batch


DLG1_PDZ3 = (
    "KVVLHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDRIISVNSVDLRAASHEQAAAALKNAGQAVTIVAQYR"
)


def test_sequence_validation_uses_terminal_pbm6():
    assert validate_pbm6("SGGKFRETRV") == "FRETRV"
    assert validate_pdz_sequence(DLG1_PDZ3) == DLG1_PDZ3
    with pytest.raises(ValueError):
        validate_pbm6("SHORT")
    with pytest.raises(ValueError):
        validate_pdz_sequence("ACDX" * 10)


def test_background_ids_are_unique():
    # Some Windows setups deny access to pytest's global temporary directory.
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = Path(temp_dir) / "background.csv"
        pd.DataFrame(
            {
                "pdz": ["DLG1", "LAP2"],
                "pdz_id": ["PDZ_1", "PDZ_1"],
                "pdz_sequence": [DLG1_PDZ3, "A" * 80],
            }
        ).to_csv(path, index=False)
        background = load_background_pdzs(path)
        assert background["pdz_id"].is_unique
        assert background["pdz_id"].tolist() == ["DLG1_PDZ_1", "LAP2_PDZ_1"]


@pytest.fixture(scope="module")
def release_models():
    models, files = load_models(device=torch.device("cpu"))
    assert len(models) == 5
    assert len(files) == 5
    return models


def test_release_models_reproduce_tsa_candidate_predictions(release_models):
    pbms = ["FRETRV", "RKETRV", "RKETSV", "RKETLM", "RKEGLV"]
    expected = [6.647281, 6.125094, 5.568636, 3.615090, 3.065750]
    observed = [
        predict_pair(DLG1_PDZ3, pbm, models=release_models)["predicted_pKd_mean"]
        for pbm in pbms
    ]
    assert observed == pytest.approx(expected, abs=1e-5)
    assert observed == sorted(observed, reverse=True)


def test_vectorized_batch_matches_single_predictions(release_models):
    pbms = ["FRETRV", "RKETRV", "RKEGLV"]
    means, stds = predict_pKd_batch(
        [DLG1_PDZ3] * len(pbms),
        pbms,
        models=release_models,
        batch_size=2,
    )
    singles = [predict_pair(DLG1_PDZ3, pbm, models=release_models) for pbm in pbms]
    assert means.tolist() == pytest.approx([row["predicted_pKd_mean"] for row in singles])
    assert stds.tolist() == pytest.approx([row["predicted_pKd_std"] for row in singles])
