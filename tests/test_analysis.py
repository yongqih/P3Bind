import pandas as pd
import pytest
import torch

from p3bind.benchmarks import BENCHMARK_MJ_MATRIX, CNNConcatModel, InteractionMapModel
from p3bind.motif import compute_enrichment, generate_random_pbm6_library
from p3bind.training import BiasedDirectionLoss


def test_main_text_neural_models_accept_padded_pdz_batches():
    pdz = torch.full((2, 100), 20, dtype=torch.long)
    pdz[:, :80] = torch.arange(80).remainder(20)
    pbm = torch.arange(12).reshape(2, 6).remainder(20)
    assert CNNConcatModel()(pdz, pbm).shape == (2,)
    assert InteractionMapModel()(pdz, pbm).shape == (2,)
    assert InteractionMapModel(use_mj=True)(pdz, pbm).shape == (2,)
    assert BENCHMARK_MJ_MATRIX.shape == (21, 21)


def test_directional_loss_matches_colab_parameters_and_censor_hinge():
    criterion = BiasedDirectionLoss(reduction="none")
    assert criterion.threshold == 4.5
    pred = torch.tensor([4.0, 5.0, 4.0])
    target = torch.tensor([5.0, 4.0, 3.1])
    censored = torch.tensor([0.0, 0.0, 1.0])
    assert criterion(pred, target, censored).tolist() == pytest.approx([9.0, 9.0, 0.81])


def test_motif_library_is_unique_and_enrichment_has_120_features():
    library = generate_random_pbm6_library(100, seed=7)
    assert library == sorted(set(library))
    scored = pd.DataFrame({"pbm6": library, "predicted_pKd_mean": range(100)})
    enrichment = compute_enrichment(scored, top_fraction=0.01)
    assert len(enrichment) == 120
    assert enrichment.top_count.sum() == 6
