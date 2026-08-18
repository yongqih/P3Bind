"""Design-ensemble architecture and inference helpers.

The architecture is kept checkpoint-compatible with the Colab training
workflow used for the manuscript's design-oriented ensemble.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Amino acid vocabulary
# =========================
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ORDER)}
PAD_IDX = len(AA_ORDER)
VOCAB_SIZE = len(AA_ORDER)
MAX_PDZ_LEN = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_valid_aa_sequence(seq: str) -> bool:
    seq = str(seq).upper()
    return set(seq).issubset(set(AA_ORDER))


# =========================
# Miyazawa-Jernigan-like matrix used by your model
# =========================
MJ_RAW_MATRIX = np.array([
    [-43, -4, 0, 1, 2, 14, 33, 8, 20, 12, 5, 3, 17, 0, 6, -4, 3, -9, 6, 23],
    [2, -31, -30, -22, -15, -14, -2, 3, 3, 22, 11, 41, 11, 11, 6, 16, -2, 4, -5, -14],
    [-2, -28, -42, -30, -30, -22, -9, -20, -8, 18, 11, 36, 27, 23, 12, 22, 12, 11, 3, -6],
    [-7, -19, -34, -43, -29, -25, -13, -18, -1, 18, 4, 41, 19, 25, -6, 26, 24, 23, 6, 10],
    [-4, -6, -26, -26, -56, -28, -24, -19, -5, 14, 11, 21, 11, 23, -1, 29, 14, 3, 3, 40],
    [28, -13, -13, -30, -35, -57, -51, -49, -13, 6, -18, 39, 30, 13, 13, 26, 46, 19, -5, -10],
    [22, 20, -16, -14, -12, -29, -83, -34, -30, -5, -39, 41, 18, 5, 1, 21, 49, 20, -30, 37],
    [20, 3, -26, -16, -27, -34, -41, -44, -15, 6, -14, 31, 21, 22, -8, 22, 23, 8, -10, 0],
    [18, 1, 2, -4, -16, -23, -22, -16, -46, 22, 1, -9, 0, 2, 8, 2, 18, 7, -10, -1],
    [8, 7, 8, 7, 16, 12, -2, -6, 16, 40, 57, -44, -54, 1, 0, 5, 4, 4, 18, 31],
    [4, 3, 3, -1, 9, -11, -17, -7, 5, 67, 26, -33, -45, 14, 2, 7, 11, 11, -5, 14],
    [5, 33, 32, 31, 12, 24, 38, 33, -7, -51, -33, 7, 25, -25, 0, -25, -13, 8, 9, 35],
    [12, 6, 7, 7, 9, 25, 20, 19, 0, -60, -43, 29, 30, 3, 9, -8, -10, 26, -9, 55],
    [3, 17, 27, 23, 29, 10, 17, 13, -4, 1, 8, -31, -0, -26, -12, -11, -13, -0, 7, 34],
    [5, 3, 3, -16, -13, 10, -15, 3, 10, -1, 4, 5, 12, -1, -13, -1, -4, 8, -11, 15],
    [-2, 12, 22, 18, 24, 22, 26, 27, -2, 3, 5, -26, -16, -6, -3, -16, -7, -0, 1, 6],
    [1, 4, 14, 25, 24, 31, 32, 30, 14, 5, 15, -11, -16, -15, -12, -13, -24, -12, 4, 23],
    [-12, -8, -2, 13, 4, -1, 12, -1, 13, 10, 11, 20, 35, -5, 15, 5, 8, -50, -2, -39],
    [8, -9, 2, 7, -2, -23, -43, -22, -18, 17, -7, 21, -1, 13, -10, 5, 13, -4, -10, -10],
    [16, 2, 6, 0, -9, -3, 52, -10, -31, 48, 23, 47, 62, 60, 17, 34, 35, 11, -5, -254]
], dtype=np.float32)

mj_mean, mj_std = MJ_RAW_MATRIX.mean(), MJ_RAW_MATRIX.std()
MJ_NORMALIZED = -1 * (MJ_RAW_MATRIX - mj_mean) / mj_std
MJ_TENSOR = torch.tensor(MJ_NORMALIZED, dtype=torch.float32)


# =========================
# Encoding functions
# =========================
def encode_single_pdz(seq, device=DEVICE, max_len=MAX_PDZ_LEN):
    seq = str(seq).upper()
    idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq]
    idx = idx[:max_len]
    idx = idx + [PAD_IDX] * max(0, max_len - len(idx))
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)


def encode_pdz_batch(seqs, device=DEVICE, max_len=MAX_PDZ_LEN):
    encoded = []
    for seq in seqs:
        seq = str(seq).upper()
        idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq]
        idx = idx[:max_len]
        idx = idx + [PAD_IDX] * max(0, max_len - len(idx))
        encoded.append(idx)
    return torch.tensor(encoded, dtype=torch.long).to(device)


def encode_pbm6(seq, device=DEVICE):
    seq = str(seq).upper()[-6:]
    if len(seq) != 6:
        raise ValueError(f"PBM input must contain at least 6 amino acids. Got: {seq}")
    idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq]
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)


# =========================
# Model architecture
# =========================
class InteractionAwareModel(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE + 1, emb_dim, padding_idx=PAD_IDX)

        # Trainable fusion weights
        self.w_learned = nn.Parameter(torch.tensor(1.5))
        self.w_mj = nn.Parameter(torch.tensor(0.5))

        self.pdz_cnn = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pep_cnn = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.interaction_conv2d = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, pdz, pep):
        # pdz: [B, L_pdz], pep: [B, 6]

        pdz_emb = self.embedding(pdz).permute(0, 2, 1)
        pep_emb = self.embedding(pep).permute(0, 2, 1)

        pdz_feat = self.pdz_cnn(pdz_emb)  # [B, 32, L]
        pep_feat = self.pep_cnn(pep_emb)  # [B, 32, 6]

        learned_map = torch.bmm(pdz_feat.permute(0, 2, 1), pep_feat)
        learned_map = learned_map / (pdz_feat.shape[1] ** 0.5)

        # MJ physical map
        pdz_onehot = F.one_hot(pdz, VOCAB_SIZE + 1)[:, :, :VOCAB_SIZE].float()
        pep_onehot = F.one_hot(pep, VOCAB_SIZE + 1)[:, :, :VOCAB_SIZE].float()

        mj_tensor = MJ_TENSOR.to(pdz.device)
        mj_transformed = torch.matmul(pdz_onehot, mj_tensor)
        mj_map = torch.matmul(mj_transformed, pep_onehot.permute(0, 2, 1))

        interaction_image = torch.stack(
            [
                self.w_learned * learned_map,
                self.w_mj * mj_map
            ],
            dim=1
        )

        out = self.interaction_conv2d(interaction_image)
        pred = self.fc(out)

        return pred, learned_map, mj_map


# =========================
# Model loading and prediction helpers
# =========================
def load_design_ensemble_models(
    model_class=InteractionAwareModel,
    model_dir="checkpoints/design_models",
    device=DEVICE,
):
    model_dir = Path(model_dir)
    device = torch.device(device)

    model_files = sorted(list(model_dir.glob("best_model_fold_*_design_m.pth")))
    if len(model_files) == 0:
        # fallback: accept any .pth or .pt files
        model_files = sorted(list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt")))

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"No model checkpoints found in {model_dir}. "
            "Expected files like best_model_fold_0_design_m.pth, or any .pth/.pt files."
        )

    models = []
    for path in model_files:
        model = model_class().to(device)
        state = torch.load(path, map_location=device, weights_only=True)

        # Your notebook saved raw model.state_dict(), but this also handles wrapped checkpoints.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Remove common prefixes if present.
        cleaned_state = {}
        for k, v in state.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            cleaned_state[new_k] = v

        model.load_state_dict(cleaned_state, strict=True)
        model.eval()
        models.append(model)

    return models, model_files


def model_device(model, fallback=DEVICE):
    """Return the device holding a model's parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def predict_pKd_single_model(
    model,
    pdz_sequence,
    pbm_or_peptide,
    device=DEVICE,
    return_maps=False,
):
    model.eval()
    device = model_device(model, fallback=device)

    pbm6 = str(pbm_or_peptide).upper()[-6:]
    pdz_sequence = str(pdz_sequence).upper()

    if len(pbm6) != 6:
        raise ValueError(f"PBM input must contain at least 6 amino acids. Got: {pbm_or_peptide}")

    invalid_pdz = set(pdz_sequence) - set(AA_ORDER)
    invalid_pbm = set(pbm6) - set(AA_ORDER)

    if len(invalid_pdz) > 0:
        raise ValueError(f"PDZ sequence contains invalid amino acids: {invalid_pdz}")

    if len(invalid_pbm) > 0:
        raise ValueError(f"PBM sequence contains invalid amino acids: {invalid_pbm}")

    pdz_t = encode_single_pdz(pdz_sequence, device=device)
    pbm_t = encode_pbm6(pbm6, device=device)

    with torch.no_grad():
        pred, learned_map, mj_map = model(pdz_t, pbm_t)

    result = {
        "pdz_sequence": pdz_sequence,
        "input_sequence": str(pbm_or_peptide).upper(),
        "pbm6_used": pbm6,
        "predicted_pKd": float(pred.item()),
    }

    if return_maps:
        result["learned_map"] = learned_map.detach().cpu().numpy()[0]
        result["mj_map"] = mj_map.detach().cpu().numpy()[0]

    return result


def predict_pKd_ensemble(
    pdz_sequence,
    pbm_or_peptide,
    models=None,
    model_class=InteractionAwareModel,
    model_dir="checkpoints/design_models",
    device=DEVICE,
    return_individual=False,
):
    if models is None:
        models, model_files = load_design_ensemble_models(
            model_class=model_class,
            model_dir=model_dir,
            device=device,
        )
    else:
        model_files = None

    if not models:
        raise ValueError("At least one trained model is required for ensemble prediction.")

    device = model_device(models[0], fallback=device)

    pbm6 = str(pbm_or_peptide).upper()[-6:]
    pdz_sequence = str(pdz_sequence).upper()

    if len(pbm6) != 6:
        raise ValueError(f"PBM input must contain at least 6 amino acids. Got: {pbm_or_peptide}")

    invalid_pdz = set(pdz_sequence) - set(AA_ORDER)
    invalid_pbm = set(pbm6) - set(AA_ORDER)

    if len(invalid_pdz) > 0:
        raise ValueError(f"PDZ sequence contains invalid amino acids: {invalid_pdz}")

    if len(invalid_pbm) > 0:
        raise ValueError(f"PBM sequence contains invalid amino acids: {invalid_pbm}")

    individual_preds = []
    for model in models:
        res = predict_pKd_single_model(
            model=model,
            pdz_sequence=pdz_sequence,
            pbm_or_peptide=pbm6,
            device=device,
            return_maps=False,
        )
        individual_preds.append(res["predicted_pKd"])

    pred_mean = float(np.mean(individual_preds))
    pred_std = float(np.std(individual_preds, ddof=1)) if len(individual_preds) > 1 else 0.0

    result = {
        "pdz_sequence": pdz_sequence,
        "input_sequence": str(pbm_or_peptide).upper(),
        "pbm6_used": pbm6,
        "predicted_pKd_mean": pred_mean,
        "predicted_pKd_std": pred_std,
        "n_models": len(models),
    }

    if return_individual:
        result["individual_predictions"] = individual_preds

    return result


def predict_pKd_batch(
    pdz_sequences,
    pbm_sequences,
    models,
    batch_size=512,
    device=None,
):
    """Vectorized ensemble prediction for aligned PDZ/PBM sequence pairs."""
    pdz_sequences = [str(seq).strip().upper() for seq in pdz_sequences]
    pbm6_sequences = [str(seq).strip().upper()[-6:] for seq in pbm_sequences]
    if len(pdz_sequences) != len(pbm6_sequences):
        raise ValueError("pdz_sequences and pbm_sequences must have the same length.")
    if not models:
        raise ValueError("At least one trained model is required for batch prediction.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not pdz_sequences:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    invalid_pdz = [i for i, seq in enumerate(pdz_sequences) if not seq or set(seq) - set(AA_ORDER)]
    invalid_pbm = [i for i, seq in enumerate(pbm6_sequences) if len(seq) != 6 or set(seq) - set(AA_ORDER)]
    if invalid_pdz:
        raise ValueError(f"Invalid PDZ sequences at row indices: {invalid_pdz[:10]}")
    if invalid_pbm:
        raise ValueError(f"Invalid PBM sequences at row indices: {invalid_pbm[:10]}")

    if device is None:
        device = model_device(models[0])
    device = torch.device(device)
    model_predictions = [[] for _ in models]

    for start in range(0, len(pdz_sequences), batch_size):
        stop = min(start + batch_size, len(pdz_sequences))
        pdz_t = encode_pdz_batch(pdz_sequences[start:stop], device=device)
        pbm_t = torch.cat(
            [encode_pbm6(seq, device=device) for seq in pbm6_sequences[start:stop]],
            dim=0,
        )
        for model_index, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                pred, _, _ = model(pdz_t, pbm_t)
            model_predictions[model_index].extend(
                pred.detach().cpu().reshape(-1).tolist()
            )

    pred_array = np.asarray(model_predictions, dtype=float)
    means = pred_array.mean(axis=0)
    stds = (
        pred_array.std(axis=0, ddof=1)
        if pred_array.shape[0] > 1
        else np.zeros(pred_array.shape[1], dtype=float)
    )
    return means, stds


def score_sequence_with_ensemble(
    seq,
    target_t,
    bg_t,
    models,
    alpha=1.0,
    device=DEVICE,
):
    if not models:
        raise ValueError("At least one trained model is required for ensemble scoring.")
    device = target_t.device
    seq_t = encode_pbm6(seq, device=device)

    target_scores = []
    bg_scores = []

    for model in models:
        model.eval()
        with torch.no_grad():
            target_pred, _, _ = model(target_t, seq_t)

            cand_expand = seq_t.repeat(bg_t.size(0), 1)
            bg_pred, _, _ = model(bg_t, cand_expand)

        target_scores.append(float(target_pred.item()))
        bg_scores.append(float(bg_pred.mean().item()))

    target_mean = float(np.mean(target_scores))
    target_std = float(np.std(target_scores, ddof=1)) if len(target_scores) > 1 else 0.0

    background_mean = float(np.mean(bg_scores))
    background_std = float(np.std(bg_scores, ddof=1)) if len(bg_scores) > 1 else 0.0

    specificity_score = target_mean - alpha * background_mean

    return {
        "pbm6": str(seq).upper()[-6:],
        "target_pKd_mean": target_mean,
        "target_pKd_std": target_std,
        "background_pKd_mean": background_mean,
        "background_pKd_std": background_std,
        "specificity_score": float(specificity_score),
    }
