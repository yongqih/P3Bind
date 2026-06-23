# Auto-exported from 03_p3bind_full_analysis_design_gnomad.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% [markdown]
# # Model Establishment

# %% Cell 1
import os
import math
import random
import argparse
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

# %% Cell 2
DATA_PATH = Path("/content/drive/MyDrive/PDZ_DL/all_data_pair_aggregated.csv")
SPLIT_PATH = Path("/content/drive/MyDrive/PDZ_DL/splits/random_split.csv")
OUT_DIR = Path("/content/drive/MyDrive/PDZ_DL/design_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = Path("/content/drive/MyDrive/PDZ_DL/design_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATA_PATH)
split_df = pd.read_csv(SPLIT_PATH)

print(df.shape)
print(df.columns.tolist())
print(split_df.shape)
print(split_df.columns.tolist())
display(df.head())
display(split_df.head())

# %% Cell 3
# =========================
# Column names
# =========================
PDZ_COL = "pdz_sequence"
PEP10_COL = "pbm_sequence_10aa"          # 10-aa peptide
PKD_COL = "pKd"                # processed target
CENSOR_COL = "is_censored_label"     # 0 = uncensored, 1 = censored
PAIR_ID_COL = "pair_id"

# If pair_id does not exist, create one
if PAIR_ID_COL not in df.columns:
    df[PAIR_ID_COL] = np.arange(len(df))

# Make sure censor column is int 0/1
df[CENSOR_COL] = pd.to_numeric(df[CENSOR_COL], errors="coerce").astype(int)

# Create PBM6 column from the C-terminal 6 aa of the 10-aa peptide
df["pbm_sequence_6aa"] = df[PEP10_COL].astype(str).str[-6:]

print(df[[PAIR_ID_COL, PDZ_COL, PEP10_COL, "pbm_sequence_6aa", PKD_COL, CENSOR_COL]].head())
print("N pairs:", len(df))
print("N PDZ:", df[PDZ_COL].nunique())
print("N PBM6:", df["pbm_sequence_6aa"].nunique())
print("Censored:", (df[CENSOR_COL] == 1).sum())
print("Uncensored:", (df[CENSOR_COL] == 0).sum())

# %% Cell 4
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ORDER)}
PAD_IDX = len(AA_ORDER)
VOCAB_SIZE = len(AA_ORDER)

def is_valid_aa_sequence(seq):
    seq = str(seq)
    return set(seq).issubset(set(AA_ORDER))


# Filter invalid sequences
df = df[
    df[PDZ_COL].apply(is_valid_aa_sequence) &
    df[PEP10_COL].apply(is_valid_aa_sequence)
].reset_index(drop=True)

print("After filtering:", df.shape)

MJ_raw_matrix = np.array([
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

mj_mean, mj_std = MJ_raw_matrix.mean(), MJ_raw_matrix.std()
MJ_NORMALIZED = -1 * (MJ_raw_matrix - mj_mean) / mj_std
MJ_TENSOR = torch.tensor(MJ_NORMALIZED, device=DEVICE)

# %% Cell 5
class PDZCTermDataset(Dataset):
    def __init__(
        self,
        df,
        pdz_col=PDZ_COL,
        pep_col=PEP10_COL,
        pkd_col=PKD_COL,
        censor_col=CENSOR_COL,
        max_len_pdz=100
    ):
        self.df = df.reset_index(drop=True)
        self.pdz_col = pdz_col
        self.pep_col = pep_col
        self.pkd_col = pkd_col
        self.censor_col = censor_col
        self.max_len_pdz = max_len_pdz

    def encode_pdz(self, seq):
        seq = str(seq)
        idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq]
        idx = idx[:self.max_len_pdz]
        idx = idx + [PAD_IDX] * max(0, self.max_len_pdz - len(idx))
        return torch.tensor(idx, dtype=torch.long)

    def encode_pep(self, seq):
        # Use C-terminal 6 amino acids from the 10-aa peptide
        seq_cut = str(seq)[-6:]
        if len(seq_cut) < 6:
            idx = [PAD_IDX] * (6 - len(seq_cut)) + [AA_TO_INT.get(aa, PAD_IDX) for aa in seq_cut]
        else:
            idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq_cut]
        return torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        pdz = self.encode_pdz(row[self.pdz_col])
        pep = self.encode_pep(row[self.pep_col])
        y = torch.tensor(row[self.pkd_col], dtype=torch.float32)
        censor = torch.tensor(row[self.censor_col], dtype=torch.float32)

        return pdz, pep, y, censor

# %% Cell 6
class BiasedDirectionLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        threshold=4.5,
        pos_weight=3.0,
        penalty_scale=3.0,
        relax_scale=0.3,
        LOD=3.1
    ):
        super().__init__()
        self.reduction = reduction
        self.threshold = threshold
        self.pos_weight = pos_weight
        self.penalty_scale = penalty_scale
        self.relax_scale = relax_scale
        self.LOD = LOD

    def forward(self, pred, target, mask):
        # mask=1: censored
        # mask=0: uncensored

        mse = (pred - target) ** 2

        is_high_true = (mask == 0) & (target >= self.threshold)
        is_low_true = (mask == 0) & (target < self.threshold)

        bad_high = is_high_true & (pred < target)
        ok_high = is_high_true & (pred >= target)

        bad_low = is_low_true & (pred > target)
        ok_low = is_low_true & (pred <= target)

        weights = torch.ones_like(pred) * self.pos_weight

        is_bad = bad_high | bad_low
        is_ok = ok_high | ok_low

        weights = torch.where(is_bad, weights * self.penalty_scale, weights)
        weights = torch.where(is_ok, weights * self.relax_scale, weights)

        loss_uncensored = mse * weights

        # For censored low-affinity observations:
        # Penalize predictions above the detection-limit label.
        diff = pred - target
        loss_censored = F.relu(diff) ** 2

        loss = torch.where(mask.bool(), loss_censored, loss_uncensored)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss

# %% [markdown]
# Loss Function:$$L = \frac{1}{N} \sum_{i=1}^{N} \left[ (1 - m_i) w_i (\hat{y}_i - y_i)^2 + m_i \max(0, \hat{y}_i - y_i)^2 \right]$$Where weights are defined as:$$w_i = w_0 \times \begin{cases} \text{penalty_scale}, & \text{if direction is bad} \\ \text{relax_scale}, & \text{if direction is ok} \\ \end{cases}$$Direction Judgment Rules:If $y_i \ge \theta$ (True High Value):bad: $\hat{y}_i < y_i$ok: $\hat{y}_i \ge y_i$If $y_i < \theta$ (True Low Value):bad: $\hat{y}_i > y_i$ok: $\hat{y}_i \le y_i$
# $$\hat{y}^* = \operatorname*{argmin}_{\hat{y}} L$$

# %% Cell 8
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

        mj_transformed = torch.matmul(pdz_onehot, MJ_TENSOR)
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

# %% Cell 9
def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if len(np.unique(y_pred)) <= 1 or len(np.unique(y_true)) <= 1:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def evaluate_model(model, loader, criterion=None, device=DEVICE):
    model.eval()

    all_preds = []
    all_y = []
    all_mask = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for pdz, pep, y, mask in loader:
            pdz = pdz.to(device)
            pep = pep.to(device)
            y = y.to(device).unsqueeze(1)
            mask = mask.to(device).unsqueeze(1)

            pred, _, _ = model(pdz, pep)

            if criterion is not None:
                loss = criterion(pred, y, mask)
                total_loss += loss.item()
                n_batches += 1

            all_preds.append(pred.cpu().numpy().reshape(-1))
            all_y.append(y.cpu().numpy().reshape(-1))
            all_mask.append(mask.cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    mask_arr = np.concatenate(all_mask)

    metrics_all = compute_regression_metrics(y_true, y_pred)

    uncensored_idx = mask_arr == 0
    if uncensored_idx.sum() > 2:
        metrics_uncensored = compute_regression_metrics(
            y_true[uncensored_idx],
            y_pred[uncensored_idx]
        )
    else:
        metrics_uncensored = {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }

    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, metrics_all, metrics_uncensored, y_true, y_pred, mask_arr

# %% Cell 10
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=OUT_DIR/'checkpoint.pth', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# %% Cell 11
def train_one_fold(
    fold_id,
    train_df,
    val_df,
    model_class=InteractionAwareModel,
    num_epochs=80,
    batch_size=64,
    lr=1e-3,
    patience=12,
    device=DEVICE,
    save_dir=OUT_DIR
):
    train_loader = DataLoader(
        PDZCTermDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        PDZCTermDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = BiasedDirectionLoss(
        threshold=4.5,
        pos_weight=3.0,
        penalty_scale=3.0,
        relax_scale=0.3,
        LOD=3.1
    )

    best_val_loss = np.inf
    best_epoch = -1
    counter = 0

    history = []

    save_path = Path(save_dir) / f"best_model_fold_{fold_id}_design_m.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for pdz, pep, y, mask in train_loader:
            pdz = pdz.to(device)
            pep = pep.to(device)
            y = y.to(device).unsqueeze(1)
            mask = mask.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred, _, _ = model(pdz, pep)
            loss = criterion(pred, y, mask)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_loss, val_metrics_all, val_metrics_unc, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion=criterion,
            device=device
        )

        record = {
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,

            "val_rmse": val_metrics_all["rmse"],
            "val_mae": val_metrics_all["mae"],
            "val_r2": val_metrics_all["r2"],
            "val_pearson": val_metrics_all["pearson"],
            "val_spearman": val_metrics_all["spearman"],

            "val_uncensored_rmse": val_metrics_unc["rmse"],
            "val_uncensored_mae": val_metrics_unc["mae"],
            "val_uncensored_r2": val_metrics_unc["r2"],
            "val_uncensored_pearson": val_metrics_unc["pearson"],
            "val_uncensored_spearman": val_metrics_unc["spearman"],
        }

        history.append(record)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), save_path)
            saved_text = "saved"
        else:
            counter += 1
            saved_text = f"no improve {counter}/{patience}"

        print(
            f"Fold {fold_id} | Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
            f"val Pearson={val_metrics_all['pearson']:.3f} | "
            f"val RMSE={val_metrics_all['rmse']:.3f} | {saved_text}"
        )

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(save_dir) / f"training_history_fold_{fold_id}_design.csv", index=False)

    # Load best model for final validation evaluation
    best_model = model_class().to(device)
    best_model.load_state_dict(torch.load(save_path, map_location=device))
    best_model.eval()

    final_val_loss, final_metrics_all, final_metrics_unc, y_true, y_pred, mask_arr = evaluate_model(
        best_model,
        val_loader,
        criterion=criterion,
        device=device
    )

    summary = {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,

        "val_rmse": final_metrics_all["rmse"],
        "val_mae": final_metrics_all["mae"],
        "val_r2": final_metrics_all["r2"],
        "val_pearson": final_metrics_all["pearson"],
        "val_spearman": final_metrics_all["spearman"],

        "val_uncensored_rmse": final_metrics_unc["rmse"],
        "val_uncensored_mae": final_metrics_unc["mae"],
        "val_uncensored_r2": final_metrics_unc["r2"],
        "val_uncensored_pearson": final_metrics_unc["pearson"],
        "val_uncensored_spearman": final_metrics_unc["spearman"],

        "model_path": str(save_path),
    }

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "is_censored": mask_arr,
    })
    pred_df.to_csv(Path(save_dir) / f"validation_predictions_fold_{fold_id}_design.csv", index=False)

    return summary, history_df

# %% Cell 12
# def get_split_column(split_df):
#     for col in ["split", "role", "set"]:
#         if col in split_df.columns:
#             return col
#     raise ValueError("Cannot find split column. Expected one of: split, role, set")


# def train_design_ensemble_from_split(
#     df,
#     split_df,
#     fold_col="fold",
#     pair_id_col=PAIR_ID_COL,
#     num_epochs=80,
#     batch_size=64,
#     lr=1e-3,
#     patience=12,
#     device=DEVICE,
#     save_dir=OUT_DIR
# ):
#     split_col = get_split_column(split_df)

#     summaries = []

#     for fold_id in sorted(split_df[fold_col].unique()):
#         print("\n" + "=" * 80)
#         print(f"Training design model fold {fold_id}")
#         print("=" * 80)

#         fold_split = split_df[split_df[fold_col] == fold_id].copy()

#         train_ids = fold_split[fold_split[split_col].isin(["train", "training"])][pair_id_col].values
#         val_ids = fold_split[fold_split[split_col].isin(["val", "valid", "validation"])][pair_id_col].values

#         if len(val_ids) == 0:
#             raise ValueError(
#                 f"No validation rows found for fold {fold_id}. "
#                 f"Check split column values: {fold_split[split_col].unique()}"
#             )

#         train_df = df[df[pair_id_col].isin(train_ids)].copy()
#         val_df = df[df[pair_id_col].isin(val_ids)].copy()

#         print("Train:", train_df.shape, "Val:", val_df.shape)

#         summary, history_df = train_one_fold(
#             fold_id=fold_id,
#             train_df=train_df,
#             val_df=val_df,
#             model_class=InteractionAwareModel,
#             num_epochs=num_epochs,
#             batch_size=batch_size,
#             lr=lr,
#             patience=patience,
#             device=device,
#             save_dir=save_dir
#         )

#         summaries.append(summary)

#     summary_df = pd.DataFrame(summaries)
#     summary_path = Path(save_dir) / "design_ensemble_validation_summary.csv"
#     summary_df.to_csv(summary_path, index=False)

#     print("\n===== Design ensemble validation summary =====")
#     display(summary_df)
#     print("Saved:", summary_path)

#     return summary_df


# design_summary_df = train_design_ensemble_from_split(
#     df=df,
#     split_df=split_df,
#     fold_col="fold",
#     pair_id_col=PAIR_ID_COL,
#     num_epochs=80,
#     batch_size=64,
#     lr=1e-3,
#     patience=12,
#     device=DEVICE,
#     save_dir=OUT_DIR
# )


# %% Cell 13
all_pdz_seqs = df[PDZ_COL].unique()


def load_design_ensemble_models(model_class=InteractionAwareModel, model_dir=OUT_DIR, device=DEVICE):
    model_files = sorted([
        f for f in Path(model_dir).glob("best_model_fold_*_design_m.pth")
    ])

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"No design model checkpoints found in {model_dir}. "
            "Expected files like best_model_fold_0_design_m.pth"
        )

    models = []

    for path in model_files:
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} design ensemble models.")
    return models, model_files


def encode_single_pdz(seq, device=DEVICE, max_len=100):
    idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in str(seq)]
    idx = idx[:max_len]
    idx = idx + [PAD_IDX] * max(0, max_len - len(idx))
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)


def encode_pdz_batch(seqs, device=DEVICE, max_len=100):
    encoded = []

    for seq in seqs:
        idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in str(seq)]
        idx = idx[:max_len]
        idx = idx + [PAD_IDX] * max(0, max_len - len(idx))
        encoded.append(idx)

    return torch.tensor(encoded, dtype=torch.long).to(device)


def encode_pbm6(seq, device=DEVICE):
    seq = str(seq)[-6:]
    idx = [AA_TO_INT.get(aa, PAD_IDX) for aa in seq]
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)


def score_sequence_with_ensemble(
    seq,
    target_t,
    bg_t,
    models,
    alpha=1.0,
    device=DEVICE
):
    seq_t = encode_pbm6(seq, device=device)

    target_scores = []
    bg_scores = []

    for model in models:
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
        "pbm6": seq,
        "target_pKd_mean": target_mean,
        "target_pKd_std": target_std,
        "background_pKd_mean": background_mean,
        "background_pKd_std": background_std,
        "specificity_score": specificity_score,
    }

# %% [markdown]
# # Peptide design and pKd prediction

# %% Cell 15
def optimize_peptide_by_ensemble(
    target_pdz,
    bg_pdzs=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    seq_len=6,
    steps=12000,
    device=DEVICE,
    alpha=1.0,
    T0=1.3,
    Tf=0.01,
    n_random_init=2000,
    top_k=50,
    save_prefix="p3bind_design_demo",
    out_dir=RESULT_DIR,
    random_seed=42,
    verbose=True,
    manual_candidates=None,
    local_refine=True,
    local_refine_top_n=20,
    local_refine_rounds=2
):
    """
    Ensemble-guided PBM6 design.

    Search strategy:
    1. Random initialization
    2. Simulated annealing
    3. Optional manual candidate scoring
    4. Optional single-mutant local refinement around top candidates
    5. Final re-ranking by specificity score

    specificity_score = target_pKd_mean - alpha * background_pKd_mean
    """

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, model_files = load_design_ensemble_models(
        model_class=model_class,
        model_dir=model_dir,
        device=device
    )

    target_t = encode_single_pdz(target_pdz, device=device)

    if bg_pdzs is None:
        bg_pdzs = [p for p in all_pdz_seqs if p != target_pdz]
        if verbose:
            print(f"bg_pdzs not specified → using all {len(bg_pdzs)} non-target PDZs.")
    else:
        if isinstance(bg_pdzs, str):
            bg_pdzs = [bg_pdzs]
        if verbose:
            print(f"Using user-specified {len(bg_pdzs)} background PDZ(s).")

    bg_t = encode_pdz_batch(bg_pdzs, device=device)

    AA = list(AA_ORDER)

    # Common hydrophobic C-terminal residues in PDZ-binding motifs
    END_ALLOWED = "LIVFC"

    def random_pbm6():
        return "".join(random.choices(AA, k=seq_len - 1)) + random.choice(END_ALLOWED)

    def mutate_pbm6(seq):
        seq_list = list(seq)
        pos = random.randint(0, seq_len - 1)

        if pos == seq_len - 1:
            allowed = END_ALLOWED
        else:
            allowed = AA

        old_aa = seq_list[pos]
        new_aa = random.choice(allowed)

        if len(allowed) > 1:
            while new_aa == old_aa:
                new_aa = random.choice(allowed)

        seq_list[pos] = new_aa
        return "".join(seq_list), pos, old_aa, new_aa

    def generate_single_mutants(seq):
        """
        Generate all one-step mutants of a PBM6 sequence.
        Last residue is restricted to END_ALLOWED.
        """
        seq = str(seq).strip().upper()
        mutants = set()

        if len(seq) != seq_len:
            return []

        for i in range(seq_len):
            allowed = END_ALLOWED if i == seq_len - 1 else AA

            for aa in allowed:
                if aa != seq[i]:
                    mutant = seq[:i] + aa + seq[i+1:]
                    mutants.add(mutant)

        return sorted(mutants)

    score_cache = {}

    def score_cached(seq):
        seq = str(seq).strip().upper()[-seq_len:]

        if len(seq) != seq_len:
            raise ValueError(f"PBM sequence must have length {seq_len}. Got: {seq}")

        if seq not in score_cache:
            score_cache[seq] = score_sequence_with_ensemble(
                seq=seq,
                target_t=target_t,
                bg_t=bg_t,
                models=models,
                alpha=alpha,
                device=device
            )
        return score_cache[seq]

    visited_records = []

    # =========================
    # 1. Random initialization
    # =========================
    if verbose:
        print(f"Scoring {n_random_init} random initial PBM6 candidates...")

    init_candidates = sorted(set(random_pbm6() for _ in range(n_random_init)))

    for seq in tqdm(init_candidates):
        rec = score_cached(seq).copy()
        rec["source"] = "random_init"
        rec["step"] = -1
        rec["temperature"] = np.nan
        rec["accepted"] = True
        rec["mutation_pos"] = np.nan
        rec["old_aa"] = ""
        rec["new_aa"] = ""
        visited_records.append(rec)

    init_best = max(
        init_candidates,
        key=lambda s: score_cached(s)["specificity_score"]
    )

    current_seq = init_best
    current_score = score_cached(current_seq)

    best_seq = current_seq
    best_score = current_score.copy()

    if verbose:
        print(
            f"Initial best seq = {best_seq}, "
            f"target={best_score['target_pKd_mean']:.3f}, "
            f"bg={best_score['background_pKd_mean']:.3f}, "
            f"score={best_score['specificity_score']:.3f}"
        )

    trajectory_records = []

    # =========================
    # 2. Simulated annealing
    # =========================
    for step in tqdm(range(steps)):
        T = T0 * ((Tf / T0) ** (step / max(steps - 1, 1)))

        candidate_seq, mut_pos, old_aa, new_aa = mutate_pbm6(current_seq)
        candidate_score = score_cached(candidate_seq)

        delta = candidate_score["specificity_score"] - current_score["specificity_score"]

        if delta > 0:
            accept = True
            accept_prob = 1.0
        else:
            accept_prob = math.exp(delta / max(T, 1e-8))
            accept = random.random() < accept_prob

        if accept:
            current_seq = candidate_seq
            current_score = candidate_score

        if current_score["specificity_score"] > best_score["specificity_score"]:
            best_seq = current_seq
            best_score = current_score.copy()

        rec = candidate_score.copy()
        rec["source"] = "simulated_annealing"
        rec["step"] = step
        rec["temperature"] = T
        rec["accepted"] = accept
        rec["mutation_pos"] = mut_pos + 1
        rec["old_aa"] = old_aa
        rec["new_aa"] = new_aa
        visited_records.append(rec)

        if step % 10 == 0 or step == steps - 1:
            traj = best_score.copy()
            traj["step"] = step
            traj["temperature"] = T
            traj["best_pbm6"] = best_seq
            traj["current_pbm6"] = current_seq
            traj["current_specificity_score"] = current_score["specificity_score"]
            trajectory_records.append(traj)

        if verbose and step % 500 == 0:
            print(
                f"Step {step:5d}, T={T:.4f}, "
                f"current={current_seq}, "
                f"best={best_seq}, "
                f"target={best_score['target_pKd_mean']:.3f}, "
                f"bg={best_score['background_pKd_mean']:.3f}, "
                f"score={best_score['specificity_score']:.3f}"
            )

    # =========================
    # 3. Optional manual candidates
    # =========================
    if manual_candidates is not None:
        if verbose:
            print(f"Scoring {len(manual_candidates)} manual candidate(s)...")

        for seq in manual_candidates:
            seq = str(seq).strip().upper()[-seq_len:]

            if len(seq) != seq_len:
                print(f"Skipping invalid manual candidate: {seq}")
                continue

            rec = score_cached(seq).copy()
            rec["source"] = "manual_candidate"
            rec["step"] = -2
            rec["temperature"] = np.nan
            rec["accepted"] = True
            rec["mutation_pos"] = np.nan
            rec["old_aa"] = ""
            rec["new_aa"] = ""
            visited_records.append(rec)

    # =========================
    # 4. First ranking before local refinement
    # =========================
    visited_df = pd.DataFrame(visited_records)

    candidate_df_pre = (
        visited_df
        .sort_values("specificity_score", ascending=False)
        .drop_duplicates(subset=["pbm6"], keep="first")
        .reset_index(drop=True)
    )

    # =========================
    # 5. Optional iterative local single-mutant refinement
    # =========================
    if local_refine:
        for refine_round in range(1, local_refine_rounds + 1):

            # Re-rank all candidates scored so far
            current_df = pd.DataFrame(visited_records)
            current_df = (
                current_df
                .sort_values("specificity_score", ascending=False)
                .drop_duplicates(subset=["pbm6"], keep="first")
                .reset_index(drop=True)
            )

            seed_seqs = current_df.head(local_refine_top_n)["pbm6"].tolist()

            local_mutants = set()
            for seq in seed_seqs:
                local_mutants.update(generate_single_mutants(seq))

            already_scored = set(current_df["pbm6"].tolist())
            local_mutants = sorted(local_mutants - already_scored)

            if verbose:
                print(
                    f"Local refinement round {refine_round}/{local_refine_rounds}: "
                    f"using top {local_refine_top_n} candidates to generate "
                    f"{len(local_mutants)} new single-mutant candidates."
                )

            if len(local_mutants) == 0:
                if verbose:
                    print("No new local mutants found. Stopping local refinement.")
                break

            before_best = current_df.iloc[0]["specificity_score"]

            for seq in tqdm(local_mutants):
                rec = score_cached(seq).copy()
                rec["source"] = f"local_refinement_round_{refine_round}"
                rec["step"] = -3 - refine_round
                rec["temperature"] = np.nan
                rec["accepted"] = True
                rec["mutation_pos"] = np.nan
                rec["old_aa"] = ""
                rec["new_aa"] = ""
                visited_records.append(rec)

            # Check improvement after this refinement round
            updated_df = pd.DataFrame(visited_records)
            updated_df = (
                updated_df
                .sort_values("specificity_score", ascending=False)
                .drop_duplicates(subset=["pbm6"], keep="first")
                .reset_index(drop=True)
            )

            after_best = updated_df.iloc[0]["specificity_score"]
            improvement = after_best - before_best

            if verbose:
                print(
                    f"Best score before round {refine_round}: {before_best:.4f}; "
                    f"after round {refine_round}: {after_best:.4f}; "
                    f"improvement: {improvement:.4f}"
                )

            # Optional early stop if no meaningful improvement
            if improvement < 1e-4:
                if verbose:
                    print("No meaningful improvement. Stopping local refinement.")
                break

    # =========================
    # 6. Final ranking
    # =========================
    visited_df = pd.DataFrame(visited_records)

    candidate_df = (
        visited_df
        .sort_values("specificity_score", ascending=False)
        .drop_duplicates(subset=["pbm6"], keep="first")
        .reset_index(drop=True)
    )

    candidate_df.insert(0, "rank", np.arange(1, len(candidate_df) + 1))

    top_candidates_df = candidate_df.head(top_k).copy()
    trajectory_df = pd.DataFrame(trajectory_records)

    top_path = out_dir / f"{save_prefix}_top{top_k}_candidates.csv"
    visited_path = out_dir / f"{save_prefix}_all_candidates_reranked.csv"
    traj_path = out_dir / f"{save_prefix}_trajectory.csv"

    top_candidates_df.to_csv(top_path, index=False)
    candidate_df.to_csv(visited_path, index=False)
    trajectory_df.to_csv(traj_path, index=False)

    final_best = top_candidates_df.iloc[0]

    if verbose:
        print("\n===== Final Result: Ensemble-guided PBM Design =====")
        print(f"Best PBM6 after final re-ranking: {final_best['pbm6']}")
        print(f"Mean target pKd: {final_best['target_pKd_mean']:.3f}")
        print(f"Mean background pKd: {final_best['background_pKd_mean']:.3f}")
        print(f"Specificity score: {final_best['specificity_score']:.3f}")
        print(f"\nSaved top candidates to: {top_path}")
        print(f"Saved all re-ranked candidates to: {visited_path}")
        print(f"Saved trajectory to: {traj_path}")
        display(top_candidates_df.head(20))

    return top_candidates_df, trajectory_df, candidate_df

# %% Cell 16
# Pick one target PDZ sequence from the dataset
target_pdz = "GFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV"
bg_pdzs = None

top_candidates_df, trajectory_df, all_candidates_df = optimize_peptide_by_ensemble(
    target_pdz=target_pdz,
    bg_pdzs=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    seq_len=6,
    steps=12000,
    device=DEVICE,
    alpha=1.0,
    T0=1.3,
    Tf=0.01,
    n_random_init=2000,
    top_k=50,
    save_prefix="target_pdz_demo_refined",
    out_dir=RESULT_DIR,
    random_seed=42,
    verbose=True,
    manual_candidates=None,
    local_refine=True,
    local_refine_top_n=20,
    local_refine_rounds=3
)

# %% Cell 17
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("figures")
outdir.mkdir(exist_ok=True)

plot_df = all_candidates_df.copy()

fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

ax.scatter(
    plot_df["background_pKd_mean"],
    plot_df["target_pKd_mean"],
    s=10,
    alpha=0.25,
    edgecolors="none"
)

# Highlight top 20
top20 = top_candidates_df.head(20)

ax.scatter(
    top20["background_pKd_mean"],
    top20["target_pKd_mean"],
    s=35,
    edgecolors="black",
    linewidth=0.5,
    label="Top 20 candidates"
)

ax.set_xlabel("Mean background predicted pKd")
ax.set_ylabel("Target predicted pKd")
ax.set_title("PDZ-specific PBM candidate ranking")

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(outdir / "Figure5_design_scatter.png", bbox_inches="tight", dpi=600)
plt.savefig(outdir / "Figure5_design_scatter.pdf", bbox_inches="tight")
plt.savefig(outdir / "Figure5_design_scatter.svg", bbox_inches="tight")

plt.show()

# %% Cell 18
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)

ax.plot(
    trajectory_df["step"],
    trajectory_df["specificity_score"],
    linewidth=1.8
)

ax.set_xlabel("Optimization step")
ax.set_ylabel("Best specificity score")
ax.set_title("Simulated annealing optimization trajectory")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(outdir / "Figure5_design_trajectory.png", bbox_inches="tight", dpi=600)
plt.savefig(outdir / "Figure5_design_trajectory.pdf", bbox_inches="tight")
plt.savefig(outdir / "Figure5_design_trajectory.svg", bbox_inches="tight")

plt.show()

# %% Cell 19
def predict_pKd_single_model(
    model,
    pdz_sequence,
    pbm_or_peptide,
    device=DEVICE,
    return_maps=False
):
    """
    Predict pKd for one PDZ sequence and one PBM/peptide sequence using a single model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained InteractionAwareModel.
    pdz_sequence : str
        PDZ domain amino acid sequence.
    pbm_or_peptide : str
        Either a PBM6 sequence or a longer peptide sequence.
        The function automatically uses the C-terminal 6 amino acids.
    return_maps : bool
        Whether to return learned interaction map and MJ map.

    Returns
    -------
    dict
        Predicted pKd and processed PBM6 sequence.
    """

    model.eval()

    # Use C-terminal 6 aa as PBM6
    pbm6 = str(pbm_or_peptide).upper()[-6:]
    pdz_sequence = str(pdz_sequence).upper()

    # Basic validation
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
        "predicted_pKd": float(pred.item())
    }

    if return_maps:
        result["learned_map"] = learned_map.detach().cpu().numpy()[0]
        result["mj_map"] = mj_map.detach().cpu().numpy()[0]

    return result

# %% Cell 20
def predict_pKd_ensemble(
    pdz_sequence,
    pbm_or_peptide,
    models=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    device=DEVICE,
    return_individual=False
):
    """
    Predict pKd for one PDZ-PBM pair using an ensemble of trained models.

    Parameters
    ----------
    pdz_sequence : str
        PDZ domain sequence.
    pbm_or_peptide : str
        PBM6 or longer peptide sequence. C-terminal 6 aa will be used.
    models : list or None
        Loaded model ensemble. If None, models will be loaded from model_dir.
    return_individual : bool
        Whether to return individual model predictions.

    Returns
    -------
    dict
        Ensemble mean predicted pKd, standard deviation, and PBM6 used.
    """

    if models is None:
        models, model_files = load_design_ensemble_models(
            model_class=model_class,
            model_dir=model_dir,
            device=device
        )
    else:
        model_files = None

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
            return_maps=False
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
        "n_models": len(models)
    }

    if return_individual:
        result["individual_predictions"] = individual_preds

    return result

# %% Cell 21
# Load ensemble once
models, model_files = load_design_ensemble_models(
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    device=DEVICE
)

# Example input
example_pdz = "EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV"
example_peptide = "FIETWV"

result = predict_pKd_ensemble(
    pdz_sequence=example_pdz,
    pbm_or_peptide=example_peptide,
    models=models,
    device=DEVICE,
    return_individual=True
)

result

# %% [markdown]
# # Single Mutation for a specific PBM

# %% Cell 23
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def score_pbm6_panel_for_target(
    target_pdz,
    pbm6_list,
    bg_pdzs=None,
    models=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    alpha=1.0,
    device=DEVICE,
):
    """
    Score a list of PBM6 sequences against one target PDZ and background PDZs.

    Output columns:
    - pbm6
    - target_pKd_mean
    - target_pKd_std
    - background_pKd_mean
    - background_pKd_std
    - specificity_score
    - n_background_pdzs
    """

    # Load ensemble if not provided
    if models is None:
        models, model_files = load_design_ensemble_models(
            model_class=model_class,
            model_dir=model_dir,
            device=device
        )
    else:
        model_files = None

    # Prepare target tensor
    target_t = encode_single_pdz(target_pdz, device=device)

    # Background PDZs: use all non-target PDZs by default
    if bg_pdzs is None:
        bg_pdzs = [p for p in all_pdz_seqs if p != target_pdz]
    elif isinstance(bg_pdzs, str):
        bg_pdzs = [bg_pdzs]

    bg_t = encode_pdz_batch(bg_pdzs, device=device)

    # Clean PBM6 list
    clean_pbm6_list = []
    for s in pbm6_list:
        s = str(s).strip().upper()
        s = s[-6:]  # allow 10-aa input; use C-terminal 6 aa
        if len(s) != 6:
            raise ValueError(f"PBM must be length 6 after trimming. Got: {s}")
        clean_pbm6_list.append(s)

    clean_pbm6_list = sorted(set(clean_pbm6_list))

    records = []

    for seq in tqdm(clean_pbm6_list, desc="Scoring PBM6 panel"):
        rec = score_sequence_with_ensemble(
            seq=seq,
            target_t=target_t,
            bg_t=bg_t,
            models=models,
            alpha=alpha,
            device=device
        ).copy()

        rec["pbm6"] = seq
        rec["n_background_pdzs"] = len(bg_pdzs)
        records.append(rec)

    out_df = pd.DataFrame(records)

    # Put pbm6 first
    cols = ["pbm6"] + [c for c in out_df.columns if c != "pbm6"]
    out_df = out_df[cols]

    return out_df


def generate_single_aa_mutants_pbm6(wt_pbm6, terminal_allowed=None):
    """
    Generate all single amino-acid mutants from a WT PBM6 sequence.

    terminal_allowed:
    - None: allow all 20 aa at every position. Use this for natural variant scan.
    - list("LIVFC"): restrict last position. Use this for design refinement.
    """

    wt_pbm6 = str(wt_pbm6).strip().upper()[-6:]

    if len(wt_pbm6) != 6:
        raise ValueError(f"WT PBM6 must be length 6. Got: {wt_pbm6}")

    mutants = []

    for pos in range(6):
        wt_aa = wt_pbm6[pos]

        if terminal_allowed is not None and pos == 5:
            allowed = terminal_allowed
        else:
            allowed = AA_LIST

        for mut_aa in allowed:
            if mut_aa == wt_aa:
                continue

            mut_pbm6 = wt_pbm6[:pos] + mut_aa + wt_pbm6[pos+1:]

            mutants.append({
                "wt_pbm6": wt_pbm6,
                "mut_pbm6": mut_pbm6,
                "position_pbm6_1based": pos + 1,
                "position_from_C_terminal": 6 - pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "mutation_pbm6": f"{wt_aa}{pos+1}{mut_aa}",
            })

    return pd.DataFrame(mutants)


def scan_single_mutant_delta_pkd(
    target_pdz,
    wt_pbm6,
    gene_name=None,
    protein_name=None,
    protein_length=None,
    bg_pdzs=None,
    models=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    alpha=1.0,
    device=DEVICE,
    delta_threshold=0.5,
    terminal_allowed=None,
    save_path=None,
):
    """
    From one known WT PBM6:
    1. Generate all single-aa mutants.
    2. Score WT and mutants.
    3. Compute delta_pKd = mutant target_pKd_mean - WT target_pKd_mean.
    4. Return full scan table and high-impact table.

    For natural human variant scan, use terminal_allowed=None.
    """

    wt_pbm6 = str(wt_pbm6).strip().upper()[-6:]

    if len(wt_pbm6) != 6:
        raise ValueError(f"WT PBM6 must be length 6. Got: {wt_pbm6}")

    # Generate all mutants
    mut_df = generate_single_aa_mutants_pbm6(
        wt_pbm6=wt_pbm6,
        terminal_allowed=terminal_allowed
    )

    pbm6_list = [wt_pbm6] + mut_df["mut_pbm6"].tolist()

    # Score WT + all mutants
    score_df = score_pbm6_panel_for_target(
        target_pdz=target_pdz,
        pbm6_list=pbm6_list,
        bg_pdzs=bg_pdzs,
        models=models,
        model_class=model_class,
        model_dir=model_dir,
        alpha=alpha,
        device=device,
    )

    score_df["pbm6"] = score_df["pbm6"].astype(str).str.upper()

    # WT row
    wt_rows = score_df[score_df["pbm6"] == wt_pbm6]
    if wt_rows.shape[0] != 1:
        raise ValueError(f"Could not uniquely find WT PBM6 score for {wt_pbm6}.")

    wt_row = wt_rows.iloc[0]

    wt_target = float(wt_row["target_pKd_mean"])
    wt_target_std = float(wt_row["target_pKd_std"])
    wt_bg = float(wt_row["background_pKd_mean"])
    wt_bg_std = float(wt_row["background_pKd_std"])
    wt_spec = float(wt_row["specificity_score"])

    # Mutant rows
    result_df = score_df[score_df["pbm6"] != wt_pbm6].copy()
    result_df = result_df.rename(columns={"pbm6": "mut_pbm6"})

    result_df = result_df.merge(
        mut_df,
        on="mut_pbm6",
        how="left"
    )

    # Metadata
    result_df["gene_name"] = gene_name
    result_df["protein_name"] = protein_name

    # WT values
    result_df["wt_target_pKd_mean"] = wt_target
    result_df["wt_target_pKd_std"] = wt_target_std
    result_df["wt_background_pKd_mean"] = wt_bg
    result_df["wt_background_pKd_std"] = wt_bg_std
    result_df["wt_specificity_score"] = wt_spec

    # Delta values
    result_df["delta_pKd"] = result_df["target_pKd_mean"] - wt_target
    result_df["delta_background_pKd"] = result_df["background_pKd_mean"] - wt_bg
    result_df["delta_specificity_score"] = result_df["specificity_score"] - wt_spec
    result_df["abs_delta_pKd"] = result_df["delta_pKd"].abs()

    # Optional protein residue mapping
    if protein_length is not None:
        # PBM6 position 1 = protein_length - 5
        # PBM6 position 6 = protein_length
        result_df["protein_position"] = (
            int(protein_length) - 6 + result_df["position_pbm6_1based"].astype(int)
        )
        result_df["protein_mutation"] = (
            result_df["wt_aa"].astype(str)
            + result_df["protein_position"].astype(int).astype(str)
            + result_df["mut_aa"].astype(str)
        )
    else:
        result_df["protein_position"] = np.nan
        result_df["protein_mutation"] = result_df["mutation_pbm6"]

    # Effect labels
    result_df["predicted_effect"] = "small_or_neutral"
    result_df.loc[result_df["delta_pKd"] >= delta_threshold, "predicted_effect"] = "predicted_gain"
    result_df.loc[result_df["delta_pKd"] <= -delta_threshold, "predicted_effect"] = "predicted_loss"

    # Sort
    result_df = result_df.sort_values("abs_delta_pKd", ascending=False).reset_index(drop=True)

    high_impact_df = result_df[
        result_df["abs_delta_pKd"] >= delta_threshold
    ].copy().reset_index(drop=True)

    # Print summary
    print("===== Single-mutant PBM6 ΔpKd scan =====")
    print(f"Gene: {gene_name}")
    print(f"WT PBM6: {wt_pbm6}")
    print(f"WT target pKd mean: {wt_target:.3f}")
    print(f"WT background pKd mean: {wt_bg:.3f}")
    print(f"WT specificity score: {wt_spec:.3f}")
    print(f"Total mutants scored: {len(result_df)}")
    print(f"High-impact mutants |ΔpKd| >= {delta_threshold}: {len(high_impact_df)}")

    print("\nTop predicted gain-of-binding mutants:")
    display(
        result_df.sort_values("delta_pKd", ascending=False).head(10)[[
            "gene_name",
            "wt_pbm6",
            "mut_pbm6",
            "mutation_pbm6",
            "protein_mutation",
            "target_pKd_mean",
            "wt_target_pKd_mean",
            "delta_pKd",
            "background_pKd_mean",
            "specificity_score",
            "predicted_effect",
        ]]
    )

    print("\nTop predicted loss-of-binding mutants:")
    display(
        result_df.sort_values("delta_pKd", ascending=True).head(10)[[
            "gene_name",
            "wt_pbm6",
            "mut_pbm6",
            "mutation_pbm6",
            "protein_mutation",
            "target_pKd_mean",
            "wt_target_pKd_mean",
            "delta_pKd",
            "background_pKd_mean",
            "specificity_score",
            "predicted_effect",
        ]]
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        result_df.to_csv(save_path, index=False)

        high_path = save_path.with_name(save_path.stem + "_high_impact.csv")
        high_impact_df.to_csv(high_path, index=False)

        print("\nSaved full scan to:", save_path)
        print("Saved high-impact scan to:", high_path)

    return result_df, high_impact_df

# %% Cell 24
target_pdz = "KVVLHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDRIISVNSVDLRAASHEQAAAALKNAGQAVTIVAQYR"
NET1_WT_PBM6 = "RKETLV"
NET1_LENGTH = None

net1_scan_df, net1_high_impact_df = scan_single_mutant_delta_pkd(
    target_pdz=target_pdz,
    wt_pbm6=NET1_WT_PBM6,
    gene_name="NET1",
    protein_name="NET1",
    protein_length=NET1_LENGTH,
    bg_pdzs=None,
    models=None,
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    alpha=1.0,
    device=DEVICE,
    delta_threshold=1,
    terminal_allowed=None,
    save_path=RESULT_DIR / "NET1_single_mutant_delta_pKd_scan.csv"
)


# %% [markdown]
# # Unrestricted PBM6 motif preference enrichment

# %% Cell 26
# ============================================================
# Best version:
# All-PDZ × 100k unrestricted PBM6 motif preference analysis
# ============================================================

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pathlib import Path
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Assumptions:
# These variables/functions/classes should already exist in your notebook:
# DEVICE, OUT_DIR, RESULT_DIR
# AA_ORDER, AA_TO_INT, PAD_IDX
# InteractionAwareModel
# load_design_ensemble_models()
# encode_single_pdz()
# ------------------------------------------------------------

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
PBM_POS_LABELS = ["-5", "-4", "-3", "-2", "-1", "0"]

BEST_MOTIF_DIR = Path(RESULT_DIR) / "best_all_pdz_100k_unrestricted_motif_preference"
BEST_MOTIF_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = BEST_MOTIF_DIR / "per_pdz_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = BEST_MOTIF_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Dataset utilities
# ============================================================

def guess_pdz_column(df):
    candidate_cols = [
        "pdz_sequence", "PDZ_sequence",
        "pdz_seq", "PDZ_seq",
        "domain_sequence", "domain_seq",
        "bait_sequence", "bait_seq",
        "PDZ", "pdz"
    ]

    for col in candidate_cols:
        if col in df.columns:
            return col

    aa_set = set(AA_LIST)
    seq_like = []

    for col in df.columns:
        vals = df[col].dropna().astype(str).head(200)
        if len(vals) == 0:
            continue

        n_ok = sum(
            len(v.strip().upper()) >= 40 and set(v.strip().upper()).issubset(aa_set)
            for v in vals
        )

        if n_ok >= max(3, int(0.1 * len(vals))):
            seq_like.append(col)

    if len(seq_like) == 1:
        return seq_like[0]

    raise ValueError(
        f"Cannot identify PDZ sequence column. Candidates: {seq_like}. "
        f"Available columns: {df.columns.tolist()}"
    )


def build_pdz_label(row):
    """
    Build readable PDZ label for figures.
    Priority:
    pdz_name > pdz_gene + domain info > pdz_uniprot > pdz_id
    """

    for c in ["pdz_name", "PDZ_name", "pdz_domain_name", "domain_name"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c]).strip()

    gene = None
    for c in ["pdz_gene", "PDZ_gene", "pdz_gene_name", "gene_name"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            gene = str(row[c]).strip()
            break

    domain = None
    for c in ["pdz_domain", "PDZ_domain", "domain", "domain_index", "pdz_domain_index"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            domain = str(row[c]).strip()
            break

    if gene is not None and domain is not None:
        if str(domain).upper().startswith("PDZ"):
            return f"{gene}_{domain}"
        return f"{gene}_PDZ{domain}"

    if gene is not None:
        return gene

    for c in ["pdz_uniprot", "PDZ_uniprot", "uniprot", "uniprot_id"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c]).strip()

    return str(row["pdz_id"])


def get_unique_pdz_sequences_from_dataset(df, min_len=40):
    pdz_table = (
        df[["pdz_gene", "pdz_sequence", "pdz_uniprot", "pdz_site"]]
        .dropna(subset=["pdz_sequence"])
        .drop_duplicates(subset=["pdz_sequence"])
        .reset_index(drop=True)
        .copy()
    )

    pdz_table["pdz_sequence"] = pdz_table["pdz_sequence"].astype(str).str.upper().str.strip()

    pdz_table = pdz_table[
        pdz_table["pdz_sequence"].apply(
            lambda s: len(s) >= min_len and set(s).issubset(set(AA_LIST))
        )
    ].reset_index(drop=True)

    pdz_table["pdz_id"] = [
        f"PDZ_{i:03d}" for i in range(len(pdz_table))
    ]

    pdz_table["pdz_seq"] = pdz_table["pdz_sequence"]

    pdz_table["pdz_label"] = (
        pdz_table["pdz_gene"].astype(str)
        + "_"
        + pdz_table["pdz_site"].astype(str)
    )

    return pdz_table


# ============================================================
# 2. PBM6 library generation and encoding
# ============================================================

def generate_random_pbm6_library(
    n=100000,
    terminal_allowed=None,
    seed=20260608,
    aa_list=AA_LIST
):
    """
    Generate random PBM6 library.

    terminal_allowed=None:
        unrestricted PBM6 library. Use this as main motif-discovery analysis.

    terminal_allowed=list("LIVFC"):
        restricted design-compatible library. Use only for sensitivity check.
    """
    rng = np.random.default_rng(seed)
    pbm6_set = set()

    while len(pbm6_set) < n:
        seq = "".join(rng.choice(aa_list, size=6))

        if terminal_allowed is not None:
            seq = seq[:5] + str(rng.choice(terminal_allowed))

        pbm6_set.add(seq)

    pbm6_list = sorted(pbm6_set)
    print(f"Generated {len(pbm6_list):,} PBM6 sequences.")
    return pbm6_list


def encode_pbm6_library_to_cpu_tensor(pbm6_list):
    """
    Encode PBM6 library once as CPU tensor.
    It will be moved to GPU batch-by-batch during scoring.
    """
    encoded = []

    aa_set = set(AA_ORDER)

    for seq in pbm6_list:
        seq = str(seq).upper().strip()[-6:]
        if len(seq) != 6:
            raise ValueError(f"PBM6 must be length 6. Got: {seq}")

        invalid = set(seq) - aa_set
        if invalid:
            raise ValueError(f"Invalid amino acids in PBM6 {seq}: {invalid}")

        encoded.append([AA_TO_INT.get(aa, PAD_IDX) for aa in seq])

    return torch.tensor(encoded, dtype=torch.long)


# ============================================================
# 3. Model prediction utilities
# ============================================================

def safe_model_predict(model, pdz_t, pep_t):
    """
    Robustly handle model output.
    Some models return pred; some return pred, aux1, aux2.
    """
    out = model(pdz_t, pep_t)

    if isinstance(out, tuple) or isinstance(out, list):
        pred = out[0]
    else:
        pred = out

    return pred


def score_one_pdz_against_encoded_library(
    target_pdz,
    pbm6_list,
    pbm6_encoded_cpu,
    models,
    batch_size=8192,
    device=DEVICE,
    use_amp=True
):
    """
    Score one PDZ against pre-encoded PBM6 library.

    Returns:
        scored_df with columns:
        pbm6, target_pKd_mean, target_pKd_std
    """

    target_pdz = str(target_pdz).upper().strip()
    target_t_single = encode_single_pdz(target_pdz, device=device)

    n = len(pbm6_list)
    mean_preds = np.zeros(n, dtype=np.float32)
    std_preds = np.zeros(n, dtype=np.float32)

    amp_enabled = bool(use_amp and str(device).startswith("cuda"))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        pep_t = pbm6_encoded_cpu[start:end].to(device, non_blocking=True)
        pdz_t = target_t_single.repeat(end - start, 1)

        batch_model_preds = []

        with torch.no_grad():
            for model in models:
                model.eval()

                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        pred = safe_model_predict(model, pdz_t, pep_t)
                else:
                    pred = safe_model_predict(model, pdz_t, pep_t)

                pred = pred.detach().float().cpu().numpy().reshape(-1)
                batch_model_preds.append(pred)

        batch_model_preds = np.vstack(batch_model_preds)

        mean_preds[start:end] = batch_model_preds.mean(axis=0)

        if batch_model_preds.shape[0] > 1:
            std_preds[start:end] = batch_model_preds.std(axis=0, ddof=1)
        else:
            std_preds[start:end] = 0.0

        del pep_t, pdz_t, batch_model_preds

    scored_df = pd.DataFrame({
        "pbm6": pbm6_list,
        "target_pKd_mean": mean_preds,
        "target_pKd_std": std_preds
    })

    return scored_df


# ============================================================
# 4. Enrichment and summary utilities
# ============================================================

def compute_position_enrichment_from_scored_df(
    scored_df,
    score_col="target_pKd_mean",
    top_frac=0.01,
    top_n=None,
    pseudocount=1.0,
    aa_list=AA_LIST
):
    """
    Compute amino-acid enrichment among top predicted PBM6 motifs.

    log2 enrichment = log2(freq_top / freq_background)
    """
    df0 = scored_df.copy()
    df0["pbm6"] = df0["pbm6"].astype(str).str.upper().str[-6:]
    df0 = df0[df0["pbm6"].str.len() == 6].copy()
    df0 = df0.sort_values(score_col, ascending=False).reset_index(drop=True)

    if top_n is None:
        top_n = max(1, int(len(df0) * top_frac))

    top_df = df0.head(top_n).copy()

    enrich_mat = pd.DataFrame(index=aa_list, columns=PBM_POS_LABELS, dtype=float)
    top_freq_mat = pd.DataFrame(index=aa_list, columns=PBM_POS_LABELS, dtype=float)
    bg_freq_mat = pd.DataFrame(index=aa_list, columns=PBM_POS_LABELS, dtype=float)

    for pos in range(6):
        pos_label = PBM_POS_LABELS[pos]

        bg_counts = pd.Series([seq[pos] for seq in df0["pbm6"]]).value_counts()
        top_counts = pd.Series([seq[pos] for seq in top_df["pbm6"]]).value_counts()

        for aa in aa_list:
            bg_count = bg_counts.get(aa, 0)
            top_count = top_counts.get(aa, 0)

            bg_freq = (bg_count + pseudocount) / (len(df0) + pseudocount * len(aa_list))
            top_freq = (top_count + pseudocount) / (len(top_df) + pseudocount * len(aa_list))

            enrich_mat.loc[aa, pos_label] = np.log2(top_freq / bg_freq)
            top_freq_mat.loc[aa, pos_label] = top_freq
            bg_freq_mat.loc[aa, pos_label] = bg_freq

    return enrich_mat, top_freq_mat, bg_freq_mat, top_df


def summarize_pdz_prediction_distribution(scored_df, top_frac=0.01):
    """
    Compute PDZ-level selectivity/promiscuity summaries.
    """
    y = scored_df["target_pKd_mean"].to_numpy(dtype=float)

    top_n = max(1, int(len(y) * top_frac))
    y_sorted = np.sort(y)[::-1]
    top_y = y_sorted[:top_n]

    summary = {
        "library_n": int(len(y)),
        "pKd_mean": float(np.mean(y)),
        "pKd_std": float(np.std(y, ddof=1)),
        "pKd_median": float(np.median(y)),
        "pKd_q75": float(np.quantile(y, 0.75)),
        "pKd_q90": float(np.quantile(y, 0.90)),
        "pKd_q95": float(np.quantile(y, 0.95)),
        "pKd_q99": float(np.quantile(y, 0.99)),
        "pKd_max": float(np.max(y)),
        "top_frac": float(top_frac),
        "top_n": int(top_n),
        "top_mean_pKd": float(np.mean(top_y)),
        "top_min_pKd": float(np.min(top_y)),
        "selectivity_gap_q99_minus_median": float(np.quantile(y, 0.99) - np.median(y)),
        "selectivity_gap_max_minus_median": float(np.max(y) - np.median(y)),
        "promiscuity_fraction_pKd_ge_4": float(np.mean(y >= 4.0)),
        "promiscuity_fraction_pKd_ge_5": float(np.mean(y >= 5.0)),
        "promiscuity_fraction_pKd_ge_6": float(np.mean(y >= 6.0)),
    }

    return summary


def flatten_enrichment_matrix_long(enrich_mat, pdz_meta):
    """
    Convert 20 x 6 enrichment matrix into long format,
    preserving PDZ metadata.
    """

    meta = pdz_meta.to_dict() if hasattr(pdz_meta, "to_dict") else dict(pdz_meta)

    rows = []

    for aa in enrich_mat.index:
        for pos in enrich_mat.columns:
            rec = meta.copy()
            rec.update({
                "position": pos,
                "amino_acid": aa,
                "log2_enrichment": float(enrich_mat.loc[aa, pos])
            })
            rows.append(rec)

    return rows


def enrichment_matrix_to_vector(enrich_mat):
    """
    Flatten 20 × 6 enrichment matrix into one vector for clustering.
    Feature names: -5_A, -5_C, ..., 0_Y
    """
    vec = {}

    for pos in PBM_POS_LABELS:
        for aa in AA_LIST:
            vec[f"{pos}_{aa}"] = float(enrich_mat.loc[aa, pos])

    return vec


# ============================================================
# 5. Plotting utilities
# ============================================================

def plot_enrichment_heatmap(
    enrich_mat,
    title,
    save_prefix,
    out_dir=FIG_DIR,
    figsize=(6.0, 6.5),
    value_label="log2 enrichment"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    sns.heatmap(
        enrich_mat.astype(float),
        cmap="vlag",
        center=0,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": value_label},
        ax=ax
    )

    ax.set_xlabel("PBM position")
    ax.set_ylabel("Amino acid")
    ax.set_title(title)

    plt.tight_layout()

    for ext in ["png", "svg"]:
        path = out_dir / f"{save_prefix}.{ext}"
        plt.savefig(path, bbox_inches="tight")

    print(f"Saved heatmap: {save_prefix}")
    plt.show()

    return fig, ax


def plot_pdz_clustermap(
    cluster_matrix_df,
    save_prefix="all_pdz_enrichment_clustermap",
    out_dir=FIG_DIR,
    figsize=(18, 14)
):
    """
    Cluster PDZ domains based on flattened motif enrichment profiles.
    Rows: PDZ domains
    Columns: position-AA features
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cg = sns.clustermap(
        cluster_matrix_df,
        method="average",
        metric="correlation",
        cmap="vlag",
        center=0,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log2 enrichment"}
    )

    cg.fig.suptitle(
        "PDZ clustering based on PBM6 amino-acid enrichment profiles",
        y=1.02
    )

    for ext in ["png", "pdf", "svg"]:
        path = out_dir / f"{save_prefix}.{ext}"
        if ext == "png":
            cg.savefig(path, dpi=600, bbox_inches="tight")
        else:
            cg.savefig(path, bbox_inches="tight")

    print(f"Saved clustermap: {save_prefix}")
    plt.show()

    return cg


def plot_pdz_selectivity_summary(
    summary_df,
    save_prefix="all_pdz_selectivity_summary",
    out_dir=FIG_DIR
):
    """
    Simple summary plots for PDZ selectivity/promiscuity.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: q99 - median
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.hist(summary_df["selectivity_gap_q99_minus_median"], bins=30)
    ax.set_xlabel("q99 predicted pKd - median predicted pKd")
    ax.set_ylabel("Number of PDZ domains")
    ax.set_title("PDZ selectivity gap distribution")
    plt.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        path = out_dir / f"{save_prefix}_gap_q99_minus_median.{ext}"
        if ext == "png":
            plt.savefig(path, bbox_inches="tight", dpi=600)
        else:
            plt.savefig(path, bbox_inches="tight")
    plt.show()

    # Plot 2: promiscuity fraction pKd >= 5
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.hist(summary_df["promiscuity_fraction_pKd_ge_5"], bins=30)
    ax.set_xlabel("Fraction of random PBM6 with predicted pKd ≥ 5")
    ax.set_ylabel("Number of PDZ domains")
    ax.set_title("Predicted PDZ promiscuity distribution")
    plt.tight_layout()

    for ext in ["png", "svg"]:
        path = out_dir / f"{save_prefix}_promiscuity_ge5.{ext}"
        plt.savefig(path, bbox_inches="tight")
    plt.show()

# %% Cell 27
# ============================================================
# Run best version:
# all PDZ × 100k unrestricted PBM6
# ============================================================

# ------------------------------------------------------------
# Step 0. Set your processed pair-level dataframe here
# ------------------------------------------------------------
# Change this to your actual processed dataframe variable name.
# For example:
# processed_df = pair_df
# processed_df = df_agg
# processed_df = processed_pairs

# processed_df = YOUR_DATAFRAME_HERE

# If you are not sure, uncomment this to inspect available variables:
# [k for k, v in globals().items() if isinstance(v, pd.DataFrame)]


# ------------------------------------------------------------
# Step 1. Extract all unique PDZ sequences
# ------------------------------------------------------------
pdz_table = get_unique_pdz_sequences_from_dataset(df, min_len=40)

display(pdz_table.head())
print(pdz_table.shape)
print(pdz_table.columns.tolist())


# ------------------------------------------------------------
# Step 2. Generate 100k unrestricted PBM6 library
# ------------------------------------------------------------
N_PBM6 = 100000
TOP_FRAC = 0.01
BATCH_SIZE = 8192
SEED = 20260608

pbm6_library = generate_random_pbm6_library(
    n=N_PBM6,
    terminal_allowed=None,   # important: unrestricted
    seed=SEED
)

pd.DataFrame({"pbm6": pbm6_library}).to_csv(
    BEST_MOTIF_DIR / "unrestricted_random_pbm6_library_100k.csv",
    index=False
)

pbm6_encoded_cpu = encode_pbm6_library_to_cpu_tensor(pbm6_library)

print("PBM6 encoded CPU tensor:", pbm6_encoded_cpu.shape)


# ------------------------------------------------------------
# Step 3. Load design ensemble
# ------------------------------------------------------------
models, model_files = load_design_ensemble_models(
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    device=DEVICE
)

print(f"Loaded {len(models)} ensemble models.")
print(model_files)


# ------------------------------------------------------------
# Step 4. Main loop over all PDZ domains
# ------------------------------------------------------------
all_enrichment_long_rows = []
all_cluster_vectors = []
all_summary_rows = []
all_top_rows = []

start_time = time.time()

for idx, row in tqdm(pdz_table.iterrows(), total=len(pdz_table), desc="All-PDZ motif preference"):
    pdz_meta = row.to_dict()

    pdz_id = row["pdz_id"]
    pdz_seq = row["pdz_seq"]
    pdz_label = row["pdz_label"]

    checkpoint_prefix = CHECKPOINT_DIR / pdz_id
    enrich_path = checkpoint_prefix.with_suffix(".enrichment.csv")
    top_path = checkpoint_prefix.with_suffix(".top_pbm6.csv")
    summary_path = checkpoint_prefix.with_suffix(".summary.json")
    vector_path = checkpoint_prefix.with_suffix(".cluster_vector.csv")

    if enrich_path.exists() and top_path.exists() and summary_path.exists() and vector_path.exists():
      enrich_mat = pd.read_csv(enrich_path, index_col=0)
      top_df = pd.read_csv(top_path)

      with open(summary_path, "r") as f:
          summary = json.load(f)

      vector_df = pd.read_csv(vector_path)
      vector = vector_df.iloc[0].to_dict()

      # Force-add current PDZ metadata, even when loading old checkpoint
      summary.update(pdz_meta)
      vector.update(pdz_meta)

      print(f"[Resume] Loaded checkpoint for {pdz_label}")

    else:
        scored_df = score_one_pdz_against_encoded_library(
            target_pdz=pdz_seq,
            pbm6_list=pbm6_library,
            pbm6_encoded_cpu=pbm6_encoded_cpu,
            models=models,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            use_amp=True
        )

        enrich_mat, top_freq_mat, bg_freq_mat, top_df = compute_position_enrichment_from_scored_df(
            scored_df=scored_df,
            score_col="target_pKd_mean",
            top_frac=TOP_FRAC,
            top_n=None,
            pseudocount=1.0,
            aa_list=AA_LIST
        )

        summary = summarize_pdz_prediction_distribution(
            scored_df=scored_df,
            top_frac=TOP_FRAC
        )

        summary.update(pdz_meta)

        vector = enrichment_matrix_to_vector(enrich_mat)
        vector.update(pdz_meta)

        enrich_mat.to_csv(enrich_path)

        for k, v in reversed(list(pdz_meta.items())):
            top_df.insert(0, k, v)

        top_df.to_csv(top_path, index=False)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame([vector]).to_csv(vector_path, index=False)

        del scored_df
        gc.collect()

        if str(DEVICE).startswith("cuda"):
            torch.cuda.empty_cache()

    all_enrichment_long_rows.extend(
        flatten_enrichment_matrix_long(
            enrich_mat=enrich_mat,
            pdz_meta=row
        )
    )

    all_summary_rows.append(summary)
    all_cluster_vectors.append(vector)
    all_top_rows.append(pd.read_csv(top_path))


elapsed = time.time() - start_time
print(f"Finished all-PDZ motif preference analysis in {elapsed / 3600:.2f} hours.")


# ------------------------------------------------------------
# Step 5. Combine and save final outputs
# ------------------------------------------------------------
all_enrichment_long_df = pd.DataFrame(all_enrichment_long_rows)
all_summary_df = pd.DataFrame(all_summary_rows)
all_cluster_matrix_df = pd.DataFrame(all_cluster_vectors)
all_top_pbm6_df = pd.concat(all_top_rows, ignore_index=True)

# Set PDZ ID as row index for clustering matrix
cluster_feature_cols = [
    c for c in all_cluster_matrix_df.columns
    if any(c.startswith(f"{pos}_") for pos in PBM_POS_LABELS)
]

cluster_matrix_df = (
    all_cluster_matrix_df
    .set_index("pdz_label")[cluster_feature_cols]
    .astype(float)
)

# Save final files
all_enrichment_long_df.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_enrichment_long.csv",
    index=False
)

all_summary_df.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_selectivity_summary.csv",
    index=False
)

all_cluster_matrix_df.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_enrichment_matrix_with_metadata.csv",
    index=False
)

cluster_matrix_df.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_enrichment_matrix_for_clustering.csv"
)

all_top_pbm6_df.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_top1pct_pbm6.csv",
    index=False
)

print("Saved final output tables.")
print("Enrichment long:", all_enrichment_long_df.shape)
print("Summary:", all_summary_df.shape)
print("Cluster matrix:", cluster_matrix_df.shape)
print("Top PBM6:", all_top_pbm6_df.shape)

# %% Cell 28
# ============================================================
# Generate figures
# ============================================================

# ------------------------------------------------------------
# Figure A. Average motif enrichment across all PDZ domains
# ------------------------------------------------------------
avg_enrich_long = (
    all_enrichment_long_df
    .groupby(["amino_acid", "position"])["log2_enrichment"]
    .mean()
    .reset_index()
)

avg_enrich_mat = (
    avg_enrich_long
    .pivot(index="amino_acid", columns="position", values="log2_enrichment")
    .loc[AA_LIST, PBM_POS_LABELS]
)

avg_enrich_mat.to_csv(
    BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_average_log2_enrichment_matrix.csv"
)

plot_enrichment_heatmap(
    avg_enrich_mat,
    title="Average PBM6 enrichment among top 1% predicted binders across PDZ domains",
    save_prefix="Figure_all_pdz_average_enrichment_top1pct_100k_unrestricted",
    out_dir=FIG_DIR,
    value_label="mean log2 enrichment"
)


# ------------------------------------------------------------
# Figure B. PDZ clustering based on enrichment profiles
# ------------------------------------------------------------
cg = plot_pdz_clustermap(
    cluster_matrix_df,
    save_prefix="Figure_all_pdz_enrichment_clustermap_top1pct_100k_unrestricted",
    out_dir=FIG_DIR,
    figsize=(18, 14)
)


# ------------------------------------------------------------
# Figure C. Selectivity / promiscuity summary
# ------------------------------------------------------------
plot_pdz_selectivity_summary(
    all_summary_df,
    save_prefix="Figure_all_pdz_100k_unrestricted_selectivity_summary",
    out_dir=FIG_DIR
)

# %% Cell 29
# ============================================================
# Representative PDZ motif preference heatmaps
# ============================================================

summary_df = all_summary_df.copy()

representative_rows = []

# Most selective by q99-median
representative_rows.append(
    summary_df.sort_values("selectivity_gap_q99_minus_median", ascending=False).iloc[0]
)

# Least selective by q99-median
representative_rows.append(
    summary_df.sort_values("selectivity_gap_q99_minus_median", ascending=True).iloc[0]
)

# Most promiscuous by fraction pKd >= 5
representative_rows.append(
    summary_df.sort_values("promiscuity_fraction_pKd_ge_5", ascending=False).iloc[0]
)

# Highest max predicted pKd
representative_rows.append(
    summary_df.sort_values("pKd_max", ascending=False).iloc[0]
)

# Highest top mean pKd
representative_rows.append(
    summary_df.sort_values("top_mean_pKd", ascending=False).iloc[0]
)

# Median selectivity example
median_gap = summary_df["selectivity_gap_q99_minus_median"].median()
summary_df["median_gap_distance"] = np.abs(summary_df["selectivity_gap_q99_minus_median"] - median_gap)
representative_rows.append(
    summary_df.sort_values("median_gap_distance", ascending=True).iloc[0]
)

rep_df = pd.DataFrame(representative_rows).drop_duplicates(subset=["pdz_id"])
rep_df.to_csv(BEST_MOTIF_DIR / "representative_pdz_for_motif_heatmaps.csv", index=False)

display_cols = [
    "pdz_id", "pdz_label",
    "pKd_median", "pKd_q99", "pKd_max",
    "top_mean_pKd",
    "selectivity_gap_q99_minus_median",
    "promiscuity_fraction_pKd_ge_5"
]
display_cols = [c for c in display_cols if c in rep_df.columns]
display(rep_df[display_cols])

for _, row in rep_df.iterrows():
    pdz_id = row["pdz_id"]
    pdz_label = row.get("pdz_label", pdz_id)

    enrich_path = CHECKPOINT_DIR / f"{pdz_id}.enrichment.csv"
    enrich_mat = pd.read_csv(enrich_path, index_col=0).loc[AA_LIST, PBM_POS_LABELS]

    title = (
        f"{pdz_label} motif enrichment\n"
        f"q99-median={row['selectivity_gap_q99_minus_median']:.2f}, "
        f"promiscuity>=5={row['promiscuity_fraction_pKd_ge_5']:.3f}"
    )

    safe_label = str(pdz_label).replace("/", "_").replace(" ", "_")

    plot_enrichment_heatmap(
        enrich_mat,
        title=title,
        save_prefix=f"Figure_representative_{safe_label}_motif_enrichment",
        out_dir=FIG_DIR,
        value_label="log2 enrichment"
    )

# %% Cell 30
# ============================================================
# PBM position importance from all-PDZ motif enrichment
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if "all_enrichment_long_df" not in globals():
    all_enrichment_long_df = pd.read_csv(
        BEST_MOTIF_DIR / "all_pdz_100k_unrestricted_enrichment_long.csv"
    )

PBM_POS_LABELS = ["-5", "-4", "-3", "-2", "-1", "0"]

df_imp = all_enrichment_long_df.copy()
df_imp["abs_log2_enrichment"] = df_imp["log2_enrichment"].abs()

pdz_position_importance = (
    df_imp
    .groupby(["pdz_id", "position"])["abs_log2_enrichment"]
    .mean()
    .reset_index()
    .rename(columns={"abs_log2_enrichment": "position_importance"})
)

pdz_position_importance["position"] = pd.Categorical(
    pdz_position_importance["position"],
    categories=PBM_POS_LABELS,
    ordered=True
)

# summary: across 260 PDZ
position_importance_summary = (
    pdz_position_importance
    .groupby("position")["position_importance"]
    .agg(["mean", "std", "median", "count"])
    .reset_index()
)

position_importance_summary["sem"] = (
    position_importance_summary["std"] / np.sqrt(position_importance_summary["count"])
)

position_importance_summary["ci95"] = 1.96 * position_importance_summary["sem"]

display(position_importance_summary)

position_importance_summary.to_csv(
    BEST_MOTIF_DIR / "pbm_position_importance_summary.csv",
    index=False
)

pdz_position_importance.to_csv(
    BEST_MOTIF_DIR / "pbm_position_importance_per_pdz.csv",
    index=False
)

# %% Cell 31
# ============================================================
# Plot PBM position importance
# ============================================================

fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

x = np.arange(len(PBM_POS_LABELS))

summary_plot = (
    position_importance_summary
    .set_index("position")
    .loc[PBM_POS_LABELS]
    .reset_index()
)

ax.bar(
    x,
    summary_plot["mean"],
    yerr=summary_plot["ci95"],
    capsize=4
)

ax.set_xticks(x)
ax.set_xticklabels(PBM_POS_LABELS)

ax.set_xlabel("PBM position")
ax.set_ylabel("Mean absolute log2 enrichment")
ax.set_title("PBM position importance inferred from top predicted binders")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = FIG_DIR / f"Figure_PBM_position_importance_from_enrichment.{ext}"
    if ext == "png":
        plt.savefig(out_path, bbox_inches="tight", dpi=600)
    else:
        plt.savefig(out_path, bbox_inches="tight")

plt.show()

# %% Cell 32
# ============================================================
# Compare C-terminal core positions vs upstream positions
# ============================================================

from scipy.stats import wilcoxon

importance_wide = (
    pdz_position_importance
    .pivot(index="pdz_id", columns="position", values="position_importance")
    .loc[:, PBM_POS_LABELS]
)

importance_wide["upstream_mean_P-5_to_P-4"] = importance_wide[["-5", "-4"]].mean(axis=1)
importance_wide["cterminal_core_mean_P-3_to_P0"] = importance_wide[["-3", "-2", "-1", "0"]].mean(axis=1)
importance_wide["cterminal_minus_upstream"] = (
    importance_wide["cterminal_core_mean_P-3_to_P0"]
    - importance_wide["upstream_mean_P-5_to_P-4"]
)

stat, pval = wilcoxon(
    importance_wide["cterminal_core_mean_P-3_to_P0"],
    importance_wide["upstream_mean_P-5_to_P-4"],
    alternative="greater"
)

print("C-terminal core mean:", importance_wide["cterminal_core_mean_P-3_to_P0"].mean())
print("Upstream mean:", importance_wide["upstream_mean_P-5_to_P-4"].mean())
print("Mean difference:", importance_wide["cterminal_minus_upstream"].mean())
print("Wilcoxon signed-rank test statistic:", stat)
print("One-sided p-value:", pval)

importance_wide.to_csv(
    BEST_MOTIF_DIR / "pbm_position_importance_cterminal_vs_upstream.csv"
)

# %% Cell 33
# ============================================================
# Paired comparison: upstream vs C-terminal core
# ============================================================

fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=300)

plot_df = importance_wide.copy()

x1 = np.zeros(len(plot_df))
x2 = np.ones(len(plot_df))

for _, row in plot_df.iterrows():
    ax.plot(
        [0, 1],
        [
            row["upstream_mean_P-5_to_P-4"],
            row["cterminal_core_mean_P-3_to_P0"]
        ],
        linewidth=0.6,
        alpha=0.25
    )

means = [
    plot_df["upstream_mean_P-5_to_P-4"].mean(),
    plot_df["cterminal_core_mean_P-3_to_P0"].mean()
]

cis = [
    1.96 * plot_df["upstream_mean_P-5_to_P-4"].std(ddof=1) / np.sqrt(len(plot_df)),
    1.96 * plot_df["cterminal_core_mean_P-3_to_P0"].std(ddof=1) / np.sqrt(len(plot_df))
]

ax.errorbar(
    [0, 1],
    means,
    yerr=cis,
    fmt="o-",
    linewidth=2.2,
    capsize=5
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["P-5 to P-4", "P-3 to P0"])
ax.set_ylabel("Mean absolute log2 enrichment")
ax.set_title("C-terminal PBM core shows stronger predicted sequence preference")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = FIG_DIR / f"Figure_Cterminal_core_vs_upstream_position_importance.{ext}"
    if ext == "png":
        plt.savefig(out_path, bbox_inches="tight", dpi=600)
    else:
        plt.savefig(out_path, bbox_inches="tight")

plt.show()

# %% [markdown]
# # GnomAD missense variants analysis

# %% Cell 35
pbm6_variant_table = pd.read_csv(
    os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "pbm6_gnomad_variants_clean.csv")
)

# %% Cell 36
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

pdz_panel_df = get_unique_pdz_sequences_from_dataset(df)

display(pdz_panel_df.head())
print(pdz_panel_df.shape)
print(pdz_panel_df.columns.tolist())

# %% Cell 37
models, model_files = load_design_ensemble_models(
    model_class=InteractionAwareModel,
    model_dir=OUT_DIR,
    device=DEVICE
)

print("Loaded models:", len(models))
print(model_files)

# %% Cell 38
def predict_one_pbm6_against_pdz_panel(
    pbm6,
    pdz_panel_df,
    models,
    device=DEVICE,
    batch_size=512
):
    """
    Predict ensemble pKd for one PBM6 against all PDZ domains.

    Returns:
        DataFrame with pdz_id, pdz_sequence, pKd_mean, pKd_std
    """
    pbm6 = str(pbm6).upper()

    if "pdz_id" not in pdz_panel_df.columns:
        raise ValueError("pdz_panel_df must contain pdz_id column.")
    if "pdz_sequence" not in pdz_panel_df.columns:
        raise ValueError("pdz_panel_df must contain pdz_sequence column.")

    pdz_ids = pdz_panel_df["pdz_id"].tolist()
    pdz_seqs = pdz_panel_df["pdz_sequence"].tolist()

    all_model_preds = []

    for model in models:
        model.eval()
        preds_this_model = []

        with torch.no_grad():
            for start in range(0, len(pdz_seqs), batch_size):
                end = min(start + batch_size, len(pdz_seqs))
                batch_pdz_seqs = pdz_seqs[start:end]

                pdz_t = encode_pdz_batch(
                    batch_pdz_seqs,
                    device=device
                )

                pbm_t_single = encode_pbm6(
                    pbm6,
                    device=device
                )

                # repeat PBM tensor to match PDZ batch size
                pbm_t = pbm_t_single.repeat(len(batch_pdz_seqs), 1)

                pred = safe_model_predict(model, pdz_t, pbm_t)
                pred = pred.detach().cpu().numpy().reshape(-1)
                preds_this_model.extend(pred.tolist())

        all_model_preds.append(preds_this_model)

    pred_arr = np.array(all_model_preds, dtype=float)  # n_models × n_pdz

    out = pdz_panel_df.copy()
    out["pbm6"] = pbm6
    out["pKd_mean"] = pred_arr.mean(axis=0)
    out["pKd_std"] = pred_arr.std(axis=0, ddof=1) if pred_arr.shape[0] > 1 else 0.0

    return out

# %% Cell 39
pbm6_variant_table["WT_PBM6"] = pbm6_variant_table["WT_PBM6"].astype(str).str.upper()
pbm6_variant_table["MUT_PBM6"] = pbm6_variant_table["MUT_PBM6"].astype(str).str.upper()

all_pbm6_to_score = sorted(
    set(pbm6_variant_table["WT_PBM6"].dropna()) |
    set(pbm6_variant_table["MUT_PBM6"].dropna())
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
all_pbm6_to_score = [
    x for x in all_pbm6_to_score
    if isinstance(x, str) and len(x) == 6 and set(x).issubset(VALID_AA)
]

print("Unique PBM6 to score:", len(all_pbm6_to_score))
print(all_pbm6_to_score[:10])

# %% Cell 40
PRED_CACHE = os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "all_wt_mut_pbm6_vs_pdz_predictions.csv")

if os.path.exists(PRED_CACHE):
    print("Loading cached predictions:", PRED_CACHE)
    pbm6_pdz_pred = pd.read_csv(PRED_CACHE)
else:
    pred_list = []

    for pbm6 in tqdm(all_pbm6_to_score, desc="Scoring PBM6 against PDZ panel"):
        pred_df = predict_one_pbm6_against_pdz_panel(
            pbm6=pbm6,
            pdz_panel_df=pdz_panel_df,
            models=models,
            device=DEVICE,
            batch_size=512
        )
        pred_list.append(pred_df)

    pbm6_pdz_pred = pd.concat(pred_list, ignore_index=True)
    pbm6_pdz_pred.to_csv(PRED_CACHE, index=False)

print(pbm6_pdz_pred.shape)
display(pbm6_pdz_pred.head())

# %% Cell 41
# PDZ metadata columns available in pbm6_pdz_pred
pdz_meta_cols = [
    "pdz_id",
    "pdz_sequence",
    "pdz_gene",
    "pdz_uniprot",
    "pdz_site",
    "pdz_label",
]

pdz_meta_cols = [c for c in pdz_meta_cols if c in pbm6_pdz_pred.columns]

# WT predictions
wt_pred = pbm6_pdz_pred.rename(columns={
    "pbm6": "WT_PBM6",
    "pKd_mean": "WT_pKd_mean",
    "pKd_std": "WT_pKd_std"
})[
    ["WT_PBM6"] + pdz_meta_cols + ["WT_pKd_mean", "WT_pKd_std"]
]

# MUT predictions
mut_pred = pbm6_pdz_pred.rename(columns={
    "pbm6": "MUT_PBM6",
    "pKd_mean": "MUT_pKd_mean",
    "pKd_std": "MUT_pKd_std"
})[
    ["MUT_PBM6", "pdz_id", "MUT_pKd_mean", "MUT_pKd_std"]
]

variant_base_cols = [
    "query_gene", "pbm_uniprot", "variant_id",
    "pbm_sequence_10aa", "WT_PBM6", "MUT_PBM6",
    "PBM_position", "PBM6_index",
    "protein_start", "protein_length",
    "ref_aa", "alt_aa",
    "aa_change_1letter",
    "hgvsp_like",
    "joint_af", "exome_af", "genome_af"
]

variant_base_cols = [c for c in variant_base_cols if c in pbm6_variant_table.columns]

variant_pdz_delta = (
    pbm6_variant_table[variant_base_cols]
    .drop_duplicates()
    .merge(wt_pred, on="WT_PBM6", how="left")
    .merge(mut_pred, on=["MUT_PBM6", "pdz_id"], how="left")
)


variant_pdz_delta["delta_pKd"] = (
    variant_pdz_delta["MUT_pKd_mean"] - variant_pdz_delta["WT_pKd_mean"]
)

variant_pdz_delta["abs_delta_pKd"] = variant_pdz_delta["delta_pKd"].abs()

variant_pdz_delta["effect_direction"] = "small_or_neutral"
variant_pdz_delta.loc[
    variant_pdz_delta["delta_pKd"] >= 0.5,
    "effect_direction"
] = "predicted_gain"
variant_pdz_delta.loc[
    variant_pdz_delta["delta_pKd"] <= -0.5,
    "effect_direction"
] = "predicted_loss"

out_long = os.path.join(
    "/content/drive/MyDrive/gnomad_pbm_outputs/",
    "variant_pdz_delta_long.csv"
)

variant_pdz_delta.to_csv(out_long, index=False)

print("Saved:", out_long)
print(variant_pdz_delta.shape)
display(variant_pdz_delta.head())

# %% Cell 42
def summarize_one_variant(g):
    idx_abs = g["abs_delta_pKd"].idxmax()
    idx_gain = g["delta_pKd"].idxmax()
    idx_loss = g["delta_pKd"].idxmin()

    top_abs = g.loc[idx_abs]
    top_gain = g.loc[idx_gain]
    top_loss = g.loc[idx_loss]

    return pd.Series({
        "n_PDZ_scored": g["pdz_id"].nunique(),

        "max_abs_delta_pKd": top_abs["abs_delta_pKd"],
        "top_abs_delta_pKd": top_abs["delta_pKd"],
        "top_abs_PDZ": top_abs["pdz_id"],
        "top_abs_WT_pKd": top_abs["WT_pKd_mean"],
        "top_abs_MUT_pKd": top_abs["MUT_pKd_mean"],

        "max_gain_delta_pKd": top_gain["delta_pKd"],
        "top_gain_PDZ": top_gain["pdz_id"],
        "top_gain_WT_pKd": top_gain["WT_pKd_mean"],
        "top_gain_MUT_pKd": top_gain["MUT_pKd_mean"],

        "max_loss_delta_pKd": top_loss["delta_pKd"],
        "top_loss_PDZ": top_loss["pdz_id"],
        "top_loss_WT_pKd": top_loss["WT_pKd_mean"],
        "top_loss_MUT_pKd": top_loss["MUT_pKd_mean"],

        "mean_abs_delta_pKd": g["abs_delta_pKd"].mean(),
        "median_abs_delta_pKd": g["abs_delta_pKd"].median(),
        "n_PDZ_abs_delta_ge_0.25": int((g["abs_delta_pKd"] >= 0.25).sum()),
        "n_PDZ_abs_delta_ge_0.5": int((g["abs_delta_pKd"] >= 0.5).sum()),
        "n_PDZ_abs_delta_ge_1.0": int((g["abs_delta_pKd"] >= 1.0).sum()),
        "n_PDZ_gain_ge_0.5": int((g["delta_pKd"] >= 0.5).sum()),
        "n_PDZ_loss_le_minus_0.5": int((g["delta_pKd"] <= -0.5).sum()),
    })

variant_id_cols = [
    "query_gene", "pbm_uniprot", "variant_id",
    "pbm_sequence_10aa", "WT_PBM6", "MUT_PBM6",
    "PBM_position", "PBM6_index",
    "protein_start", "protein_length",
    "ref_aa", "alt_aa",
    "aa_change_1letter", "hgvsp_like",
    "joint_af", "exome_af", "genome_af"
]

variant_id_cols = [c for c in variant_id_cols if c in variant_pdz_delta.columns]

variant_effect_summary = (
    variant_pdz_delta
    .groupby(variant_id_cols, dropna=False)
    .apply(summarize_one_variant)
    .reset_index()
)

# impact class
variant_effect_summary["predicted_variant_effect"] = "small_or_neutral"
variant_effect_summary.loc[
    variant_effect_summary["max_abs_delta_pKd"] >= 0.5,
    "predicted_variant_effect"
] = "moderate_perturbation"
variant_effect_summary.loc[
    variant_effect_summary["max_abs_delta_pKd"] >= 1.0,
    "predicted_variant_effect"
] = "strong_perturbation"

# AF class
def classify_af(af):
    if pd.isna(af):
        return "unknown"
    if af < 1e-4:
        return "ultra_rare_AF<1e-4"
    if af < 1e-3:
        return "rare_1e-4_to_1e-3"
    if af < 1e-2:
        return "low_frequency_1e-3_to_1e-2"
    return "common_AF>=1e-2"

variant_effect_summary["joint_af_class"] = variant_effect_summary["joint_af"].apply(classify_af)

out_summary = os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "variant_effect_summary.csv")
variant_effect_summary.to_csv(out_summary, index=False)

print("Saved:", out_summary)
print(variant_effect_summary.shape)

display(
    variant_effect_summary
    .sort_values("max_abs_delta_pKd", ascending=False)
    .head(20)
)

# %% Cell 43
top_variants = (
    variant_effect_summary
    .sort_values("max_abs_delta_pKd", ascending=False)
    .head(30)
)

display(top_variants[[
    "query_gene", "variant_id",
    "WT_PBM6", "MUT_PBM6", "PBM_position", "hgvsp_like",
    "joint_af", "joint_af_class",
    "max_abs_delta_pKd", "top_abs_delta_pKd", "top_abs_PDZ",
    "n_PDZ_abs_delta_ge_0.5"
]])

# %% Cell 44
position_effect = (
    variant_effect_summary
    .groupby("PBM_position")
    .agg(
        n_variants=("variant_id", "nunique"),
        mean_max_abs_delta=("max_abs_delta_pKd", "mean"),
        median_max_abs_delta=("max_abs_delta_pKd", "median"),
        n_strong=("predicted_variant_effect", lambda x: (x == "strong_perturbation").sum())
    )
    .reindex(["P-5", "P-4", "P-3", "P-2", "P-1", "P0"])
    .reset_index()
)

display(position_effect)

position_effect.to_csv(
    os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "position_level_variant_effect_summary.csv"),
    index=False
)

# %% Cell 45
af_effect = (
    variant_effect_summary
    .groupby("joint_af_class")
    .agg(
        n_variants=("variant_id", "nunique"),
        mean_max_abs_delta=("max_abs_delta_pKd", "mean"),
        median_max_abs_delta=("max_abs_delta_pKd", "median"),
        n_strong=("predicted_variant_effect", lambda x: (x == "strong_perturbation").sum())
    )
    .reset_index()
)

display(af_effect)

af_effect.to_csv(
    os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "af_level_variant_effect_summary.csv"),
    index=False
)

# %% Cell 46
top_variant_ids = (
    variant_effect_summary
    .sort_values("max_abs_delta_pKd", ascending=False)
    .head(20)["variant_id"]
    .tolist()
)

heatmap_df = variant_pdz_delta[
    variant_pdz_delta["variant_id"].isin(top_variant_ids)
].copy()

heatmap_top_list = []

for vid, g in heatmap_df.groupby("variant_id"):
    gg = g.sort_values("abs_delta_pKd", ascending=False).head(20)
    heatmap_top_list.append(gg)

heatmap_top_df = pd.concat(heatmap_top_list, ignore_index=True)

heatmap_top_df.to_csv(
    os.path.join("/content/drive/MyDrive/gnomad_pbm_outputs/", "top_variant_pdz_delta_heatmap_data.csv"),
    index=False
)

display(heatmap_top_df.head())
print(heatmap_top_df.shape)
