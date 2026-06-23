# Auto-exported from 02_p3bind_training_and_design.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% Cell 0
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

# %% Cell 1
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

# %% Cell 2
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

# %% Cell 3
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

# %% Cell 4
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

# %% Cell 5
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

# %% Cell 7
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

# %% Cell 8
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

# %% Cell 9
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

# %% Cell 10
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

# %% Cell 11
def get_split_column(split_df):
    for col in ["split", "role", "set"]:
        if col in split_df.columns:
            return col
    raise ValueError("Cannot find split column. Expected one of: split, role, set")


def train_design_ensemble_from_split(
    df,
    split_df,
    fold_col="fold",
    pair_id_col=PAIR_ID_COL,
    num_epochs=80,
    batch_size=64,
    lr=1e-3,
    patience=12,
    device=DEVICE,
    save_dir=OUT_DIR
):
    split_col = get_split_column(split_df)

    summaries = []

    for fold_id in sorted(split_df[fold_col].unique()):
        print("\n" + "=" * 80)
        print(f"Training design model fold {fold_id}")
        print("=" * 80)

        fold_split = split_df[split_df[fold_col] == fold_id].copy()

        train_ids = fold_split[fold_split[split_col].isin(["train", "training"])][pair_id_col].values
        val_ids = fold_split[fold_split[split_col].isin(["val", "valid", "validation"])][pair_id_col].values

        if len(val_ids) == 0:
            raise ValueError(
                f"No validation rows found for fold {fold_id}. "
                f"Check split column values: {fold_split[split_col].unique()}"
            )

        train_df = df[df[pair_id_col].isin(train_ids)].copy()
        val_df = df[df[pair_id_col].isin(val_ids)].copy()

        print("Train:", train_df.shape, "Val:", val_df.shape)

        summary, history_df = train_one_fold(
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            model_class=InteractionAwareModel,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            save_dir=save_dir
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = Path(save_dir) / "design_ensemble_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Design ensemble validation summary =====")
    display(summary_df)
    print("Saved:", summary_path)

    return summary_df


design_summary_df = train_design_ensemble_from_split(
    df=df,
    split_df=split_df,
    fold_col="fold",
    pair_id_col=PAIR_ID_COL,
    num_epochs=80,
    batch_size=64,
    lr=1e-3,
    patience=12,
    device=DEVICE,
    save_dir=OUT_DIR
)

# %% Cell 12
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

# %% Cell 13
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

# %% Cell 14
# Pick one target PDZ sequence from the dataset
target_pdz = "KVVLHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDRIISVNSVDLRAASHEQAAAALKNAGQAVTIVAQYR"
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

# %% Cell 15
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

# %% Cell 16
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

# %% Cell 17
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

# %% Cell 18
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

# %% Cell 19
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
