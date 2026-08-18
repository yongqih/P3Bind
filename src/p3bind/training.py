"""Local training utilities for the design-oriented P3Bind ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import InteractionAwareModel, encode_pbm6, encode_single_pdz
from .validation import load_pair_dataset


class BiasedDirectionLoss(nn.Module):
    """Directional loss used by the manuscript's design ensemble.

    Uncensored observations receive larger penalties for high-affinity
    underprediction and low-affinity overprediction. Censored observations
    use a hinge-squared penalty only above their detection-limit label.
    """

    def __init__(
        self,
        threshold: float = 4.5,
        pos_weight: float = 3.0,
        penalty_scale: float = 3.0,
        relax_scale: float = 0.3,
        LOD: float = 3.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.threshold = threshold
        self.pos_weight = pos_weight
        self.penalty_scale = penalty_scale
        self.relax_scale = relax_scale
        self.LOD = LOD
        self.reduction = reduction

    def forward(self, pred, target, censored):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        censored = censored.reshape(-1).bool()
        mse = (pred - target) ** 2

        high_true = (~censored) & (target >= self.threshold)
        low_true = (~censored) & (target < self.threshold)
        bad_direction = (high_true & (pred < target)) | (low_true & (pred > target))
        good_direction = (high_true & (pred >= target)) | (low_true & (pred <= target))

        weights = torch.full_like(pred, self.pos_weight)
        weights = torch.where(bad_direction, weights * self.penalty_scale, weights)
        weights = torch.where(good_direction, weights * self.relax_scale, weights)
        uncensored_loss = mse * weights
        censored_loss = F.relu(pred - target) ** 2
        loss = torch.where(censored, censored_loss, uncensored_loss)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class DesignPairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.pdz = torch.cat(
            [encode_single_pdz(seq, device=torch.device("cpu")) for seq in self.frame["pdz_sequence"]],
            dim=0,
        )
        self.pbm = torch.cat(
            [encode_pbm6(seq, device=torch.device("cpu")) for seq in self.frame["pbm6"]],
            dim=0,
        )
        self.target = torch.tensor(self.frame["pKd"].to_numpy(dtype=np.float32))
        censor_column = "is_censored_label" if "is_censored_label" in self.frame else "is_censored"
        self.censored = torch.tensor(self.frame[censor_column].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return self.pdz[index], self.pbm[index], self.target[index], self.censored[index]


@dataclass
class TrainResult:
    fold: int
    best_epoch: int
    best_val_loss: float
    best_val_rmse: float
    checkpoint: Path


@torch.no_grad()
def evaluate_design_model(model, loader, device, criterion):
    model.eval()
    truth = []
    predictions = []
    losses = []
    for pdz, pbm, target, censored in loader:
        pred, _, _ = model(pdz.to(device), pbm.to(device))
        losses.append(criterion(pred, target.to(device), censored.to(device)).item())
        truth.append(target.numpy())
        predictions.append(pred.detach().cpu().reshape(-1).numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predictions)
    return float(np.mean(losses)), float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_design_fold(
    dataset_path: str | Path,
    split_path: str | Path,
    fold: int,
    output_dir: str | Path,
    *,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> TrainResult:
    """Train one random-split design-ensemble member without notebook state."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    data = load_pair_dataset(dataset_path)
    split = pd.read_csv(split_path)
    fold_split = split.loc[split["fold"] == fold, ["pair_id", "split"]]
    data = data.merge(fold_split, on="pair_id", how="inner", validate="one_to_one")
    train_data = data.loc[data["split"] == "train"]
    val_data = data.loc[data["split"] == "val"]
    if train_data.empty or val_data.empty:
        raise ValueError(f"Fold {fold} must contain non-empty train and validation sets.")

    train_loader = DataLoader(
        DesignPairDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DesignPairDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = InteractionAwareModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = BiasedDirectionLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"best_model_fold_{fold}_design_m.pth"

    best_loss = float("inf")
    best_rmse = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for pdz, pbm, target, censored in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, _, _ = model(pdz.to(device), pbm.to(device))
            loss = criterion(pred, target.to(device), censored.to(device))
            loss.backward()
            optimizer.step()

        val_loss, val_rmse = evaluate_design_model(model, val_loader, device, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_rmse = val_rmse
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    return TrainResult(fold, best_epoch, best_loss, best_rmse, checkpoint)
