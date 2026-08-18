"""Main-text baseline and neural benchmark workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from .model import AA_ORDER, AA_TO_INT, MAX_PDZ_LEN, PAD_IDX
from .validation import load_pair_dataset


BENCHMARK_MJ_MATRIX = torch.tensor(
    np.loadtxt(Path(__file__).with_name("data") / "benchmark_mj_matrix.csv", delimiter=","),
    dtype=torch.float32,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        pearson = spearman = float("nan")
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson,
        "spearman": spearman,
    }


def _pbm_one_hot(sequences: pd.Series) -> np.ndarray:
    out = np.zeros((len(sequences), 6, len(AA_ORDER)), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str)):
        for position, aa in enumerate(sequence.upper()[-6:]):
            if aa in AA_TO_INT:
                out[row, position, AA_TO_INT[aa]] = 1.0
    return out.reshape(len(sequences), -1)


def _pdz_composition(sequences: pd.Series) -> np.ndarray:
    out = np.zeros((len(sequences), len(AA_ORDER)), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str)):
        for aa in sequence.upper():
            if aa in AA_TO_INT:
                out[row, AA_TO_INT[aa]] += 1
        total = out[row].sum()
        if total:
            out[row] /= total
    return out


def run_baseline_benchmarks(
    dataset_path: str | Path,
    split_dir: str | Path,
    *,
    settings=("random", "pbm_heldout", "pdz_heldout"),
    folds=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the mean/Ridge/RF baselines from the benchmark Colab."""
    data = load_pair_dataset(dataset_path)
    target_column = "pKd_label" if "pKd_label" in data else "pKd"
    configurations = {
        "mean": ("mean", None),
        "ridge_pbm_only": ("ridge", "pbm"),
        "ridge_pdz_pbm_concat": ("ridge", "paired"),
        "rf_pbm_only": ("rf", "pbm"),
        "rf_pdz_only": ("rf", "pdz"),
        "rf_pdz_pbm_concat": ("rf", "paired"),
    }
    metrics_rows, prediction_rows = [], []
    for setting in settings:
        assignment = pd.read_csv(Path(split_dir) / f"{setting}_split.csv")
        selected_folds = sorted(assignment["fold"].unique()) if folds is None else folds
        for fold in selected_folds:
            fold_assignment = assignment.loc[assignment["fold"] == fold, ["pair_id", "split"]]
            frame = data.merge(fold_assignment, on="pair_id", validate="one_to_one")
            fit = frame.loc[frame["split"].isin(["train", "val"])]
            test = frame.loc[frame["split"] == "test"]
            pdz_fit, pdz_test = _pdz_composition(fit.pdz_sequence), _pdz_composition(test.pdz_sequence)
            pbm_fit, pbm_test = _pbm_one_hot(fit.pbm6), _pbm_one_hot(test.pbm6)
            features = {
                None: (np.zeros((len(fit), 1)), np.zeros((len(test), 1))),
                "pdz": (pdz_fit, pdz_test),
                "pbm": (pbm_fit, pbm_test),
                "paired": (np.c_[pdz_fit, pbm_fit], np.c_[pdz_test, pbm_test]),
            }
            for name, (kind, mode) in configurations.items():
                model = {
                    "mean": DummyRegressor(strategy="mean"),
                    "ridge": Ridge(alpha=1.0),
                    "rf": RandomForestRegressor(
                        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
                    ),
                }[kind]
                x_fit, x_test = features[mode]
                model.fit(x_fit, fit[target_column])
                prediction = model.predict(x_test)
                result = regression_metrics(test[target_column], prediction)
                result.update(
                    setting=setting, fold=int(fold), model=name,
                    n_train=len(fit), n_test=len(test),
                )
                metrics_rows.append(result)
                block = test[["pair_id", "pdz_sequence", "pbm6", target_column]].copy()
                censor_column = "is_censored_label" if "is_censored_label" in test else "is_censored"
                block["is_censored"] = test[censor_column].to_numpy()
                block.insert(0, "model", name)
                block.insert(0, "fold", int(fold))
                block.insert(0, "setting", setting)
                block["y_pred"] = prediction
                prediction_rows.append(block.rename(columns={target_column: "y_true"}))
    return pd.DataFrame(metrics_rows), pd.concat(prediction_rows, ignore_index=True)


class BenchmarkDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.pdz = torch.tensor(
            [[AA_TO_INT.get(aa, PAD_IDX) for aa in seq[:MAX_PDZ_LEN]] +
             [PAD_IDX] * max(0, MAX_PDZ_LEN - len(seq)) for seq in frame.pdz_sequence],
            dtype=torch.long,
        )
        self.pbm = torch.tensor(
            [[AA_TO_INT.get(aa, PAD_IDX) for aa in seq] for seq in frame.pbm6],
            dtype=torch.long,
        )
        target_column = "pKd_label" if "pKd_label" in frame else "pKd"
        self.target = torch.tensor(frame[target_column].to_numpy(np.float32))
        self.pair_id = torch.tensor(frame.pair_id.to_numpy(np.int64))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.pdz[index], self.pbm[index], self.target[index], self.pair_id[index]


class CNNConcatModel(nn.Module):
    def __init__(self, embed_dim=32, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(21, embed_dim, padding_idx=PAD_IDX)
        self.pdz_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1),
        )
        self.pbm_mlp = nn.Sequential(
            nn.Linear(6 * embed_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, pdz, pbm):
        pdz_feature = self.pdz_cnn(self.embedding(pdz).transpose(1, 2)).squeeze(-1)
        pbm_feature = self.pbm_mlp(self.embedding(pbm).flatten(1))
        return self.regressor(torch.cat([pdz_feature, pbm_feature], 1)).squeeze(-1)


class InteractionMapModel(nn.Module):
    def __init__(self, use_mj=False, embed_dim=32, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.use_mj = use_mj
        self.embedding = nn.Embedding(21, embed_dim, padding_idx=PAD_IDX)
        self.pdz_proj, self.pbm_proj = nn.Linear(embed_dim, hidden_dim), nn.Linear(embed_dim, hidden_dim)
        if use_mj:
            self.register_buffer("mj_matrix", BENCHMARK_MJ_MATRIX.clone())
        channels = 2 if use_mj else 1
        self.map_cnn = nn.Sequential(
            nn.Conv2d(channels, 32, (5, 3), padding=(2, 1)), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (5, 3), padding=(2, 1)), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.pdz_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(64), nn.AdaptiveMaxPool1d(1),
        )
        self.pbm_mlp = nn.Sequential(
            nn.Linear(6 * embed_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, pdz, pbm):
        pdz_embedding, pbm_embedding = self.embedding(pdz), self.embedding(pbm)
        learned = torch.einsum(
            "blh,bmh->blm", self.pdz_proj(pdz_embedding), self.pbm_proj(pbm_embedding)
        ) / np.sqrt(self.pdz_proj.out_features)
        mask = (pdz != PAD_IDX).float().unsqueeze(-1) * (pbm != PAD_IDX).float().unsqueeze(1)
        learned = learned * mask
        maps = [learned]
        if self.use_mj:
            maps.append(self.mj_matrix[pdz.unsqueeze(-1), pbm.unsqueeze(1)] * mask)
        map_feature = self.map_cnn(torch.stack(maps, 1)).flatten(1)
        pdz_feature = self.pdz_cnn(pdz_embedding.transpose(1, 2)).squeeze(-1)
        pbm_feature = self.pbm_mlp(pbm_embedding.flatten(1))
        return self.regressor(torch.cat([map_feature, pdz_feature, pbm_feature], 1)).squeeze(-1)


NEURAL_MODELS = {
    "cnn_concat": CNNConcatModel,
    "interaction_map": InteractionMapModel,
    "interaction_map_mj": lambda: InteractionMapModel(use_mj=True),
}


@torch.no_grad()
def _predict_neural(model, loader, device):
    model.eval()
    truth, prediction, pair_ids = [], [], []
    for pdz, pbm, target, pair_id in loader:
        truth.append(target.numpy())
        prediction.append(model(pdz.to(device), pbm.to(device)).cpu().numpy())
        pair_ids.append(pair_id.numpy())
    return np.concatenate(truth), np.concatenate(prediction), np.concatenate(pair_ids)


def train_neural_benchmark(
    dataset_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    setting: str,
    fold: int,
    model_name: str,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42,
    device=None,
) -> tuple[dict, pd.DataFrame]:
    """Train/evaluate one Colab-equivalent neural benchmark fold."""
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"model_name must be one of {sorted(NEURAL_MODELS)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed + fold)
    torch.cuda.manual_seed_all(seed + fold)
    np.random.seed(seed + fold)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load_pair_dataset(dataset_path)
    assignment = pd.read_csv(split_path)
    assignment = assignment.loc[assignment.fold == fold, ["pair_id", "split"]]
    frame = data.merge(assignment, on="pair_id", validate="one_to_one")
    subsets = {name: frame.loc[frame.split == name].copy() for name in ("train", "val", "test")}
    loaders = {
        name: DataLoader(BenchmarkDataset(part), batch_size=batch_size, shuffle=name == "train", num_workers=0,
                         pin_memory=device.type == "cuda")
        for name, part in subsets.items()
    }
    model = NEURAL_MODELS[model_name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{model_name}_{setting}_fold{fold}.pt"
    best_rmse, best_epoch, stale = float("inf"), -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for pdz, pbm, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(pdz.to(device), pbm.to(device)), target.to(device))
            loss.backward()
            if model_name != "cnn_concat":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        y_val, pred_val, _ = _predict_neural(model, loaders["val"], device)
        val_rmse = regression_metrics(y_val, pred_val)["rmse"]
        if val_rmse < best_rmse:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    y_true, y_pred, pair_ids = _predict_neural(model, loaders["test"], device)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(setting=setting, fold=fold, model=model_name, best_epoch=best_epoch,
                   best_val_rmse=best_rmse, n_train=len(subsets["train"]), n_test=len(subsets["test"]))
    predictions = pd.DataFrame(
        {"setting": setting, "fold": fold, "model": model_name, "pair_id": pair_ids,
         "y_true": y_true, "y_pred": y_pred}
    )
    censor_column = "is_censored_label" if "is_censored_label" in subsets["test"] else "is_censored"
    predictions = predictions.merge(
        subsets["test"][["pair_id", censor_column]].rename(columns={censor_column: "is_censored"}),
        on="pair_id", how="left", validate="one_to_one",
    )
    return metrics, predictions
