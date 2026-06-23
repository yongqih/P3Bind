# Auto-exported from 01_affinity_prediction_benchmark.ipynb.
# NOTE: This is a faithful notebook export. Some paths may need to be set using the reproducibility guide.


# %% Cell 0
import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')


df = pd.read_csv("/content/drive/MyDrive/PDZ_DL/all_data.csv")

df = df.rename(columns={
    "bait_gene_name": "pdz_gene",
    "bait_molecule_id": "pdz_uniprot",
    "bait_site": "pdz_site",
    "prey_gene_name": "pbm_gene",
    "prey_molecule_id": "pbm_uniprot",
    "prey_site": "pbm_site",
    "prey_sequence": "pbm_sequence_10aa",
    "mask": "is_censored"
})

df["pdz_sequence"] = df["pdz_sequence"].str.strip().str.upper()
df["pbm_sequence_10aa"] = df["pbm_sequence_10aa"].str.strip().str.upper()
df["pKd"] = pd.to_numeric(df["pKd"], errors="coerce")
df["is_censored"] = df["is_censored"].astype(int)

df = df.drop_duplicates().reset_index(drop=True)

def aggregate_replicates(g):
    uncensored = g[g["is_censored"] == 0]
    censored = g[g["is_censored"] == 1]

    row = g.iloc[0].copy()

    row["n_records"] = len(g)
    row["n_uncensored"] = len(uncensored)
    row["n_censored"] = len(censored)
    row["has_mixed_censoring"] = int(len(uncensored) > 0 and len(censored) > 0)

    if len(uncensored) > 0:
        values = uncensored["pKd"].astype(float)

        row["pKd_label"] = values.median()
        row["is_censored_label"] = 0

        row["pKd_mean_uncensored"] = values.mean()
        row["pKd_median_uncensored"] = values.median()
        row["pKd_std_uncensored"] = values.std(ddof=1) if len(values) > 1 else 0.0
        row["pKd_min_uncensored"] = values.min()
        row["pKd_max_uncensored"] = values.max()
        row["pKd_range_uncensored"] = values.max() - values.min()
    else:
        row["pKd_label"] = 3.1
        row["is_censored_label"] = 1

        row["pKd_mean_uncensored"] = np.nan
        row["pKd_median_uncensored"] = np.nan
        row["pKd_std_uncensored"] = np.nan
        row["pKd_min_uncensored"] = np.nan
        row["pKd_max_uncensored"] = np.nan
        row["pKd_range_uncensored"] = np.nan

    row["high_variance"] = int(
        row["is_censored_label"] == 0 and row["pKd_range_uncensored"] >= 1.0
    )

    return row

df_pair = (
    df
    .groupby(["pdz_sequence", "pbm_sequence_10aa"], group_keys=False)
    .apply(aggregate_replicates)
    .reset_index(drop=True)
)



# %% Cell 1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from pathlib import Path

# =========================
# Load data
# =========================
df = pd.read_csv("/content/drive/MyDrive/PDZ_DL/all_data_pair_aggregated.csv").reset_index(drop=True)
df["pair_id"] = np.arange(len(df))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold

SEED = 42
SPLIT_DIR = Path("/content/drive/MyDrive/PDZ_DL/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Prepare columns
# =========================

df = df.reset_index(drop=True).copy()

if "pair_id" not in df.columns:
    df["pair_id"] = np.arange(len(df))

if "pbm_sequence_6aa" not in df.columns:
    if "pbm_sequence_10aa" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence_10aa"].astype(str).str.upper().str[-6:]
    elif "prey_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["prey_sequence"].astype(str).str.upper().str[-6:]
    elif "pbm_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence"].astype(str).str.upper().str[-6:]
    else:
        raise ValueError("Cannot find PBM sequence column.")

df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.upper().str.strip()
df["pbm_sequence_6aa"] = df["pbm_sequence_6aa"].astype(str).str.upper().str.strip()


# =========================
# Split helper
# =========================

def make_split_file(df, splitter, split_name, groups=None, val_fraction=0.1):
    rows = []

    for fold, (trainval_idx, test_idx) in enumerate(splitter.split(df, groups=groups)):
        trainval_idx = np.array(trainval_idx)
        test_idx = np.array(test_idx)

        rng = np.random.default_rng(SEED + fold)
        shuffled = trainval_idx.copy()
        rng.shuffle(shuffled)

        val_size = max(1, int(val_fraction * len(shuffled)))
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]

        for i in train_idx:
            rows.append({
                "pair_id": int(df.loc[i, "pair_id"]),
                "split": "train",
                "fold": fold,
                "setting": split_name
            })

        for i in val_idx:
            rows.append({
                "pair_id": int(df.loc[i, "pair_id"]),
                "split": "val",
                "fold": fold,
                "setting": split_name
            })

        for i in test_idx:
            rows.append({
                "pair_id": int(df.loc[i, "pair_id"]),
                "split": "test",
                "fold": fold,
                "setting": split_name
            })

    split_df = pd.DataFrame(rows)

    out_path = SPLIT_DIR / f"{split_name}_split.csv"
    split_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(split_df.groupby(["fold", "split"]).size())

    return split_df


# =========================
# 1. Random split
# =========================

random_splitter = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

random_split = make_split_file(
    df=df,
    splitter=random_splitter,
    split_name="random",
    groups=None
)


# =========================
# 2. PDZ-held-out split
# =========================

pdz_splitter = GroupKFold(n_splits=5)

pdz_heldout_split = make_split_file(
    df=df,
    splitter=pdz_splitter,
    split_name="pdz_heldout",
    groups=df["pdz_sequence"]
)


# =========================
# 3. PBM6-held-out split
# =========================

pbm_splitter = GroupKFold(n_splits=5)

pbm_heldout_split = make_split_file(
    df=df,
    splitter=pbm_splitter,
    split_name="pbm_heldout",
    groups=df["pbm_sequence_6aa"]
)

# %% Cell 2
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

SEED = 42

RESULT_DIR = Path("/content/drive/MyDrive/PDZ_DL/splits/results/baselines")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


def one_hot_pbm6(seq):
    seq = str(seq).upper()[-6:]
    x = np.zeros((6, len(AA)), dtype=float)

    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0

    return x.flatten()


def aa_composition(seq):
    seq = str(seq).upper()
    counts = np.zeros(len(AA), dtype=float)

    for aa in seq:
        if aa in AA_TO_IDX:
            counts[AA_TO_IDX[aa]] += 1

    total = counts.sum()
    if total > 0:
        counts = counts / total

    return counts


def build_features(df_sub, mode):
    features = []

    for _, row in df_sub.iterrows():
        pbm_feat = one_hot_pbm6(row["pbm_sequence_6aa"])
        pdz_feat = aa_composition(row["pdz_sequence"])

        if mode == "pbm_only":
            feat = pbm_feat
        elif mode == "pdz_only":
            feat = pdz_feat
        elif mode == "pdz_pbm_concat":
            feat = np.concatenate([pdz_feat, pbm_feat])
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        features.append(feat)

    return np.vstack(features)

# %% Cell 3
def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan, np.nan

    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]

    return pearson, spearman


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, spearman = safe_corr(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }

# %% Cell 4
if "pKd_label" in df.columns:
    TARGET_COL = "pKd_label"
elif "pKd" in df.columns:
    TARGET_COL = "pKd"
else:
    raise ValueError("Cannot find pKd_label or pKd column.")

if "is_censored_label" in df.columns:
    CENSOR_COL = "is_censored_label"
elif "is_censored" in df.columns:
    CENSOR_COL = "is_censored"
elif "mask" in df.columns:
    CENSOR_COL = "mask"
else:
    CENSOR_COL = None

print("Target column:", TARGET_COL)
print("Censor column:", CENSOR_COL)

# %% Cell 5
SPLIT_FILES = {
    "random": SPLIT_DIR / "random_split.csv",
    "pdz_heldout": SPLIT_DIR / "pdz_heldout_split.csv",
    "pbm_heldout": SPLIT_DIR / "pbm_heldout_split.csv",
}

models_to_run = {
    "mean": {
        "type": "dummy",
        "feature_mode": None
    },
    "ridge_pbm_only": {
        "type": "ridge",
        "feature_mode": "pbm_only"
    },
    "ridge_pdz_pbm_concat": {
        "type": "ridge",
        "feature_mode": "pdz_pbm_concat"
    },
    "rf_pbm_only": {
        "type": "rf",
        "feature_mode": "pbm_only"
    },
    "rf_pdz_only": {
        "type": "rf",
        "feature_mode": "pdz_only"
    },
    "rf_pdz_pbm_concat": {
        "type": "rf",
        "feature_mode": "pdz_pbm_concat"
    },
}


def create_model(model_type):
    if model_type == "dummy":
        return DummyRegressor(strategy="mean")

    if model_type == "ridge":
        return Ridge(alpha=1.0)

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model type: {model_type}")


all_metrics = []
all_predictions = []

for setting, split_path in SPLIT_FILES.items():
    split_df = pd.read_csv(split_path)

    print(f"\n========== Setting: {setting} ==========")

    for fold in sorted(split_df["fold"].unique()):
        fold_split = split_df[split_df["fold"] == fold]

        train_ids = fold_split[fold_split["split"] == "train"]["pair_id"].values
        val_ids = fold_split[fold_split["split"] == "val"]["pair_id"].values
        test_ids = fold_split[fold_split["split"] == "test"]["pair_id"].values

        train_df = df[df["pair_id"].isin(train_ids)].copy()
        val_df = df[df["pair_id"].isin(val_ids)].copy()
        test_df = df[df["pair_id"].isin(test_ids)].copy()

        # baseline 不调参，所以 train + val 一起训练，test 评估
        fit_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        y_train = fit_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values

        for model_name, config in models_to_run.items():
            print(f"Running {setting}, fold {fold}, model {model_name}")

            model = create_model(config["type"])

            if config["type"] == "dummy":
                X_train = np.zeros((len(fit_df), 1))
                X_test = np.zeros((len(test_df), 1))
            else:
                X_train = build_features(fit_df, config["feature_mode"])
                X_test = build_features(test_df, config["feature_mode"])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate_regression(y_test, y_pred)
            metrics.update({
                "setting": setting,
                "fold": fold,
                "model": model_name,
                "n_train": len(fit_df),
                "n_test": len(test_df),
            })

            all_metrics.append(metrics)

            pred_df = pd.DataFrame({
                "setting": setting,
                "fold": fold,
                "model": model_name,
                "pair_id": test_df["pair_id"].values,
                "pdz_sequence": test_df["pdz_sequence"].values,
                "pbm_sequence_6aa": test_df["pbm_sequence_6aa"].values,
                "y_true": y_test,
                "y_pred": y_pred,
            })

            if CENSOR_COL is not None and CENSOR_COL in test_df.columns:
                pred_df[CENSOR_COL] = test_df[CENSOR_COL].values

            all_predictions.append(pred_df)

# %% Cell 6
metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)

metrics_path = RESULT_DIR / "baseline_metrics.csv"
predictions_path = RESULT_DIR / "baseline_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("Saved metrics to:", metrics_path)
print("Saved predictions to:", predictions_path)

display(metrics_df.head())
display(predictions_df.head())

# %% Cell 7
summary = (
    metrics_df
    .groupby(["setting", "model"])
    [["rmse", "mae", "r2", "pearson", "spearman"]]
    .agg(["mean", "std"])
)

summary_path = RESULT_DIR / "baseline_summary.csv"
summary.to_csv(summary_path)

display(summary)
print("Saved summary to:", summary_path)

# %% Cell 8
if CENSOR_COL is not None and CENSOR_COL in predictions_df.columns:
    uncensored_pred = predictions_df[predictions_df[CENSOR_COL] == 0].copy()

    uncensored_metrics = []

    for (setting, fold, model), g in uncensored_pred.groupby(["setting", "fold", "model"]):
        if len(g) < 5:
            continue

        m = evaluate_regression(g["y_true"].values, g["y_pred"].values)
        m.update({
            "setting": setting,
            "fold": fold,
            "model": model,
            "n_test_uncensored": len(g),
        })

        uncensored_metrics.append(m)

    uncensored_metrics_df = pd.DataFrame(uncensored_metrics)

    uncensored_path = RESULT_DIR / "baseline_metrics_uncensored_only.csv"
    uncensored_metrics_df.to_csv(uncensored_path, index=False)

    uncensored_summary = (
        uncensored_metrics_df
        .groupby(["setting", "model"])
        [["rmse", "mae", "r2", "pearson", "spearman"]]
        .agg(["mean", "std"])
    )

    uncensored_summary_path = RESULT_DIR / "baseline_summary_uncensored_only.csv"
    uncensored_summary.to_csv(uncensored_summary_path)

    display(uncensored_summary)
    print("Saved uncensored metrics to:", uncensored_path)
    print("Saved uncensored summary to:", uncensored_summary_path)

else:
    print("No censor column found. Skipping uncensored-only evaluation.")

# %% [markdown]
# DL model
# PyTorch CNN-concat baseline

# %% Cell 10
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# =========================
# Basic settings
# =========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SPLIT_DIR = Path("/content/drive/MyDrive/PDZ_DL/splits")
RESULT_DIR = Path("/content/drive/MyDrive/PDZ_DL/results/dl_cnn_concat")
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training settings
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8


# %% Cell 11
df = pd.read_csv("/content/drive/MyDrive/PDZ_DL/all_data_pair_aggregated.csv")
df = df.reset_index(drop=True).copy()

if "pair_id" not in df.columns:
    df["pair_id"] = np.arange(len(df))

if "pKd_label" in df.columns:
    TARGET_COL = "pKd_label"
elif "pKd" in df.columns:
    TARGET_COL = "pKd"
else:
    raise ValueError("Cannot find pKd_label or pKd column.")

if "is_censored_label" in df.columns:
    CENSOR_COL = "is_censored_label"
elif "is_censored" in df.columns:
    CENSOR_COL = "is_censored"
elif "mask" in df.columns:
    CENSOR_COL = "mask"
else:
    CENSOR_COL = None

if "pbm_sequence_6aa" not in df.columns:
    if "pbm_sequence_10aa" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence_10aa"].astype(str).str.upper().str[-6:]
    elif "prey_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["prey_sequence"].astype(str).str.upper().str[-6:]
    elif "pbm_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence"].astype(str).str.upper().str[-6:]
    else:
        raise ValueError("Cannot find PBM sequence column.")

df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.upper().str.strip()
df["pbm_sequence_6aa"] = df["pbm_sequence_6aa"].astype(str).str.upper().str.strip()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df.dropna(subset=["pdz_sequence", "pbm_sequence_6aa", TARGET_COL]).reset_index(drop=True)
df["pair_id"] = np.arange(len(df))

print("Data shape:", df.shape)
print("Target:", TARGET_COL)
print("Censor:", CENSOR_COL)
print("Unique PDZ:", df["pdz_sequence"].nunique())
print("Unique PBM6:", df["pbm_sequence_6aa"].nunique())
print(df[["pair_id", "pdz_sequence", "pbm_sequence_6aa", TARGET_COL]].head())

# %% Cell 12
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA)}  # 0 reserved for padding
PAD_IDX = 0

MAX_PDZ_LEN = int(df["pdz_sequence"].str.len().quantile(0.99))
MAX_PDZ_LEN = max(MAX_PDZ_LEN, 80)

print("MAX_PDZ_LEN:", MAX_PDZ_LEN)


def encode_sequence(seq, max_len):
    seq = str(seq).upper()
    arr = np.zeros(max_len, dtype=np.int64)

    for i, aa in enumerate(seq[:max_len]):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr


def encode_pbm6(seq):
    seq = str(seq).upper()[-6:]
    arr = np.zeros(6, dtype=np.int64)

    for i, aa in enumerate(seq):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr

# %% Cell 13
class PDZPBMDataset(Dataset):
    def __init__(self, df_sub, target_col):
        self.df = df_sub.reset_index(drop=True).copy()
        self.target_col = target_col

        self.pdz_encoded = np.stack([
            encode_sequence(seq, MAX_PDZ_LEN)
            for seq in self.df["pdz_sequence"].values
        ])

        self.pbm_encoded = np.stack([
            encode_pbm6(seq)
            for seq in self.df["pbm_sequence_6aa"].values
        ])

        self.y = self.df[target_col].astype(float).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdz = torch.tensor(self.pdz_encoded[idx], dtype=torch.long)
        pbm = torch.tensor(self.pbm_encoded[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return {
            "pdz": pdz,
            "pbm": pbm,
            "y": y,
            "pair_id": int(self.df.loc[idx, "pair_id"])
        }


def make_loader(df_sub, shuffle=False):
    dataset = PDZPBMDataset(df_sub, TARGET_COL)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    return loader

# %% Cell 14
class CNNConcatModel(nn.Module):
    def __init__(
        self,
        vocab_size=21,
        embed_dim=32,
        pdz_channels=64,
        pbm_hidden=64,
        dropout=0.25
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        # PDZ CNN encoder
        self.pdz_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, pdz_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(pdz_channels),

            nn.Conv1d(pdz_channels, pdz_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(pdz_channels),

            nn.AdaptiveMaxPool1d(1)
        )

        # PBM6 encoder: embedding flatten + MLP
        self.pbm_mlp = nn.Sequential(
            nn.Linear(6 * embed_dim, pbm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pbm_hidden, pbm_hidden),
            nn.ReLU()
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(pdz_channels + pbm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, pdz, pbm):
        # pdz: [B, L]
        # pbm: [B, 6]

        pdz_emb = self.embedding(pdz)          # [B, L, E]
        pdz_emb = pdz_emb.transpose(1, 2)      # [B, E, L]
        pdz_feat = self.pdz_cnn(pdz_emb)       # [B, C, 1]
        pdz_feat = pdz_feat.squeeze(-1)        # [B, C]

        pbm_emb = self.embedding(pbm)          # [B, 6, E]
        pbm_feat = pbm_emb.reshape(pbm_emb.size(0), -1)  # [B, 6E]
        pbm_feat = self.pbm_mlp(pbm_feat)      # [B, H]

        feat = torch.cat([pdz_feat, pbm_feat], dim=1)
        pred = self.regressor(feat).squeeze(-1)

        return pred

# %% Cell 15
def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan, np.nan

    return pearsonr(y_true, y_pred)[0], spearmanr(y_true, y_pred)[0]


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, spearman = safe_corr(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }


@torch.no_grad()
def predict_model(model, loader):
    model.eval()

    all_y = []
    all_pred = []
    all_pair_ids = []

    for batch in loader:
        pdz = batch["pdz"].to(DEVICE)
        pbm = batch["pbm"].to(DEVICE)
        y = batch["y"].cpu().numpy()
        pair_id = batch["pair_id"].cpu().numpy()

        pred = model(pdz, pbm).detach().cpu().numpy()

        all_y.append(y)
        all_pred.append(pred)
        all_pair_ids.append(pair_id)

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_pair_ids = np.concatenate(all_pair_ids)

    return all_y, all_pred, all_pair_ids

# %% Cell 16
def train_one_model(train_df, val_df, setting, fold):
    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    model = CNNConcatModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss()

    best_val_rmse = np.inf
    best_epoch = -1
    patience_counter = 0

    best_path = MODEL_DIR / f"cnn_concat_{setting}_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            pdz = batch["pdz"].to(DEVICE)
            pbm = batch["pbm"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            optimizer.zero_grad()

            pred = model(pdz, pbm)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        y_val, pred_val, _ = predict_model(model, val_loader)
        val_metrics = evaluate_regression(y_val, pred_val)
        val_rmse = val_metrics["rmse"]

        mean_train_loss = float(np.mean(train_losses))

        print(
            f"[{setting} fold {fold}] "
            f"Epoch {epoch:03d} | "
            f"Train loss {mean_train_loss:.4f} | "
            f"Val RMSE {val_rmse:.4f} | "
            f"Val Pearson {val_metrics['pearson']:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    return model, best_epoch, best_val_rmse

# %% Cell 17
SPLIT_FILES = {
    "random": SPLIT_DIR / "random_split.csv",
    "pdz_heldout": SPLIT_DIR / "pdz_heldout_split.csv",
    "pbm_heldout": SPLIT_DIR / "pbm_heldout_split.csv",
}

# 如果只想先测试 random：
# SPLIT_FILES = {
#     "random": SPLIT_DIR / "random_split.csv",
# }

all_metrics = []
all_predictions = []

for setting, split_path in SPLIT_FILES.items():
    split_df = pd.read_csv(split_path)

    print(f"\n==============================")
    print(f"Running setting: {setting}")
    print(f"==============================")

    for fold in sorted(split_df["fold"].unique()):
        fold_split = split_df[split_df["fold"] == fold]

        train_ids = fold_split[fold_split["split"] == "train"]["pair_id"].values
        val_ids = fold_split[fold_split["split"] == "val"]["pair_id"].values
        test_ids = fold_split[fold_split["split"] == "test"]["pair_id"].values

        train_df = df[df["pair_id"].isin(train_ids)].copy()
        val_df = df[df["pair_id"].isin(val_ids)].copy()
        test_df = df[df["pair_id"].isin(test_ids)].copy()

        print(f"\nSetting {setting}, fold {fold}")
        print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

        model, best_epoch, best_val_rmse = train_one_model(
            train_df=train_df,
            val_df=val_df,
            setting=setting,
            fold=fold
        )

        test_loader = make_loader(test_df, shuffle=False)
        y_test, y_pred, pair_ids = predict_model(model, test_loader)

        metrics = evaluate_regression(y_test, y_pred)
        metrics.update({
            "setting": setting,
            "fold": fold,
            "model": "cnn_concat",
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df)
        })

        all_metrics.append(metrics)

        pred_df = pd.DataFrame({
            "setting": setting,
            "fold": fold,
            "model": "cnn_concat",
            "pair_id": pair_ids,
            "y_true": y_test,
            "y_pred": y_pred
        })

        meta_cols = ["pair_id", "pdz_sequence", "pbm_sequence_6aa"]
        if CENSOR_COL is not None and CENSOR_COL in df.columns:
            meta_cols.append(CENSOR_COL)

        pred_df = pred_df.merge(
            df[meta_cols],
            on="pair_id",
            how="left"
        )

        all_predictions.append(pred_df)

        print("Test metrics:", metrics)

# %% Cell 18
dl_metrics_df = pd.DataFrame(all_metrics)
dl_predictions_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)

metrics_path = RESULT_DIR / "dl_cnn_concat_metrics.csv"
pred_path = RESULT_DIR / "dl_cnn_concat_predictions.csv"

dl_metrics_df.to_csv(metrics_path, index=False)
dl_predictions_df.to_csv(pred_path, index=False)

print("Saved metrics to:", metrics_path)
print("Saved predictions to:", pred_path)

display(dl_metrics_df)
display(dl_predictions_df.head())

# %% Cell 19
dl_summary = (
    dl_metrics_df
    .groupby(["setting", "model"])
    [["rmse", "mae", "r2", "pearson", "spearman"]]
    .agg(["mean", "std"])
)

summary_path = RESULT_DIR / "dl_cnn_concat_summary.csv"
dl_summary.to_csv(summary_path)

display(dl_summary)
print("Saved summary to:", summary_path)

# %% Cell 20
if CENSOR_COL is not None and CENSOR_COL in dl_predictions_df.columns:
    uncensored_pred = dl_predictions_df[dl_predictions_df[CENSOR_COL] == 0].copy()

    uncensored_metrics = []

    for (setting, fold, model), g in uncensored_pred.groupby(["setting", "fold", "model"]):
        if len(g) < 5:
            continue

        m = evaluate_regression(g["y_true"].values, g["y_pred"].values)
        m.update({
            "setting": setting,
            "fold": fold,
            "model": model,
            "n_test_uncensored": len(g)
        })

        uncensored_metrics.append(m)

    dl_uncensored_metrics_df = pd.DataFrame(uncensored_metrics)

    uncensored_metrics_path = RESULT_DIR / "dl_cnn_concat_metrics_uncensored_only.csv"
    dl_uncensored_metrics_df.to_csv(uncensored_metrics_path, index=False)

    dl_uncensored_summary = (
        dl_uncensored_metrics_df
        .groupby(["setting", "model"])
        [["rmse", "mae", "r2", "pearson", "spearman"]]
        .agg(["mean", "std"])
    )

    uncensored_summary_path = RESULT_DIR / "dl_cnn_concat_summary_uncensored_only.csv"
    dl_uncensored_summary.to_csv(uncensored_summary_path)

    display(dl_uncensored_summary)
    print("Saved uncensored metrics to:", uncensored_metrics_path)
    print("Saved uncensored summary to:", uncensored_summary_path)

else:
    print("No censor column found. Skipping uncensored-only evaluation.")

# %% Cell 21
import matplotlib.pyplot as plt

for setting in dl_predictions_df["setting"].unique():
    g = dl_predictions_df[dl_predictions_df["setting"] == setting].copy()

    plt.figure(figsize=(5, 5))
    plt.scatter(g["y_true"], g["y_pred"], alpha=0.3, s=10)

    min_val = min(g["y_true"].min(), g["y_pred"].min())
    max_val = max(g["y_true"].max(), g["y_pred"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Observed pKd")
    plt.ylabel("Predicted pKd")
    plt.title(f"CNN-concat: {setting}")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Interaction-aware model

# %% Cell 23
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SPLIT_DIR = Path("/content/drive/MyDrive/PDZ_DL/splits")
RESULT_DIR = Path("/content/drive/MyDrive/PDZ_DL/results/dl_interaction_map")
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8


# %% Cell 24
df = df.reset_index(drop=True).copy()

if "pair_id" not in df.columns:
    df["pair_id"] = np.arange(len(df))

if "pKd_label" in df.columns:
    TARGET_COL = "pKd_label"
elif "pKd" in df.columns:
    TARGET_COL = "pKd"
else:
    raise ValueError("Cannot find pKd_label or pKd column.")

if "is_censored_label" in df.columns:
    CENSOR_COL = "is_censored_label"
elif "is_censored" in df.columns:
    CENSOR_COL = "is_censored"
elif "mask" in df.columns:
    CENSOR_COL = "mask"
else:
    CENSOR_COL = None

if "pbm_sequence_6aa" not in df.columns:
    if "pbm_sequence_10aa" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence_10aa"].astype(str).str.upper().str[-6:]
    elif "prey_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["prey_sequence"].astype(str).str.upper().str[-6:]
    elif "pbm_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence"].astype(str).str.upper().str[-6:]
    else:
        raise ValueError("Cannot find PBM sequence column.")

df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.upper().str.strip()
df["pbm_sequence_6aa"] = df["pbm_sequence_6aa"].astype(str).str.upper().str.strip()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df.dropna(subset=["pdz_sequence", "pbm_sequence_6aa", TARGET_COL]).reset_index(drop=True)
df["pair_id"] = np.arange(len(df))

print("Data shape:", df.shape)
print("Target:", TARGET_COL)
print("Censor:", CENSOR_COL)
print("Unique PDZ:", df["pdz_sequence"].nunique())
print("Unique PBM6:", df["pbm_sequence_6aa"].nunique())

# %% Cell 25
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA)}
PAD_IDX = 0

MAX_PDZ_LEN = int(df["pdz_sequence"].str.len().quantile(0.99))
MAX_PDZ_LEN = max(MAX_PDZ_LEN, 80)

print("MAX_PDZ_LEN:", MAX_PDZ_LEN)


def encode_sequence(seq, max_len):
    seq = str(seq).upper()
    arr = np.zeros(max_len, dtype=np.int64)

    for i, aa in enumerate(seq[:max_len]):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr


def encode_pbm6(seq):
    seq = str(seq).upper()[-6:]
    arr = np.zeros(6, dtype=np.int64)

    for i, aa in enumerate(seq):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr

# %% Cell 26
class PDZPBMDataset(Dataset):
    def __init__(self, df_sub, target_col):
        self.df = df_sub.reset_index(drop=True).copy()
        self.target_col = target_col

        self.pdz_encoded = np.stack([
            encode_sequence(seq, MAX_PDZ_LEN)
            for seq in self.df["pdz_sequence"].values
        ])

        self.pbm_encoded = np.stack([
            encode_pbm6(seq)
            for seq in self.df["pbm_sequence_6aa"].values
        ])

        self.y = self.df[target_col].astype(float).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdz = torch.tensor(self.pdz_encoded[idx], dtype=torch.long)
        pbm = torch.tensor(self.pbm_encoded[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return {
            "pdz": pdz,
            "pbm": pbm,
            "y": y,
            "pair_id": int(self.df.loc[idx, "pair_id"])
        }


def make_loader(df_sub, shuffle=False):
    dataset = PDZPBMDataset(df_sub, TARGET_COL)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    return loader

# %% Cell 27
class InteractionMapModel(nn.Module):
    def __init__(
        self,
        vocab_size=21,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.25
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        # 把 embedding 投影到 interaction space
        self.pdz_proj = nn.Linear(embed_dim, hidden_dim)
        self.pbm_proj = nn.Linear(embed_dim, hidden_dim)

        # 处理 interaction map: [B, 1, L, 6]
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        # 额外保留 sequence-level features
        self.pdz_seq_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1)
        )

        self.pbm_mlp = nn.Sequential(
            nn.Linear(6 * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, pdz, pbm):
        # pdz: [B, L]
        # pbm: [B, 6]

        pdz_emb = self.embedding(pdz)      # [B, L, E]
        pbm_emb = self.embedding(pbm)      # [B, 6, E]

        # learned interaction map
        pdz_h = self.pdz_proj(pdz_emb)     # [B, L, H]
        pbm_h = self.pbm_proj(pbm_emb)     # [B, 6, H]

        # dot product interaction: [B, L, 6]
        interaction = torch.einsum("blh,bmh->blm", pdz_h, pbm_h)
        interaction = interaction / np.sqrt(pdz_h.size(-1))

        # mask padding positions in PDZ
        pdz_mask = (pdz != PAD_IDX).float().unsqueeze(-1)  # [B, L, 1]
        interaction = interaction * pdz_mask

        interaction = interaction.unsqueeze(1)  # [B, 1, L, 6]
        map_feat = self.map_cnn(interaction).squeeze(-1).squeeze(-1)  # [B, 64]

        # sequence-level PDZ feature
        pdz_emb_cnn = pdz_emb.transpose(1, 2)  # [B, E, L]
        pdz_seq_feat = self.pdz_seq_cnn(pdz_emb_cnn).squeeze(-1)  # [B, 64]

        # PBM feature
        pbm_flat = pbm_emb.reshape(pbm_emb.size(0), -1)
        pbm_feat = self.pbm_mlp(pbm_flat)  # [B, 64]

        feat = torch.cat([map_feat, pdz_seq_feat, pbm_feat], dim=1)
        pred = self.regressor(feat).squeeze(-1)

        return pred

# %% Cell 28
def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan, np.nan

    return pearsonr(y_true, y_pred)[0], spearmanr(y_true, y_pred)[0]


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, spearman = safe_corr(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }


@torch.no_grad()
def predict_model(model, loader):
    model.eval()

    all_y = []
    all_pred = []
    all_pair_ids = []

    for batch in loader:
        pdz = batch["pdz"].to(DEVICE)
        pbm = batch["pbm"].to(DEVICE)

        y = batch["y"].cpu().numpy()
        pair_id = batch["pair_id"].cpu().numpy()

        pred = model(pdz, pbm).detach().cpu().numpy()

        all_y.append(y)
        all_pred.append(pred)
        all_pair_ids.append(pair_id)

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_pair_ids = np.concatenate(all_pair_ids)

    return all_y, all_pred, all_pair_ids

# %% Cell 29
def train_one_model(train_df, val_df, setting, fold):
    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    model = InteractionMapModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss()

    best_val_rmse = np.inf
    best_epoch = -1
    patience_counter = 0

    best_path = MODEL_DIR / f"interaction_map_{setting}_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            pdz = batch["pdz"].to(DEVICE)
            pbm = batch["pbm"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            optimizer.zero_grad()

            pred = model(pdz, pbm)
            loss = loss_fn(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_losses.append(loss.item())

        y_val, pred_val, _ = predict_model(model, val_loader)
        val_metrics = evaluate_regression(y_val, pred_val)
        val_rmse = val_metrics["rmse"]

        mean_train_loss = float(np.mean(train_losses))

        print(
            f"[{setting} fold {fold}] "
            f"Epoch {epoch:03d} | "
            f"Train loss {mean_train_loss:.4f} | "
            f"Val RMSE {val_rmse:.4f} | "
            f"Val Pearson {val_metrics['pearson']:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    return model, best_epoch, best_val_rmse

# %% Cell 30
SPLIT_FILES = {
    "random": SPLIT_DIR / "random_split.csv",
    "pdz_heldout": SPLIT_DIR / "pdz_heldout_split.csv",
    "pbm_heldout": SPLIT_DIR / "pbm_heldout_split.csv",
}

# 如果只想先测试：
# SPLIT_FILES = {
#     "random": SPLIT_DIR / "random_split.csv",
# }

all_metrics = []
all_predictions = []

for setting, split_path in SPLIT_FILES.items():
    split_df = pd.read_csv(split_path)

    print(f"\n==============================")
    print(f"Running setting: {setting}")
    print(f"==============================")

    for fold in sorted(split_df["fold"].unique()):
        fold_split = split_df[split_df["fold"] == fold]

        train_ids = fold_split[fold_split["split"] == "train"]["pair_id"].values
        val_ids = fold_split[fold_split["split"] == "val"]["pair_id"].values
        test_ids = fold_split[fold_split["split"] == "test"]["pair_id"].values

        train_df = df[df["pair_id"].isin(train_ids)].copy()
        val_df = df[df["pair_id"].isin(val_ids)].copy()
        test_df = df[df["pair_id"].isin(test_ids)].copy()

        print(f"\nSetting {setting}, fold {fold}")
        print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

        model, best_epoch, best_val_rmse = train_one_model(
            train_df=train_df,
            val_df=val_df,
            setting=setting,
            fold=fold
        )

        test_loader = make_loader(test_df, shuffle=False)
        y_test, y_pred, pair_ids = predict_model(model, test_loader)

        metrics = evaluate_regression(y_test, y_pred)
        metrics.update({
            "setting": setting,
            "fold": fold,
            "model": "interaction_map",
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df)
        })

        all_metrics.append(metrics)

        pred_df = pd.DataFrame({
            "setting": setting,
            "fold": fold,
            "model": "interaction_map",
            "pair_id": pair_ids,
            "y_true": y_test,
            "y_pred": y_pred
        })

        meta_cols = ["pair_id", "pdz_sequence", "pbm_sequence_6aa"]
        if CENSOR_COL is not None and CENSOR_COL in df.columns:
            meta_cols.append(CENSOR_COL)

        pred_df = pred_df.merge(
            df[meta_cols],
            on="pair_id",
            how="left"
        )

        all_predictions.append(pred_df)

        print("Test metrics:", metrics)

# %% Cell 31
interaction_metrics_df = pd.DataFrame(all_metrics)
interaction_predictions_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)

metrics_path = RESULT_DIR / "interaction_map_metrics.csv"
pred_path = RESULT_DIR / "interaction_map_predictions.csv"

interaction_metrics_df.to_csv(metrics_path, index=False)
interaction_predictions_df.to_csv(pred_path, index=False)

print("Saved metrics to:", metrics_path)
print("Saved predictions to:", pred_path)

display(interaction_metrics_df)
display(interaction_predictions_df.head())

# %% Cell 32
interaction_summary = (
    interaction_metrics_df
    .groupby(["setting", "model"])
    [["rmse", "mae", "r2", "pearson", "spearman"]]
    .agg(["mean", "std"])
)

summary_path = RESULT_DIR / "interaction_map_summary.csv"
interaction_summary.to_csv(summary_path)

display(interaction_summary)
print("Saved summary to:", summary_path)

# %% Cell 33
if CENSOR_COL is not None and CENSOR_COL in interaction_predictions_df.columns:
    uncensored_pred = interaction_predictions_df[
        interaction_predictions_df[CENSOR_COL] == 0
    ].copy()

    uncensored_metrics = []

    for (setting, fold, model), g in uncensored_pred.groupby(["setting", "fold", "model"]):
        if len(g) < 5:
            continue

        m = evaluate_regression(g["y_true"].values, g["y_pred"].values)
        m.update({
            "setting": setting,
            "fold": fold,
            "model": model,
            "n_test_uncensored": len(g)
        })

        uncensored_metrics.append(m)

    interaction_uncensored_metrics_df = pd.DataFrame(uncensored_metrics)

    uncensored_metrics_path = RESULT_DIR / "interaction_map_metrics_uncensored_only.csv"
    interaction_uncensored_metrics_df.to_csv(uncensored_metrics_path, index=False)

    interaction_uncensored_summary = (
        interaction_uncensored_metrics_df
        .groupby(["setting", "model"])
        [["rmse", "mae", "r2", "pearson", "spearman"]]
        .agg(["mean", "std"])
    )

    uncensored_summary_path = RESULT_DIR / "interaction_map_summary_uncensored_only.csv"
    interaction_uncensored_summary.to_csv(uncensored_summary_path)

    display(interaction_uncensored_summary)
    print("Saved uncensored metrics to:", uncensored_metrics_path)
    print("Saved uncensored summary to:", uncensored_summary_path)

else:
    print("No censor column found. Skipping uncensored-only evaluation.")

# %% Cell 34
import matplotlib.pyplot as plt

for setting in interaction_predictions_df["setting"].unique():
    g = interaction_predictions_df[interaction_predictions_df["setting"] == setting].copy()

    plt.figure(figsize=(5, 5))
    plt.scatter(g["y_true"], g["y_pred"], alpha=0.3, s=10)

    min_val = min(g["y_true"].min(), g["y_pred"].min())
    max_val = max(g["y_true"].max(), g["y_pred"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Observed pKd")
    plt.ylabel("Predicted pKd")
    plt.title(f"Interaction-map model: {setting}")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# MJ model

# %% Cell 36
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SPLIT_DIR = Path("/content/drive/MyDrive/PDZ_DL/splits")

RESULT_DIR = Path("/content/drive/MyDrive/PDZ_DL/results/dl_interaction_map_mj")
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

# %% Cell 37
df = df.reset_index(drop=True).copy()

if "pair_id" not in df.columns:
    df["pair_id"] = np.arange(len(df))

if "pKd_label" in df.columns:
    TARGET_COL = "pKd_label"
elif "pKd" in df.columns:
    TARGET_COL = "pKd"
else:
    raise ValueError("Cannot find pKd_label or pKd column.")

if "is_censored_label" in df.columns:
    CENSOR_COL = "is_censored_label"
elif "is_censored" in df.columns:
    CENSOR_COL = "is_censored"
elif "mask" in df.columns:
    CENSOR_COL = "mask"
else:
    CENSOR_COL = None

if "pbm_sequence_6aa" not in df.columns:
    if "pbm_sequence_10aa" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence_10aa"].astype(str).str.upper().str[-6:]
    elif "prey_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["prey_sequence"].astype(str).str.upper().str[-6:]
    elif "pbm_sequence" in df.columns:
        df["pbm_sequence_6aa"] = df["pbm_sequence"].astype(str).str.upper().str[-6:]
    else:
        raise ValueError("Cannot find PBM sequence column.")

df["pdz_sequence"] = df["pdz_sequence"].astype(str).str.upper().str.strip()
df["pbm_sequence_6aa"] = df["pbm_sequence_6aa"].astype(str).str.upper().str.strip()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df.dropna(subset=["pdz_sequence", "pbm_sequence_6aa", TARGET_COL]).reset_index(drop=True)
df["pair_id"] = np.arange(len(df))

print("Data shape:", df.shape)
print("Target:", TARGET_COL)
print("Censor:", CENSOR_COL)
print("Unique PDZ:", df["pdz_sequence"].nunique())
print("Unique PBM6:", df["pbm_sequence_6aa"].nunique())

# %% Cell 38
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA)}
PAD_IDX = 0

MAX_PDZ_LEN = int(df["pdz_sequence"].str.len().quantile(0.99))
MAX_PDZ_LEN = max(MAX_PDZ_LEN, 80)

print("MAX_PDZ_LEN:", MAX_PDZ_LEN)


def encode_sequence(seq, max_len):
    seq = str(seq).upper()
    arr = np.zeros(max_len, dtype=np.int64)

    for i, aa in enumerate(seq[:max_len]):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr


def encode_pbm6(seq):
    seq = str(seq).upper()[-6:]
    arr = np.zeros(6, dtype=np.int64)

    for i, aa in enumerate(seq):
        arr[i] = AA_TO_IDX.get(aa, PAD_IDX)

    return arr

# %% Cell 39
class PDZPBMDataset(Dataset):
    def __init__(self, df_sub, target_col):
        self.df = df_sub.reset_index(drop=True).copy()
        self.target_col = target_col

        self.pdz_encoded = np.stack([
            encode_sequence(seq, MAX_PDZ_LEN)
            for seq in self.df["pdz_sequence"].values
        ])

        self.pbm_encoded = np.stack([
            encode_pbm6(seq)
            for seq in self.df["pbm_sequence_6aa"].values
        ])

        self.y = self.df[target_col].astype(float).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdz = torch.tensor(self.pdz_encoded[idx], dtype=torch.long)
        pbm = torch.tensor(self.pbm_encoded[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return {
            "pdz": pdz,
            "pbm": pbm,
            "y": y,
            "pair_id": int(self.df.loc[idx, "pair_id"])
        }


def make_loader(df_sub, shuffle=False):
    dataset = PDZPBMDataset(df_sub, TARGET_COL)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    return loader

# %% Cell 40
MJ_VALUES = {
    ("C","C"):-5.44, ("C","M"):-4.99, ("C","F"):-5.80, ("C","I"):-5.50, ("C","L"):-5.83,
    ("C","V"):-4.96, ("C","W"):-4.95, ("C","Y"):-4.16, ("C","A"):-3.57, ("C","G"):-3.16,
    ("C","T"):-3.11, ("C","S"):-2.86, ("C","N"):-2.59, ("C","Q"):-2.85, ("C","D"):-2.41,
    ("C","E"):-2.27, ("C","H"):-3.60, ("C","R"):-2.57, ("C","K"):-1.95, ("C","P"):-3.07,

    ("M","M"):-5.46, ("M","F"):-6.56, ("M","I"):-6.02, ("M","L"):-6.41, ("M","V"):-5.32,
    ("M","W"):-5.55, ("M","Y"):-4.91, ("M","A"):-3.94, ("M","G"):-3.39, ("M","T"):-3.51,
    ("M","S"):-3.03, ("M","N"):-2.95, ("M","Q"):-3.30, ("M","D"):-2.57, ("M","E"):-2.89,
    ("M","H"):-3.98, ("M","R"):-3.12, ("M","K"):-2.48, ("M","P"):-3.45,

    ("F","F"):-7.26, ("F","I"):-6.84, ("F","L"):-7.28, ("F","V"):-6.29, ("F","W"):-6.16,
    ("F","Y"):-5.66, ("F","A"):-4.81, ("F","G"):-4.13, ("F","T"):-4.28, ("F","S"):-4.02,
    ("F","N"):-3.75, ("F","Q"):-4.10, ("F","D"):-3.48, ("F","E"):-3.56, ("F","H"):-4.77,
    ("F","R"):-3.98, ("F","K"):-3.36, ("F","P"):-4.25,

    ("I","I"):-6.54, ("I","L"):-7.04, ("I","V"):-6.05, ("I","W"):-5.78, ("I","Y"):-5.25,
    ("I","A"):-4.58, ("I","G"):-3.78, ("I","T"):-4.03, ("I","S"):-3.52, ("I","N"):-3.24,
    ("I","Q"):-3.67, ("I","D"):-3.17, ("I","E"):-3.27, ("I","H"):-4.14, ("I","R"):-3.63,
    ("I","K"):-3.01, ("I","P"):-3.76,

    ("L","L"):-7.37, ("L","V"):-6.48, ("L","W"):-6.14, ("L","Y"):-5.67, ("L","A"):-4.91,
    ("L","G"):-4.16, ("L","T"):-4.34, ("L","S"):-3.92, ("L","N"):-3.74, ("L","Q"):-4.04,
    ("L","D"):-3.40, ("L","E"):-3.59, ("L","H"):-4.54, ("L","R"):-4.03, ("L","K"):-3.37,
    ("L","P"):-4.20,

    ("V","V"):-5.52, ("V","W"):-5.18, ("V","Y"):-4.62, ("V","A"):-4.04, ("V","G"):-3.38,
    ("V","T"):-3.46, ("V","S"):-3.05, ("V","N"):-2.83, ("V","Q"):-3.07, ("V","D"):-2.48,
    ("V","E"):-2.67, ("V","H"):-3.58, ("V","R"):-3.07, ("V","K"):-2.49, ("V","P"):-3.32,

    ("W","W"):-5.06, ("W","Y"):-4.66, ("W","A"):-3.82, ("W","G"):-3.42, ("W","T"):-3.22,
    ("W","S"):-2.99, ("W","N"):-3.07, ("W","Q"):-3.11, ("W","D"):-2.84, ("W","E"):-2.99,
    ("W","H"):-4.03, ("W","R"):-3.41, ("W","K"):-2.69, ("W","P"):-3.73,

    ("Y","Y"):-4.17, ("Y","A"):-3.36, ("Y","G"):-3.01, ("Y","T"):-3.01, ("Y","S"):-2.78,
    ("Y","N"):-2.76, ("Y","Q"):-2.97, ("Y","D"):-2.76, ("Y","E"):-2.79, ("Y","H"):-3.52,
    ("Y","R"):-3.16, ("Y","K"):-2.60, ("Y","P"):-3.19,

    ("A","A"):-2.72, ("A","G"):-2.31, ("A","T"):-2.32, ("A","S"):-2.01, ("A","N"):-1.84,
    ("A","Q"):-1.89, ("A","D"):-1.70, ("A","E"):-1.51, ("A","H"):-2.41, ("A","R"):-1.83,
    ("A","K"):-1.31, ("A","P"):-2.03,

    ("G","G"):-2.24, ("G","T"):-2.08, ("G","S"):-1.82, ("G","N"):-1.74, ("G","Q"):-1.66,
    ("G","D"):-1.59, ("G","E"):-1.22, ("G","H"):-2.15, ("G","R"):-1.72, ("G","K"):-1.15,
    ("G","P"):-1.87,

    ("T","T"):-2.12, ("T","S"):-1.96, ("T","N"):-1.88, ("T","Q"):-1.90, ("T","D"):-1.80,
    ("T","E"):-1.74, ("T","H"):-2.42, ("T","R"):-1.90, ("T","K"):-1.31, ("T","P"):-1.90,

    ("S","S"):-1.67, ("S","N"):-1.58, ("S","Q"):-1.49, ("S","D"):-1.63, ("S","E"):-1.48,
    ("S","H"):-2.11, ("S","R"):-1.62, ("S","K"):-1.05, ("S","P"):-1.57,

    ("N","N"):-1.68, ("N","Q"):-1.71, ("N","D"):-1.68, ("N","E"):-1.51, ("N","H"):-2.08,
    ("N","R"):-1.64, ("N","K"):-1.21, ("N","P"):-1.53,

    ("Q","Q"):-1.54, ("Q","D"):-1.46, ("Q","E"):-1.42, ("Q","H"):-1.98, ("Q","R"):-1.80,
    ("Q","K"):-1.29, ("Q","P"):-1.73,

    ("D","D"):-1.21, ("D","E"):-1.02, ("D","H"):-2.32, ("D","R"):-2.29, ("D","K"):-1.68,
    ("D","P"):-1.33,

    ("E","E"):-0.91, ("E","H"):-2.15, ("E","R"):-2.27, ("E","K"):-1.80, ("E","P"):-1.26,

    ("H","H"):-3.05, ("H","R"):-2.16, ("H","K"):-1.35, ("H","P"):-2.25,

    ("R","R"):-1.55, ("R","K"):-1.59, ("R","P"):-1.70,

    ("K","K"):-0.12, ("K","P"):-0.97,

    ("P","P"):-1.75,
}


def build_mj_matrix():
    mat = torch.zeros((21, 21), dtype=torch.float32)

    values = []

    for (a, b), v in MJ_VALUES.items():
        ia = AA_TO_IDX[a]
        ib = AA_TO_IDX[b]

        mat[ia, ib] = v
        mat[ib, ia] = v

        values.append(v)

    values = torch.tensor(values, dtype=torch.float32)
    mean = values.mean()
    std = values.std()

    for i in range(1, 21):
        for j in range(1, 21):
            if mat[i, j] != 0:
                mat[i, j] = (mat[i, j] - mean) / std

    return mat


MJ_MATRIX = build_mj_matrix()
print("MJ matrix shape:", MJ_MATRIX.shape)

# %% Cell 41
class InteractionMapMJModel(nn.Module):
    def __init__(
        self,
        vocab_size=21,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.25
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        self.pdz_proj = nn.Linear(embed_dim, hidden_dim)
        self.pbm_proj = nn.Linear(embed_dim, hidden_dim)

        self.register_buffer("mj_matrix", MJ_MATRIX)

        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.pdz_seq_cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1)
        )

        self.pbm_mlp = nn.Sequential(
            nn.Linear(6 * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, pdz, pbm):
        pdz_emb = self.embedding(pdz)
        pbm_emb = self.embedding(pbm)

        pdz_h = self.pdz_proj(pdz_emb)
        pbm_h = self.pbm_proj(pbm_emb)

        learned_map = torch.einsum("blh,bmh->blm", pdz_h, pbm_h)
        learned_map = learned_map / np.sqrt(pdz_h.size(-1))

        mj_map = self.mj_matrix[pdz.unsqueeze(-1), pbm.unsqueeze(1)]

        pdz_mask = (pdz != PAD_IDX).float().unsqueeze(-1)
        pbm_mask = (pbm != PAD_IDX).float().unsqueeze(1)
        mask = pdz_mask * pbm_mask

        learned_map = learned_map * mask
        mj_map = mj_map * mask

        interaction_maps = torch.stack([learned_map, mj_map], dim=1)

        map_feat = self.map_cnn(interaction_maps).squeeze(-1).squeeze(-1)

        pdz_emb_cnn = pdz_emb.transpose(1, 2)
        pdz_seq_feat = self.pdz_seq_cnn(pdz_emb_cnn).squeeze(-1)

        pbm_flat = pbm_emb.reshape(pbm_emb.size(0), -1)
        pbm_feat = self.pbm_mlp(pbm_flat)

        feat = torch.cat([map_feat, pdz_seq_feat, pbm_feat], dim=1)
        pred = self.regressor(feat).squeeze(-1)

        return pred

# %% Cell 42
def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan, np.nan

    return pearsonr(y_true, y_pred)[0], spearmanr(y_true, y_pred)[0]


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, spearman = safe_corr(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }


@torch.no_grad()
def predict_model(model, loader):
    model.eval()

    all_y = []
    all_pred = []
    all_pair_ids = []

    for batch in loader:
        pdz = batch["pdz"].to(DEVICE)
        pbm = batch["pbm"].to(DEVICE)

        y = batch["y"].cpu().numpy()
        pair_id = batch["pair_id"].cpu().numpy()

        pred = model(pdz, pbm).detach().cpu().numpy()

        all_y.append(y)
        all_pred.append(pred)
        all_pair_ids.append(pair_id)

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_pair_ids = np.concatenate(all_pair_ids)

    return all_y, all_pred, all_pair_ids

# %% Cell 43
def train_one_mj_model(train_df, val_df, setting, fold):
    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    model = InteractionMapMJModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss()

    best_val_rmse = np.inf
    best_epoch = -1
    patience_counter = 0

    best_path = MODEL_DIR / f"interaction_map_mj_{setting}_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            pdz = batch["pdz"].to(DEVICE)
            pbm = batch["pbm"].to(DEVICE)
            y = batch["y"].to(DEVICE)

            optimizer.zero_grad()

            pred = model(pdz, pbm)
            loss = loss_fn(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())

        y_val, pred_val, _ = predict_model(model, val_loader)
        val_metrics = evaluate_regression(y_val, pred_val)
        val_rmse = val_metrics["rmse"]

        mean_train_loss = float(np.mean(train_losses))

        print(
            f"[MJ {setting} fold {fold}] "
            f"Epoch {epoch:03d} | "
            f"Train loss {mean_train_loss:.4f} | "
            f"Val RMSE {val_rmse:.4f} | "
            f"Val Pearson {val_metrics['pearson']:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    return model, best_epoch, best_val_rmse

# %% Cell 44
SPLIT_FILES = {
    "random": SPLIT_DIR / "random_split.csv",
    "pdz_heldout": SPLIT_DIR / "pdz_heldout_split.csv",
    "pbm_heldout": SPLIT_DIR / "pbm_heldout_split.csv",
}

# 如果只想先测试 random，就用下面这个：
# SPLIT_FILES = {
#     "random": SPLIT_DIR / "random_split.csv",
# }

mj_all_metrics = []
mj_all_predictions = []

for setting, split_path in SPLIT_FILES.items():
    split_df = pd.read_csv(split_path)

    print(f"\n==============================")
    print(f"Running MJ setting: {setting}")
    print(f"==============================")

    for fold in sorted(split_df["fold"].unique()):
        fold_split = split_df[split_df["fold"] == fold]

        train_ids = fold_split[fold_split["split"] == "train"]["pair_id"].values
        val_ids = fold_split[fold_split["split"] == "val"]["pair_id"].values
        test_ids = fold_split[fold_split["split"] == "test"]["pair_id"].values

        train_df = df[df["pair_id"].isin(train_ids)].copy()
        val_df = df[df["pair_id"].isin(val_ids)].copy()
        test_df = df[df["pair_id"].isin(test_ids)].copy()

        print(f"\nMJ setting {setting}, fold {fold}")
        print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

        model, best_epoch, best_val_rmse = train_one_mj_model(
            train_df=train_df,
            val_df=val_df,
            setting=setting,
            fold=fold
        )

        test_loader = make_loader(test_df, shuffle=False)
        y_test, y_pred, pair_ids = predict_model(model, test_loader)

        metrics = evaluate_regression(y_test, y_pred)
        metrics.update({
            "setting": setting,
            "fold": fold,
            "model": "interaction_map_mj",
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df)
        })

        mj_all_metrics.append(metrics)

        pred_df = pd.DataFrame({
            "setting": setting,
            "fold": fold,
            "model": "interaction_map_mj",
            "pair_id": pair_ids,
            "y_true": y_test,
            "y_pred": y_pred
        })

        meta_cols = ["pair_id", "pdz_sequence", "pbm_sequence_6aa"]
        if CENSOR_COL is not None and CENSOR_COL in df.columns:
            meta_cols.append(CENSOR_COL)

        pred_df = pred_df.merge(
            df[meta_cols],
            on="pair_id",
            how="left"
        )

        mj_all_predictions.append(pred_df)

        print("MJ test metrics:", metrics)

# %% Cell 45
mj_metrics_df = pd.DataFrame(mj_all_metrics)
mj_predictions_df = pd.concat(mj_all_predictions, axis=0).reset_index(drop=True)

metrics_path = RESULT_DIR / "interaction_map_mj_metrics.csv"
pred_path = RESULT_DIR / "interaction_map_mj_predictions.csv"

mj_metrics_df.to_csv(metrics_path, index=False)
mj_predictions_df.to_csv(pred_path, index=False)

print("Saved MJ metrics to:", metrics_path)
print("Saved MJ predictions to:", pred_path)

display(mj_metrics_df)
display(mj_predictions_df.head())

# %% Cell 46
mj_summary = (
    mj_metrics_df
    .groupby(["setting", "model"])
    [["rmse", "mae", "r2", "pearson", "spearman"]]
    .agg(["mean", "std"])
)

summary_path = RESULT_DIR / "interaction_map_mj_summary.csv"
mj_summary.to_csv(summary_path)

display(mj_summary)
print("Saved MJ summary to:", summary_path)

# %% Cell 47
if CENSOR_COL is not None and CENSOR_COL in mj_predictions_df.columns:
    mj_uncensored_pred = mj_predictions_df[
        mj_predictions_df[CENSOR_COL] == 0
    ].copy()

    mj_uncensored_metrics = []

    for (setting, fold, model), g in mj_uncensored_pred.groupby(["setting", "fold", "model"]):
        if len(g) < 5:
            continue

        m = evaluate_regression(g["y_true"].values, g["y_pred"].values)
        m.update({
            "setting": setting,
            "fold": fold,
            "model": model,
            "n_test_uncensored": len(g)
        })

        mj_uncensored_metrics.append(m)

    mj_uncensored_metrics_df = pd.DataFrame(mj_uncensored_metrics)

    uncensored_metrics_path = RESULT_DIR / "interaction_map_mj_metrics_uncensored_only.csv"
    mj_uncensored_metrics_df.to_csv(uncensored_metrics_path, index=False)

    mj_uncensored_summary = (
        mj_uncensored_metrics_df
        .groupby(["setting", "model"])
        [["rmse", "mae", "r2", "pearson", "spearman"]]
        .agg(["mean", "std"])
    )

    uncensored_summary_path = RESULT_DIR / "interaction_map_mj_summary_uncensored_only.csv"
    mj_uncensored_summary.to_csv(uncensored_summary_path)

    display(mj_uncensored_summary)
    print("Saved MJ uncensored metrics to:", uncensored_metrics_path)
    print("Saved MJ uncensored summary to:", uncensored_summary_path)

else:
    print("No censor column found. Skipping MJ uncensored-only evaluation.")

# %% Cell 48
import matplotlib.pyplot as plt

for setting in mj_predictions_df["setting"].unique():
    g = mj_predictions_df[mj_predictions_df["setting"] == setting].copy()

    plt.figure(figsize=(5, 5))
    plt.scatter(g["y_true"], g["y_pred"], alpha=0.3, s=10)

    min_val = min(g["y_true"].min(), g["y_pred"].min())
    max_val = max(g["y_true"].max(), g["y_pred"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Observed pKd")
    plt.ylabel("Predicted pKd")
    plt.title(f"Interaction-map + MJ model: {setting}")

    plt.tight_layout()
    plt.show()
