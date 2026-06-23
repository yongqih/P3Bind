
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
PBM_POSITIONS = ["P-5", "P-4", "P-3", "P-2", "P-1", "P0"]


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_processed_csv(path_or_name: str | Path) -> Path:
    """Resolve a processed CSV, falling back to a .example.csv for smoke tests."""
    path = Path(path_or_name)
    if not path.is_absolute():
        path = PROCESSED_DIR / path
    if path.exists():
        return path

    example = path.with_name(path.stem + ".example" + path.suffix)
    if example.exists():
        print(
            f"[warning] {path.name} was not found. Using {example.name} instead. "
            "Replace example files with final manuscript result tables for exact reproduction.",
            file=sys.stderr,
        )
        return example

    raise FileNotFoundError(
        f"Could not find {path} or {example}. See docs/FIGURE_REPRODUCTION.md for expected schemas."
    )


def read_csv(path_or_name: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve_processed_csv(path_or_name))


def save_current_figure(filename: str, dpi: int = 300) -> Path:
    ensure_output_dirs()
    out = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {out}")
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of the expected columns were found: {candidates}. Available: {list(df.columns)}")
    return None


def normalize_position_labels(values):
    """Return labels ordered as P-5...P0 when possible."""
    mapping = {
        "-5": "P-5", "-4": "P-4", "-3": "P-3", "-2": "P-2", "-1": "P-1", "0": "P0",
        "1": "P-5", "2": "P-4", "3": "P-3", "4": "P-2", "5": "P-1", "6": "P0",
        "P-5": "P-5", "P-4": "P-4", "P-3": "P-3", "P-2": "P-2", "P-1": "P-1", "P0": "P0",
        "p-5": "P-5", "p-4": "P-4", "p-3": "P-3", "p-2": "P-2", "p-1": "P-1", "p0": "P0",
    }
    return [mapping.get(str(v), str(v)) for v in values]


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")


def plot_heatmap(ax, matrix, x_labels, y_labels, title, colorbar_label="value", cmap="coolwarm"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    return im


def metric_table_to_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Normalize performance table to model, split, mean, std columns for one metric."""
    metric = metric.lower()
    if {"model", "split", "metric", "mean"}.issubset(df.columns):
        out = df[df["metric"].astype(str).str.lower() == metric].copy()
        if len(out) == 0:
            raise ValueError(f"Metric {metric} not found in long-format table.")
        if "std" not in out.columns:
            out["std"] = 0.0
        return out[["model", "split", "mean", "std"]]
    if {"model", "split", "metric", "value"}.issubset(df.columns):
        out = df[df["metric"].astype(str).str.lower() == metric].copy()
        if len(out) == 0:
            raise ValueError(f"Metric {metric} not found in long-format table.")
        out = out.rename(columns={"value": "mean"})
        if "std" not in out.columns:
            out["std"] = 0.0
        return out[["model", "split", "mean", "std"]]
    mean_col = metric
    std_col = f"{metric}_std"
    if f"{metric}_mean" in df.columns:
        mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        raise ValueError(f"Could not find columns for metric {metric}. Available: {list(df.columns)}")
    out = df[["model", "split", mean_col]].rename(columns={mean_col: "mean"}).copy()
    out["std"] = df[std_col] if std_col in df.columns else 0.0
    return out


def grouped_metric_bars(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str):
    mdf = metric_table_to_wide(df, metric)
    splits = list(dict.fromkeys(mdf["split"].astype(str).tolist()))
    models = list(dict.fromkeys(mdf["model"].astype(str).tolist()))
    x = np.arange(len(splits))
    width = min(0.8 / max(1, len(models)), 0.18)
    offsets = (np.arange(len(models)) - (len(models)-1)/2) * width
    for i, model in enumerate(models):
        sub = mdf[mdf["model"].astype(str) == model]
        means = []
        stds = []
        for split in splits:
            row = sub[sub["split"].astype(str) == split]
            if len(row):
                means.append(float(row["mean"].iloc[0])); stds.append(float(row["std"].iloc[0]))
            else:
                means.append(np.nan); stds.append(0.0)
        ax.bar(x + offsets[i], means, width, yerr=stds, label=model, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return ax
