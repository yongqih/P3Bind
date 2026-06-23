# Reproducibility guide

This repository is intended to be opened directly in PyCharm and run as a normal Python project.

## 1. Files that must be supplied by the user

For **inference reproduction**:

```text
checkpoints/design_models/best_model_fold_0_design_m.pth
checkpoints/design_models/best_model_fold_1_design_m.pth
checkpoints/design_models/best_model_fold_2_design_m.pth
checkpoints/design_models/best_model_fold_3_design_m.pth
checkpoints/design_models/best_model_fold_4_design_m.pth
data/processed/background_pdz.csv
```

For **training/benchmark reproduction**:

```text
data/raw/all_data.csv                         # optional raw table used by early benchmark notebook
data/raw/all_data_pair_aggregated.csv         # main aggregated PDZ-PBM affinity table
data/splits/random_split.csv                  # split file used by training/design notebooks
```

For **gnomAD analyses**, the notebooks reference additional files originally stored in Google Drive under `gnomad_pbm_outputs/`. Put those under:

```text
data/processed/gnomad/
results/predictions/gnomad/
```

## 2. Paths to change in exported notebook scripts

The exported scripts in `scripts/notebook_exports/` are faithful exports from the notebooks. They may still contain Colab paths. Replace these path variables before running them:

```python
DATA_PATH = Path("data/raw/all_data_pair_aggregated.csv")
SPLIT_PATH = Path("data/splits/random_split.csv")
OUT_DIR = Path("checkpoints/design_models")
RESULT_DIR = Path("results")
BEST_MOTIF_DIR = Path("results/best_all_pdz_100k_unrestricted_motif_preference")
PRED_CACHE = Path("results/predictions/all_wt_mut_pbm6_vs_pdz_predictions.csv")
```

The curated scripts in `scripts/00_check_setup.py` through `scripts/04_prepare_background_pdz.py` do not require editing paths; paths can be passed as command-line arguments.

## 3. Checkpoint policy

Commit only final inference-ready checkpoints if the repository is intended to be runnable by others. Do not commit intermediate training checkpoints unless needed for reproducibility.

Recommended committed files:

```text
checkpoints/design_models/best_model_fold_0_design_m.pth
...
checkpoints/design_models/best_model_fold_4_design_m.pth
```

Do not commit:

```text
.ipynb_checkpoints/
wandb/
training cache files
large raw private datasets
temporary checkpoints
```

If checkpoint files are large, use Git LFS.
