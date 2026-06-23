# Reproducibility guide

This repository is intended to be opened directly in PyCharm and run as a normal Python project.

## 1. Files required for inference reproduction

Place final inference-ready checkpoints here:

```text
checkpoints/design_models/best_model_fold_0_design_m.pth
checkpoints/design_models/best_model_fold_1_design_m.pth
checkpoints/design_models/best_model_fold_2_design_m.pth
checkpoints/design_models/best_model_fold_3_design_m.pth
checkpoints/design_models/best_model_fold_4_design_m.pth
```

Place the background PDZ set here:

```text
data/processed/background_pdz.csv
```

Required columns:

```text
pdz_id,pdz_sequence
```

Then run:

```bash
python scripts/00_check_setup.py
python scripts/01_predict_single.py --pdz-seq <PDZ_SEQUENCE> --pbm YRETRV
python scripts/02_batch_predict.py --input-csv examples/example_pairs.csv
python scripts/03_specificity_profile.py --target-pdz-seq <PDZ_SEQUENCE> --pbm YRETRV --background-csv data/processed/background_pdz.csv
```

## 2. Files required for figure reproduction

The final manuscript contains seven figures. Figure 1 and Figure 3 are conceptual schematics and should be archived under `results/figures/source/`. The remaining computational figures can be regenerated from processed CSV files under `data/processed/`.

For final paper-code release, place these processed manuscript result tables under `data/processed/`:

```text
data/processed/pair_level_dataset.csv                     # Figure 2
data/processed/benchmark_results.csv                      # Figure 4
data/processed/motif_enrichment_matrix.csv                # Figure 5A
data/processed/motif_position_importance.csv              # Figure 5B
data/processed/motif_cluster_matrix.csv                   # Figure 5C
data/processed/representative_motif_profiles.csv          # Figure 5D
data/processed/design_candidates.csv                      # Figure 6B/D
data/processed/optimization_trajectory.csv                # Figure 6C
data/processed/pbm_variant_effects.csv                    # Figure 7B/C/F
data/processed/variant_pdz_delta_matrix.csv               # Figure 7D
data/processed/representative_variant_profile.csv         # Figure 7E
```

Then run:

```bash
python scripts/figures/run_all_figures.py
```

The scripts save outputs to:

```text
results/figures/
results/tables/
```

See `docs/FIGURE_REPRODUCTION.md` for expected CSV schemas.

## 3. Files required for full training/benchmark reproduction

Full model retraining is optional for the public release. If full retraining is desired, place the raw/aggregated affinity table and split files here:

```text
data/raw/all_data.csv                         # optional raw table used by early benchmark notebook
data/raw/all_data_pair_aggregated.csv         # main aggregated PDZ-PBM affinity table
data/splits/random_split.csv                  # split file used by training/design notebooks
```

The cleaned notebooks in `notebooks/` and exported scripts in `scripts/notebook_exports/` preserve the original analysis workflow. Some exported notebook scripts may still contain Colab/Google Drive paths and should be updated to use local paths:

```python
DATA_PATH = Path("data/raw/all_data_pair_aggregated.csv")
SPLIT_PATH = Path("data/splits/random_split.csv")
OUT_DIR = Path("checkpoints/design_models")
RESULT_DIR = Path("results")
```

## 4. Checkpoint policy

Commit only final inference-ready checkpoints if the repository is intended to be runnable by others. Do not commit intermediate training checkpoints unless needed for a specific reproduction claim.

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
local virtual environments
```

Use Git LFS for `.pth`, `.pt`, and `.ckpt` files.
