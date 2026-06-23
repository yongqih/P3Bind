# P3Bind

**P3Bind** is an interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction and PBM motif design.

Created and developed by **Yongqi Huang**.

This repository is organized as a reproducible Python/PyCharm project, not as a web app. It contains reusable model code, command-line scripts, cleaned notebooks, and instructions for reproducing training, prediction, specificity profiling, and figure generation.

## Repository structure

```text
P3Bind/
├── src/p3bind/                 # importable Python package
│   ├── model.py                # model architecture, sequence encoding, ensemble loading
│   └── core.py                 # prediction, batch prediction, specificity helpers
├── scripts/                    # PyCharm/terminal runnable scripts
│   ├── 00_check_setup.py
│   ├── 01_predict_single.py
│   ├── 02_batch_predict.py
│   ├── 03_specificity_profile.py
│   ├── 04_prepare_background_pdz.py
│   └── notebook_exports/       # code exported from original notebooks
├── notebooks/                  # cleaned notebooks with outputs removed
├── data/
│   ├── raw/                    # place raw training data here, not committed by default
│   ├── processed/              # processed files such as background_pdz.csv
│   └── splits/                 # train/validation/test split CSVs
├── checkpoints/design_models/  # final P3Bind .pth weights for inference
├── results/                    # generated predictions/figures
├── examples/                   # small example inputs
└── docs/                       # reproducibility notes
```

## Quick start in PyCharm

1. Clone the repository.
2. Open the folder in PyCharm.
3. Create a Python 3.10 environment.
4. Install the package in editable mode:

```bash
pip install -e .
```

5. Check the setup:

```bash
python scripts/00_check_setup.py
```

## Required files for exact inference reproduction

The GitHub repository can contain code only, but to run real P3Bind inference you need:

```text
checkpoints/design_models/best_model_fold_0_design_m.pth
checkpoints/design_models/best_model_fold_1_design_m.pth
checkpoints/design_models/best_model_fold_2_design_m.pth
checkpoints/design_models/best_model_fold_3_design_m.pth
checkpoints/design_models/best_model_fold_4_design_m.pth

data/processed/background_pdz.csv
```

`background_pdz.csv` must contain:

```csv
pdz_id,pdz_sequence
LAP2_PDZ_1,EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV
```

If model weights are not committed, provide a release download link or put them in `checkpoints/design_models/` manually after cloning.

## Run a single prediction

```bash
python scripts/01_predict_single.py   --pdz-seq EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV   --pbm YRETRV
```

## Run batch prediction

Input CSV must contain `pdz_sequence` and `pbm6` columns.

```bash
python scripts/02_batch_predict.py   --input-csv examples/example_pairs.csv   --output-csv results/predictions/example_predictions.csv
```

## Run specificity/off-target profiling

```bash
python scripts/03_specificity_profile.py   --target-pdz-seq EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV   --pbm YRETRV   --background-csv data/processed/background_pdz.csv
```

## Reproducing notebook analyses

Cleaned notebooks are stored in `notebooks/`. Their exported Python versions are in `scripts/notebook_exports/`.

The original notebooks used Google Drive absolute paths. For local/PyCharm reproduction, update these paths to the local `data/`, `checkpoints/`, and `results/` folders. See `docs/REPRODUCIBILITY.md`.

## Citation

If you use this code or model predictions, please cite or acknowledge:

> P3Bind: An interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction and motif design. Yongqi Huang.
