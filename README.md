# P3Bind

P3Bind is an interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction, PBM candidate prioritization, and natural PBM variant-effect analysis.

This repository is intended as the reproducible code release for the P3Bind manuscript. It provides the model architecture, inference utilities, training/analysis scripts, cleaned notebooks, fixed data splits, and final inference checkpoints required to reproduce the main computational workflow in a local Python/PyCharm environment.

## Repository scope

This repository provides only the release inputs needed for reproducibility:

- `data/raw/all_data_pair_aggregated.csv`
- `data/processed/background_pdz.csv`
- `data/splits/*.csv`
- `checkpoints/design_models/best_model_fold_*_design_m.pth`
- source code, scripts, notebooks, and documentation

Intermediate result tables, generated figures, temporary training outputs, and cache files are intentionally not included. Users can regenerate those outputs from the provided data, split files, checkpoints, and scripts.

## Repository structure

```text
P3Bind/
├── src/p3bind/                  # importable P3Bind package
│   ├── model.py                 # model architecture and sequence encoding
│   ├── core.py                  # prediction, specificity, batch scoring utilities
│   └── __init__.py
├── scripts/
│   ├── 00_check_setup.py
│   ├── 01_predict_single.py
│   ├── 02_batch_predict.py
│   ├── 03_specificity_profile.py
│   ├── 04_mutation_scan.py
│   ├── analysis/                # scripts to regenerate processed result tables
│   ├── figures/                 # scripts to regenerate manuscript figures
│   └── train/                   # training and benchmark scripts
├── notebooks/                   # cleaned notebooks for transparency
├── data/
│   ├── raw/
│   │   └── all_data_pair_aggregated.csv
│   ├── processed/
│   │   └── background_pdz.csv
│   └── splits/
│       └── *.csv                # fixed random, PBM-heldout, and PDZ-heldout split files
├── checkpoints/
│   └── design_models/
│       └── best_model_fold_*_design_m.pth
├── results/
│   ├── tables/                  # generated tables, not tracked by Git
│   └── figures/                 # generated figures, not tracked by Git
├── docs/
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Installation

Clone the repository and install the package in editable mode.

```bash
git clone https://github.com/yongqih/P3Bind.git
cd P3Bind
pip install -e .
```

Alternatively, install dependencies directly:

```bash
pip install -r requirements.txt
```

Python 3.10 is recommended.

## Required release files

The following files are expected to be present for full local reproducibility:

```text
data/raw/all_data_pair_aggregated.csv
data/processed/background_pdz.csv
data/splits/*.csv
checkpoints/design_models/best_model_fold_0_design_m.pth
checkpoints/design_models/best_model_fold_1_design_m.pth
checkpoints/design_models/best_model_fold_2_design_m.pth
checkpoints/design_models/best_model_fold_3_design_m.pth
checkpoints/design_models/best_model_fold_4_design_m.pth
```

The five `.pth` files are final inference-ready design-ensemble checkpoints. Intermediate training checkpoints are not required.

## Quick start

Check the installation:

```bash
python scripts/00_check_setup.py
```

Run single-pair prediction:

```bash
python scripts/01_predict_single.py \
  --pdz-seq EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV \
  --pbm YRETRV
```

Run batch prediction:

```bash
python scripts/02_batch_predict.py --input-csv examples/example_pairs.csv
```

Run specificity / off-target profiling:

```bash
python scripts/03_specificity_profile.py \
  --target-pdz-seq EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV \
  --pbm YRETRV \
  --background-csv data/processed/background_pdz.csv
```

Run PBM single-mutant scan:

```bash
python scripts/04_mutation_scan.py \
  --target-pdz-seq EIRVRVEKDPELGFSISGGVGGRGNPFRPDDDGIFVTRVQPEGPASKLLQPGDKIIQANGYSFINIEHGQAVSLLKTFQNTVELIIVREV \
  --pbm YRETRV \
  --background-csv data/processed/background_pdz.csv
```

Generated tables are written to `results/tables/`.

## Reproducing manuscript analyses and figures

This repository does not include precomputed intermediate figure tables. Instead, users can regenerate the processed result tables and manuscript figures from:

- `data/raw/all_data_pair_aggregated.csv`
- `data/processed/background_pdz.csv`
- `data/splits/*.csv`
- `checkpoints/design_models/*.pth`

The fixed split files should be used for all benchmark comparisons to match the manuscript evaluation settings.

A typical reproduction workflow is:

```bash
python scripts/00_check_setup.py

# Regenerate analysis tables
python scripts/analysis/prepare_dataset_summary.py
python scripts/analysis/run_benchmark_models.py
python scripts/analysis/run_motif_landscape.py
python scripts/analysis/run_design_demo.py
python scripts/analysis/run_variant_effect_analysis.py

# Regenerate figures from generated tables
python scripts/figures/run_all_figures.py
```

Generated intermediate tables will be saved to:

```text
results/tables/
```

Generated figures will be saved to:

```text
results/figures/
```

Full benchmark reproduction may take substantial time because baseline and neural models are retrained from the aggregated dataset using the provided split files. Direct inference and design-related analyses can be run from the provided final ensemble checkpoints without retraining.

## Data and checkpoint policy

Included:

- Aggregated ProfAff-derived pair-level dataset
- Background PDZ panel used for specificity scoring
- Fixed train/test split files
- Final design-ensemble checkpoints
- Source code and cleaned notebooks

Not included:

- Precomputed manuscript figure tables
- Generated figures
- Temporary training logs
- Intermediate model checkpoints
- Cache files
- Local environment files

This keeps the repository compact while allowing users to regenerate analysis outputs from the released inputs.

## Citation

If you use P3Bind or results generated from this repository, please cite the P3Bind manuscript and this GitHub repository.

```text
P3Bind: An interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction and motif design
```

## Author

Yongqi Huang  
University of Pennsylvania  
Contact: yongqi@seas.upenn.edu
