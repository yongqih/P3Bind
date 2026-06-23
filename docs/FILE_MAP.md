# File map

## Core package

- `src/p3bind/model.py`: P3Bind model architecture, amino-acid encoding, ensemble checkpoint loading, and ensemble pKd prediction helpers.
- `src/p3bind/core.py`: reusable APIs for single prediction, batch prediction, specificity/off-target profiling, and PBM single-mutant scans.

## Curated runnable scripts

- `scripts/00_check_setup.py`: verifies Python, PyTorch, CUDA status, and expected repository paths.
- `scripts/01_predict_single.py`: command-line inference for one PDZ–PBM pair.
- `scripts/02_batch_predict.py`: batch inference from a CSV containing `pdz_sequence` and `pbm6`.
- `scripts/03_specificity_profile.py`: target/background PDZ specificity profiling.
- `scripts/04_mutation_scan.py`: PBM6 single-mutant scan, optionally with background specificity deltas.
- `scripts/05_prepare_background_pdz.py`: utility for preparing `background_pdz.csv`.

## Manuscript figure scripts

- `results/figures/source/`: source/final artwork for schematic Figure 1 and Figure 3.
- `scripts/figures/plot_fig2_dataset_characterization.py`: Figure 2 dataset construction and characterization.
- `scripts/figures/plot_fig4_model_performance.py`: Figure 4 benchmark/model performance comparison.
- `scripts/figures/plot_fig5_motif_landscape.py`: Figure 5 motif preference landscape.
- `scripts/figures/plot_fig6_pbm_design.py`: Figure 6 PDZ-specific PBM design demonstration.
- `scripts/figures/plot_fig7_gnomad_variants.py`: Figure 7 gnomAD-P3Bind PBM variant-effect analysis.
- `scripts/figures/run_all_figures.py`: runs all computational figure scripts.

## Notebooks and exported code

- `notebooks/`: cleaned notebooks with outputs removed.
- `scripts/notebook_exports/`: code exported from the original notebooks for transparency and full analysis reconstruction.

## Data/checkpoints/results

- `data/raw/`: optional raw training data; not committed by default.
- `data/processed/`: background PDZ table and processed manuscript result tables.
- `data/splits/`: optional train/validation/test split files.
- `checkpoints/design_models/`: final inference-ready `.pth` checkpoints.
- `results/predictions/`: generated prediction CSVs.
- `results/figures/`: generated manuscript figures.
- `results/tables/`: generated supplemental/top-hit tables.
