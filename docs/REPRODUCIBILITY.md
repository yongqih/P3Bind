# Reproducibility guide

## Scope

This repository implements the analyses explicitly used in the manuscript body. ESM is excluded because it was an internal ablation. Files under `notebooks/` and `scripts/notebook_exports/` preserve historical Colab provenance and may contain `/content/drive` paths; use the curated local modules and scripts instead.

## Release validation

```powershell
python scripts/00_check_setup.py
python -m pytest -q
```

The setup report checks Python/PyTorch/CUDA, the five checkpoints, 45,595 dataset rows, 260 PDZ sequences, 424 PBM6 sequences, censor counts, five folds per setting, and held-out group isolation.

The measurement table is aggregated by PDZ sequence and assayed 10-aa peptide. Different 10-aa peptides can share the same terminal PBM6; those are not duplicate observations.

## Benchmark protocol

All models use the committed fold assignments:

- random split
- PBM6-held-out split
- PDZ-sequence-held-out split

Traditional baselines fit on train+validation and evaluate on test. Neural models select the checkpoint with the lowest validation RMSE and evaluate the untouched test set. Metrics are RMSE, MAE, R², Pearson correlation, and Spearman correlation.

```powershell
python scripts/analysis/run_baseline_benchmarks.py
python scripts/train/run_neural_benchmarks.py --device cuda
```

## Design ensemble

The committed five-model ensemble is checkpoint-compatible with the Colab design architecture. Its directional loss uses threshold 4.5, base positive weight 3.0, bad-direction multiplier 3.0, relaxed-direction multiplier 0.3, and a one-sided censored loss above the 3.1 label.

```powershell
python scripts/train/train_design_ensemble.py --device cuda
```

Newly trained checkpoints are written to `results/training/design_models/`; the committed inference weights are not overwritten.

## Downstream workflows

- `scripts/analysis/run_motif_landscape.py`: shared random PBM6 library, top 1%, pseudocount-smoothed log2 enrichment, 120-feature PDZ profiles.
- `scripts/analysis/run_design_demo.py`: target-minus-background specificity objective, random initialization, simulated annealing, and local single-mutant refinement; terminal residue constrained to L/I/V/F/C.
- `scripts/analysis/run_variant_effect_analysis.py`: WT/MUT PBM6 scores across the PDZ panel and ΔpKd summaries.

## PyCharm

Select the project virtual environment and run scripts with the repository root as working directory. No Google Drive mount, notebook magic, or Colab-only package installation is used. Relative defaults are anchored to `REPO_ROOT`, making the scripts robust to PyCharm run configurations.
