# P3Bind

P3Bind is an interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction, motif preference analysis, specificity-aware PBM design, and natural PBM variant-effect analysis.

This release is a standard Python project intended to run locally in PyCharm or from a terminal. The historical Colab notebooks are retained only as provenance snapshots; Google Drive mounting and notebook state are not required. ESM experiments were internal ablations and are intentionally outside this main-text release.

## Install and open in PyCharm

Use Python 3.10–3.13. On Windows, a short project/interpreter path is recommended because PyTorch contains deeply nested package paths.

```powershell
git clone https://github.com/yongqih/P3Bind.git
cd P3Bind
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/00_check_setup.py
```

In PyCharm, open the repository root, select `.venv\Scripts\python.exe`, and keep the working directory at the repository root. All curated scripts resolve data and checkpoints relative to the repository, so they also work when launched from another working directory.

## Included release assets

- `data/raw/all_data_pair_aggregated.csv`: 45,595 aggregated PDZ–10-aa-peptide measurements; model features use the terminal PBM6.
- `data/splits/{random,pbm_heldout,pdz_heldout}_split.csv`: fixed five-fold assignments.
- `data/processed/background_pdz.csv`: 260-sequence PDZ specificity panel.
- `checkpoints/design_models/best_model_fold_*_design_m.pth`: five-model design ensemble.
- `src/p3bind/`: importable inference, benchmark, training, design, motif, variant, and validation APIs.

Generated outputs under `results/` are ignored by Git.

## Inference

```powershell
python scripts/01_predict_single.py --pdz-seq <PDZ_SEQUENCE> --pbm YRETRV
python scripts/02_batch_predict.py --input-csv examples/example_pairs.csv
python scripts/03_specificity_profile.py --target-pdz-seq <PDZ_SEQUENCE> --pbm YRETRV
python scripts/04_mutation_scan.py --target-pdz-seq <PDZ_SEQUENCE> --pbm YRETRV
```

Batch input accepts `pdz_sequence` plus any one of `pbm6`, `pbm_sequence_10aa`, or `peptide`.

## Main-text reproduction

Traditional baselines:

```powershell
python scripts/analysis/run_baseline_benchmarks.py
```

CNN-concat, learned interaction-map, and interaction-map+MJ benchmarks (GPU recommended):

```powershell
python scripts/train/run_neural_benchmarks.py --device cuda
```

Design ensemble training (writes new checkpoints under `results/training/`):

```powershell
python scripts/train/train_design_ensemble.py --device cuda
```

Downstream analyses:

```powershell
python scripts/analysis/run_motif_landscape.py --device cuda
python scripts/analysis/run_design_demo.py --target-pdz-seq <PDZ_SEQUENCE>
python scripts/analysis/run_variant_effect_analysis.py --variants-csv <CLEANED_VARIANTS.csv>
```

The full motif landscape and all 45 neural benchmark folds are compute intensive. Use `--limit-pdz`, `--folds`, `--models`, or fewer epochs for smoke tests.

## Manuscript figures

The current manuscript order is Figure 1 overview, Figure 2 architecture, Figure 3 performance, Figure 4 motif landscape, Figure 5 natural variants, Figure 6 TSA validation, Supplementary Figure S1 dataset, and Supplementary Figure S2 design. Figure 1 and Figure 2 are schematics; computational plotting entry points are under `scripts/figures/`.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/FIGURE_REPRODUCTION.md](docs/FIGURE_REPRODUCTION.md).

## Tests

```powershell
python -m pytest -q
```

The tests validate the released dataset/splits and reproduce the five published DLG1-PDZ3 TSA candidate predictions from the committed ensemble weights.

## Citation

Please cite the P3Bind manuscript and this repository.

> P3Bind: An interaction-aware sequence framework for quantitative PDZ–PBM affinity prediction and motif design
