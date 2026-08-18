# File map

## Package

- `src/p3bind/model.py`: checkpoint-compatible design architecture, encoding, loading, scalar and vectorized ensemble inference.
- `src/p3bind/core.py`: validation, single/batch prediction, specificity profiling, mutation scan.
- `src/p3bind/benchmarks.py`: main-text traditional and neural benchmark implementations.
- `src/p3bind/training.py`: directional censored loss and five-fold design training.
- `src/p3bind/motif.py`: random PBM6 library scoring and positional enrichment.
- `src/p3bind/design.py`: specificity-aware sequence optimization.
- `src/p3bind/variants.py`: natural variant panel scoring and summaries.
- `src/p3bind/validation.py`: dataset and split integrity checks.

## Curated scripts

- `scripts/00_check_setup.py` through `05_prepare_background_pdz.py`: setup and user-facing inference utilities.
- `scripts/analysis/`: baseline, motif, design, and natural-variant workflows.
- `scripts/train/`: neural benchmarks and design-ensemble training.
- `scripts/figures/`: current main/supplementary plotting entry points.

## Provenance only

- `notebooks/`
- `scripts/notebook_exports/`

These snapshots are not the recommended local execution path.
