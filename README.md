# PDZ-PBM Affinity Prediction and Motif Design

This repository contains code for benchmarking and designing PDZ-binding motifs using sequence-based machine learning and interaction-aware neural network models.

## Overview

We formulate PDZ-PBM binding affinity prediction as a supervised regression task:

PDZ sequence + C-terminal PBM6 sequence → predicted pKd

The repository includes:
- non-deep-learning baselines;
- CNN-concat neural baseline;
- learned residue-pair interaction map model;
- interaction-map + Miyazawa-Jernigan contact potential model;
- evaluation under random, PBM-heldout, and PDZ-heldout splits;
- tools for PDZ-specific PBM candidate design.

## Model architectures

1. Random forest baseline  
2. CNN-concat model  
3. Interaction-map model  
4. Interaction-map + MJ potential model  

## Data

The processed modeling dataset should be placed at:

data/processed/all_data_pair_aggregated.csv

The expected columns are:

- pdz_sequence
- pbm_sequence_6aa
- pKd_label
- is_censored_label
- pair_id

## Reproduce results

### 1. Generate splits

```bash
python scripts/01_make_splits.py
```

### 2. Run baselines
```bash
python scripts/02_run_baselines.py
```

### 3. Train CNN-concat model
```bash
python scripts/03_train_cnn_concat.py
```

### 4. Train interaction-map model
```bash
python scripts/04_train_interaction_map.py
```
### 5. Train interaction-map + MJ model
```bash
python scripts/05_train_interaction_map_mj.py
```

| Model                | Random Pearson | PBM-heldout Pearson | PDZ-heldout Pearson |
| -------------------- | -------------: | ------------------: | ------------------: |
| RF PDZ+PBM           |          0.799 |               0.762 |               0.357 |
| CNN-concat           |          0.621 |               0.621 |               0.456 |
| Interaction-map      |          0.734 |               0.675 |               0.483 |
| Interaction-map + MJ |          0.740 |               0.730 |               0.488 |

## Citation
Manuscript in preparation

## License
MIT License.

