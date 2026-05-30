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
