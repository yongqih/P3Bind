# P3Bind

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
[1]	Lee, Ho-Jin, and Jie J. Zheng. "PDZ domains and their binding partners: structure, specificity, and modification." Cell communication and Signaling 8.1 (2010): 8.

[2]	Christensen, Nikolaj R., et al. "PDZ domains as drug targets." Advanced therapeutics 2.7 (2019): 1800143.

[3]	Manjunath, G. P., Praveena L. Ramanujam, and Sanjeev Galande. "Structure function relations in PDZ-domain-containing proteins: Implications for protein networks in cellular signalling." Journal of biosciences 43.1 (2018): 155-171.

[4]	Harris, Baruch Z., and Wendell A. Lim. "Mechanism and role of PDZ domains in signaling complex assembly." Journal of cell science 114.18 (2001): 3219-3231.

[5]	Cha, Boyoung, et al. "PDZ domain-dependent regulation of NHE3 protein by both internal Class II and C-terminal Class I PDZ-binding motifs." Journal of Biological Chemistry 292.20 (2017): 8279-8290.

[6]	Gogl, Gergo, et al. "Quantitative fragmentomics allow affinity mapping of interactomes." Nature communications 13.1 (2022): 5472.

[7]	Honrubia, Jose M., et al. "Interaction between SARS-CoV PBM and Cellular PDZ Domains Leading to Virus Virulence." Viruses 16.8 (2024): 1214.

[8]	Wang, Conan K., et al. "Extensions of PDZ domains as important structural and functional elements." Protein & cell 1.8 (2010): 737-751.

[9]	Cong, Shuang, and Yang Zhou. "A review of convolutional neural network architectures and their optimizations." Artificial Intelligence Review 56.3 (2023): 1905.




