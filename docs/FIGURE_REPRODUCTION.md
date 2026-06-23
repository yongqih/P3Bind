
# Manuscript figure reproduction

This repository is organized to match the figure order in the current manuscript.
The computational figure scripts read processed CSV files from `data/processed/` and save outputs to `results/figures/`.

The manuscript contains seven figures:

| Manuscript figure | Content | Reproduction status in this repo |
|---|---|---|
| Figure 1 | P3Bind prediction/design framework overview | Schematic; archive source/final artwork under `results/figures/source/` |
| Figure 2 | Dataset construction and characterization | `scripts/figures/plot_fig2_dataset_characterization.py` |
| Figure 3 | Neural architectures | Schematic; archive source/final artwork under `results/figures/source/` |
| Figure 4 | Baseline and neural model performance | `scripts/figures/plot_fig4_model_performance.py` |
| Figure 5 | Motif preference landscape | `scripts/figures/plot_fig5_motif_landscape.py` |
| Figure 6 | PDZ-specific PBM design demonstration | `scripts/figures/plot_fig6_pbm_design.py` |
| Figure 7 | gnomAD-P3Bind PBM variant-effect analysis | `scripts/figures/plot_fig7_gnomad_variants.py` |

Run all computational figure scripts:

```bash
python scripts/figures/run_all_figures.py
```

If the exact manuscript CSV is missing, scripts fall back to `.example.csv` files for a smoke test. Replace `.example.csv` files with the final processed result tables before paper submission or archival release.

## Figure 1: Overview of P3Bind prediction and design framework

Figure 1 is a schematic workflow diagram. It is not regenerated from a data table. Archive the editable source or final image here:

```text
results/figures/source/figure1_overview_workflow.svg
```

## Figure 2: Dataset construction and characterization

Script:

```bash
python scripts/figures/plot_fig2_dataset_characterization.py
```

Default input:

```text
data/processed/pair_level_dataset.csv
```

Minimum useful schema:

```csv
pdz_sequence,pbm10,pbm6,pKd,censored
SEQUENCE,AAAAATSV,ATSV,3.10,1
SEQUENCE,BBBBYRETRV,YRETRV,5.90,0
```

Accepted alternative pKd columns include `pkd`, `target_pKd`, `target_pKd_mean`, `pKd_median`, or `y`. Accepted censoring columns include `censored`, `is_censored`, `censored_pair`, or `is_censored_pair`.

## Figure 3: Neural architectures for PDZ-PBM affinity prediction

Figure 3 is a schematic architecture diagram. It is not regenerated from a data table. Archive the editable source or final image here:

```text
results/figures/source/figure3_neural_architectures.svg
```

## Figure 4: Performance comparison of baseline and neural models

Script:

```bash
python scripts/figures/plot_fig4_model_performance.py
```

Default input:

```text
data/processed/benchmark_results.csv
```

Recommended schema matching Table 1 / Figure 4:

```csv
model,split,pearson,pearson_std,rmse,rmse_std
RF, PBM-only,Random,0.331,0.004,0.511,0.006
Interaction-map + MJ,PBM-heldout,0.730,0.046,0.403,0.033
```

Long format is also accepted:

```csv
model,split,metric,mean,std
Interaction-map + MJ,PBM-heldout,pearson,0.730,0.046
```

## Figure 5: P3Bind motif preference landscape

Script:

```bash
python scripts/figures/plot_fig5_motif_landscape.py
```

Required/default inputs:

```text
data/processed/motif_enrichment_matrix.csv
data/processed/motif_position_importance.csv
```

Optional inputs for panels C and D:

```text
data/processed/motif_cluster_matrix.csv
data/processed/representative_motif_profiles.csv
```

`motif_enrichment_matrix.csv` schema:

```csv
position,amino_acid,log2_enrichment
P-5,A,-0.12
P0,V,2.10
```

`motif_position_importance.csv` schema:

```csv
position,mean_abs_log2_enrichment,ci_low,ci_high
P-2,1.84,1.74,1.93
```

## Figure 6: PDZ-specific PBM candidate design

Script:

```bash
python scripts/figures/plot_fig6_pbm_design.py
```

Required/default input:

```text
data/processed/design_candidates.csv
```

Optional trajectory input:

```text
data/processed/optimization_trajectory.csv
```

Expected candidate schema:

```csv
final_rank,pbm6,target_pKd_mean,target_pKd_std,background_pKd_mean,background_pKd_std,specificity_score
1,YRETRV,5.998,0.120,3.781,0.310,2.217
```

## Figure 7: gnomAD-P3Bind PBM variant effects

Script:

```bash
python scripts/figures/plot_fig7_gnomad_variants.py
```

Required/default input:

```text
data/processed/pbm_variant_effects.csv
```

Optional inputs:

```text
data/processed/variant_pdz_delta_matrix.csv
data/processed/representative_variant_profile.csv
```

`pbm_variant_effects.csv` schema:

```csv
variant_id,gene,wt_pbm6,mut_pbm6,pbm_position,max_abs_delta_pKd,af_bin
SLC15A5_LWETAL_LWEIAL,SLC15A5,LWETAL,LWEIAL,P-2,2.9,<1e-5
```

`variant_pdz_delta_matrix.csv` is a wide table with one row per top variant and one column per affected PDZ domain:

```csv
variant_id,DLG1_PDZ1,MAGI1_PDZ1,LNX2_PDZ2
SLC15A5_LWETAL_LWEIAL,2.1,-1.2,0.5
```
