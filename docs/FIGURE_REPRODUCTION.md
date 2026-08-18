# Figure reproduction

The current manuscript numbering is:

| Figure | Content | Entry point |
|---|---|---|
| 1 | P3Bind overview | schematic source under `results/figures/source/` |
| 2 | neural architecture | schematic source under `results/figures/source/` |
| 3 | model performance | `scripts/figures/plot_fig3_model_performance.py` |
| 4 | motif preference landscape | `scripts/figures/plot_fig4_motif_landscape.py` |
| 5 | natural PBM variants | `scripts/figures/plot_fig5_gnomad_variants.py` |
| 6 | TSA validation | `scripts/figures/plot_fig6_tsa_validation.py` |
| S1 | dataset characterization | `scripts/figures/plot_supp_fig_s1_dataset.py` |
| S2 | PDZ-specific PBM design | `scripts/figures/plot_supp_fig_s2_design.py` |

Run an individual script with `--help` to see its expected table arguments. Defaults resolve under `data/processed/`, while figures are saved to `results/figures/`.

Key generated tables are produced by the analysis scripts:

- `run_baseline_benchmarks.py` and `run_neural_benchmarks.py`: fold metrics/predictions for Figure 3.
- `run_motif_landscape.py`: long enrichment, position importance, and 120-feature matrix for Figure 4.
- `run_variant_effect_analysis.py`: long variant-by-PDZ effects and per-variant summary for Figure 5.
- `run_design_demo.py`: ranked candidates and optimization trajectory for Supplementary Figure S2.

Figure 6 expects a prepared `tsa_validation_summary.csv` with candidate name, predicted pKd, experimental response (for example `delta_Tm`), and optional response SD. Raw instrument melt curves remain source experimental data and are not silently transformed by the plotting script.
