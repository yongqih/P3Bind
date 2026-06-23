# Processed data tables

This directory stores lightweight processed tables required for inference and manuscript figure reproduction.

Required for specificity/off-target prediction:

```text
background_pdz.csv
```

Required for final manuscript figure reproduction:

```text
pair_level_dataset.csv                     # Figure 2
benchmark_results.csv                      # Figure 4
motif_enrichment_matrix.csv                # Figure 5A
motif_position_importance.csv              # Figure 5B
motif_cluster_matrix.csv                   # Figure 5C
representative_motif_profiles.csv          # Figure 5D
design_candidates.csv                      # Figure 6B/D
optimization_trajectory.csv                # Figure 6C
pbm_variant_effects.csv                    # Figure 7B/C/F
variant_pdz_delta_matrix.csv               # Figure 7D
representative_variant_profile.csv         # Figure 7E
```

`.example.csv` files are included only to document schemas and smoke-test scripts. Replace them with final processed manuscript tables for exact figure reproduction.
