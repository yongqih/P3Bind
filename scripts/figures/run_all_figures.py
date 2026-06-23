
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Manuscript figure order:
# Fig. 1 = overview schematic, Fig. 3 = architecture schematic.
# These are manually assembled schematics; place source/final images under results/figures/source/.
# Computationally regenerated figures are listed below.
FIGURE_SCRIPTS = [
    "plot_fig2_dataset_characterization.py",
    "plot_fig4_model_performance.py",
    "plot_fig5_motif_landscape.py",
    "plot_fig6_pbm_design.py",
    "plot_fig7_gnomad_variants.py",
]

this_dir = Path(__file__).resolve().parent
for script in FIGURE_SCRIPTS:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(this_dir / script)], check=True)
print("\nAll computational manuscript figure scripts completed.")
print("Note: Figure 1 and Figure 3 are schematic diagrams and should be provided as source/final artwork files.")
