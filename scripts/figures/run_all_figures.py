
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Figure 1 (overview) and Figure 2 (architecture) are schematics.
FIGURE_SCRIPTS = [
    "plot_fig3_model_performance.py",
    "plot_fig4_motif_landscape.py",
    "plot_fig5_gnomad_variants.py",
    "plot_fig6_tsa_validation.py",
    "plot_supp_fig_s1_dataset.py",
    "plot_supp_fig_s2_design.py",
]

this_dir = Path(__file__).resolve().parent
for script in FIGURE_SCRIPTS:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(this_dir / script)], check=True)
print("\nAll computational manuscript figure scripts completed.")
print("Note: Figure 1 and Figure 2 are schematics and should be provided as source/final artwork files.")
