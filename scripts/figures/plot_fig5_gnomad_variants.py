from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("plot_fig7_gnomad_variants.py")), run_name="__main__")
