from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("plot_fig6_pbm_design.py")), run_name="__main__")
