import argparse
import json
from pathlib import Path
import sys
import torch

from p3bind.core import DEFAULT_CHECKPOINT_DIR, REPO_ROOT, load_models
from p3bind.validation import validate_pair_dataset, validate_split_file


parser = argparse.ArgumentParser(description="Validate the local P3Bind release and Python environment.")
parser.add_argument("--skip-checkpoints", action="store_true", help="Skip loading release checkpoints")
args = parser.parse_args()

report = {
    "python": sys.version,
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "repo_root": str(REPO_ROOT),
}

dataset_path = REPO_ROOT / "data" / "raw" / "all_data_pair_aggregated.csv"
report["dataset"] = validate_pair_dataset(dataset_path)
report["splits"] = {}
for setting in ["random", "pbm_heldout", "pdz_heldout"]:
    report["splits"][setting] = validate_split_file(
        dataset_path,
        REPO_ROOT / "data" / "splits" / f"{setting}_split.csv",
        expected_setting=setting,
    )

if not args.skip_checkpoints:
    models, files = load_models(DEFAULT_CHECKPOINT_DIR, device=torch.device("cpu"))
    report["checkpoints"] = {
        "count": len(models),
        "files": [str(path.relative_to(REPO_ROOT)) for path in files],
    }
    if len(models) != 5:
        raise ValueError(f"Expected five design-ensemble checkpoints, found {len(models)}.")

print(json.dumps(report, indent=2))
