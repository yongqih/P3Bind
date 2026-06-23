from pathlib import Path
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Repo root:", Path(__file__).resolve().parents[1])
print("Expected checkpoints:", Path(__file__).resolve().parents[1] / "checkpoints" / "design_models")
