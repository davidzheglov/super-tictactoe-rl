#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -r requirements-remote-gpu.txt
python - <<'PY'
import torch
import torchrl
print("PyTorch:", torch.__version__)
print("TorchRL:", getattr(torchrl, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.ones((64, 64), device="cuda")
    y = torch.relu(x).sum()
    torch.cuda.synchronize()
    print("CUDA smoke:", float(y.cpu()))
PY
