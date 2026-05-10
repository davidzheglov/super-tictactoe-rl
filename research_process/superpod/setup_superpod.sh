#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# HKUST SuperPOD exposes software through environment modules. Module names can
# vary over time, so these are best-effort and can be overridden with PYTHON_BIN.
module purge >/dev/null 2>&1 || true
module load python >/dev/null 2>&1 || true
module load cuda >/dev/null 2>&1 || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-superpod}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -r requirements-remote-gpu.txt

python - <<'PY'
import sys
import torch
import torchrl

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("TorchRL:", getattr(torchrl, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.ones((64, 64), device="cuda")
    y = torch.relu(x).sum()
    torch.cuda.synchronize()
    print("CUDA smoke:", float(y.cpu()))
else:
    print("CUDA smoke skipped: no GPU is visible on this node.")
PY

echo "SuperPOD environment ready: $VENV_DIR"
