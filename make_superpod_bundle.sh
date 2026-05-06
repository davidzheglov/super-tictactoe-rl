#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/super_ttt_superpod_bundle.tar.gz}"

tar -czf "$OUT" \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude=".venv-superpod" \
  --exclude="__pycache__" \
  --exclude="*/__pycache__" \
  --exclude=".DS_Store" \
  --exclude="runs" \
  --exclude="slurm_logs" \
  --exclude="tic-tac-toe" \
  --exclude="req.txt" \
  --exclude="models/*.pt*" \
  --exclude="models/*.pkl" \
  --exclude="models/*.csv" \
  --exclude="models/*.done" \
  --exclude="models/*.json" \
  -C "$(dirname "$ROOT")" \
  "$(basename "$ROOT")"

echo "Wrote $OUT"
