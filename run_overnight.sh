#!/usr/bin/env bash
# Overnight training: DQN on both GPUs + Q-learning (CPU) in parallel.
# Run from repo root:  bash run_overnight.sh
# Results go to runs/overnight_dqn/ and runs/overnight_qlearning/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Detect python binary ──────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: neither python3 nor python found in PATH"
    exit 1
fi
echo "Using python: $PY  ($(${PY} --version 2>&1))"

# Force Python to flush stdout/stderr immediately (no buffering when redirected to file)
export PYTHONUNBUFFERED=1

# ── Activate venv if present in repo root ────────────────────────────────────
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
    echo "Activated venv: $ROOT/.venv"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DQN_STOCH_DIR="runs/overnight_dqn/dqn_stochastic_${TIMESTAMP}"
DQN_DET_DIR="runs/overnight_dqn/dqn_deterministic_${TIMESTAMP}"
QL_DIR="runs/overnight_qlearning/qlearning_${TIMESTAMP}"

mkdir -p "$DQN_STOCH_DIR/checkpoints"
mkdir -p "$DQN_DET_DIR/checkpoints"
mkdir -p "$QL_DIR/snapshots"

echo "============================================================"
echo "Overnight training  |  $(date)"
echo "  GPU 0  → DQN stochastic     → $DQN_STOCH_DIR"
echo "  GPU 1  → DQN deterministic  → $DQN_DET_DIR"
echo "  CPU    → Q-learning (75k)   → $QL_DIR"
echo "============================================================"

# ── GPU 0: DQN stochastic placement, mixed curriculum ────────────────────────
CUDA_VISIBLE_DEVICES=0 $PY -m train_torch_dqn \
    --episodes 300000 \
    --placement-mode stochastic \
    --opponent mixed \
    --mixed-heuristic-prob 0.45 \
    --mixed-line-prob 0.30 \
    --mixed-self-prob 0.20 \
    --mixed-random-prob 0.05 \
    --start-state-mode mixed \
    --start-state-min-plies 4 \
    --start-state-max-plies 18 \
    --hidden-size 256 \
    --batch-size 512 \
    --replay-size 200000 \
    --lr 3e-4 \
    --eps-start 1.0 \
    --eps-end 0.05 \
    --eps-decay-frac 0.70 \
    --target-update-episodes 500 \
    --shaping-scale 0.03 \
    --eval-interval 5000 \
    --eval-games 50 \
    --save-interval 10000 \
    --log-interval 1000 \
    --checkpoint-dir "$DQN_STOCH_DIR/checkpoints" \
    --save-path "$DQN_STOCH_DIR/dqn_stochastic.pt" \
    --log-csv "$DQN_STOCH_DIR/dqn_stochastic_log.csv" \
    --seed 0 \
    --device cuda \
    > "$DQN_STOCH_DIR/train.log" 2>&1 &
PID_GPU0=$!
echo "Started DQN stochastic on GPU 0  (PID $PID_GPU0)"

# ── GPU 1: DQN deterministic placement, heuristic-focused curriculum ─────────
CUDA_VISIBLE_DEVICES=1 $PY -m train_torch_dqn \
    --episodes 300000 \
    --placement-mode deterministic \
    --opponent mixed \
    --mixed-heuristic-prob 0.65 \
    --mixed-line-prob 0.20 \
    --mixed-self-prob 0.10 \
    --mixed-random-prob 0.05 \
    --start-state-mode mixed \
    --start-state-min-plies 4 \
    --start-state-max-plies 18 \
    --hidden-size 256 \
    --batch-size 512 \
    --replay-size 200000 \
    --lr 3e-4 \
    --eps-start 1.0 \
    --eps-end 0.05 \
    --eps-decay-frac 0.70 \
    --target-update-episodes 500 \
    --shaping-scale 0.03 \
    --eval-interval 5000 \
    --eval-games 50 \
    --save-interval 10000 \
    --log-interval 1000 \
    --checkpoint-dir "$DQN_DET_DIR/checkpoints" \
    --save-path "$DQN_DET_DIR/dqn_deterministic.pt" \
    --log-csv "$DQN_DET_DIR/dqn_deterministic_log.csv" \
    --seed 1 \
    --device cuda \
    > "$DQN_DET_DIR/train.log" 2>&1 &
PID_GPU1=$!
echo "Started DQN deterministic on GPU 1  (PID $PID_GPU1)"

# ── CPU: Q-learning with snapshots + Q-value stats ───────────────────────────
$PY -m train_qlearning \
    --episodes 75000 \
    --alpha 0.05 \
    --gamma 0.99 \
    --eps-start 0.9 \
    --eps-end 0.05 \
    --shaping-scale 0.03 \
    --save-interval 5000 \
    --log-interval 1000 \
    --snapshot-interval 15000 \
    --snapshot-dir "$QL_DIR/snapshots" \
    --save-path "$QL_DIR/q_table.pkl" \
    --log-csv "$QL_DIR/q_learning_log.csv" \
    --seed 0 \
    > "$QL_DIR/train.log" 2>&1 &
PID_QL=$!
echo "Started Q-learning on CPU       (PID $PID_QL)"

echo ""
echo "All three processes launched. Logs:"
echo "  tail -f $DQN_STOCH_DIR/train.log"
echo "  tail -f $DQN_DET_DIR/train.log"
echo "  tail -f $QL_DIR/train.log"
echo ""
echo "Waiting for all to complete (or Ctrl-C to detach and let them run)..."
wait $PID_GPU0 && echo "DQN stochastic  DONE" || echo "DQN stochastic  FAILED (exit $?)"
wait $PID_GPU1 && echo "DQN deterministic DONE" || echo "DQN deterministic FAILED (exit $?)"
wait $PID_QL   && echo "Q-learning      DONE" || echo "Q-learning      FAILED (exit $?)"
echo "All done  |  $(date)"
