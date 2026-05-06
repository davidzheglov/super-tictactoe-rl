#!/usr/bin/env bash
#SBATCH --job-name=superttt_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_ROOT="${RUN_ROOT:-$HOME/super_ttt_runs}"
SWEEP_NAME="${SWEEP_NAME:-manual_sweep}"
VENV_DIR="${VENV_DIR:-.venv-superpod}"
NEURAL_CONFIG="${NEURAL_CONFIG:-superpod/experiments_neural.tsv}"
Q_CONFIG="${Q_CONFIG:-superpod/experiments_q.tsv}"
BENCH_GAMES="${BENCH_GAMES:-100}"
MCTS_ROLLOUTS="${MCTS_ROLLOUTS:-4}"

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export SUPER_TTT_DISABLE_TF_AGENTS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

python superpod/benchmark_sweep.py \
  --run-root "$RUN_ROOT" \
  --sweep-name "$SWEEP_NAME" \
  --neural-config "$NEURAL_CONFIG" \
  --q-config "$Q_CONFIG" \
  --games "$BENCH_GAMES" \
  --mcts-rollouts "$MCTS_ROLLOUTS" \
  --device auto \
  --output-dir "$RUN_ROOT/$SWEEP_NAME/benchmarks"
