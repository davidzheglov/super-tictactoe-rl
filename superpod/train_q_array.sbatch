#!/usr/bin/env bash
#SBATCH --job-name=superttt_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_FILE="${Q_CONFIG_FILE:-superpod/experiments_q.tsv}"
RUN_ROOT="${RUN_ROOT:-$HOME/super_ttt_runs}"
SWEEP_NAME="${SWEEP_NAME:-manual_sweep}"
VENV_DIR="${VENV_DIR:-.venv-superpod}"

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export SUPER_TTT_DISABLE_TF_AGENTS=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

line="$(
  awk -v id="${SLURM_ARRAY_TASK_ID:-0}" 'BEGIN { FS="\t" } NR > 1 && $0 !~ /^#/ { if (i == id) { print; exit } i++ }' "$CONFIG_FILE"
)"
if [[ -z "$line" ]]; then
  echo "No Q-learning row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit 2
fi

IFS=$'\t' read -r name seed episodes alpha gamma eps_start eps_end save_interval log_interval stop_after_seconds <<< "$line"

out_dir="$RUN_ROOT/$SWEEP_NAME/$name"
mkdir -p "$out_dir"

echo "Experiment: $name"
echo "Algorithm: q_learning"
echo "Seed: $seed"
echo "Episodes: $episodes"
echo "Run root: $RUN_ROOT"
echo "Output: $out_dir"

stop_args=()
if [[ "$stop_after_seconds" != "0" && "$stop_after_seconds" != "0.0" ]]; then
  stop_args=(--stop-after-seconds "$stop_after_seconds")
fi

python train_qlearning.py \
  --episodes "$episodes" \
  --alpha "$alpha" \
  --gamma "$gamma" \
  --eps-start "$eps_start" \
  --eps-end "$eps_end" \
  --seed "$seed" \
  --save-path "$out_dir/q_table.pkl" \
  --log-csv "$out_dir/q_learning_log.csv" \
  --done-file "$out_dir/q_learning.done" \
  --save-interval "$save_interval" \
  --log-interval "$log_interval" \
  --resume \
  --skip-if-done \
  "${stop_args[@]}"
