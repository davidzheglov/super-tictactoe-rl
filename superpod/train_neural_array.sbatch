#!/usr/bin/env bash
#SBATCH --job-name=superttt_neural
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_FILE="${CONFIG_FILE:-superpod/experiments_neural.tsv}"
RUN_ROOT="${RUN_ROOT:-$HOME/super_ttt_runs}"
SWEEP_NAME="${SWEEP_NAME:-manual_sweep}"
VENV_DIR="${VENV_DIR:-.venv-superpod}"

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

line="$(
  awk -v id="${SLURM_ARRAY_TASK_ID:-0}" 'BEGIN { FS="\t" } NR > 1 && $0 !~ /^#/ { if (i == id) { print; exit } i++ }' "$CONFIG_FILE"
)"
if [[ -z "$line" ]]; then
  echo "No experiment row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit 2
fi

IFS=$'\t' read -r name algo seed episodes opponent player_mode mixed_self mixed_heuristic mixed_line mixed_basic mixed_random batch_or_size lr hidden_size save_interval log_interval shaping_scale shaping_clip shaping_defense_weight forfeit_penalty stop_after_seconds <<< "$line"

out_dir="$RUN_ROOT/$SWEEP_NAME/$name"
mkdir -p "$out_dir"

echo "Experiment: $name"
echo "Algorithm: $algo"
echo "Seed: $seed"
echo "Episodes: $episodes"
echo "Opponent: $opponent"
echo "Run root: $RUN_ROOT"
echo "Output: $out_dir"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

stop_args=()
if [[ "$stop_after_seconds" != "0" && "$stop_after_seconds" != "0.0" ]]; then
  stop_args=(--stop-after-seconds "$stop_after_seconds")
fi

if [[ "$algo" == "ppo" ]]; then
  python train_torchrl_ppo.py \
    --episodes "$episodes" \
    --batch-episodes "$batch_or_size" \
    --update-epochs 4 \
    --minibatch-size 1024 \
    --lr "$lr" \
    --hidden-size "$hidden_size" \
    --opponent "$opponent" \
    --agent-player-mode "$player_mode" \
    --mixed-self-prob "$mixed_self" \
    --mixed-heuristic-prob "$mixed_heuristic" \
    --mixed-line-prob "$mixed_line" \
    --mixed-basic-prob "$mixed_basic" \
    --mixed-random-prob "$mixed_random" \
    --shaping-scale "$shaping_scale" \
    --shaping-clip "$shaping_clip" \
    --shaping-defense-weight "$shaping_defense_weight" \
    --forfeit-penalty "$forfeit_penalty" \
    --device auto \
    --seed "$seed" \
    --save-path "$out_dir/super_ttt_agent_torchrl.pt" \
    --log-csv "$out_dir/ppo_log.csv" \
    --done-file "$out_dir/ppo.done" \
    --save-interval "$save_interval" \
    --log-interval "$log_interval" \
    --resume \
    --skip-if-done \
    "${stop_args[@]}"
elif [[ "$algo" == "dqn" ]]; then
  python train_torch_dqn.py \
    --episodes "$episodes" \
    --batch-size "$batch_or_size" \
    --lr "$lr" \
    --hidden-size "$hidden_size" \
    --opponent "$opponent" \
    --agent-player-mode "$player_mode" \
    --mixed-self-prob "$mixed_self" \
    --mixed-heuristic-prob "$mixed_heuristic" \
    --mixed-line-prob "$mixed_line" \
    --mixed-basic-prob "$mixed_basic" \
    --mixed-random-prob "$mixed_random" \
    --shaping-scale "$shaping_scale" \
    --shaping-clip "$shaping_clip" \
    --shaping-defense-weight "$shaping_defense_weight" \
    --forfeit-penalty "$forfeit_penalty" \
    --device auto \
    --seed "$seed" \
    --save-path "$out_dir/dqn_agent_torch.pt" \
    --log-csv "$out_dir/dqn_log.csv" \
    --done-file "$out_dir/dqn.done" \
    --save-interval "$save_interval" \
    --log-interval "$log_interval" \
    --resume \
    --skip-if-done \
    "${stop_args[@]}"
else
  echo "Unknown neural algorithm: $algo" >&2
  exit 3
fi
