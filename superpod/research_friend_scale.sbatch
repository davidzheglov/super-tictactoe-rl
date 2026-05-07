#!/usr/bin/env bash
# Submit with:
#   sbatch --account "$SPOD_ACCOUNT" --partition normal superpod/research_friend_scale.sbatch
#
# The default PPO budget mirrors the reference folder's long PPO scale:
# 3000 updates * 512 games/update = 1,536,000 games per PPO variant.
# Override PPO_EPISODES and DET_PPO_EPISODES at submission time for shorter runs.
# Example:
#   sbatch --account "$SPOD_ACCOUNT" --partition normal \
#     --export "ALL,PPO_EPISODES=300000,DET_PPO_EPISODES=300000" \
#     superpod/research_friend_scale.sbatch
#SBATCH --job-name=superttt_friend
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

VENV_DIR="${VENV_DIR:-.venv-superpod}"
RUN_ROOT="${RUN_ROOT:-$HOME/super_ttt_runs}"
SWEEP_NAME="${SWEEP_NAME:-research_friend_scale_${SLURM_JOB_ID:-manual}}"
PPO_EPISODES="${PPO_EPISODES:-1536000}"
DET_PPO_EPISODES="${DET_PPO_EPISODES:-1536000}"
BC_SAMPLES="${BC_SAMPLES:-200000}"
BC_EPOCHS="${BC_EPOCHS:-8}"
OUTPUT_DIR="$RUN_ROOT/$SWEEP_NAME"

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Output: $OUTPUT_DIR"
echo "PPO episodes: $PPO_EPISODES"
echo "Deterministic PPO episodes: $DET_PPO_EPISODES"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

python run_remote_training.py \
  --neural-backend torchrl \
  --gpus 0,1 \
  --output-dir "$OUTPUT_DIR" \
  --only behavior_clone,ppo,ppo_deterministic \
  --enable-behavior-clone \
  --include-deterministic-ppo \
  --bc-samples "$BC_SAMPLES" \
  --bc-epochs "$BC_EPOCHS" \
  --ppo-episodes "$PPO_EPISODES" \
  --deterministic-ppo-episodes "$DET_PPO_EPISODES" \
  --ppo-batch-episodes 512 \
  --ppo-update-epochs 4 \
  --ppo-lr 0.0003 \
  --ppo-entropy-coef 0.05 \
  --save-interval 51200 \
  --log-interval 5120 \
  --ppo-opponent mixed \
  --mixed-self-prob 0.10 \
  --mixed-heuristic-prob 0.65 \
  --mixed-line-prob 0.20 \
  --mixed-basic-prob 0.00 \
  --mixed-random-prob 0.05 \
  --start-state-mode mixed \
  --start-state-min-plies 4 \
  --start-state-max-plies 18 \
  --shaping-scale 0.04 \
  --shaping-clip 2.0 \
  --shaping-defense-weight 0.85 \
  --forfeit-penalty 0.025

echo "Job finished: $(date)"
