#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${SPOD_ACCOUNT:-}" ]]; then
  echo "Set SPOD_ACCOUNT to your SuperPOD project/account before submitting." >&2
  echo "Example: export SPOD_ACCOUNT=<your_account>" >&2
  exit 2
fi

mkdir -p slurm_logs

NEURAL_CONFIG="${NEURAL_CONFIG:-superpod/experiments_neural.tsv}"
Q_CONFIG="${Q_CONFIG:-superpod/experiments_q.tsv}"
RUN_ROOT="${RUN_ROOT:-$HOME/super_ttt_runs}"
SWEEP_NAME="${SWEEP_NAME:-superttt_$(date +%Y%m%d_%H%M%S)}"
VENV_DIR="${VENV_DIR:-.venv-superpod}"

SPOD_PARTITION="${SPOD_PARTITION:-normal}"
SPOD_Q_PARTITION="${SPOD_Q_PARTITION:-cpu}"
BENCH_PARTITION="${BENCH_PARTITION:-normal}"

NEURAL_TIME="${NEURAL_TIME:-12:00:00}"
Q_TIME="${Q_TIME:-08:00:00}"
BENCH_TIME="${BENCH_TIME:-04:00:00}"

MAX_PARALLEL="${MAX_PARALLEL:-8}"
Q_MAX_PARALLEL="${Q_MAX_PARALLEL:-4}"
RUN_NEURAL="${RUN_NEURAL:-1}"
RUN_Q="${RUN_Q:-1}"
SUBMIT_BENCHMARK="${SUBMIT_BENCHMARK:-1}"

count_rows() {
  awk 'BEGIN { n=0 } NR > 1 && $0 !~ /^#/ && NF { n++ } END { print n }' "$1"
}

echo "Sweep name: $SWEEP_NAME"
echo "Run root: $RUN_ROOT"
echo "Neural config: $NEURAL_CONFIG"
echo "Q config: $Q_CONFIG"

neural_job_id=""
q_job_id=""

if [[ "$RUN_NEURAL" == "1" ]]; then
  neural_n="$(count_rows "$NEURAL_CONFIG")"
  if [[ "$neural_n" -gt 0 ]]; then
    neural_job_id="$(
      sbatch --parsable \
        --account "$SPOD_ACCOUNT" \
        --partition "$SPOD_PARTITION" \
        --time "$NEURAL_TIME" \
        --array "0-$((neural_n - 1))%$MAX_PARALLEL" \
        --export "ALL,CONFIG_FILE=$NEURAL_CONFIG,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=$SWEEP_NAME,VENV_DIR=$VENV_DIR" \
        superpod/train_neural_array.sbatch
    )"
    echo "Submitted neural array: $neural_job_id ($neural_n tasks, %$MAX_PARALLEL)"
  fi
fi

if [[ "$RUN_Q" == "1" ]]; then
  q_n="$(count_rows "$Q_CONFIG")"
  if [[ "$q_n" -gt 0 ]]; then
    q_job_id="$(
      sbatch --parsable \
        --account "$SPOD_ACCOUNT" \
        --partition "$SPOD_Q_PARTITION" \
        --time "$Q_TIME" \
        --array "0-$((q_n - 1))%$Q_MAX_PARALLEL" \
        --export "ALL,Q_CONFIG_FILE=$Q_CONFIG,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=$SWEEP_NAME,VENV_DIR=$VENV_DIR" \
        superpod/train_q_array.sbatch
    )"
    echo "Submitted Q-learning array: $q_job_id ($q_n tasks, %$Q_MAX_PARALLEL)"
  fi
fi

if [[ "$SUBMIT_BENCHMARK" == "1" ]]; then
  deps=()
  [[ -n "$neural_job_id" ]] && deps+=("$neural_job_id")
  [[ -n "$q_job_id" ]] && deps+=("$q_job_id")
  dep_arg=()
  if [[ "${#deps[@]}" -gt 0 ]]; then
    dep_arg=(--dependency "afterok:$(IFS=:; echo "${deps[*]}")")
  fi
  bench_job_id="$(
    sbatch --parsable \
      --account "$SPOD_ACCOUNT" \
      --partition "$BENCH_PARTITION" \
      --time "$BENCH_TIME" \
      "${dep_arg[@]}" \
      --export "ALL,NEURAL_CONFIG=$NEURAL_CONFIG,Q_CONFIG=$Q_CONFIG,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=$SWEEP_NAME,VENV_DIR=$VENV_DIR" \
      superpod/benchmark_sweep.sbatch
  )"
  echo "Submitted benchmark job: $bench_job_id"
fi

cat <<EOF

Monitor:
  squeue -u $USER
  tail -f slurm_logs/superttt_neural_*.out
  tail -f slurm_logs/superttt_q_*.out

Results:
  $RUN_ROOT/$SWEEP_NAME
EOF
