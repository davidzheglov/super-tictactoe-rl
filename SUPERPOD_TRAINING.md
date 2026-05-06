# HKUST SuperPOD Training Guide

This project should use SuperPOD as a parallel experiment machine, not as one
large single-GPU job. Super Tic-Tac-Toe is stochastic, the environment step is
Python-heavy, and our A2 run showed low GPU utilization because the GPU waits
for game simulation. SuperPOD still helps a lot: run several short curricula at
once, then pick checkpoints by evaluation against the smart heuristic and
line-builder baselines.

## Why This Plan

Your friend's point is important: in a stochastic game, more episodes do not
automatically mean a better policy. More data reduces variance, but self-play
can also shift the training distribution, become conservative, or forget useful
lines. That is why this sweep includes:

- short smart-heuristic and line-builder curriculum runs, to test whether the
  agent can beat human rules quickly;
- mixed opponent runs, the default serious setting: self-play plus smart
  heuristic plus line-builder plus random;
- pure self-play controls, to measure whether mixed training actually helps;
- multiple seeds, because one stochastic run is not publication-quality;
- benchmark selection, because the best checkpoint may not be the last one.

The default neural table is intentionally short. After the first sweep finishes,
copy rows in `superpod/experiments_neural.tsv` to add seeds or longer budgets
only if the benchmark curves justify it.

## Documentation Checked

HKUST's SuperPOD page says jobs must be run through SLURM, and the current quick
start examples use `sbatch`, `squeue`, and `scancel`. The same pages describe
GPU requests using `--gpus` or `--gpus-per-node`, partitions such as `normal`
and `cpu`, and optional project accounting with `--account`.

Useful references:

- SuperPOD overview and getting-started page:
  https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod
- SLURM usage page:
  https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/slurm
- First SLURM job page:
  https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job
- Partition/resource page:
  https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/partition

## First-Time Setup with SCP or rsync

Connect to the HKUST network or VPN, then SSH to the SuperPOD login node. HKUST
lists SuperPOD access under the getting-started instructions. Use the actual
login host given to your account if it differs from `superpod.ust.hk`.

From this local project folder:

```bash
cd /path/to/super_tictactoe_rl
bash make_superpod_bundle.sh
scp /tmp/super_ttt_superpod_bundle.tar.gz <your_itsc_username>@superpod.ust.hk:~/
```

On SuperPOD:

```bash
ssh <your_itsc_username>@superpod.ust.hk
mkdir -p ~/super_tictactoe_rl
tar -xzf ~/super_ttt_superpod_bundle.tar.gz -C ~/super_tictactoe_rl --strip-components=1
cd ~/super_tictactoe_rl
bash superpod/setup_superpod.sh
```

If the login node cannot reach PyPI or the PyTorch wheel index, use HKUST's
recommended module or container workflow and keep the same SLURM scripts.

## Submit the Full Sweep

Set your project account. HKUST requires an account/project allocation for
scheduled jobs. The scripts follow HKUST's SLURM guidance: GPU jobs use
`normal` by default, CPU Q-learning uses `cpu`, and no script asks for manual
memory because the docs say memory is allocated proportionally:
https://hkust-hpc-docs.readthedocs.io/latest/kb/slurm/slurm-how-to-submit-and-run-batch-jobs-G75o-i.html

```bash
cd ~/super-tictactoe-rl
export SPOD_ACCOUNT=<your_superpod_account>
export SPOD_PARTITION=normal
export SPOD_Q_PARTITION=cpu
export MAX_PARALLEL=8
export Q_MAX_PARALLEL=4
export RUN_ROOT=$HOME/super_ttt_runs
bash superpod/submit_sweep.sh
```

The default `RUN_ROOT` is in `$HOME` so both GPU and CPU jobs can read it. If
you move outputs to `/scratch`, remember that HKUST documents the CPU partition
as not mounting `/scratch`; either skip Q-learning, run Q on a GPU-accessible
partition, or copy outputs before CPU benchmarking. Partition limits are listed
here:
https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/partition

## Monitor Jobs

```bash
squeue -u $USER
tail -f slurm_logs/superttt_neural_*.out
tail -f slurm_logs/superttt_q_*.out
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,AllocTRES%60
```

Each training task writes logs and checkpoints under:

```text
$RUN_ROOT/$SWEEP_NAME/<experiment_name>/
```

Done markers:

```bash
find "$RUN_ROOT" -name "*.done" -print
find "$RUN_ROOT" -name "*.json" -print -exec cat {} \;
```

Cancel a sweep if needed:

```bash
scancel <job_id>
```

## Benchmark After Training

`submit_sweep.sh` submits a dependent benchmark job by default. To run it
manually:

```bash
export SPOD_ACCOUNT=<your_superpod_account>
export RUN_ROOT=$HOME/super_ttt_runs
export SWEEP_NAME=<the_sweep_name_printed_by_submit_sweep>
sbatch \
  --account "$SPOD_ACCOUNT" \
  --partition normal \
  --export "ALL,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=$SWEEP_NAME" \
  superpod/benchmark_sweep.sbatch
```

Benchmark outputs:

```text
$RUN_ROOT/$SWEEP_NAME/benchmarks/benchmark_summary.csv
$RUN_ROOT/$SWEEP_NAME/benchmarks/benchmark_raw.csv
$RUN_ROOT/$SWEEP_NAME/benchmarks/missing_checkpoints.csv
```

The key success criterion is practical: beat the smart heuristic baseline
reliably and beat the line-builder baseline. For a report, use win rates over at
least 100 games per matchup with alternating first player and multiple random
seeds.

## Plot Learning Curves

After the dependent benchmark finishes:

```bash
python analyze_training.py \
  --run-dir "$RUN_ROOT/$SWEEP_NAME" \
  --output-dir "$RUN_ROOT/$SWEEP_NAME/figures"
```

This writes PPO loss/entropy curves, DQN loss/epsilon curves, Q-table growth,
and benchmark win-rate plots.

## Download Results

From your local machine:

```bash
rsync -avz <your_itsc_username>@superpod.ust.hk:$HOME/super_ttt_runs/ ./superpod_results/
```

If you used `/scratch`, copy or rsync it before the scratch cleanup policy can
remove it.

## Laptop Fallback

The same code runs locally. Use smaller budgets first:

```bash
python run_remote_training.py \
  --neural-backend torchrl \
  --neural-device auto \
  --gpus 0 \
  --output-dir runs/local_mixed \
  --ppo-episodes 3000 \
  --ppo-batch-episodes 8 \
  --dqn-episodes 3000 \
  --q-episodes 10000 \
  --save-interval 500 \
  --q-save-interval 1000 \
  --log-interval 100

python benchmark.py \
  --agents random,basic,line,heuristic,ppo,dqn,q \
  --games 100 \
  --ppo-path runs/local_mixed/ppo_seed0/super_ttt_agent_torchrl.pt \
  --dqn-path runs/local_mixed/dqn_seed0/dqn_agent_torch.pt \
  --q-path runs/local_mixed/q_learning_seed0/q_table.pkl \
  --output-dir runs/local_mixed/benchmarks

python analyze_training.py --run-dir runs/local_mixed
```
