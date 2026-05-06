# HKUST SuperPOD Training Guide

This project should use SuperPOD as a parallel experiment machine, not as one
large single-GPU job. Super Tic-Tac-Toe is stochastic, the environment step is
Python-heavy, and our A2 run showed low GPU utilization because the GPU waits
for game simulation. SuperPOD still helps a lot: run many independent seeds and
curricula at once, then pick checkpoints by evaluation against heuristic and
MCTS baselines.

## Why This Plan

Your friend's point is important: in a stochastic game, more episodes do not
automatically mean a better policy. More data reduces variance, but self-play
can also shift the training distribution, become conservative, or forget useful
lines. That is why this sweep includes:

- short heuristic curriculum runs, to test whether the agent can beat human
  rules quickly;
- mixed opponent runs, the default serious setting: self-play plus heuristic
  plus random;
- pure self-play controls, to measure whether mixed training actually helps;
- multiple seeds, because one stochastic run is not publication-quality;
- benchmark selection, because the best checkpoint may not be the last one.

The default neural table has 10 tasks because HKUST lists a 10 queued/running
job limit per user for the GPU partitions. After the first sweep finishes, copy
rows in `superpod/experiments_neural.tsv` to add seeds or longer budgets.

## First-Time Setup

Connect to the HKUST network or VPN, then SSH to the SuperPOD login node. HKUST
lists the login host as `superpod.ust.hk`:
https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/usage-tips/login

```bash
ssh <your_itsc_username>@superpod.ust.hk
git clone git@github.com:davidzheglov/super-tictactoe-rl.git
cd super-tictactoe-rl
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

The key success criterion is practical: beat the heuristic baseline reliably,
then compare against rollout MCTS. For a report, use win rates over at least
100 games per matchup with alternating first player and multiple random seeds.

## Download Results

From your local machine:

```bash
rsync -avz <your_itsc_username>@<superpod_login_host>:$HOME/super_ttt_runs/ ./superpod_results/
```

If you used `/scratch`, copy or rsync it before the scratch cleanup policy can
remove it.
