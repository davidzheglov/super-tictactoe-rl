# HKUST SuperPOD Training Guide

This project should use SuperPOD as a parallel experiment machine, not as one
large single-GPU job. Super Tic-Tac-Toe is stochastic, the environment step is
Python-heavy, and our A2 run showed low GPU utilization because the GPU waits
for game simulation. SuperPOD still helps a lot: run several short curricula at
once, then pick checkpoints by evaluation against the smart heuristic and
line-builder baselines.

## Why This Plan

The reference solution's point is important: in a stochastic game, more episodes
do not automatically mean a better policy. More data reduces variance, but
self-play can also shift the training distribution, become conservative, or
forget useful lines. That is why this sweep includes:

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

## Reference Script Pattern

The ignored `tic-tac-toe/` reference folder uses ordinary Slurm batch scripts:

- login/copy target: `jtangbx@superpod.ust.hk`;
- Slurm project account in the scripts: `mscbdtsuperpod`;
- GPU partition: `normal`;
- PPO jobs: `--gpus-per-node=2`, `--time=12:00:00` to `24:00:00`;
- setup style: activate an environment, print `nvidia-smi`, then run training;
- reference PPO scale: `--updates 3000 --episodes 512`, meaning
  `1,536,000` rollout games.

For our project, the matching SuperPOD script is
`superpod/research_reference_scale.sbatch`. By default it runs the reference-scale
budget on both PPO variants. You can override it down to `300000` games with
environment variables at submission time.

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
scp /tmp/sttt2_superpod_bundle.tar.gz jtangbx@superpod.ust.hk:~/
```

On SuperPOD:

```bash
ssh jtangbx@superpod.ust.hk
mkdir -p ~/sttt2
tar -xzf ~/sttt2_superpod_bundle.tar.gz -C ~/sttt2 --strip-components=1
cd ~/sttt2
```

Build the Python environment on a compute/GPU node, following the same pattern
used by `tic-tac-toe/job.sh`:

```bash
srun --account=mscbdtsuperpod --partition=normal --gpus-per-node=1 --time=00:30:00 --pty bash
cd ~/sttt2
bash superpod/setup_superpod.sh
exit
```

If the login node cannot reach PyPI or the PyTorch wheel index, use HKUST's
recommended module or container workflow and keep the same SLURM scripts.

## Submit Focused PPO Runs

Use this for a SuperPOD run with the same budget scale as the reference scripts:

```bash
cd ~/sttt2
export SPOD_ACCOUNT=mscbdtsuperpod
export RUN_ROOT=$HOME/super_ttt_runs
sbatch \
  --account "$SPOD_ACCOUNT" \
  --partition normal \
  --export "ALL,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=research_reference_scale_1536k" \
  superpod/research_reference_scale.sbatch
```

Use this for the smaller `300000`-game version:

```bash
cd ~/sttt2
export SPOD_ACCOUNT=mscbdtsuperpod
export RUN_ROOT=$HOME/super_ttt_runs
sbatch \
  --account "$SPOD_ACCOUNT" \
  --partition normal \
  --export "ALL,RUN_ROOT=$RUN_ROOT,SWEEP_NAME=research_superpod_300k,PPO_EPISODES=300000,DET_PPO_EPISODES=300000" \
  superpod/research_reference_scale.sbatch
```

Both commands request 2 GPUs in the Slurm script. The runner first performs
behavior cloning, then trains stochastic PPO on one GPU and deterministic PPO on
the other GPU. Both use PPO entropy coefficient `0.05` and vectorized rollout
collection, so each PPO update gathers the whole `--ppo-batch-episodes` batch
concurrently instead of playing one game to completion before starting the next.

## Submit the Full Sweep

Set your project account. HKUST requires an account/project allocation for
scheduled jobs. The scripts follow HKUST's SLURM guidance: GPU jobs use
`normal` by default, CPU Q-learning uses `cpu`, and no script asks for manual
memory because the docs say memory is allocated proportionally:
https://hkust-hpc-docs.readthedocs.io/latest/kb/slurm/slurm-how-to-submit-and-run-batch-jobs-G75o-i.html

```bash
cd ~/sttt2
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
tail -f slurm_logs/superttt_reference_*.out
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
