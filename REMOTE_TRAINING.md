# Remote GPU Training Guide

This guide is for an Ubuntu server with NVIDIA A2 GPUs.

## Is This Worth Doing?

Yes. The neural-network updates in PPO and DQN use PyTorch/CUDA, and the PPO
bonus path validates a TorchRL action-masked environment before training. Your
server has two A2 cards. Environment simulation is still Python/NumPy and
therefore CPU-bound, so speedup will not be linear with GPU count. Still,
overnight remote training is much more practical than MacBook CPU training.

Your `nvidia-smi` shows both GPUs are already doing work. GPU 0 has a large
process using about 6.7 GB. If those processes are not yours, use only GPU 1 or
wait until the GPUs are free.

## What Gets Trained

One command can run:

- TorchRL PPO self-play agent on one GPU
- PyTorch DQN baseline on one GPU
- Tabular Q-learning baseline on CPU

The PyTorch PPO/DQN jobs use mixed opponent training by default:

- 50% self-play
- 40% heuristic opponent
- 10% random opponent

Override this with `--ppo-opponent heuristic`, `--dqn-opponent heuristic`, or
`--mixed-self-prob/--mixed-heuristic-prob/--mixed-random-prob`.

Each trainer is resumable and cached:

- Checkpoints are saved every N episodes.
- CSV logs are appended.
- `.done` markers skip completed jobs.
- If a run is interrupted, rerun the same command and it resumes.
- Tests are cached by source-code hash.

## Server Setup

On the server:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip tmux
nvidia-smi
```

`nvidia-smi` must work before PyTorch/TorchRL can use the GPU.

## Transfer From Local Machine

From your local machine, in the parent folder that contains `super_tictactoe_rl`:

```bash
cd /Users/davidzheglov/Desktop/projects/rl/content/super_tictactoe_rl
bash make_remote_bundle.sh
scp ../super_tictactoe_rl_remote.zip USER@SERVER:~/
```

On the server:

```bash
unzip super_tictactoe_rl_remote.zip
cd super_tictactoe_rl
```

## Install Remote Dependencies

```bash
bash setup_remote_gpu.sh
```

This creates `.venv`, installs a CUDA-enabled PyTorch wheel plus
`requirements-remote-gpu.txt`, and prints the PyTorch-visible GPUs.

## Start Overnight Training

Use `tmux` so the job survives SSH disconnects:

```bash
tmux new -s superttt
cd ~/super_tictactoe_rl
source .venv/bin/activate
python run_remote_training.py --gpus 0,1 --output-dir runs/overnight
```

The default PPO backend is TorchRL. Use `--neural-backend torch` only if you
want to skip the TorchRL environment validation and run the plain PyTorch PPO
baseline.

Detach:

```text
Ctrl-b then d
```

Reconnect:

```bash
tmux attach -t superttt
```

If GPU 0 is still busy, use only GPU 1:

```bash
python run_remote_training.py --gpus 1 --output-dir runs/overnight
```

If both GPUs are busy, run only CPU Q-learning or wait:

```bash
python run_remote_training.py --only q_learning --output-dir runs/overnight
```

Pure heuristic-opponent curriculum:

```bash
python run_remote_training.py \
  --neural-backend torchrl \
  --gpus 0,1 \
  --ppo-opponent heuristic \
  --dqn-opponent heuristic \
  --output-dir runs/heuristic_curriculum
```

## Useful Run Sizes

Default overnight command:

```bash
python run_remote_training.py --gpus 0,1 --output-dir runs/overnight
```

It runs:

- PPO: 300,000 episodes
- DQN: 150,000 episodes
- Q-learning: 75,000 episodes

Longer PPO-focused run:

```bash
python run_remote_training.py \
  --only ppo \
  --gpus 1 \
  --ppo-episodes 500000 \
  --ppo-batch-episodes 64 \
  --ppo-minibatch-size 1024 \
  --ppo-lr 2e-4 \
  --output-dir runs/ppo_500k
```

Smoke test:

```bash
python run_remote_training.py \
  --ppo-episodes 20 \
  --dqn-episodes 20 \
  --q-episodes 20 \
  --output-dir runs/smoke
```

## Monitoring

In another SSH session:

```bash
nvidia-smi
tail -f ~/super_tictactoe_rl/runs/overnight/logs/ppo.log
tail -f ~/super_tictactoe_rl/runs/overnight/logs/dqn.log
tail -f ~/super_tictactoe_rl/runs/overnight/logs/q_learning.log
```

## Results To Download

After training, download:

```bash
scp -r USER@SERVER:~/super_tictactoe_rl/runs/overnight ./remote_results
```

The PPO checkpoint is:

```text
runs/overnight/ppo_seed0/super_ttt_agent_torchrl.pt
```

Copy `super_ttt_agent_torchrl.pt` into your local `models/` folder, then run:

```bash
python app.py --model-path models/super_ttt_agent_torchrl.pt
```

Run cross-play benchmarks on the server:

```bash
python benchmark.py \
  --agents random,heuristic,mcts,ppo,dqn,q \
  --games 100 \
  --output-dir runs/overnight_torch/benchmark
```

The benchmark writes:

```text
runs/overnight_torch/benchmark/benchmark_raw.csv
runs/overnight_torch/benchmark/benchmark_summary.csv
```

## Re-running Is Safe

You can run the same command again. It will:

- Skip cached tests if source did not change.
- Resume partial checkpoints.
- Skip completed jobs with `.done` markers.
