"""One-command remote training orchestrator.

This script is designed for an Ubuntu GPU server. It runs tests once per source
hash, launches resumable training jobs, writes logs, and skips completed work on
later invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent


@dataclass
class Job:
    name: str
    cmd: List[str]
    log_path: Path
    done_file: Path
    gpu: Optional[str] = None


def source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def run_tests_once(cache_dir: Path, force: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / f"tests_{source_hash()}.ok"
    if marker.exists() and not force:
        print(f"[tests] cached OK: {marker.name}")
        return
    print("[tests] running tests.py")
    subprocess.check_call([sys.executable, str(ROOT / "tests.py")], cwd=str(ROOT))
    marker.write_text(f"ok {time.time()}\n", encoding="utf-8")
    print(f"[tests] cached result: {marker.name}")


def torch_gpu_works(gpu: str) -> bool:
    """Return True if PyTorch can run forward/backward kernels on a GPU."""
    code = r"""
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
device = torch.device("cuda")
x = torch.randn((128, 128), device=device, requires_grad=True)
layer = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 96),
).to(device)
y = layer(x).relu().mean()
y.backward()
torch.cuda.synchronize()
print("torch gpu ok", float(y.detach().cpu()))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if proc.returncode == 0:
        print(f"[gpu-check] GPU {gpu}: PyTorch CUDA kernels OK")
        return True
    print(f"[gpu-check] GPU {gpu}: PyTorch CUDA kernels FAILED")
    print(proc.stdout[-2000:])
    return False


def launch_job(job: Job, dry_run: bool = False) -> subprocess.Popen:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if job.done_file.exists():
        print(f"[{job.name}] done marker exists; skipping")
        return None  # type: ignore[return-value]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    print(f"[{job.name}] command: {' '.join(job.cmd)}")
    print(f"[{job.name}] log: {job.log_path}")
    if dry_run:
        return None  # type: ignore[return-value]
    log_f = job.log_path.open("a", buffering=1, encoding="utf-8")
    log_f.write(f"\n===== launch {time.ctime()} gpu={job.gpu} =====\n")
    proc = subprocess.Popen(
        job.cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._super_ttt_log_f = log_f  # type: ignore[attr-defined]
    return proc


def wait_processes(processes: Dict[str, subprocess.Popen]) -> None:
    failures = []
    while processes:
        for name, proc in list(processes.items()):
            ret = proc.poll()
            if ret is None:
                continue
            log_f = getattr(proc, "_super_ttt_log_f", None)
            if log_f is not None:
                log_f.write(f"===== exit {time.ctime()} code={ret} =====\n")
                log_f.close()
            print(f"[{name}] exited with code {ret}")
            if ret != 0:
                failures.append((name, ret))
            processes.pop(name)
        if processes:
            time.sleep(20)
    if failures:
        raise SystemExit(f"Training failures: {failures}")


def wait_one_wave(processes: Dict[str, subprocess.Popen]) -> None:
    if processes:
        wait_processes(processes)


def make_jobs(args: argparse.Namespace) -> List[Job]:
    run_dir = Path(args.output_dir).resolve()
    logs = run_dir / "logs"
    py = sys.executable
    common = ["--resume", "--skip-if-done"]
    stop_args = []
    if args.stop_after_seconds > 0:
        stop_args = ["--stop-after-seconds", str(args.stop_after_seconds)]

    ppo_dir = run_dir / "ppo_seed0"
    ppo_det_dir = run_dir / "ppo_deterministic_seed0"
    dqn_dir = run_dir / "dqn_seed0"
    q_dir = run_dir / "q_learning_seed0"
    bc_dir = run_dir / "behavior_clone_seed0"
    use_torchrl = args.neural_backend == "torchrl"
    ppo_script = "train_torchrl_ppo.py" if use_torchrl else "train_torch_ppo.py"
    dqn_script = "train_torch_dqn.py"
    ppo_save_name = "super_ttt_agent_torchrl.pt" if use_torchrl else "super_ttt_agent_torch.pt"
    dqn_save_name = "dqn_agent_torch.pt"
    common_opponent_args = [
        "--agent-player-mode",
        args.agent_player_mode,
        "--placement-mode",
        args.placement_mode,
        "--mixed-self-prob",
        str(args.mixed_self_prob),
        "--mixed-heuristic-prob",
        str(args.mixed_heuristic_prob),
        "--mixed-line-prob",
        str(args.mixed_line_prob),
        "--mixed-basic-prob",
        str(args.mixed_basic_prob),
        "--mixed-random-prob",
        str(args.mixed_random_prob),
        "--shaping-scale",
        str(args.shaping_scale),
        "--shaping-clip",
        str(args.shaping_clip),
        "--shaping-defense-weight",
        str(args.shaping_defense_weight),
        "--forfeit-penalty",
        str(args.forfeit_penalty),
        "--start-state-mode",
        args.start_state_mode,
        "--start-state-min-plies",
        str(args.start_state_min_plies),
        "--start-state-max-plies",
        str(args.start_state_max_plies),
    ]
    ppo_extra_args = [
        "--checkpoint-dir",
        str(ppo_dir / "checkpoints"),
    ]
    ppo_init_checkpoint = args.ppo_init_checkpoint
    if args.enable_behavior_clone and not ppo_init_checkpoint:
        ppo_init_checkpoint = str(bc_dir / "behavior_clone_torchrl.pt")
    if ppo_init_checkpoint:
        ppo_extra_args.extend(["--init-checkpoint", ppo_init_checkpoint])
    ppo_opponent_args = ["--opponent", args.ppo_opponent, *common_opponent_args]
    dqn_opponent_args = [
        "--opponent",
        args.dqn_opponent,
        *common_opponent_args,
        "--checkpoint-dir",
        str(dqn_dir / "checkpoints"),
    ]

    jobs: List[Job] = []
    if args.enable_behavior_clone:
        jobs.append(
            Job(
                name="behavior_clone",
                gpu=None,
                log_path=logs / "behavior_clone.log",
                done_file=bc_dir / "behavior_clone.done",
                cmd=[
                    py,
                    str(ROOT / "train_behavior_clone.py"),
                    "--samples",
                    str(args.bc_samples),
                    "--epochs",
                    str(args.bc_epochs),
                    "--batch-size",
                    str(args.bc_batch_size),
                    "--lr",
                    str(args.bc_lr),
                    "--teacher",
                    args.bc_teacher,
                    "--placement-mode",
                    args.placement_mode,
                    "--device",
                    args.neural_device,
                    "--seed",
                    "0",
                    "--save-path",
                    str(bc_dir / "behavior_clone_torchrl.pt"),
                    "--log-csv",
                    str(bc_dir / "behavior_clone_log.csv"),
                    "--done-file",
                    str(bc_dir / "behavior_clone.done"),
                    "--skip-if-done",
                ],
            )
        )

    jobs.append(
        Job(
            name="ppo",
            gpu=None,
            log_path=logs / "ppo.log",
            done_file=ppo_dir / "ppo.done",
            cmd=[
                py,
                str(ROOT / ppo_script),
                "--episodes",
                str(args.ppo_episodes),
                "--batch-episodes",
                str(args.ppo_batch_episodes),
                "--rollout-mode",
                args.ppo_rollout_mode,
                "--update-epochs",
                str(args.ppo_update_epochs),
                "--minibatch-size",
                str(args.ppo_minibatch_size),
                "--lr",
                str(args.ppo_lr),
                "--entropy-coef",
                str(args.ppo_entropy_coef),
                *ppo_opponent_args,
                *ppo_extra_args,
                "--device",
                args.neural_device,
                "--seed",
                "0",
                "--save-path",
                str(ppo_dir / ppo_save_name),
                "--log-csv",
                str(ppo_dir / "ppo_log.csv"),
                "--done-file",
                str(ppo_dir / "ppo.done"),
                "--save-interval",
                str(args.save_interval),
                "--log-interval",
                str(args.log_interval),
                *common,
                *stop_args,
            ],
        )
    )
    if args.include_deterministic_ppo:
        ppo_det_extra_args = [
            "--opponent",
            "heuristic",
            "--agent-player-mode",
            args.agent_player_mode,
            "--placement-mode",
            "deterministic",
            "--shaping-scale",
            str(args.shaping_scale),
            "--shaping-clip",
            str(args.shaping_clip),
            "--shaping-defense-weight",
            str(args.shaping_defense_weight),
            "--forfeit-penalty",
            "0.0",
            "--start-state-mode",
            args.start_state_mode,
            "--start-state-min-plies",
            str(args.start_state_min_plies),
            "--start-state-max-plies",
            str(args.start_state_max_plies),
            "--checkpoint-dir",
            str(ppo_det_dir / "checkpoints"),
        ]
        if ppo_init_checkpoint:
            ppo_det_extra_args.extend(["--init-checkpoint", ppo_init_checkpoint])
        jobs.append(
            Job(
                name="ppo_deterministic",
                gpu=None,
                log_path=logs / "ppo_deterministic.log",
                done_file=ppo_det_dir / "ppo_deterministic.done",
                cmd=[
                    py,
                    str(ROOT / ppo_script),
                    "--episodes",
                    str(args.deterministic_ppo_episodes),
                    "--batch-episodes",
                    str(args.ppo_batch_episodes),
                    "--rollout-mode",
                    args.ppo_rollout_mode,
                    "--update-epochs",
                    str(args.ppo_update_epochs),
                    "--minibatch-size",
                    str(args.ppo_minibatch_size),
                    "--lr",
                    str(args.ppo_lr),
                    "--entropy-coef",
                    str(args.ppo_entropy_coef),
                    *ppo_det_extra_args,
                    "--device",
                    args.neural_device,
                    "--seed",
                    "0",
                    "--save-path",
                    str(ppo_det_dir / ppo_save_name),
                    "--log-csv",
                    str(ppo_det_dir / "ppo_log.csv"),
                    "--done-file",
                    str(ppo_det_dir / "ppo_deterministic.done"),
                    "--save-interval",
                    str(args.save_interval),
                    "--log-interval",
                    str(args.log_interval),
                    *common,
                    *stop_args,
                ],
            )
        )
    jobs.extend(
        [
        Job(
            name="dqn",
            gpu=None,
            log_path=logs / "dqn.log",
            done_file=dqn_dir / "dqn.done",
            cmd=[
                py,
                str(ROOT / dqn_script),
                "--episodes",
                str(args.dqn_episodes),
                "--batch-size",
                str(args.dqn_batch_size),
                "--lr",
                str(args.dqn_lr),
                *dqn_opponent_args,
                "--device",
                args.neural_device,
                "--seed",
                "0",
                "--save-path",
                str(dqn_dir / dqn_save_name),
                "--log-csv",
                str(dqn_dir / "dqn_log.csv"),
                "--done-file",
                str(dqn_dir / "dqn.done"),
                "--save-interval",
                str(args.save_interval),
                "--log-interval",
                str(args.log_interval),
                *common,
                *stop_args,
            ],
        ),
        Job(
            name="q_learning",
            gpu=None,
            log_path=logs / "q_learning.log",
            done_file=q_dir / "q_learning.done",
            cmd=[
                py,
                str(ROOT / "train_qlearning.py"),
                "--episodes",
                str(args.q_episodes),
                "--seed",
                "0",
                "--save-path",
                str(q_dir / "q_table.pkl"),
                "--log-csv",
                str(q_dir / "q_learning_log.csv"),
                "--done-file",
                str(q_dir / "q_learning.done"),
                "--save-interval",
                str(args.q_save_interval),
                "--log-interval",
                str(args.log_interval),
                "--shaping-scale",
                str(args.shaping_scale),
                "--shaping-clip",
                str(args.shaping_clip),
                "--shaping-defense-weight",
                str(args.shaping_defense_weight),
                "--forfeit-penalty",
                str(args.forfeit_penalty),
                *common,
                *stop_args,
            ],
        ),
        ]
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all remote training jobs.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "runs" / "overnight"))
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids for neural jobs.")
    parser.add_argument(
        "--neural-backend",
        type=str,
        default="torchrl",
        choices=["torch", "torchrl"],
        help="PPO trainer backend. Default uses TorchRL env validation plus PyTorch PPO updates.",
    )
    parser.add_argument(
        "--neural-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Device argument passed to PPO/DQN trainers.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Do not run backend GPU kernel checks before GPU jobs.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow PPO/DQN to continue on CPU if no requested GPU passes checks.",
    )
    parser.add_argument("--only", type=str, default="all", help="Comma list: all,ppo,dqn,q_learning")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-tests", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--q-save-interval", type=int, default=2500)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--ppo-episodes", type=int, default=6000)
    parser.add_argument("--ppo-batch-episodes", type=int, default=32)
    parser.add_argument(
        "--ppo-rollout-mode",
        type=str,
        default="vectorized",
        choices=["vectorized", "sequential"],
        help="PPO rollout collection mode.",
    )
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-lr", type=float, default=2.0e-4)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-init-checkpoint", type=str, default="")
    parser.add_argument("--enable-behavior-clone", action="store_true")
    parser.add_argument("--bc-samples", type=int, default=200000)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--bc-lr", type=float, default=1.0e-3)
    parser.add_argument("--bc-teacher", type=str, default="mixed", choices=["heuristic", "line", "basic", "mixed"])
    parser.add_argument("--include-deterministic-ppo", action="store_true")
    parser.add_argument("--deterministic-ppo-episodes", type=int, default=48000)
    parser.add_argument(
        "--ppo-opponent",
        type=str,
        default="mixed",
        choices=["self", "random", "heuristic", "line", "basic", "mixed"],
    )
    parser.add_argument("--dqn-episodes", type=int, default=6000)
    parser.add_argument("--dqn-batch-size", type=int, default=512)
    parser.add_argument("--dqn-lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--dqn-opponent",
        type=str,
        default="mixed",
        choices=["self", "random", "heuristic", "line", "basic", "mixed"],
    )
    parser.add_argument(
        "--agent-player-mode",
        type=str,
        default="alternate",
        choices=["alternate", "random", "x", "o"],
    )
    parser.add_argument("--mixed-self-prob", type=float, default=0.2)
    parser.add_argument("--mixed-heuristic-prob", type=float, default=0.45)
    parser.add_argument("--mixed-line-prob", type=float, default=0.3)
    parser.add_argument("--mixed-basic-prob", type=float, default=0.0)
    parser.add_argument("--mixed-random-prob", type=float, default=0.05)
    parser.add_argument("--placement-mode", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    parser.add_argument(
        "--start-state-mode",
        type=str,
        default="mixed",
        choices=["none", "random", "heuristic", "line", "basic", "mixed"],
    )
    parser.add_argument("--start-state-min-plies", type=int, default=4)
    parser.add_argument("--start-state-max-plies", type=int, default=18)
    parser.add_argument("--shaping-scale", type=float, default=0.03)
    parser.add_argument("--shaping-clip", type=float, default=2.0)
    parser.add_argument("--shaping-defense-weight", type=float, default=0.75)
    parser.add_argument("--forfeit-penalty", type=float, default=0.02)
    parser.add_argument("--q-episodes", type=int, default=15000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_tests:
        run_tests_once(run_dir / ".cache", force=args.force_tests)

    requested = {"ppo", "dqn", "q_learning"} if args.only == "all" else set(args.only.split(","))
    if args.enable_behavior_clone:
        requested.add("behavior_clone")
    if args.include_deterministic_ppo:
        requested.add("ppo_deterministic")
    jobs = [job for job in make_jobs(args) if job.name in requested]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.neural_device == "cpu":
        print("[gpu-check] neural-device=cpu; hiding GPUs for PPO/DQN")
        gpus = []
    elif gpus and not args.skip_gpu_check:
        working_gpus = [gpu for gpu in gpus if torch_gpu_works(gpu)]
        if not working_gpus:
            print("[gpu-check] No requested GPU passed the PyTorch kernel check.")
            if args.allow_cpu_fallback:
                print("[gpu-check] Falling back to CPU for PPO/DQN.")
            else:
                raise SystemExit(
                    "No requested GPU is usable for PyTorch. Install CUDA-enabled torch "
                    "or pass --allow-cpu-fallback explicitly."
                )
        gpus = working_gpus

    setup_jobs = [job for job in jobs if job.name == "behavior_clone"]
    gpu_jobs = [job for job in jobs if job.name in {"ppo", "ppo_deterministic", "dqn"}]
    cpu_jobs = [job for job in jobs if job.name not in {"ppo", "ppo_deterministic", "dqn", "behavior_clone"}]
    for i, job in enumerate(gpu_jobs):
        job.gpu = gpus[i % len(gpus)] if gpus else None

    if setup_jobs:
        setup_processes: Dict[str, subprocess.Popen] = {}
        for job in setup_jobs:
            job.gpu = gpus[0] if gpus else None
            proc = launch_job(job, dry_run=args.dry_run)
            if proc is not None:
                setup_processes[job.name] = proc
        if args.dry_run:
            pass
        else:
            wait_processes(setup_processes)

    cpu_processes: Dict[str, subprocess.Popen] = {}
    for job in cpu_jobs:
        proc = launch_job(job, dry_run=args.dry_run)
        if proc is not None:
            cpu_processes[job.name] = proc
    if args.dry_run:
        for job in gpu_jobs:
            launch_job(job, dry_run=True)
        print("Dry run complete.")
        return

    wave_size = max(len(gpus), 1)
    for start in range(0, len(gpu_jobs), wave_size):
        wave: Dict[str, subprocess.Popen] = {}
        for job in gpu_jobs[start : start + wave_size]:
            proc = launch_job(job, dry_run=False)
            if proc is not None:
                wave[job.name] = proc
        wait_one_wave(wave)
    wait_processes(cpu_processes)
    print(f"All requested jobs completed or were already cached in {run_dir}")


if __name__ == "__main__":
    main()
