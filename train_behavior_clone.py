"""Behavior-clone scripted Super Tic-Tac-Toe policies into a PPO network."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent
    from .env import SuperTicTacToeEnv
    from .torch_models import TorchPolicyValueNet, mask_logits, resolve_torch_device
    from .utils import hidden_sizes_from_arg, project_root, random_legal_action, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent
    from env import SuperTicTacToeEnv
    from torch_models import TorchPolicyValueNet, mask_logits, resolve_torch_device
    from utils import hidden_sizes_from_arg, project_root, random_legal_action, set_global_seeds


def choose_teacher(args: argparse.Namespace, rng: np.random.Generator) -> str:
    if args.teacher != "mixed":
        return args.teacher
    labels = np.asarray(["heuristic", "line", "basic"], dtype=object)
    probs = np.asarray([args.teacher_heuristic_prob, args.teacher_line_prob, args.teacher_basic_prob])
    probs = probs / max(float(np.sum(probs)), 1.0e-12)
    return str(rng.choice(labels, p=probs))


def teacher_action(
    env: SuperTicTacToeEnv,
    teacher: str,
    heuristic: HeuristicAgent,
    line: LineBuilderAgent,
    basic: BasicHeuristicAgent,
    random_agent: RandomAgent,
) -> int:
    if teacher == "heuristic":
        return heuristic.select_action(env)
    if teacher == "line":
        return line.select_action(env)
    if teacher == "basic":
        return basic.select_action(env)
    return random_agent.select_action(env)


def collect_samples(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(args.seed)
    env = SuperTicTacToeEnv(seed=args.seed, placement_mode=args.placement_mode)
    heuristic = HeuristicAgent(seed=args.seed + 11)
    line = LineBuilderAgent(seed=args.seed + 13)
    basic = BasicHeuristicAgent(seed=args.seed + 17)
    random_agent = RandomAgent(seed=args.seed + 19)

    observations: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    counts = {"heuristic": 0, "line": 0, "basic": 0, "random_execute": 0, "games": 0}

    while len(actions) < args.samples:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        counts["games"] += 1
        while not done and len(actions) < args.samples:
            selected_teacher = choose_teacher(args, rng)
            action_mask = env.legal_action_mask()
            action = teacher_action(env, selected_teacher, heuristic, line, basic, random_agent)
            observations.append(obs.copy())
            masks.append(action_mask.copy())
            actions.append(int(action))
            counts[selected_teacher] += 1

            executed_action = action
            if args.explore_prob > 0.0 and rng.random() < args.explore_prob:
                executed_action = random_legal_action(action_mask, rng)
                counts["random_execute"] += 1
            obs, _, terminated, truncated, _ = env.step(executed_action)
            done = bool(terminated or truncated)

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
        np.asarray(actions, dtype=np.int64),
        counts,
    )


def append_csv_row(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: str,
    model: TorchPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    samples: int,
    extra: Dict[str, object],
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": "torchrl_ppo",
        "framework": "TorchRL",
        "pretraining": "behavior_clone",
        "episodes": 0,
        "bc_samples": int(samples),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hidden_size": int(args.hidden_size),
        "seed": int(args.seed),
        "updated_at_unix": time.time(),
        **extra,
    }
    torch.save(payload, save_path)
    with (save_path.with_suffix(save_path.suffix + ".json")).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": "torchrl_ppo",
                "framework": "TorchRL",
                "pretraining": "behavior_clone",
                "checkpoint": str(save_path),
                "bc_samples": int(samples),
                "hidden_size": int(args.hidden_size),
                "seed": int(args.seed),
                **extra,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Behavior-clone heuristic actions into PPO policy.")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu", "mps"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--placement-mode", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    parser.add_argument("--teacher", type=str, default="mixed", choices=["heuristic", "line", "basic", "mixed"])
    parser.add_argument("--teacher-heuristic-prob", type=float, default=0.70)
    parser.add_argument("--teacher-line-prob", type=float, default=0.25)
    parser.add_argument("--teacher-basic-prob", type=float, default=0.05)
    parser.add_argument("--explore-prob", type=float, default=0.08)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "behavior_clone_torchrl.pt"))
    parser.add_argument("--log-csv", type=str, default=str(root / "models" / "behavior_clone_log.csv"))
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--skip-if-done", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done_file = Path(args.done_file or args.save_path + ".done")
    if args.skip_if_done and done_file.exists():
        print(f"Done marker exists; skipping behavior cloning: {done_file}")
        return

    set_global_seeds(args.seed)
    device = resolve_torch_device(args.device)
    started_at = time.time()
    print(f"Behavior cloning device: {device}")
    print(f"Collecting {args.samples} teacher states ({args.teacher}, {args.placement_mode})")
    observations, masks, actions, counts = collect_samples(args)

    model = TorchPolicyValueNet(hidden_sizes=hidden_sizes_from_arg(args.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 101)
    indices = np.arange(actions.shape[0])
    final_loss = 0.0
    final_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        model.train()
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            obs_t = torch.as_tensor(observations[batch_indices], dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(masks[batch_indices], dtype=torch.bool, device=device)
            action_t = torch.as_tensor(actions[batch_indices], dtype=torch.long, device=device)
            logits, _ = model(obs_t)
            masked_logits = mask_logits(logits, mask_t)
            loss = F.cross_entropy(masked_logits, action_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(masked_logits, dim=-1)
                total_correct += int((pred == action_t).sum().item())
            total_loss += float(loss.item()) * len(batch_indices)
            total_seen += len(batch_indices)

        final_loss = total_loss / max(total_seen, 1)
        final_accuracy = total_correct / max(total_seen, 1)
        row = {
            "time_unix": time.time(),
            "algo": "behavior_clone",
            "epoch": epoch,
            "samples": int(actions.shape[0]),
            "loss": final_loss,
            "accuracy": final_accuracy,
            "device": str(device),
            "placement_mode": args.placement_mode,
            "teacher": args.teacher,
            "elapsed_seconds": time.time() - started_at,
            **counts,
        }
        append_csv_row(args.log_csv, row)
        print(
            f"epoch={epoch}/{args.epochs} loss={final_loss:.4f} "
            f"accuracy={final_accuracy:.3f} samples={actions.shape[0]}"
        )

    extra = {
        "completed": True,
        "final_bc_loss": final_loss,
        "final_bc_accuracy": final_accuracy,
        "placement_mode": args.placement_mode,
        "teacher": args.teacher,
        **counts,
    }
    save_checkpoint(args.save_path, model, optimizer, args, int(actions.shape[0]), extra)
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(
        f"completed samples={actions.shape[0]} accuracy={final_accuracy:.6f} path={args.save_path}\n",
        encoding="utf-8",
    )
    print(f"Behavior cloning completed: {args.save_path}")


if __name__ == "__main__":
    main()
