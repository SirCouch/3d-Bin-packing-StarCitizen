"""Regression-smoke runner for the manifest-generator refactor (v2).

Trains a small-category model from scratch for a fixed number of episodes
under a single seed, dumping reward + success-rate curves to JSON. Used
to compare v1 (pre-refactor) vs v2 (post-refactor) generator distributions.

Usage:
    py tools/regression_smoke.py --label v1_baseline --episodes 200 --seed 17
    py tools/regression_smoke.py --label v2_post     --episodes 200 --seed 17

Outputs:
    tasks/regression_smoke_<label>.json
    tasks/regression_smoke_<label>.log         (stdout from train_agent)
    tasks/regression_smoke_<label>.pt          (throwaway checkpoint, not the production model)
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from packing_core.utils import train_agent  # noqa: E402
from train_ensemble import categorize_ships  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_manifest(seed: int) -> list:
    """Capture one manifest sample for documentation."""
    from scu_manifest_generator import generate_scu_manifest
    set_all_seeds(seed)
    return generate_scu_manifest(
        grids_list=[(4, 4, 4)],
        target_fill_ratio=0.8,
        difficulty="medium",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="run label, e.g. 'v1_baseline'")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    out_dir = REPO_ROOT / "tasks"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"regression_smoke_{args.label}.json"
    ckpt = out_dir / f"regression_smoke_{args.label}.pt"

    # Refuse to overwrite the production model paths.
    forbidden = {"small_gnn_model.pt", "medium_gnn_model.pt", "large_gnn_model.pt"}
    if ckpt.name in forbidden:
        raise SystemExit(f"refusing to write checkpoint to production path {ckpt}")

    set_all_seeds(args.seed)

    small_ships, _, _ = categorize_ships()
    if not small_ships:
        raise SystemExit("no small ships categorized — check ships_cargo_grids.json")

    print(f"[smoke:{args.label}] seed={args.seed} episodes={args.episodes} "
          f"small_ships={len(small_ships)}")
    print(f"[smoke:{args.label}] checkpoint -> {ckpt}")

    # Match train_ensemble small-stage hyperparameters exactly.
    actor, critic, metrics = train_agent(
        possible_ships=small_ships,
        num_episodes=args.episodes,
        gamma=0.95,
        gae_lambda=0.95,
        lr_actor=1e-4,
        lr_critic=1e-4,
        batch_size=32,
        replay_buffer_size=10000,
        weight_decay=1e-4,
        ppo_epochs=4,
        ppo_clip=0.2,
        entropy_coeff=0.02,
        target_critic_tau=0.01,
        print_interval=50,
        save_interval=args.episodes,  # save once at the end
        checkpoint_path=str(ckpt),
        hidden_dim=128,
        resume=False,
    )

    rewards = list(metrics.get("episode_rewards", []))
    actor_losses = list(metrics.get("actor_losses", []))
    critic_losses = list(metrics.get("critic_losses", []))

    summary = {
        "label": args.label,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_completed": len(rewards),
        "reward_mean_first50": float(np.mean(rewards[:50])) if rewards else None,
        "reward_mean_last50": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else None,
        "reward_max": float(max(rewards)) if rewards else None,
        "actor_loss_mean": float(np.mean(actor_losses)) if actor_losses else None,
        "critic_loss_mean": float(np.mean(critic_losses)) if critic_losses else None,
        "episode_rewards": [float(r) for r in rewards],
        "sample_manifest_seed17": sample_manifest(17),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[smoke:{args.label}] wrote {out_json}")
    f50 = summary["reward_mean_first50"]
    l50 = summary["reward_mean_last50"]
    f50s = f"{f50:.3f}" if f50 is not None else "n/a"
    l50s = f"{l50:.3f}" if l50 is not None else "n/a"
    print(f"[smoke:{args.label}] first50_mean={f50s} last50_mean={l50s}")


if __name__ == "__main__":
    main()
