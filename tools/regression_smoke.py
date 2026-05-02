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
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from packing_core.utils import train_agent  # noqa: E402
from train_ensemble import categorize_ships  # noqa: E402


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # CUBLAS_WORKSPACE_CONFIG must be set before any CUDA op to make
        # cuBLAS deterministic (see torch.use_deterministic_algorithms docs).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    p.add_argument("--stage", choices=["small", "medium", "large"], default="small")
    p.add_argument("--deterministic", action="store_true",
                   help="enable torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG")
    p.add_argument("--profile", action="store_true",
                   help="run training under cProfile; save top-30 to <label>_profile.txt")
    args = p.parse_args()

    out_dir = REPO_ROOT / "tasks"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"regression_smoke_{args.label}.json"
    ckpt = out_dir / f"regression_smoke_{args.label}.pt"

    # Refuse to overwrite the production model paths.
    forbidden = {"small_gnn_model.pt", "medium_gnn_model.pt", "large_gnn_model.pt"}
    if ckpt.name in forbidden:
        raise SystemExit(f"refusing to write checkpoint to production path {ckpt}")

    set_all_seeds(args.seed, deterministic=args.deterministic)

    small_ships, medium_ships, large_ships = categorize_ships()
    stage_ships = {"small": small_ships, "medium": medium_ships, "large": large_ships}[args.stage]
    if not stage_ships:
        raise SystemExit(f"no {args.stage} ships categorized — check ships_cargo_grids.json")

    hidden_dim = 128 if args.stage in ("small", "medium") else 256
    print(f"[smoke:{args.label}] stage={args.stage} seed={args.seed} "
          f"episodes={args.episodes} ships={len(stage_ships)} "
          f"deterministic={args.deterministic} hidden_dim={hidden_dim}")
    print(f"[smoke:{args.label}] checkpoint -> {ckpt}")

    # Match train_ensemble per-stage hyperparameters.
    train_kwargs = dict(
        possible_ships=stage_ships,
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
        hidden_dim=hidden_dim,
        resume=False,
    )

    t_start = time.perf_counter()
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        actor, critic, metrics = train_agent(**train_kwargs)
        profiler.disable()
        prof_path = out_dir / f"regression_smoke_{args.label}_profile.txt"
        with open(prof_path, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats(30)
            f.write("\n\n=== sort by tottime ===\n\n")
            stats.sort_stats("tottime")
            stats.print_stats(30)
        print(f"[smoke:{args.label}] cProfile -> {prof_path}")
    else:
        actor, critic, metrics = train_agent(**train_kwargs)

    t_elapsed = time.perf_counter() - t_start
    rewards = list(metrics.get("episode_rewards", []))
    actor_losses = list(metrics.get("actor_losses", []))
    critic_losses = list(metrics.get("critic_losses", []))

    summary = {
        "label": args.label,
        "stage": args.stage,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "episodes_requested": args.episodes,
        "episodes_completed": len(rewards),
        "wall_clock_sec": t_elapsed,
        "sec_per_episode": t_elapsed / max(1, len(rewards)),
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
