import json
from pathlib import Path

import torch

from packing_core.utils import train_agent
from scu_manifest_generator import get_total_usable_volume


REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPS_PATH = REPO_ROOT / "ships_cargo_grids.json"


def _grid_from_json(grid):
    dims = (
        grid["dimensions"][0],
        grid["dimensions"][1],
        grid["dimensions"][2],
    )
    blocked = grid.get("blocked", [])
    return [dims, grid["name"], blocked] if blocked else [dims, grid["name"]]


def categorize_ships(ships_path=SHIPS_PATH):
    try:
        with open(ships_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        print("Failed to load ships:", exc)
        return [], [], []

    small_ships = []
    medium_ships = []
    large_ships = []

    for ship in data["ships"]:
        ship_format = [_grid_from_json(grid) for grid in ship["grids"]]
        total_vol = get_total_usable_volume(ship_format)
        num_grids = len(ship["grids"])

        volumes = [get_total_usable_volume([grid]) for grid in ship_format]
        max_vol_ratio = max(volumes) / sum(volumes) if sum(volumes) > 0 else 1.0

        # Duplicate multi-grid layouts so they remain visible in the pool.
        is_irregular = num_grids > 1 and max_vol_ratio < 0.9
        copies = 3 if is_irregular else (2 if num_grids > 1 else 1)

        for _ in range(copies):
            if total_vol <= 64:
                small_ships.append(ship_format)
            elif total_vol <= 256:
                medium_ships.append(ship_format)
            else:
                large_ships.append(ship_format)

    print(
        f"Categorized {len(small_ships)} small ships, "
        f"{len(medium_ships)} medium ships, and "
        f"{len(large_ships)} large ships."
    )
    return small_ships, medium_ships, large_ships


def train_ensemble():
    small_ships, medium_ships, large_ships = categorize_ships()

    shared_params = {
        "gamma": 0.95,
        "gae_lambda": 0.95,
        # Shared backbone parameters receive both optimizer steps.
        "lr_actor": 1e-4,
        "lr_critic": 1e-4,
        "batch_size": 32,
        "replay_buffer_size": 10000,
        "weight_decay": 1e-4,
        "ppo_epochs": 4,
        "ppo_clip": 0.2,
        "entropy_coeff": 0.02,
        "target_critic_tau": 0.01,
    }

    stages = [
        (
            "SMALL SHIPS",
            small_ships,
            REPO_ROOT / "small_gnn_model.pt",
            500,
            100,
            100,
            128,
        ),
        (
            "MEDIUM SHIPS",
            medium_ships,
            REPO_ROOT / "medium_gnn_model.pt",
            3000,
            200,
            500,
            128,
        ),
        (
            "LARGE/MASSIVE SHIPS",
            large_ships,
            REPO_ROOT / "large_gnn_model.pt",
            2500,
            100,
            500,
            256,
        ),
    ]

    for (
        name,
        ships,
        checkpoint,
        episodes,
        print_interval,
        save_interval,
        hidden,
    ) in stages:
        if not ships:
            continue

        resume = checkpoint.exists()
        if resume:
            try:
                data = torch.load(checkpoint, map_location="cpu")
                completed = data.get("episode", -1) + 1
                if completed >= episodes:
                    print(
                        f"--- Skipping {name}: checkpoint already at "
                        f"episode {completed}/{episodes} ---"
                    )
                    continue
                print(
                    f"--- Resuming {name} from episode "
                    f"{completed}/{episodes} ---"
                )
            except Exception as exc:
                print(
                    f"[warn] Could not inspect {checkpoint}: {exc}. "
                    "Starting fresh."
                )
                resume = False
        else:
            print(f"--- Training {name} from scratch ---")

        result = train_agent(
            possible_ships=ships,
            num_episodes=episodes,
            print_interval=print_interval,
            save_interval=save_interval,
            checkpoint_path=checkpoint,
            hidden_dim=hidden,
            resume=resume,
            **shared_params,
        )
        if isinstance(result, tuple) and len(result) == 3 and result[2].get("stopped"):
            print(
                f"[pause] Training halted during {name}. "
                "Delete STOP_TRAINING and rerun to resume."
            )
            return


def run_post_training_eval():
    """Run the canonical ensemble report after all training stages."""
    try:
        from evaluate_model import generate_full_ensemble_evaluation_report

        print("\n" + "=" * 70)
        print("Training complete. Running ensemble evaluation...")
        print("=" * 70)
        generate_full_ensemble_evaluation_report()
    except Exception as exc:
        print(f"[eval] Post-training evaluation failed: {exc}")


if __name__ == "__main__":
    train_ensemble()
    run_post_training_eval()
