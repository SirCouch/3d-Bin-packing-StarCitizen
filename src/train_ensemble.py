import json
import torch
import numpy as np
from packing_core.utils import train_agent, load_trained_model, pack_single_manifest

def categorize_ships():
    try:
        with open('ships_cargo_grids.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print("Failed to load ships:", e)
        return [], [], []

    small_ships = []
    medium_ships = []
    large_ships = []

    for ship in data['ships']:
        # Format for environment: [[(dim_x, dim_y, dim_z), "Name"], ...]
        ship_format = [[(g['dimensions'][0], g['dimensions'][1], g['dimensions'][2]), g['name']] for g in ship['grids']]
        
        total_vol = sum(g['dimensions'][0] * g['dimensions'][1] * g['dimensions'][2] for g in ship['grids'])
        num_grids = len(ship['grids'])
        
        # Calculate volume variance if multiple grids
        vols = [g['dimensions'][0] * g['dimensions'][1] * g['dimensions'][2] for g in ship['grids']]
        max_vol_ratio = max(vols) / sum(vols) if sum(vols) > 0 else 1.0

        # Stratify sampling: ships with multiple grids or high variance get duplicated
        # This ensures the model encounters irregular ships frequently
        is_irregular = num_grids > 1 and max_vol_ratio < 0.9
        copies = 3 if is_irregular else (2 if num_grids > 1 else 1)
        
        for _ in range(copies):
            if total_vol <= 64:
                small_ships.append(ship_format)
            elif total_vol <= 256:
                medium_ships.append(ship_format)
            else:
                large_ships.append(ship_format)

    print(f"Categorized {len(small_ships)} Small ships, {len(medium_ships)} Medium ships, and {len(large_ships)} Large ships.")
    return small_ships, medium_ships, large_ships

def train_ensemble():
    small_ships, medium_ships, large_ships = categorize_ships()

    # Tuned PPO/GAE hyperparameters
    shared_params = dict(
        gamma=0.95,
        gae_lambda=0.95,
        lr_actor=1e-4,          # lower — shared backbone gets 2x gradient signal
        lr_critic=1e-4,
        batch_size=32,
        replay_buffer_size=10000,
        weight_decay=1e-4,
        ppo_epochs=4,
        ppo_clip=0.2,
        entropy_coeff=0.02,     # higher — more exploration with cross-attention arch
        target_critic_tau=0.01,
    )

    import os
    stages = [
        ("SMALL SHIPS", small_ships, "small_gnn_model.pt", 500, 100, 100, 128),
        ("MEDIUM SHIPS", medium_ships, "medium_gnn_model.pt", 3000, 200, 500, 128),
        ("LARGE/MASSIVE SHIPS", large_ships, "large_gnn_model.pt", 2500, 100, 500, 256),
    ]

    for name, ships, ckpt, eps, print_iv, save_iv, hidden in stages:
        if not ships:
            continue
        resume = os.path.exists(ckpt)
        if resume:
            import torch
            try:
                data = torch.load(ckpt, map_location='cpu')
                if data.get('episode', -1) + 1 >= eps:
                    print(f"--- Skipping {name}: checkpoint already at episode {data['episode']+1}/{eps} ---")
                    continue
                print(f"--- Resuming {name} from episode {data.get('episode', -1)+1}/{eps} ---")
            except Exception as e:
                print(f"[warn] Could not inspect {ckpt}: {e}. Starting fresh.")
                resume = False
        else:
            print(f"--- Training {name} from scratch ---")

        result = train_agent(
            possible_ships=ships,
            num_episodes=eps,
            print_interval=print_iv,
            save_interval=save_iv,
            checkpoint_path=ckpt,
            hidden_dim=hidden,
            resume=resume,
            **shared_params,
        )
        if isinstance(result, tuple) and len(result) == 3 and result[2].get('stopped'):
            print(f"[pause] Training halted during {name}. Delete STOP_TRAINING file and rerun to resume.")
            return

def run_post_training_eval():
    """Kick off the full ensemble evaluation report after training completes."""
    try:
        from evaluate_model import generate_full_ensemble_evaluation_report
        print("\n" + "=" * 70)
        print("Training complete. Running ensemble evaluation...")
        print("=" * 70)
        generate_full_ensemble_evaluation_report()
    except Exception as e:
        print(f"[eval] Post-training evaluation failed: {e}")


if __name__ == "__main__":
    train_ensemble()
    run_post_training_eval()
