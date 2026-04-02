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

    if small_ships:
        print("--- Training Specialized Model: SMALL SHIPS ---")
        train_agent(
            possible_ships=small_ships,
            num_episodes=500,
            print_interval=100,
            save_interval=100,
            checkpoint_path="small_gnn_model.pt",
            hidden_dim=128,
            **shared_params,
        )

    if medium_ships:
        print("--- Training Specialized Model: MEDIUM SHIPS ---")
        train_agent(
            possible_ships=medium_ships,
            num_episodes=2000,
            print_interval=200,
            save_interval=400,
            checkpoint_path="medium_gnn_model.pt",
            hidden_dim=128,
            **shared_params,
        )

    if large_ships:
        print("--- Training Specialized Model: LARGE/MASSIVE SHIPS ---")
        train_agent(
            possible_ships=large_ships,
            num_episodes=3000,
            print_interval=100,
            save_interval=300,
            checkpoint_path="large_gnn_model.pt",
            hidden_dim=256,
            **shared_params,
        )

if __name__ == "__main__":
    train_ensemble()
