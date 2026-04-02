"""
Legacy entry point — all logic now lives in packing_core/.
This file re-exports the public API for backwards compatibility.
"""
import json
import torch
from packing_core.models import ActorGNN, CriticGNN
from packing_core.drl_env import DRLBinPackingEnv
from packing_core.mer_manager import MERManager
from packing_core.box3d import Box3D
from packing_core.utils import (
    train_agent,
    load_trained_model,
    pack_single_manifest,
    evaluate_agent,
    visualize_packing,
    ReplayBuffer,
    load_ships_from_json,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    ships = load_ships_from_json()
    print(f"Loaded {len(ships)} ships for training.")

    actor, critic, stats = train_agent(
        possible_ships=ships,
        num_episodes=1000,
        checkpoint_path="multi_gnn_model_checkpoint.pt",
        print_interval=10,
        save_interval=50,
    )
