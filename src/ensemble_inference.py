import os
import torch
import numpy as np
from packing_core.utils import load_trained_model, pack_single_manifest

class EnsembleRouter:
    def __init__(self, small_ckpt="small_gnn_model.pt", medium_ckpt="medium_gnn_model.pt", large_ckpt="large_gnn_model.pt"):
        print("Initializing Specialized Ensemble Router...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine path relative to root
        base_path = "." if os.path.exists(small_ckpt) else ".."
            
        small_path = os.path.join(base_path, small_ckpt)
        medium_path = os.path.join(base_path, medium_ckpt)
        large_path = os.path.join(base_path, large_ckpt)

        # Load models
        try:
            print(f"Loading Small Model from {small_path}...")
            self.actor_small, _ = load_trained_model(checkpoint_path=small_path, device=self.device)
            print("Small Model Online.")
        except Exception as e:
            print(f"Warning: Small Model offline: {e}")
            self.actor_small = None

        try:
            print(f"Loading Medium Model from {medium_path}...")
            self.actor_medium, _ = load_trained_model(checkpoint_path=medium_path, device=self.device)
            print("Medium Model Online.")
        except Exception as e:
            print(f"Warning: Medium Model offline: {e}")
            self.actor_medium = None

        try:
            print(f"Loading Large Model from {large_path}...")
            self.actor_large, _ = load_trained_model(checkpoint_path=large_path, device=self.device)
            print("Large Model Online.")
        except Exception as e:
            print(f"Warning: Large Model offline: {e}")
            self.actor_large = None

    def route_manifest(self, ship_grids, manifest):
        """
        Calculates the total volume of the ship and routes the payload to the specialized model.
        """
        # Calculate total SCU volume across all grids
        total_vol = 0
        for grid in ship_grids:
            dims = grid[0]
            total_vol += dims[0] * dims[1] * dims[2]

        print(f"Incoming Ship Volume: {total_vol} SCU. Routing to specialized model...")

        # Select model based on breakpoints
        if total_vol <= 64 and self.actor_small is not None:
            print("Routed to: SMALL MODEL")
            selected_actor = self.actor_small
        elif total_vol <= 256 and self.actor_medium is not None:
            print("Routed to: MEDIUM MODEL")
            selected_actor = self.actor_medium
        elif self.actor_large is not None:
            print("Routed to: LARGE MODEL")
            selected_actor = self.actor_large
        else:
            # Fallback
            print("WARNING: Falling back to first available model due to offline specialized models.")
            selected_actor = self.actor_large or self.actor_medium or self.actor_small

        if selected_actor is None:
            raise ValueError("All specialized models are offline. Cannot route request.")

        # Run inference using the selected specialist
        return pack_single_manifest(selected_actor, ship_grids, manifest, device=self.device)
