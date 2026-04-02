# Star Citizen 3D Bin-Packing (GNN)

An advanced 3D bin-packing solver tailored for Star Citizen cargo grids, using Graph Attention Networks (GAT) and Proximal Policy Optimization (PPO) to optimize cargo placement across entire ships.

## Overview

This project solves 3D bin-packing within Star Citizen's Standard Cargo Unit (SCU) system. An ensemble of three specialized GNN models learns to efficiently pack 7 SCU container types (1–32 SCU) into ship cargo grids while enforcing:

- **Z-Axis Rotation** — largest surface area always faces the grid floor
- **Heavy-Stacking Physics** — heavier boxes cannot rest on lighter ones
- **Priority Accessibility** — higher-priority cargo is placed closer to the front for unloading
- **Physical Fit Validation** — manifests only contain containers that geometrically fit the ship's grids

## Architecture

### Ensemble System

Three size-specialized models, each with a shared Actor-Critic GNN backbone:

| Model | Ships | Episodes | Hidden Dim |
|-------|-------|----------|-----------|
| Small | ≤64 SCU | 500 | 128 |
| Medium | 65–256 SCU | 2,000 | 128 |
| Large | >256 SCU | 3,000 | 256 |

`EnsembleRouter` (`src/ensemble_inference.py`) selects the appropriate model based on total ship volume.

### Training Algorithm

- **PPO** with clipped surrogate objective (ε=0.2) and clipped value loss
- **GAE** (λ=0.95) for advantage estimation
- **Entropy bonus** (β=0.02) to prevent premature policy collapse
- **Target critic** with soft updates (τ=0.01) for stable value bootstrapping
- **Reward normalization** via running mean/variance (Welford's algorithm)
- **Shared GNN backbone** between actor and critic with cross-attention MER scoring
- **Performance-weighted ship sampling** — focuses training on ships with lowest success rates
- **Curriculum learning** — difficulty ramps from very-easy to very-hard across training

### Core Modules (`src/packing_core/`)

| Module | Purpose |
|--------|---------|
| `models.py` | `SharedGNNBackbone` (3-layer GAT), `ActorGNN` (cross-attention + MER scoring), `CriticGNN` (value head) |
| `drl_env.py` | `DRLBinPackingEnv` — Gym-like environment with multi-grid state, MER management, constraint enforcement |
| `mer_manager.py` | Maximal Empty Rectangle tracking per grid (Box3D objects, up to 6 new MERs per placement) |
| `box3d.py` | 3D geometry primitive — overlap, containment, fit checks (GPU tensors) |
| `utils.py` | `train_agent()`, `pack_single_manifest()`, `load_trained_model()`, GAE/batching utilities |

### Performance (very-hard difficulty, 28 ships)

| Category | Avg Success Rate | Avg Volume Utilization |
|----------|-----------------|----------------------|
| Small (12 ships) | 94% | 70% |
| Medium (12 ships) | 78% | 55% |
| Large (4 ships) | 72% | 51% |
| **Global** | **84%** | **61%** |

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- PyTorch Geometric
- Flask (for the API)
- NumPy, Matplotlib

### Installation

```bash
git clone https://github.com/SirCouch/3d-Bin-packing-StarCitizen.git
cd 3d-Bin-packing-StarCitizen
pip install torch torch-geometric flask numpy matplotlib
```

### Usage

#### Train the Ensemble
```bash
python src/train_ensemble.py
```

#### Evaluate Model Performance
```bash
python src/evaluate_model.py
```

#### Start the API Server (port 8000)
```bash
python src/api_backend.py
```

#### API Example
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "ship_grids": [[[4, 10, 2], "Cargo Module 1"], [[4, 6, 2], "Front Module"]],
    "manifest": [
      {"scu_type": "4 SCU", "quantity": 2, "priority": 1},
      {"scu_type": "1 SCU", "quantity": 5, "priority": 2}
    ]
  }'
```

### Ship Data

`ships_cargo_grids.json` contains 28 Star Citizen ships with their cargo grid dimensions, from the Avenger Titan (8 SCU) to the C2 Hercules (696 SCU).

## Contributing

Contributions, suggestions, and feedback are welcome!