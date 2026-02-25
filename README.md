# Star Citizen 3D Bin-Packing (GNN)

An advanced 3D bin-packing solver tailored for Star Citizen cargo grids, utilizing Graph Neural Networks (GNN) and Reinforcement Learning to optimize cargo placement.

## Overview

This project aims to solve the complex problem of 3D bin-packing within the specific constraints of Star Citizen's Standard Cargo Unit (SCU) system. It features a GNN-based model that learns to efficiently pack various SCU container sizes into ship cargo grids while considering priority, accessibility, and volume utilization.

## Project Structure

- `src/`: Core application logic.
  - `gnn_bin_packing.py`: The final GNN-based reinforcement learning model.
  - `scu_manifest_generator.py`: Utility for generating realistic Star Citizen cargo manifests and training data.
  - `api_backend.py`: Flask-based API for integrating the solver with external applications or web interfaces.
- `archive/`: Experimental models and legacy testing environments (including DRL and ILP approaches).

## Key Features

- **GNN-based Solver**: Uses Graph Attention Networks (GAT) to represent the spatial relationships between cargo items and potential placement locations.
- **SCU Aware**: Specifically designed for Star Citizen cargo units (1 SCU, 2 SCU, 8 SCU, etc.).
- **GPU Accelerated**: Utilizes Numba for CUDA-accelerated placement validation.
- **Priority Handling**: Considers cargo priority for optimized unloading and accessibility.

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- PyTorch Geometric
- Numba (for GPU acceleration)
- Flask (for the API)
- PuLP (for optimization constraints)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/3d-Bin-packing-StarCitizen.git
   cd 3d-Bin-packing-StarCitizen
   ```

2. Install dependencies:
   ```bash
   pip install torch torch-geometric flask numba pulp numpy matplotlib
   ```

### Usage

#### Running the GNN Model
To train or run the GNN bin-packing solver:
```bash
python src/gnn_bin_packing.py
```

#### Starting the API Backend
To start the Flask server for external integration:
```bash
python src/api_backend.py
```

## Contributing

This project is a work-in-progress. Contributions, suggestions, and feedback are welcome!
