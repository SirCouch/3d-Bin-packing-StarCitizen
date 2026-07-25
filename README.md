# Star Citizen 3D Bin Packing

A graph-neural-network and PPO solver for placing Star Citizen SCU cargo
across one or more ship cargo grids. The solver combines a learned
MER-and-rotation policy with deterministic geometry checks for overlap,
support, stacking order, unloading priority, and irregular blocked space.

The repository now has one public evaluation report:
[EVALUATION.md](EVALUATION.md). For implementation details and the project
timeline, see [Current architecture](docs/ARCHITECTURE.md) and
[Project history](docs/HISTORY.md).

## Current status

The current system is a three-model specialist ensemble:

| Specialist | Routed usable volume | Training episodes | Hidden width |
|---|---:|---:|---:|
| Small | up to 64 SCU | 500 | 128 |
| Medium | 65-256 SCU | 3,000 | 128 |
| Large | above 256 SCU | 2,500 | 256 |

The published V6 benchmark evaluated 43 ships with 10 hard manifests per
ship. It reported 95.70% mean item-placement success and 66.45% mean usable
volume utilization. Those are macro averages across ships; read the
[full report](EVALUATION.md) for the protocol, per-ship results, and
comparability limits.

## Repository layout

| Path | Purpose |
|---|---|
| `src/packing_core/` | Geometry, MER tracking, graph environment, GNN models, PPO training, and inference helpers |
| `src/train_ensemble.py` | Train or resume the three specialists |
| `src/evaluate_model.py` | Canonical evaluator and `EVALUATION.md` generator |
| `src/ensemble_inference.py` | Volume-based specialist router |
| `src/api_backend.py` | Flask inference API |
| `src/scu_manifest_generator.py` | Drop-off-oriented synthetic cargo generation |
| `tests/` | Geometry, graph, manifest, evaluation, and API regression tests |
| `tools/` | Reusable extraction, ablation, and regression-smoke utilities |
| `evaluation/results/` | Machine-readable data retained for published evaluations |
| `ships_cargo_grids.json` | Runtime ship geometry, including optional blocker cuboids |
| `ships_cargo_grids_positioned.json` | Extraction/reference data with source positions |
| `docs/` | Architecture and project-evolution documentation |

Generated plots and logs belong in `artifacts/` and are ignored by Git.
Historical experiments and superseded scripts belong in `archive/`, which is
also ignored. Trained `.pt` files are intentionally not versioned.

## Installation

Python 3.9 or newer is required. This repository does not currently pin a
dependency lock file, so install the runtime and test dependencies directly:

```bash
pip install torch torch-geometric flask numpy matplotlib requests pytest
```

PyTorch Geometric installation can vary by PyTorch/CUDA combination. Confirm
that `import torch_geometric` works before starting a long training run.

## Common commands

Run commands from the repository root:

```bash
python src/train_ensemble.py
python src/evaluate_model.py --episodes 10 --seed 17
python src/api_backend.py
pytest tests --ignore=tests/test_api.py
python tools/regression_smoke.py --label local_check --episodes 20 --seed 17
```

`tests/test_api.py` is a live-server integration check. Run it separately
after starting `python src/api_backend.py`.

The evaluator supports partial runs without separate scripts:

```bash
python src/evaluate_model.py --category small --episodes 10
python src/evaluate_model.py --category small --category medium
python src/evaluate_model.py --json evaluation/results/latest.json
```

By default, training and inference use these root-level checkpoints:

- `small_gnn_model.pt`
- `medium_gnn_model.pt`
- `large_gnn_model.pt`

Training resumes a checkpoint when it exists. Create a `STOP_TRAINING` file
to request a checkpointed stop at the next episode boundary.

## API

Start the server on `127.0.0.1:8000`:

```bash
python src/api_backend.py
```

Example request:

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "ship_grids": [
      [[4, 10, 2], "Cargo Module 1"],
      [[4, 6, 2], "Front Module"]
    ],
    "manifest": [
      {"scu_type": "4 SCU", "quantity": 2, "priority": 1},
      {"scu_type": "1 SCU", "quantity": 5, "priority": 2}
    ]
  }'
```

Each grid may instead be an object with `name`, `dimensions`, and an optional
`blocked` list. A blocker contains `position`, `dimensions`, and optionally
`supports` to control whether cargo may use its upper face as support.

## Important model boundaries

- Only rotation around the Z axis is considered; height stays fixed.
- The policy selects a maximal empty region and one of two X/Y orientations.
  Deterministic checks decide whether the action is physically legal.
- Priority is a hard front-to-back ordering constraint within each grid, not
  a full unloading-path simulation.
- The vector-environment wrapper is experimental and is not connected to the
  current training loop.
- Published checkpoints and reports are experiment artifacts, not formal
  semantic releases.

These boundaries, the exact reward function, graph schema, PPO update, and
known technical debt are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
