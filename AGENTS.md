# Repository Guidelines

## Project Structure & Module Organization

This repository implements a Star Citizen 3D bin-packing solver using GNN/PPO models.

- `src/packing_core/`: core geometry, environment, MER management, model definitions, and training utilities.
- `src/`: runnable entry points such as `train_ensemble.py`, `evaluate_model.py`, `ensemble_inference.py`, and `api_backend.py`.
- `tests/`: pytest checks for graph construction plus an API integration script.
- `tools/`: evaluation, ablation, charting, and data extraction helpers.
- `frontend/viewer.html`: standalone packing visualization UI; `frontend/vendor/` contains local browser dependencies.
- `ships_cargo_grids*.json`: ship cargo grid datasets. Model checkpoints and reports live at the repo root.

## Build, Test, and Development Commands

Install dependencies manually; there is currently no `requirements.txt` or `pyproject.toml`.

```bash
pip install torch torch-geometric flask numpy matplotlib requests pytest
python src/train_ensemble.py        # train small/medium/large specialist models
python src/evaluate_model.py        # evaluate models against ship data
python src/api_backend.py           # start Flask API on port 8000
pytest tests/test_multi_grid_graph.py
python tests/test_api.py            # requires the API server to be running
python tools/regression_smoke.py    # run targeted regression smoke checks
```

## Coding Style & Naming Conventions

Use Python 3.9+ and 4-space indentation. Prefer clear `snake_case` for functions, variables, and modules; use `PascalCase` for classes such as `DRLBinPackingEnv` and `Box3D`. Keep tensor/device handling explicit, especially in `packing_core`, and avoid broad rewrites when changing training or inference behavior. Preserve existing module-level constants such as `NODE_TYPE_*` naming.

## Testing Guidelines

Add pytest tests under `tests/` using `test_*.py` filenames and focused `test_*` functions. Prefer deterministic, small fixtures for geometry and graph behavior. API behavior can be checked with `tests/test_api.py`, but it is an integration script and requires `python src/api_backend.py` in a separate shell first.

## Commit & Pull Request Guidelines

Recent commits use concise imperative summaries, for example: `Cap interior-anchor probe to 256 positions to bound worst case`. Keep subjects specific and under roughly 72 characters when possible. Pull requests should describe the behavior change, list commands run, call out model/checkpoint impacts, and include screenshots when modifying `frontend/viewer.html` or generated visual outputs.

## Agent-Specific Instructions

Do not overwrite trained `.pt` checkpoints, JSON datasets, or evaluation reports unless the task explicitly requires regenerating them. For non-trivial changes, update or add a focused regression test and verify with the smallest relevant command before broader training or evaluation runs.
