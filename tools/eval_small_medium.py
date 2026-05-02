"""Eval small + medium models. Safe to run alongside large-stage training.

Both small_gnn_model.pt and medium_gnn_model.pt are frozen on disk while
the training process is on the large stage.

Usage:  py tools/eval_small_medium.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_model import categorize_ships_for_eval, evaluate_model_on_ships  # noqa: E402

SHIPS_FILE = REPO_ROOT / "ships_cargo_grids.json"


def eval_one(label: str, checkpoint: Path, ships: dict, n_per_ship: int = 10) -> None:
    if not checkpoint.exists():
        print(f"[skip {label}] missing checkpoint: {checkpoint}")
        return
    if not ships:
        print(f"[skip {label}] no ships in this category")
        return

    print(f"\n=== Eval: {label} vs {len(ships)} ships ({n_per_ship} eps each, hard difficulty) ===")
    results, success, util = evaluate_model_on_ships(
        str(checkpoint), ships, num_episodes_per_ship=n_per_ship,
    )
    print(f"\n--- {label} summary ---")
    print(f"  ships:       {len(results)}")
    print(f"  avg success: {np.mean(success):6.2f}%   median: {np.median(success):6.2f}%")
    print(f"  avg util:    {np.mean(util):6.2f}%   median: {np.median(util):6.2f}%")
    print(f"  worst ship:  {min(success):6.2f}%")
    print(f"  best ship:   {max(success):6.2f}%")
    print(f"\n--- {label} per-ship ---")
    for r in sorted(results, key=lambda x: x["avg_success"]):
        print(f"  {r['ship_name']:30s} grids={r['grids_count']} "
              f"vol={r['total_vol']:>4d}  "
              f"success={r['avg_success']:6.2f}%  util={r['avg_utilization']:6.2f}%")


def main() -> None:
    data = json.loads(SHIPS_FILE.read_text())
    small_ships, medium_ships, _ = categorize_ships_for_eval(data)
    eval_one("SMALL",  REPO_ROOT / "small_gnn_model.pt",  small_ships)
    eval_one("MEDIUM", REPO_ROOT / "medium_gnn_model.pt", medium_ships)


if __name__ == "__main__":
    main()
