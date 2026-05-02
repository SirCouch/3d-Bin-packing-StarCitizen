"""One-off eval of small_gnn_model.pt against all small ships.

Safe to run alongside training (small checkpoint is frozen while training is
on medium/large). Loads ~100 MB on GPU; allows shared GPU use.

Usage:  py tools/eval_small_only.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_model import categorize_ships_for_eval, evaluate_model_on_ships  # noqa: E402

SHIPS_FILE = REPO_ROOT / "ships_cargo_grids.json"
CHECKPOINT = REPO_ROOT / "small_gnn_model.pt"


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"missing checkpoint: {CHECKPOINT}")
    data = json.loads(SHIPS_FILE.read_text())
    small_ships, _, _ = categorize_ships_for_eval(data)
    if not small_ships:
        raise SystemExit("no small ships found in ships_cargo_grids.json")

    print(f"=== Eval: small_gnn_model.pt vs {len(small_ships)} small ships ===")
    print(f"(10 episodes per ship, hard difficulty)\n")

    results, success, util = evaluate_model_on_ships(
        str(CHECKPOINT), small_ships, num_episodes_per_ship=10,
    )

    print(f"\n=== Summary ===")
    print(f"Ships evaluated:          {len(results)}")
    print(f"Avg success rate:         {np.mean(success):.2f}% (median {np.median(success):.2f}%)")
    print(f"Avg volume utilization:   {np.mean(util):.2f}% (median {np.median(util):.2f}%)")
    print(f"Min success (worst ship): {min(success):.2f}%")
    print(f"Max success (best ship):  {max(success):.2f}%")

    print(f"\n=== Per-ship breakdown ===")
    for r in sorted(results, key=lambda x: x["avg_success"]):
        print(f"  {r['ship_name']:30s} grids={r['grids_count']} "
              f"vol={r['total_vol']:>4d}  success={r['avg_success']:6.2f}%  "
              f"util={r['avg_utilization']:6.2f}%")


if __name__ == "__main__":
    main()
