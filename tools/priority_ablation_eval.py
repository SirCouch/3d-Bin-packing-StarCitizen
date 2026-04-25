"""Priority-diversity ablation: same manifest, three priority configs.

Run this from the repo root:
    py tools/priority_ablation_eval.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from packing_core.utils import load_trained_model, pack_single_manifest
from scu_manifest_generator import (
    SCU_DEFINITIONS,
    generate_scu_manifest,
    manifest_to_item_list,
)

SHIP_NAME = "Polaris"
GRIDS_LIST = [((6, 12, 4), "Right"), ((6, 12, 4), "Left")]
CHECKPOINT = "large_gnn_model.pt"
SEED = 1234


def make_manifest(spec):
    """Build a manifest from a spec dict {scu_type: {priority: count, ...}, ...}."""
    out = []
    for scu_type, prio_counts in spec.items():
        for priority, count in prio_counts.items():
            for _ in range(count):
                out.append({"scu_type": scu_type, "quantity": 1, "priority": priority})
    out.sort(key=lambda x: (x["priority"], -SCU_DEFINITIONS[x["scu_type"]]["volume"]))
    return out


# Manifests reconstructed from the user's pre-patch eval output.
# Same cargo composition (46 items, 190 SCU), three different priority assignments.
RUN_1_SPEC = {  # 4 tiers — pre-patch result: 47.8% (22 placed)
    "24 SCU": {1: 2},
    "8 SCU":  {2: 8},
    "4 SCU":  {1: 6, 3: 5},
    "2 SCU":  {3: 3, 4: 6},
    "1 SCU":  {2: 9, 3: 7},
}
RUN_2_SPEC = {  # 2 tiers — pre-patch result: 89.1% (41 placed)
    "24 SCU": {1: 2},
    "8 SCU":  {1: 5, 2: 3},
    "4 SCU":  {1: 6, 2: 5},
    "2 SCU":  {2: 9},
    "1 SCU":  {1: 9, 2: 7},
}
RUN_3_SPEC = {  # 3 tiers — pre-patch result: 76.1% (35 placed)
    "24 SCU": {1: 2},
    "8 SCU":  {1: 5, 3: 3},
    "4 SCU":  {1: 6, 2: 5},
    "2 SCU":  {3: 9},
    "1 SCU":  {2: 16},
}


def format_result(label, num_groups, result):
    metrics = result["metrics"]
    placed = int(metrics["success_rate"] * metrics["total_items"])
    lines = []
    lines.append("=" * 60)
    lines.append(f"{label}  (priority groups: {num_groups})")
    lines.append("=" * 60)
    lines.append(f"Success rate:       {metrics['success_rate']*100:.1f}% ({placed} placed)")
    lines.append(f"Volume utilization: {metrics['volume_utilization']*100:.1f}%")
    lines.append(f"Unplaced items:     {metrics['total_items'] - placed}")
    lines.append("=" * 60)

    by_grid = {}
    for p in result["placements"]:
        by_grid.setdefault(p["grid_name"], []).append(p)
    for gname, plist in by_grid.items():
        lines.append(f"\n[Grid: {gname}]  {len(plist)} items")
        for p in plist:
            scu = p.get("scu_type", "?")
            scu_num = scu.replace(" SCU", "").strip()
            pos = p["position"]
            dims = p["dimensions"]
            lines.append(
                f"  P{p['priority']}  {scu_num:>4} SCU  "
                f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  "
                f"dims=({int(dims[0])}x{int(dims[1])}x{int(dims[2])})"
            )

    diag = result.get("diagnostics") or {}
    missed = diag.get("missed_placements", [])
    truly = diag.get("skipped_items", [])
    if missed or truly:
        lines.append("\n" + "=" * 60)
        lines.append("DIAGNOSTICS")
        lines.append("=" * 60)
    if missed:
        lines.append(f"\nMER tracking missed {len(missed)} feasible placement(s):")
        for m in missed:
            lines.append(
                f"  item#{m['item_idx']} dims={m['dims']} P{m['priority']} "
                f"→ could fit at {m['feasible_position']} on '{m['grid_name']}' (rot={m['rotation']})"
            )
    if truly:
        lines.append(f"\nNo-fit given current layout (upstream packing likely suboptimal): {len(truly)}")
        for s in truly:
            lines.append(f"  item#{s['item_idx']} dims={s['dims']} P{s['priority']}")

    return "\n".join(lines)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    grid_dims_only = [g[0] for g in GRIDS_LIST]
    print(f"Ship: {SHIP_NAME}  grids={grid_dims_only}  ship_vol={sum(g[0]*g[1]*g[2] for g in grid_dims_only)} SCU")

    actor, _ = load_trained_model(CHECKPOINT)

    runs = [
        ("RUN 1 — 4 priority tiers (pre-patch: 47.8%)", 4, RUN_1_SPEC),
        ("RUN 2 — 2 priority tiers (pre-patch: 89.1%)", 2, RUN_2_SPEC),
        ("RUN 3 — 3 priority tiers (pre-patch: 76.1%)", 3, RUN_3_SPEC),
    ]
    for label, num_groups, spec in runs:
        manifest = make_manifest(spec)
        total_items = sum(e["quantity"] for e in manifest)
        total_scu = sum(SCU_DEFINITIONS[e["scu_type"]]["volume"] for e in manifest)
        print(f"\n>>> {label}: {total_items} items, {total_scu} SCU")
        result = pack_single_manifest(actor, GRIDS_LIST, manifest, diagnose=True)
        print(format_result(label, num_groups, result))


if __name__ == "__main__":
    main()
