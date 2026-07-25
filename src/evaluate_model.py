"""Canonical checkpoint evaluation and GitHub report generation.

This is the only general-purpose evaluation entry point. It can evaluate any
subset of the small/medium/large specialists, writes the repository's canonical
``EVALUATION.md`` report, and can optionally persist the same data as JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from packing_core.utils import load_trained_model, pack_single_manifest
from scu_manifest_generator import (
    BULK_ONLY_RUN_TYPE,
    generate_scu_manifest,
    get_grid_dimensions,
    get_total_usable_volume,
    is_bulk_only_manifest_target,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHIPS_PATH = REPO_ROOT / "ships_cargo_grids.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "EVALUATION.md"
DEFAULT_CHECKPOINTS = {
    "small": REPO_ROOT / "small_gnn_model.pt",
    "medium": REPO_ROOT / "medium_gnn_model.pt",
    "large": REPO_ROOT / "large_gnn_model.pt",
}
CATEGORY_ORDER = ("small", "medium", "large")
CATEGORY_LABELS = {
    "small": "Small Ships",
    "medium": "Medium Ships",
    "large": "Large Ships",
}


def _grid_from_json(grid):
    dims = tuple(grid["dimensions"])
    blocked = grid.get("blocked", [])
    return [dims, grid["name"], blocked] if blocked else [dims, grid["name"]]


def categorize_ships_for_eval(data):
    """Split ship definitions using the same routing thresholds as inference."""
    categorized = {category: {} for category in CATEGORY_ORDER}

    for ship in data["ships"]:
        grids = [_grid_from_json(grid) for grid in ship["grids"]]
        total_vol = get_total_usable_volume(grids)

        if total_vol <= 64:
            category = "small"
        elif total_vol <= 256:
            category = "medium"
        else:
            category = "large"
        categorized[category][ship["name"]] = grids

    return tuple(categorized[category] for category in CATEGORY_ORDER)


def set_evaluation_seed(seed):
    """Seed every RNG used by manifest generation and model evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def evaluate_model_on_ships(
    checkpoint_path,
    ships_dict,
    num_episodes_per_ship=10,
    difficulty="hard",
    target_fill_ratio=0.8,
):
    """Evaluate one specialist and return per-ship and aggregate metric lists.

    Success and utilization are first averaged across episodes for each ship.
    The returned aggregate lists therefore support macro-averaging across
    ships, which is the reporting convention used by this repository.
    """
    checkpoint_path = _resolve_repo_path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint {checkpoint_path} not found.")
        return [], [], []

    print(f"\nLoading model from {checkpoint_path}...")
    actor, _ = load_trained_model(checkpoint_path)

    results = []
    overall_success = []
    overall_utilization = []

    for ship_name, grids_list in ships_dict.items():
        print(f"Evaluating {ship_name} ({len(grids_list)} grids)...")

        total_vol = int(get_total_usable_volume(grids_list))
        dummy_dims = (total_vol, 1, 1)
        run_type = (
            BULK_ONLY_RUN_TYPE
            if is_bulk_only_manifest_target(
                [get_grid_dimensions(grid) for grid in grids_list],
                ship_name=ship_name,
            )
            else None
        )

        ship_success_rates = []
        ship_utilizations = []
        for _ in range(num_episodes_per_ship):
            manifest = generate_scu_manifest(
                grid_dims=dummy_dims,
                grids_list=grids_list,
                target_fill_ratio=target_fill_ratio,
                difficulty=difficulty,
                run_type=run_type,
            )
            result = pack_single_manifest(actor, grids_list, manifest)
            ship_success_rates.append(result["metrics"]["success_rate"])
            ship_utilizations.append(result["metrics"]["volume_utilization"])

        avg_success = float(np.mean(ship_success_rates) * 100)
        avg_utilization = float(np.mean(ship_utilizations) * 100)
        overall_success.append(avg_success)
        overall_utilization.append(avg_utilization)
        results.append(
            {
                "ship_name": ship_name,
                "grids_count": len(grids_list),
                "total_vol": total_vol,
                "avg_success": avg_success,
                "avg_utilization": avg_utilization,
            }
        )

    return results, overall_success, overall_utilization


def _metric_summary(success, utilization):
    if not success:
        return {
            "ships": 0,
            "avg_success": 0.0,
            "median_success": 0.0,
            "avg_utilization": 0.0,
            "median_utilization": 0.0,
        }
    return {
        "ships": len(success),
        "avg_success": float(np.mean(success)),
        "median_success": float(np.median(success)),
        "avg_utilization": float(np.mean(utilization)),
        "median_utilization": float(np.median(utilization)),
    }


def build_evaluation_payload(
    ships_path=DEFAULT_SHIPS_PATH,
    checkpoint_paths=None,
    categories=None,
    num_episodes_per_ship=10,
    difficulty="hard",
    target_fill_ratio=0.8,
    seed=17,
    label="Current checkpoints",
):
    """Run the requested specialists and return a JSON-serializable payload."""
    if num_episodes_per_ship < 1:
        raise ValueError("num_episodes_per_ship must be at least 1")
    if not 0 < target_fill_ratio <= 1:
        raise ValueError("target_fill_ratio must be in (0, 1]")

    ships_path = _resolve_repo_path(ships_path)
    data = json.loads(ships_path.read_text(encoding="utf-8"))
    categorized_tuple = categorize_ships_for_eval(data)
    ships_by_category = dict(zip(CATEGORY_ORDER, categorized_tuple))

    checkpoint_paths = dict(
        DEFAULT_CHECKPOINTS if checkpoint_paths is None else checkpoint_paths
    )
    selected_categories = tuple(categories or CATEGORY_ORDER)
    unknown = set(selected_categories) - set(CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")

    set_evaluation_seed(seed)
    category_payloads = []
    global_success = []
    global_utilization = []

    for category in selected_categories:
        checkpoint = _resolve_repo_path(checkpoint_paths[category])
        results, success, utilization = evaluate_model_on_ships(
            checkpoint,
            ships_by_category[category],
            num_episodes_per_ship=num_episodes_per_ship,
            difficulty=difficulty,
            target_fill_ratio=target_fill_ratio,
        )
        summary = _metric_summary(success, utilization)
        category_payloads.append(
            {
                "category": category,
                "label": CATEGORY_LABELS[category],
                "checkpoint": _display_path(checkpoint),
                "results": results,
                "summary": summary,
            }
        )
        global_success.extend(success)
        global_utilization.extend(utilization)

    return {
        "schema_version": 1,
        "label": label,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ships_path": _display_path(ships_path),
        "protocol": {
            "episodes_per_ship": num_episodes_per_ship,
            "difficulty": difficulty,
            "target_fill_ratio": target_fill_ratio,
            "seed": seed,
            "policy": "greedy argmax",
            "aggregation": "macro average of per-ship episode means",
        },
        "categories": category_payloads,
        "global_summary": _metric_summary(global_success, global_utilization),
    }


def render_evaluation_report(payload):
    """Render an evaluation payload as the canonical GitHub Markdown report."""
    protocol = payload["protocol"]
    global_summary = payload["global_summary"]
    lines = [
        "# Model Evaluation",
        "",
        "> This is the repository's canonical evaluation report. Historical",
        "> experiments are summarized in [Project history](docs/HISTORY.md).",
        "",
        f"**Snapshot:** {payload['label']}  ",
        f"**Generated:** {payload['generated_at']}  ",
        f"**Ship data:** `{payload['ships_path']}`",
        "",
        "## Protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Episodes per ship | {protocol['episodes_per_ship']} |",
        f"| Manifest difficulty | `{protocol['difficulty']}` |",
        f"| Requested fill ratio | {protocol['target_fill_ratio']:.0%} |",
        f"| Random seed | {protocol['seed']} |",
        f"| Policy | {protocol['policy']} |",
        f"| Aggregation | {protocol['aggregation']} |",
        "",
        "Success rate is the fraction of manifest items placed. Volume",
        "utilization is placed SCU divided by usable grid volume, excluding",
        "blocker cuboids. The manifest generator applies a difficulty",
        "multiplier and random jitter after the requested fill ratio, so the",
        "requested value is not itself the expected utilization.",
        "",
        "## Summary",
        "",
        "| Category | Checkpoint | Ships | Avg success | Avg utilization |",
        "|---|---|---:|---:|---:|",
    ]

    for category in payload["categories"]:
        summary = category["summary"]
        lines.append(
            f"| {category['label']} | `{category['checkpoint']}` | "
            f"{summary['ships']} | {summary['avg_success']:.2f}% | "
            f"{summary['avg_utilization']:.2f}% |"
        )

    lines.extend(
        [
            f"| **Global** | - | **{global_summary['ships']}** | "
            f"**{global_summary['avg_success']:.2f}%** | "
            f"**{global_summary['avg_utilization']:.2f}%** |",
            "",
            "## Per-ship results",
            "",
        ]
    )

    for category in payload["categories"]:
        lines.extend(
            [
                f"### {category['label']}",
                "",
                f"Checkpoint: `{category['checkpoint']}`",
                "",
            ]
        )
        if not category["results"]:
            lines.extend(["> Checkpoint missing or no ships evaluated.", ""])
            continue

        lines.extend(
            [
                "| Ship | Grids | Usable SCU | Success | Utilization |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for result in sorted(category["results"], key=lambda item: item["avg_success"]):
            lines.append(
                f"| {result['ship_name']} | {result['grids_count']} | "
                f"{result['total_vol']} | {result['avg_success']:.2f}% | "
                f"{result['avg_utilization']:.2f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reporting limits",
            "",
            "- Manifest generation is stochastic. A seed improves repeatability,",
            "  but device kernels and dependency versions can still affect a run.",
            "  The report does not include confidence intervals.",
            "- These are macro averages: a small ship and a very large ship each",
            "  contribute one equally weighted observation.",
            "- Evaluation uses greedy actions, while training samples from the",
            "  policy distribution.",
            "- Results measure the model, manifest generator, geometry engine,",
            "  ship dataset, and checkpoint together. Changing any one of those",
            "  changes the benchmark.",
            "",
            "## Reproduce or refresh",
            "",
            "```bash",
            "python src/evaluate_model.py --episodes 10 --seed 17",
            "```",
            "",
            "Use `--category small` (repeatable for multiple categories),",
            "`--small-checkpoint`, `--medium-checkpoint`, and",
            "`--large-checkpoint` to evaluate archived checkpoint generations.",
            "Pass `--json evaluation/results/latest.json` when machine-readable",
            "output is also needed.",
            "",
            "See [Current architecture](docs/ARCHITECTURE.md) for the model and",
            "environment contract.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_full_ensemble_evaluation_report(
    report_path=DEFAULT_REPORT_PATH,
    ships_path=DEFAULT_SHIPS_PATH,
    checkpoint_paths=None,
    categories=None,
    num_episodes_per_ship=10,
    difficulty="hard",
    target_fill_ratio=0.8,
    seed=17,
    label="Current checkpoints",
    json_path=None,
):
    """Run evaluation and write the canonical Markdown report."""
    payload = build_evaluation_payload(
        ships_path=ships_path,
        checkpoint_paths=checkpoint_paths,
        categories=categories,
        num_episodes_per_ship=num_episodes_per_ship,
        difficulty=difficulty,
        target_fill_ratio=target_fill_ratio,
        seed=seed,
        label=label,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_evaluation_report(payload), encoding="utf-8")

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["global_summary"]
    print(f"\nEvaluation report: {report_path}")
    if json_path is not None:
        print(f"Evaluation data:   {json_path}")
    print("=" * 70)
    print(f"GLOBAL ({summary['ships']} ships)")
    print(f"  Success rate:       {summary['avg_success']:.2f}%")
    print(f"  Volume utilization: {summary['avg_utilization']:.2f}%")
    print("=" * 70)
    return payload


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate specialist checkpoints and refresh EVALUATION.md."
    )
    parser.add_argument("--ships", default=str(DEFAULT_SHIPS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--fill-ratio", type=float, default=0.8)
    parser.add_argument("--label", default="Current checkpoints")
    parser.add_argument(
        "--category",
        action="append",
        choices=CATEGORY_ORDER,
        help="Evaluate one category; repeat the option to select more than one.",
    )
    parser.add_argument(
        "--small-checkpoint", default=str(DEFAULT_CHECKPOINTS["small"])
    )
    parser.add_argument(
        "--medium-checkpoint", default=str(DEFAULT_CHECKPOINTS["medium"])
    )
    parser.add_argument(
        "--large-checkpoint", default=str(DEFAULT_CHECKPOINTS["large"])
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    checkpoints = {
        "small": args.small_checkpoint,
        "medium": args.medium_checkpoint,
        "large": args.large_checkpoint,
    }
    generate_full_ensemble_evaluation_report(
        report_path=args.report,
        ships_path=args.ships,
        checkpoint_paths=checkpoints,
        categories=args.category,
        num_episodes_per_ship=args.episodes,
        difficulty=args.difficulty,
        target_fill_ratio=args.fill_ratio,
        seed=args.seed,
        label=args.label,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    main()
