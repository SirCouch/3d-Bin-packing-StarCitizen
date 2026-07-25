import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_model import categorize_ships_for_eval, render_evaluation_report


ROOT = Path(__file__).resolve().parents[1]


def test_ship_dataset_is_valid_json():
    data = json.loads((ROOT / "ships_cargo_grids.json").read_text(encoding="utf-8"))

    assert len(data["ships"]) == 43
    assert all(ship["name"] and ship["grids"] for ship in data["ships"])


def test_evaluation_categories_use_usable_volume():
    data = {
        "ships": [
            {
                "name": "Small",
                "grids": [{"name": "Main", "dimensions": [4, 4, 4]}],
            },
            {
                "name": "Medium after blocker",
                "grids": [
                    {
                        "name": "Main",
                        "dimensions": [10, 10, 3],
                        "blocked": [
                            {
                                "position": [0, 0, 0],
                                "dimensions": [10, 10, 1],
                            }
                        ],
                    }
                ],
            },
            {
                "name": "Large",
                "grids": [{"name": "Main", "dimensions": [257, 1, 1]}],
            },
        ]
    }

    small, medium, large = categorize_ships_for_eval(data)

    assert list(small) == ["Small"]
    assert list(medium) == ["Medium after blocker"]
    assert list(large) == ["Large"]


def test_report_renderer_includes_protocol_summary_and_ship_rows():
    payload = {
        "label": "Fixture",
        "generated_at": "2026-07-25T12:00:00-04:00",
        "ships_path": "ships_cargo_grids.json",
        "protocol": {
            "episodes_per_ship": 2,
            "difficulty": "hard",
            "target_fill_ratio": 0.8,
            "seed": 17,
            "policy": "greedy argmax",
            "aggregation": "macro average of per-ship episode means",
        },
        "categories": [
            {
                "category": "small",
                "label": "Small Ships",
                "checkpoint": "small_gnn_model.pt",
                "results": [
                    {
                        "ship_name": "Fixture Ship",
                        "grids_count": 1,
                        "total_vol": 64,
                        "avg_success": 95.0,
                        "avg_utilization": 70.0,
                    }
                ],
                "summary": {
                    "ships": 1,
                    "avg_success": 95.0,
                    "median_success": 95.0,
                    "avg_utilization": 70.0,
                    "median_utilization": 70.0,
                },
            }
        ],
        "global_summary": {
            "ships": 1,
            "avg_success": 95.0,
            "median_success": 95.0,
            "avg_utilization": 70.0,
            "median_utilization": 70.0,
        },
    }

    report = render_evaluation_report(payload)

    assert report.startswith("# Model Evaluation")
    assert "| Episodes per ship | 2 |" in report
    assert "| **Global** | - | **1** | **95.00%** | **70.00%** |" in report
    assert "| Fixture Ship | 1 | 64 | 95.00% | 70.00% |" in report


def test_root_has_one_canonical_evaluation_report():
    evaluation_markdown = sorted(
        path.name
        for path in ROOT.glob("*.md")
        if "eval" in path.stem.lower()
    )

    assert evaluation_markdown == ["EVALUATION.md"]


def test_published_v6_summary_matches_retained_raw_results():
    raw = json.loads(
        (ROOT / "evaluation" / "results" / "v6.json").read_text(encoding="utf-8")
    )
    success = [
        value
        for category in raw["categories"]
        for value in category["success"]
    ]
    utilization = [
        value
        for category in raw["categories"]
        for value in category["utilization"]
    ]

    assert len(success) == 43
    assert np.mean(success) == pytest.approx(95.70, abs=0.01)
    assert np.mean(utilization) == pytest.approx(66.45, abs=0.01)

    report = (ROOT / "EVALUATION.md").read_text(encoding="utf-8")
    assert "**95.70%**" in report
    assert "**66.45%**" in report
