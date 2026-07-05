from types import SimpleNamespace

import pytest
import torch

from src.packing_core.drl_env import DRLBinPackingEnv
from src.packing_core.utils import pack_single_manifest
from src.scu_manifest_generator import get_grid_category, get_total_usable_volume


def _blocker(position, dimensions, supports=True):
    return {
        "position": position,
        "dimensions": dimensions,
        "supports": supports,
    }


def test_blockers_are_seeded_and_carve_mer_space():
    grid = [
        (
            (4, 4, 1),
            "stepped",
            [_blocker([2, 0, 0], [2, 2, 1])],
        )
    ]
    env = DRLBinPackingEnv(grids_list=grid)
    env.reset(cargo_manifest=[(4, 4, 1, 10.0, 1)])

    blockers = [p for p in env.placed_items if p.get("is_blocker")]
    assert len(blockers) == 1
    assert env.total_ship_volume == pytest.approx(12.0)
    assert env.successful_placements == 0

    blocker_box = blockers[0]["box"]
    assert all(not mer["box"].overlaps(blocker_box) for mer in env._get_all_mers())
    assert not env.get_feasibility_mask().any()


def test_supporting_blockers_exempt_weight_and_priority_constraints():
    env = DRLBinPackingEnv(
        grids_list=[
            (
                (2, 2, 2),
                "supported",
                [_blocker([0, 0, 0], [1, 1, 1], supports=True)],
            )
        ],
        max_stack_weight=10.0,
    )
    env.reset(cargo_manifest=[])

    assert env._check_constraints_fast(
        position=(0, 0, 1),
        dimensions=(1, 1, 1),
        weight=1000.0,
        priority=5,
        grid_idx=0,
        grid_dims=env.grids[0]["dims"],
        grid_items=[p for p in env.placed_items if p["grid_idx"] == 0],
    )


def test_non_supporting_blockers_do_not_count_as_floor_support():
    env = DRLBinPackingEnv(
        grids_list=[
            (
                (2, 2, 2),
                "void",
                [_blocker([0, 0, 0], [1, 1, 1], supports=False)],
            )
        ],
    )
    env.reset(cargo_manifest=[])

    assert not env._check_constraints_fast(
        position=(0, 0, 1),
        dimensions=(1, 1, 1),
        weight=1.0,
        priority=1,
        grid_idx=0,
        grid_dims=env.grids[0]["dims"],
        grid_items=[p for p in env.placed_items if p["grid_idx"] == 0],
    )


class _GreedyActor:
    def __call__(self, _state, feasibility_mask):
        probs = feasibility_mask.to(dtype=torch.float)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        return SimpleNamespace(probs=probs)


def test_pack_single_manifest_excludes_blockers_from_output_and_utilization():
    grid = [
        (
            (2, 2, 1),
            "notched",
            [_blocker([1, 0, 0], [1, 1, 1])],
        )
    ]

    result = pack_single_manifest(
        _GreedyActor(),
        grid,
        [{"scu_type": "1 SCU", "quantity": 1, "priority": 1}],
    )

    assert result["metrics"]["items_placed"] == 1
    assert result["metrics"]["volume_utilization"] == pytest.approx(1 / 3)
    assert len(result["placements"]) == 1
    assert result["placements"][0]["scu_type"] == "1 SCU"


def test_usable_volume_helpers_account_for_blocked_space():
    grid = [((5, 5, 4), "mostly blocked", [_blocker([0, 0, 0], [5, 2, 4])])]

    assert get_total_usable_volume(grid) == pytest.approx(60.0)
    assert get_grid_category(grid) == "small"
