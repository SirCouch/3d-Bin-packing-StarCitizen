from collections import deque

import pytest

from src.packing_core.drl_env import DRLBinPackingEnv
from src.packing_core.utils import compute_ship_sampling_weights


def test_ship_sampling_uses_v4_performance_weighting():
    ships = [
        [((4, 4, 4), "easy")],
        [((4, 4, 4), "weak")],
        [((4, 4, 4), "unexplored")],
    ]
    history = {
        0: deque([1.0, 0.9, 0.95], maxlen=20),
        1: deque([0.2, 0.3, 0.25], maxlen=20),
        2: deque([0.4, 0.5], maxlen=20),
    }

    weights = compute_ship_sampling_weights(ships, history)

    assert weights[0] == pytest.approx(0.525)
    assert weights[1] == pytest.approx(0.875)
    assert weights[2] == pytest.approx(1.0)


def test_grid_balance_reward_is_reduced_for_two_grid_ships():
    one_grid = DRLBinPackingEnv(grids_list=[((4, 4, 4), "main")])
    two_grid = DRLBinPackingEnv(grids_list=[
        ((4, 4, 4), "left"),
        ((4, 4, 4), "right"),
    ])
    four_grid = DRLBinPackingEnv(grids_list=[
        ((4, 4, 4), "a"),
        ((4, 4, 4), "b"),
        ((4, 4, 4), "c"),
        ((4, 4, 4), "d"),
    ])

    assert one_grid._grid_balance_weight() == 0.0
    assert two_grid._grid_balance_weight() < four_grid._grid_balance_weight()
    assert two_grid._grid_balance_weight() == pytest.approx(0.75)
    assert four_grid._grid_balance_weight() == pytest.approx(2.0)
