from types import SimpleNamespace

import pytest
import torch

from src.packing_core.utils import evaluate_agent


class _FakeBox:
    def __init__(self, volume):
        self.volume = volume


class _FakeActor:
    def __init__(self):
        self.seen_states = []

    def __call__(self, state, feasibility_mask):
        self.seen_states.append(state)
        assert state == "state-after-dummy-step"
        return SimpleNamespace(probs=torch.tensor([1.0]))


class _FakeEvalEnv:
    def reset(self, difficulty=None):
        self.grids = [{"dims": torch.tensor([10.0, 1.0, 1.0])}]
        self.cargo_manifest = [
            (1.0, 1.0, 1.0, 10.0, 1),
            (2.0, 1.0, 1.0, 10.0, 1),
            (4.0, 1.0, 1.0, 10.0, 1),
        ]
        self.total_items = len(self.cargo_manifest)
        self.current_item_idx = 0
        self.successful_placements = 0
        self.placed_items = []
        self.step_count = 0
        return "initial-state"

    def get_feasibility_mask(self):
        if self.step_count == 0:
            return torch.tensor([False])
        if self.step_count == 1:
            return torch.tensor([True])
        return torch.tensor([False])

    def step(self, action):
        if self.step_count == 0:
            self.step_count += 1
            self.current_item_idx = 1
            self.successful_placements = 1
            self.placed_items.append({"box": _FakeBox(1.0)})
            return "state-after-dummy-step", 1.0, False, {"feasible": True}

        if self.step_count == 1:
            self.step_count += 1
            self.current_item_idx = 2
            return "state-after-skipped-placement", -5.0, False, {"feasible": False}

        self.step_count += 1
        self.current_item_idx = 3
        return "done-state", -5.0, True, {"feasible": False}


def test_evaluate_agent_refreshes_state_after_dummy_step_and_counts_placed_volume():
    actor = _FakeActor()

    _, success_rate, volume_utilization = evaluate_agent(
        _FakeEvalEnv(),
        actor,
        num_episodes=1,
    )

    assert actor.seen_states == ["state-after-dummy-step"]
    assert success_rate == pytest.approx(1 / 3)
    assert volume_utilization == pytest.approx(0.1)
