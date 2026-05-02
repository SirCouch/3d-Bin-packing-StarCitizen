"""SyncVectorEnv — wraps N DRLBinPackingEnv instances for batched rollout.

Phase 5 / Refactor A foundation. Single-process, batched-Python-loop. No
subprocesses (D1=a). Each env keeps its own state on CPU; the actor/critic
forwards still run per-env for now (batched GAT forwards are a follow-up
once SyncVectorEnv is proven correct).

Auto-reset semantics: when an env's episode finishes, the next `step_all`
call sees a freshly-reset state for that slot. The episode boundary is
recorded in `infos[i]["episode_done"]=True` so the rollout loop can:
  - close the current trajectory segment for env i
  - start a new GAE segment at step T+1

The rollout loop is responsible for choosing the action for the post-reset
state (i.e., it must call the actor on the new state, not reuse the
just-sampled action that was for the pre-reset state).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .drl_env import DRLBinPackingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EnvFactory = Callable[[int], DRLBinPackingEnv]
ResetSpec = Callable[[int], dict]  # int -> kwargs for env.reset()


class SyncVectorEnv:
    """Run N DRLBinPackingEnv instances in lockstep with auto-reset."""

    def __init__(
        self,
        env_factory: EnvFactory,
        num_envs: int,
        reset_spec: ResetSpec,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self._reset_spec = reset_spec
        self.envs: List[DRLBinPackingEnv] = [env_factory(i) for i in range(num_envs)]
        # Per-env metadata that PyG `Batch` cannot carry safely (dicts, ints).
        self._states: List[Optional[Data]] = [None] * num_envs
        self._reset_all()

    # ------------------------------------------------------------------ reset
    def _reset_one(self, idx: int) -> Data:
        kwargs = self._reset_spec(idx)
        state = self.envs[idx].reset(**kwargs)
        self._states[idx] = state
        return state

    def _reset_all(self) -> None:
        for i in range(self.num_envs):
            self._reset_one(i)

    def reset_all(self) -> List[Data]:
        """Force-reset every env. Used at the start of a rollout phase."""
        self._reset_all()
        return list(self._states)

    # ------------------------------------------------------------------- step
    def step_all(self, actions: List[int]) -> Tuple[
        List[Data], List[float], List[bool], List[dict]
    ]:
        """Step all N envs with the given per-env actions. On done, auto-reset
        the slot and return the post-reset state in `next_states[i]`. The done
        flag is preserved in `dones[i]` and `infos[i]["episode_done"]=True` so
        the caller can close the trajectory segment."""
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        next_states: List[Data] = [None] * self.num_envs
        rewards: List[float] = [0.0] * self.num_envs
        dones: List[bool] = [False] * self.num_envs
        infos: List[dict] = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            state, reward, done, info = self.envs[i].step(actions[i])
            rewards[i] = float(reward)
            dones[i] = bool(done)
            info = dict(info)  # local copy
            if done:
                info["episode_done"] = True
                info["terminal_state"] = state  # caller may want it for GAE bootstrap
                next_states[i] = self._reset_one(i)
            else:
                info["episode_done"] = False
                next_states[i] = state
                self._states[i] = state
            infos[i] = info
        return next_states, rewards, dones, infos

    # ----------------------------------------------------------------- access
    def get_states(self) -> List[Data]:
        return list(self._states)

    def get_feasibility_masks(self) -> List[torch.Tensor]:
        """Per-env feasibility mask. CPU bool tensors (see lessons.md)."""
        return [env.get_feasibility_mask() for env in self.envs]

    # ---------------------------------------------------- batched forward API
    def forward_actor_batched(
        self,
        actor,
        states: List[Data],
        masks: List[torch.Tensor],
    ):
        """Run the actor over N envs. Returns a list of N Categorical
        distributions (one per env). For Phase 5 we loop sequentially; this
        exposes a stable API that a Phase 5b GAT-batched implementation can
        later optimize without changing callers."""
        return [actor(states[i], masks[i]) for i in range(self.num_envs)]

    def forward_critic_batched(self, critic, states: List[Data]) -> torch.Tensor:
        """Run the critic over N envs. Returns a length-N tensor of state values."""
        values = []
        for i in range(self.num_envs):
            v = critic(states[i])
            if isinstance(v, torch.Tensor):
                v = v.view(-1)[0] if v.numel() else torch.zeros((), device=device)
            values.append(v)
        return torch.stack([v.detach() if v.requires_grad else v for v in values])

    def __len__(self) -> int:
        return self.num_envs


def make_default_env_factory(grids_list_pool: List[List[Tuple[int, int, int]]]) -> EnvFactory:
    """Factory that hands each env slot a deterministic ship from the pool.
    Slot i picks pool[i % len(pool)]. Used for testing — production callers
    should sample ships per their own weighting (e.g., performance-weighted)."""
    def _factory(slot_idx: int) -> DRLBinPackingEnv:
        grids = grids_list_pool[slot_idx % len(grids_list_pool)]
        return DRLBinPackingEnv(grids_list=grids)
    return _factory
