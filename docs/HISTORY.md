# Project History

This timeline reconstructs the project from Git history, current source,
checkpoint metadata, saved evaluation reports, and regression notes. Git
commit hashes identify changes that are reproducible from the repository.
V4-V6 checkpoint labels are experiment names rather than Git tags.

## Timeline

| Date | Commit / artifact | Milestone |
|---|---|---|
| 2025-02-02 | `c9ec4ad` | First mixed-integer linear programming prototype |
| 2025-02-02 | `7833181` | Big-M adjustment and a separate model test environment |
| 2025-02-06 to 2025-02-08 | `34dce7e`, `3e933f2` | API work and same-size overlap fixes |
| 2025-02-25 | `fcef501` | First neural actor-critic packing experiment |
| 2025-05-13 | `1239ba5` | Monolithic GAT + MER reinforcement-learning solver |
| 2025-05-24 | `91b515b` | No-feasible-MER infinite-loop fix |
| 2025-05-25 | `57c4316` | Refined training/evaluation and extracted manifest generation |
| 2026-02-24 | `894ade7` | Moved active code into `src/` and archived older approaches |
| 2026-04-02 | `e9e5bcb` | Modular `packing_core`, PPO/GAE, shared backbone, specialist ensemble |
| 2026-04-24 | `736cf3f` | Support physics, reward shaping, resume support, and major environment speedups |
| 2026-04-25 | `daeaa4d` | Fixed priority-staged MER coverage and interior-anchor placement |
| 2026-05-01 | `a34c0a7` | Drop-off-first manifest generator and regression harness |
| 2026-05-01 | `a95d784` | Float-primary geometry; measured 1.79x training speedup |
| 2026-05-01 | `58ba6e3` | CPU boolean feasibility masks; measured 1.80x cumulative speedup |
| 2026-05-01 | `43202c6` | Added, but did not integrate, `SyncVectorEnv` |
| 2026-05-02 | `44cc7cf`, `37c947f` | Bounded interior-anchor search and fixed stale rollout state |
| 2026-07-05 | `c7c6a8c` | Irregular-grid blockers, usable-volume accounting, and V6 evaluation |
| 2026-07-25 | current cleanup | One canonical evaluation report and full architecture/history documentation |

## Phase 1: exact optimization prototype

The first implementation (`Model.py`) used PuLP and the CBC solver. It modeled:

- one binary grid assignment per item;
- X/Y/Z placement variables;
- six binary separation choices for every item pair;
- Big-M non-overlap and grid-bound constraints;
- route-order constraints along the Y axis;
- post-solve stacking and access validation;
- CUDA/Numba/CuPy helpers for placement validation;
- database integration through PostgreSQL.

This formulation could express exact pairwise constraints, but binary
separation variables grow quadratically with cargo count. That made it a poor
fit for interactive packing of large manifests. The February 8 changes fixed
same-size boxes overlapping and improved diagnostics, but did not change the
fundamental scaling problem.

## Phase 2: flat actor-critic experiment

On February 25, 2025, `DRL model testing/Model DRL.py` replaced search over a
large mixed-integer program with a Gym-style sequential environment.

The model was a conventional feed-forward actor and critic:

- flattened environment state;
- a categorical distribution over a fixed action set;
- feasibility masking in the actor;
- Monte Carlo returns and advantage-weighted policy gradients;
- separate actor and critic Adam optimizers;
- gradient clipping and periodic checkpoints.

This established the sequential decision framing used today, but a flat state
did not naturally represent a variable number of cargo items, grids, or empty
spaces.

## Phase 3: graph state and maximal empty regions

The May 13, 2025 monolith, `GNN Bin packing.py`, introduced the core ideas that
survived:

- `Box3D` geometry;
- maximal empty regions split after every placement;
- a graph containing the container, current item, placed items, and MERs;
- a three-layer Graph Attention Network;
- a learned score for feasible MERs;
- a graph-pooled critic;
- curriculum learning and a replay-buffer-based training loop.

The first graph version operated on a single container and stored MERs as
tensor-heavy dictionaries. It also built expensive adjacency relationships
inside one large file. The May 24 fix made no-MER states terminal instead of
allowing an infinite loop. The May 25 revision extracted cargo-manifest
generation and refined training and evaluation.

## Phase 4: source/archive split

The February 24, 2026 cleanup moved the active implementation under `src/`,
placed the MILP and early DRL implementations in `archive/`, and created a
basic README. This was the first clear separation between active and
historical code, although the active GNN implementation was still a large
monolith.

## Phase 5: modular PPO specialist ensemble

Commit `e9e5bcb` on April 2, 2026 is the architectural break between the old
prototype and the current system.

### Module split

The monolith became:

- `packing_core/box3d.py`
- `packing_core/mer_manager.py`
- `packing_core/drl_env.py`
- `packing_core/models.py`
- `packing_core/utils.py`

The old entry point became a small backward-compatible re-export module.

### Multi-grid graph

The environment moved from one container to a whole ship. Grid nodes, a ship
node, cross-grid edges, grid ownership for MERs/items, and dynamic routing
allowed one policy step to choose space anywhere on the ship.

### Specialist models

A single universal checkpoint was replaced by small, medium, and large
specialists. Small and medium use 128 hidden features; large uses 256.
`EnsembleRouter` selects by total ship volume.

### PPO stack

The training loop moved to clipped PPO with:

- Generalized Advantage Estimation;
- clipped value loss;
- entropy regularization;
- reward normalization;
- a soft-updated target critic;
- a backbone object shared by actor and critic;
- performance-weighted ship sampling;
- a staged difficulty curriculum.

Manifest generation also began filtering out SCU types that could not
physically fit any selected grid.

## Phase 6: physical validity and reward shaping

Commit `736cf3f` on April 24 added the constraints and optimizations that made
the solver meaningfully physical:

- 99% bottom support, preventing floating boxes;
- no-heavy-on-light stacking;
- priority clustering and adjacency reward;
- largest-first reservation for very large containers;
- per-grid volume normalization;
- grid-balancing reward;
- cached scalar coordinates and graph/MER metadata;
- graceful `STOP_TRAINING` checkpoints and resume support;
- headless plot generation;
- multi-grid graph regression tests.

The commit records an approximately 26x reduction in environment-step time
from removing repeated tensor `.item()` synchronization in hot loops. That
number is a targeted step benchmark, not an end-to-end training speedup.

## Phase 7: MER coverage corrections

Priority-ordered cargo exposed two false-negative paths:

1. The MER minimum-volume threshold had been proportional to ship size, which
   deleted every 1/2-SCU pocket on large grids.
2. The feasibility mask tested only a MER's origin. A priority or support
   conflict at that corner hid the entire region even when another anchor
   inside it was legal.

Commit `daeaa4d` changed the minimum to an absolute 1 SCU and added interior
anchor search to both masking and placement. A same-cargo Polaris ablation
recorded three priority configurations improving from 47.8%, 89.1%, and 76.1%
success to 100% without retraining. This demonstrated that geometry coverage,
not only policy quality, controls measured model performance.

## Phase 8: realistic contract distribution

Commit `a34c0a7` replaced the first manifest distribution with the current
drop-off-first generator.

The old generator selected container sizes too heavily by volume, produced too
little 1/2-SCU filler on larger ships, allowed rare 24/32-SCU containers in
unrealistic medium scenarios, and assigned priority at the entry level. The
new generator:

- samples one to four delivery locations;
- assigns one priority to each location;
- uses bounded location shares;
- performs a bulk fill followed by a small-container remainder;
- carries explicit normal, mixed, bulk, and bulk-only run types;
- validates physical fit;
- is covered by a 14-test distribution contract.

The distribution created roughly three times as many individual boxes in some
scenarios, making equal success percentages harder to achieve. Large training
was reduced from 5,000 to 2,500 episodes after convergence was observed in the
2,000-2,500 range.

## Phase 9: geometry performance work

Profiling showed that tensor construction and scalar synchronization inside
`Box3D` and `MERManager` dominated training.

Commit `a95d784` made boxes float-primary and tensors lazy. A fixed-seed,
100-episode medium smoke run changed from 1,308.4 seconds to 732.7 seconds
while retaining a bit-identical final-50 reward mean: a measured 1.79x
end-to-end speedup.

Commit `58ba6e3` changed the feasibility mask from NumPy float arrays to CPU
boolean tensors. Keeping the mask on CPU was faster for single-environment
training because a GPU `.any()` would force a device synchronization on every
step. The cumulative smoke result was 728.2 seconds, or 1.80x the original
baseline.

`SyncVectorEnv` was added next as an interface foundation, but vectorized PPO
integration did not land. It remains unused by `train_agent`.

## Phase 10: bounded anchor search

Interior anchor search fixed correctness but introduced a new worst case. On
dense long grids, one mask calculation could scan hundreds of positions for
dozens of MER/rotation pairs and compare each position with about 90 boxes.
One episode was observed stalling for more than 20 minutes.

Commit `44cc7cf` introduced a 256-probe bound. Commit `37c947f` tightened it to
32 and fixed a stale-state bug in the no-feasible fallback path. The bound is
an explicit performance/completeness tradeoff that remains in the current
environment.

## Phase 11: irregular cargo grids and V6

Commit `c7c6a8c` added blocker-aware ship geometry:

- blocker schema and API validation;
- MER carving during reset;
- optional support surfaces;
- usable-volume routing, manifests, rewards, and metrics;
- exclusion of blockers from public placements;
- tests for blocker carving, support behavior, evaluation metrics, bulk-only
  Hull-C generation, and grid-balance weighting.

The ship split changed from 15/16/12 in V4 to 13/17/13 under the updated
geometry. V5 trained against blocker-aware data but regressed on the long
Ironclad/Polaris lanes. V6 restored the earlier long-lane curriculum while
retaining blockers and recovered those cases.

## Evaluation generations

The following figures are macro averages from the saved reports. They are not
one continuous apples-to-apples leaderboard: manifest distribution, ship
geometry, ship count, and evaluation execution changed along the way.

| Generation | Ships | Avg success | Avg utilization | Main distinction |
|---|---:|---:|---:|---|
| V1 cycle | 40 | 85.9% | 64.0% | Pre-v2 manifest benchmark after April reward/physics work |
| V2 cycle | 40 | 91.2% | 62.6% | Denser drop-off-first manifests |
| V3 | 40 | 93.9% | 64.4% | Improved retrained specialists |
| V4 | 43 | 94.6% | 64.9% | Three new large ships; strong long-lane behavior |
| V5 rerun | 43 | 92.14% | 62.84% | Blocker-aware geometry; long-lane regression |
| V6 | 43 | 95.70% | 66.45% | V4-style curriculum restored with blocker support |

V5 was evaluated more than once because a long evaluation process exited
during large ships. The complete per-ship persisted rerun is the historical
figure used here.

See [EVALUATION.md](../EVALUATION.md) for the current published snapshot and
all V6 per-ship values.

## What persisted across every redesign

Several domain requirements survived the transitions from MILP to flat DRL to
GNN/PPO:

- cargo must stay within a declared grid;
- boxes must not overlap;
- container orientation is restricted to the cargo-grid convention;
- heavier cargo must not be placed on lighter cargo;
- delivery priority affects front-to-back accessibility;
- the system must return concrete coordinates, not only a score.

What changed was where decisions are made. The MILP attempted to solve all
coordinates jointly. The first DRL model selected from a flat action space.
The current solver delegates empty-space construction and hard validity to the
environment, leaving the GNN to rank a changing set of MER/rotation choices.

## Repository cleanup policy

The current repository treats:

- `EVALUATION.md` as the only maintained GitHub evaluation report;
- `evaluation/results/` as machine-readable evidence for published reports;
- `docs/` as maintained design/history documentation;
- `artifacts/` as ignored generated output;
- `archive/` as ignored local history and superseded scripts;
- root-level `.pt` files as local active checkpoints.

Legacy reports were moved to the ignored archive during the July 2026 cleanup,
not overwritten. Their durable metrics and protocol caveats are consolidated
here and in the canonical report.
