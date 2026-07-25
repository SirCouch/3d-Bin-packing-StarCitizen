# Model Evaluation

> This is the repository's single maintained evaluation report. Historical
> implementation and checkpoint changes are documented in
> [Project history](docs/HISTORY.md).

## Published snapshot

| Field | Value |
|---|---|
| Generation | V6 |
| Evaluation completed | 2026-07-05 |
| Ship dataset | 43 blocker-aware ship definitions at commit `c7c6a8c` |
| Episodes | 10 generated manifests per ship |
| Difficulty | `hard` |
| Requested fill ratio | 80% |
| Policy | Greedy `argmax` over feasible MER/rotation actions |
| Aggregation | Macro average of per-ship episode means |
| Raw data | [`evaluation/results/v6.json`](evaluation/results/v6.json) |

The small and medium results came from the completed V6 partial run. Large
ships were evaluated one ship at a time with results persisted after each
ship. All 43 ships completed; no large-ship evaluation failed.

The original V6 runner did not record a random seed. Results are therefore a
published stochastic snapshot, not a bit-for-bit reproducibility claim. The
current unified evaluator defaults to seed 17 for future reports.

## Metric definitions

**Success rate** is the fraction of individual manifest containers placed.
Every ship is evaluated independently, and its 10 episode rates are averaged.

**Volume utilization** is placed cargo SCU divided by usable ship-grid volume.
Blocker cuboids are removed from the denominator and are not counted as cargo.

The generator applies a 0.9 multiplier at hard difficulty. An 80% requested
fill therefore targets roughly 72% of usable volume before +/-3 SCU jitter.
Utilization should be interpreted against that generated cargo target, not as
an attempt to fill every ship to 100%.

## Global results

| Category | Ships | Checkpoint | Avg success | Avg utilization |
|---|---:|---|---:|---:|
| Small | 13 | `checkpoints_v6/small_gnn_model (4).pt` | 99.03% | 68.31% |
| Medium | 17 | `checkpoints_v6/medium_gnn_model (6).pt` | 91.53% | 63.04% |
| Large | 13 | `checkpoints_v6/large_gnn_model (9).pt` | 97.83% | 69.04% |
| **Global** | **43** | - | **95.70%** | **66.45%** |

The global median was 98.00% success and 66.25% utilization.

## Per-ship results

### Small ships

| Ship | Grids | Usable SCU | Success | Utilization |
|---|---:|---:|---:|---:|
| Freelancer DUR/MIS | 3 | 36 | 95.08% | 61.39% |
| Cutlass Black | 2 | 46 | 96.67% | 65.65% |
| Avenger Renegade | 1 | 8 | 98.00% | 60.00% |
| Zeus Mk II ES | 1 | 32 | 98.57% | 69.38% |
| Hammerhead | 1 | 40 | 99.09% | 70.50% |
| C1 Spirit | 2 | 64 | 100.00% | 70.31% |
| Hull-A | 2 | 32 | 100.00% | 66.25% |
| 400i | 1 | 42 | 100.00% | 68.57% |
| Avenger Titan | 1 | 8 | 100.00% | 85.00% |
| Apollo | 2 | 32 | 100.00% | 69.06% |
| Clipper | 1 | 12 | 100.00% | 61.67% |
| Prowler Utility | 2 | 32 | 100.00% | 71.88% |
| Shiv | 1 | 32 | 100.00% | 68.44% |
| **Category average** | **13** | - | **99.03%** | **68.31%** |

### Medium ships

| Ship | Grids | Usable SCU | Success | Utilization |
|---|---:|---:|---:|---:|
| A2 Hercules | 1 | 216 | 83.13% | 65.00% |
| RAFT | 1 | 192 | 83.63% | 61.04% |
| Mercury Star Runner | 2 | 114 | 84.10% | 51.23% |
| Valkyrie | 1 | 90 | 86.75% | 63.67% |
| Zeus Mk II CL | 1 | 128 | 87.09% | 57.58% |
| Corsair | 1 | 72 | 89.07% | 57.36% |
| Freelancer MAX | 3 | 120 | 90.02% | 59.17% |
| Perseus | 1 | 96 | 90.21% | 61.35% |
| Constellation Andromeda Phoenix | 1 | 80 | 92.64% | 64.88% |
| Constellation Andromeda Aquila | 1 | 96 | 92.84% | 65.21% |
| Asgard | 1 | 180 | 92.95% | 66.00% |
| Constellation Taurus | 2 | 174 | 95.93% | 67.07% |
| Constellation Andromeda | 1 | 96 | 96.57% | 65.83% |
| Freelancer | 3 | 66 | 96.69% | 63.94% |
| Starlancer TAC | 2 | 96 | 96.79% | 64.38% |
| Retaliator Cargo Module | 2 | 74 | 98.15% | 67.30% |
| Starlancer MAX | 4 | 224 | 99.52% | 70.62% |
| **Category average** | **17** | - | **91.53%** | **63.04%** |

### Large ships

| Ship | Grids | Usable SCU | Success | Utilization |
|---|---:|---:|---:|---:|
| C2 Hercules | 2 | 696 | 92.21% | 63.33% |
| 890 Jump | 7 | 388 | 92.57% | 60.23% |
| M2 Hercules | 2 | 522 | 93.39% | 64.20% |
| Ironclad Assault | 2 | 1,440 | 95.65% | 68.75% |
| Starfarer/Gemini | 3 | 291 | 98.88% | 68.42% |
| Caterpillar | 5 | 576 | 99.39% | 70.31% |
| Polaris | 2 | 576 | 99.69% | 71.08% |
| Carrack | 9 | 456 | 100.00% | 71.62% |
| Hermes | 2 | 288 | 100.00% | 71.98% |
| Hull-B | 8 | 512 | 100.00% | 71.91% |
| Hull-C | 8 | 4,608 | 100.00% | 71.81% |
| Ironclad | 9 | 2,200 | 100.00% | 72.00% |
| Railen | 6 | 640 | 100.00% | 71.84% |
| **Category average** | **13** | - | **97.83%** | **69.04%** |

## Evolution of measured performance

These generations are useful for understanding direction, but they are not a
single controlled leaderboard. V2 changed manifest density, V4 added three
ships, V5 changed physical grid geometry, and V6 combined earlier small/medium
results with a persisted per-ship large run.

| Generation | Ships | Avg success | Avg utilization | Material protocol change |
|---|---:|---:|---:|---|
| V1 cycle | 40 | 85.9% | 64.0% | Pre-v2 manifest distribution |
| V2 cycle | 40 | 91.2% | 62.6% | Drop-off-first, filler-heavy manifests |
| V3 | 40 | 93.9% | 64.4% | Retrained specialists |
| V4 | 43 | 94.6% | 64.9% | Added Ironclad, Ironclad Assault, and Railen |
| V5 persisted rerun | 43 | 92.14% | 62.84% | Blocker-aware geometry; long-lane regression |
| **V6** | **43** | **95.70%** | **66.45%** | Restored V4-style curriculum with blockers retained |

V6's most important recovery was on long cargo lanes:

| Ship | V4 success | V5 rerun success | V6 success |
|---|---:|---:|---:|
| Ironclad Assault | 82.8% | 59.5% | 95.7% |
| Ironclad | 87.4% | 70.3% | 100.0% |
| Polaris | 93.7% | 76.0% | 99.7% |

Medium ships remain the current floor. A2 Hercules, RAFT, and Mercury Star
Runner all averaged below 85% success in V6. The benchmark does not establish
whether those misses are primarily policy ranking, bounded anchor search,
priority ordering, or manifest/geometry mismatch.

## Reporting limits

- No confidence intervals were retained, and the V6 seed is unknown.
- A macro average gives a tiny ship and Hull-C equal weight.
- Success is item-count based. Missing one 32-SCU box and one 1-SCU box has
  the same success-rate effect, although utilization changes differently.
- Greedy evaluation differs from sampled training behavior.
- The 32-probe interior-anchor cap can hide legal deep placements.
- Hull-C uses a named bulk-only evaluation distribution that the unnamed
  training pool does not apply.
- The specialist router and manifest generator use different category
  boundaries for ships between 257 and 512 SCU.
- V6 describes the July 5 ship data and checkpoints. Later edits to
  `ships_cargo_grids.json` can make a fresh run non-comparable.

## Reproduce or refresh

The current evaluator is the only general report runner:

```bash
python src/evaluate_model.py --episodes 10 --seed 17
```

To evaluate the archived V6 checkpoints explicitly:

```bash
python src/evaluate_model.py \
  --label "V6 checkpoints, seeded rerun" \
  --small-checkpoint "checkpoints_v6/small_gnn_model (4).pt" \
  --medium-checkpoint "checkpoints_v6/medium_gnn_model (6).pt" \
  --large-checkpoint "checkpoints_v6/large_gnn_model (9).pt" \
  --episodes 10 \
  --seed 17 \
  --json evaluation/results/latest.json
```

Use `--category small`, repeated as needed, for a partial run. Refreshing the
report overwrites this file by design; review the diff and retain the protocol
metadata with any published result.

See [Current architecture](docs/ARCHITECTURE.md) for the exact model,
environment, reward, and checkpoint contracts.
