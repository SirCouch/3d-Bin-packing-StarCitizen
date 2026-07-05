# V6 Checkpoint Evaluation
**Date Generated:** 2026-07-05 14:48:44

**Run config:** checkpoints loaded from `checkpoints_v6/`; 10 episodes per ship, hard difficulty, current blocker-aware ship geometry, V4-style curriculum restored.

**Execution note:** small/medium use the earlier V6 partial run; large was evaluated per ship and persisted after each ship.

## Global Results

| Category | Ships | Avg Success | Avg Utilization |
|---|---:|---:|---:|
| Small | 13 | 99.03% | 68.31% |
| Medium | 17 | 91.53% | 63.04% |
| Large | 13 | 97.83% | 69.04% |
| **Global** | **43** | **95.70%** | **66.45%** |

Long-lane recovery targets are back above the V4/V5 regression band: Ironclad reached 100.0% success, Ironclad Assault reached 95.7%, and Polaris reached 99.7%.

## Small Ships - V6

**Checkpoint:** `checkpoints_v6/small_gnn_model (4).pt`

| Ship | Grids | Usable Vol | V6 SR | V6 Util |
|---|---:|---:|---:|---:|
| Freelancer DUR/MIS | 3 | 36 | 95.1% | 61.4% |
| Cutlass Black | 2 | 46 | 96.7% | 65.7% |
| Avenger Renegade | 1 | 8 | 98.0% | 60.0% |
| Zeus Mk II ES | 1 | 32 | 98.6% | 69.4% |
| Hammerhead | 1 | 40 | 99.1% | 70.5% |
| C1 Spirit | 2 | 64 | 100.0% | 70.3% |
| Hull-A | 2 | 32 | 100.0% | 66.2% |
| 400i | 1 | 42 | 100.0% | 68.6% |
| Avenger Titan | 1 | 8 | 100.0% | 85.0% |
| Apollo | 2 | 32 | 100.0% | 69.1% |
| Clipper | 1 | 12 | 100.0% | 61.7% |
| Prowler Utility | 2 | 32 | 100.0% | 71.9% |
| Shiv | 1 | 32 | 100.0% | 68.4% |
| **Category avg** | **13** | - | **99.0%** | **68.3%** |

## Medium Ships - V6

**Checkpoint:** `checkpoints_v6/medium_gnn_model (6).pt`

| Ship | Grids | Usable Vol | V6 SR | V6 Util |
|---|---:|---:|---:|---:|
| A2 Hercules | 1 | 216 | 83.1% | 65.0% |
| RAFT | 1 | 192 | 83.6% | 61.0% |
| Mercury Star Runner | 2 | 114 | 84.1% | 51.2% |
| Valkyrie | 1 | 90 | 86.7% | 63.7% |
| Zeus Mk II CL | 1 | 128 | 87.1% | 57.6% |
| Corsair | 1 | 72 | 89.1% | 57.4% |
| Freelancer MAX | 3 | 120 | 90.0% | 59.2% |
| Perseus | 1 | 96 | 90.2% | 61.4% |
| Constellation Andromeda Phoenix | 1 | 80 | 92.6% | 64.9% |
| Constellation Andromeda Aquila | 1 | 96 | 92.8% | 65.2% |
| Asgard | 1 | 180 | 93.0% | 66.0% |
| Constellation Taurus | 2 | 174 | 95.9% | 67.1% |
| Constellation Andromeda | 1 | 96 | 96.6% | 65.8% |
| Freelancer | 3 | 66 | 96.7% | 63.9% |
| Starlancer TAC | 2 | 96 | 96.8% | 64.4% |
| Retaliator Cargo Module | 2 | 74 | 98.1% | 67.3% |
| Starlancer MAX | 4 | 224 | 99.5% | 70.6% |
| **Category avg** | **17** | - | **91.5%** | **63.0%** |

## Large Ships - V6

**Checkpoint:** `checkpoints_v6/large_gnn_model (9).pt`

| Ship | Grids | Usable Vol | V6 SR | V6 Util |
|---|---:|---:|---:|---:|
| C2 Hercules | 2 | 696 | 92.2% | 63.3% |
| 890 Jump | 7 | 388 | 92.6% | 60.2% |
| M2 Hercules | 2 | 522 | 93.4% | 64.2% |
| Ironclad Assault | 2 | 1440 | 95.7% | 68.8% |
| Starfarer/Gemini | 3 | 291 | 98.9% | 68.4% |
| Caterpillar | 5 | 576 | 99.4% | 70.3% |
| Polaris | 2 | 576 | 99.7% | 71.1% |
| Carrack | 9 | 456 | 100.0% | 71.6% |
| Hermes | 2 | 288 | 100.0% | 72.0% |
| Hull-B | 8 | 512 | 100.0% | 71.9% |
| Hull-c | 8 | 4608 | 100.0% | 71.8% |
| Ironclad | 9 | 2200 | 100.0% | 72.0% |
| Railen | 6 | 640 | 100.0% | 71.8% |
| **Category avg** | **13** | - | **97.8%** | **69.0%** |

## Full V6 Summary

- **Evaluated ships:** 43
- **Average Success Rate:** 95.70%
- **Median Success Rate:** 98.00%
- **Average Volume Utilization:** 66.45%
- **Median Volume Utilization:** 66.25%
- **Large failures:** 0
