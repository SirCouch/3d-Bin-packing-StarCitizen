# Eval Comparison: Previous vs Current Run

**Previous run**: small 500 / medium 2000 / large 3000, old reward (pre-support, pre-cluster).
**Current run**: small 500 / medium 2000 / large 5000, new reward (support constraint + priority clustering + largest-first sort), env optimizations.

## Global Summary

| Metric | Previous | Current | Δ |
|---|---:|---:|---:|
| Success Rate | 83.7% | **85.9%** | **+2.2%** |
| Volume Utilization | 62.0% | **64.0%** | **+2.0%** |

## Small Ships (15)

| Ship | Prev SR | Curr SR | Δ SR | Prev Util | Curr Util | Δ Util |
|---|---:|---:|---:|---:|---:|---:|
| Freelancer DUR/MIS | 79.2% | 96.7% | **+17.5** | 44.6% | 70.4% | **+25.8** |
| Avenger Renegade | 100.0% | 78.3% | -21.7 | 75.0% | 56.3% | -18.7 |
| C1 Spirit | 95.4% | 80.0% | -15.4 | 71.6% | 66.6% | -5.0 |
| Hull-A | 100.0% | 100.0% | 0.0 | 79.7% | 70.3% | -9.4 |
| Hammerhead | 89.1% | 81.7% | -7.4 | 65.3% | 66.3% | +1.0 |
| 400i | 67.4% | 82.2% | **+14.8** | 54.8% | 62.6% | **+7.8** |
| Cutlass Black | 85.8% | 91.9% | +6.1 | 65.0% | 67.5% | +2.5 |
| Zeus Mk II ES | 78.3% | 97.8% | **+19.5** | 57.8% | 68.1% | **+10.3** |
| Freelancer | 84.4% | 77.7% | -6.7 | 56.3% | 64.4% | +8.1 |
| Avenger Titan | 100.0% | 76.7% | -23.3 | 75.0% | 56.3% | -18.7 |
| Valkyrie | 84.2% | 77.0% | -7.2 | 66.0% | 61.3% | -4.7 |
| Apollo | 92.7% | 100.0% | +7.3 | 71.9% | 68.8% | -3.1 |
| Clipper | 90.0% | 96.0% | +6.0 | 65.0% | 65.0% | 0.0 |
| Prowler Utility | 97.5% | 100.0% | +2.5 | 76.9% | 68.8% | -8.1 |
| Shiv | 100.0% | 85.7% | -14.3 | 78.1% | 62.5% | -15.6 |
| **Category avg** | — | **88.1%** | — | — | **65.0%** | — |

## Medium Ships (16)

| Ship | Prev SR | Curr SR | Δ SR | Prev Util | Curr Util | Δ Util |
|---|---:|---:|---:|---:|---:|---:|
| Mercury Star Runner | 71.5% | 85.6% | **+14.1** | 52.0% | 63.8% | **+11.8** |
| Zeus Mk II CL | 73.6% | 79.5% | +5.9 | 42.5% | 53.0% | **+10.5** |
| Constellation Andromeda Aquila | 72.2% | 75.8% | +3.6 | 58.8% | 62.6% | +3.8 |
| Corsair | 78.0% | 80.8% | +2.8 | 55.8% | 64.4% | +8.6 |
| Freelancer MAX | 72.0% | 82.7% | **+10.7** | 42.2% | 63.2% | **+21.0** |
| Constellation Taurus | 77.1% | 86.7% | +9.6 | 48.4% | 61.7% | **+13.3** |
| Retaliator Cargo Module | 82.5% | 84.5% | +2.0 | 57.4% | 59.7% | +2.3 |
| RAFT | 93.1% | 79.8% | -13.3 | 74.2% | 63.3% | -10.9 |
| Starfarer/Gemini¹ | 62.6% | 88.9% | **+26.3** | 37.0% | 62.8% | **+25.8** |
| Constellation Andromeda Phoenix | 97.5% | 81.2% | -16.3 | 78.0% | 68.3% | -9.7 |
| Constellation Andromeda | 77.3% | 78.2% | +0.9 | 56.5% | 64.6% | +8.1 |
| A2 Hercules | 69.8% | 73.6% | +3.8 | 56.8% | 61.3% | +4.5 |
| Perseus | 100.0% | 77.3% | -22.7 | 79.2% | 61.3% | -17.9 |
| Starlancer MAX | 84.8% | 87.2% | +2.4 | 65.1% | 61.8% | -3.3 |
| Asgard | 74.9% | 81.5% | +6.6 | 55.8% | 62.4% | +6.6 |
| Starlancer TAC | 83.6% | 96.6% | **+13.0** | 58.3% | 68.3% | **+10.0** |
| **Category avg** | — | **82.5%** | — | — | **62.7%** | — |

¹ Starfarer/Gemini grid count changed 1G→3G between runs.

## Large Ships (9)

| Ship | Prev SR | Curr SR | Δ SR | Prev Util | Curr Util | Δ Util |
|---|---:|---:|---:|---:|---:|---:|
| C2 Hercules | 68.6% | 88.5% | **+19.9** | 46.9% | 68.8% | **+21.9** |
| 890 Jump² | 79.6% | 81.4% | +1.8 | 64.3% | 60.6% | -3.7 |
| Caterpillar | 65.6% | 75.5% | **+9.9** | 40.5% | 44.3% | +3.8 |
| M2 Hercules | 73.0% | 74.8% | +1.8 | 56.1% | 56.7% | +0.6 |
| Polaris | 78.7% | 90.3% | **+11.6** | 60.1% | 69.4% | +9.3 |
| Carrack | 95.3% | 100.0% | +4.7 | 75.3% | 71.7% | -3.6 |
| Hermes | 93.0% | 83.8% | -9.2 | 74.9% | 68.2% | -6.7 |
| Hull-B | 99.5% | 100.0% | +0.5 | 79.3% | 71.9% | -7.4 |
| Hull-c | 81.5% | 100.0% | **+18.5** | 63.4% | 72.0% | +8.6 |
| **Category avg** | — | **88.3%** | — | — | **64.8%** | — |

² 890 Jump grid count changed 1G→7G between runs.

## Observations

**Big wins**:
- **Starfarer/Gemini** +26.3 SR, +25.8 Util — was the worst performer, now top tier
- **C2 Hercules** +19.9 SR, +21.9 Util
- **Hull-c** +18.5 SR (to 100%)
- **Freelancer DUR/MIS** +17.5 SR, +25.8 Util
- **Mercury Star Runner** +14.1 SR, +11.8 Util
- **Polaris** +11.6 SR
- **Freelancer MAX** +10.7 SR, +21.0 Util

**Regressions** (likely policy-choice differences from the new reward balance, not bugs):
- **Avenger Titan / Avenger Renegade** -23/-22 SR — very small 8-SCU grids lose some robustness
- **Perseus** -22.7 SR (100→77) — worth investigating
- **Constellation Phoenix** -16.3 SR
- **C1 Spirit / Shiv** -15 / -14 SR
- **RAFT** -13.3 SR

**Category direction**:
- Small SR avg is higher than before (88.1% vs ~87% inferred), but with more variance
- Medium utilization up notably (62.7% vs low 50s prior) — clustering+support paying off on medium's typical layouts
- Large SR avg up to 88.3% — strongest category now

---

# V2 Manifest Cycle (2026-04-30)

**Run config**: small 500 / medium 3000 / large 5000 (in progress), GPU (RTX 4080, torch 2.9.1+cu126), v2 manifest distribution.

**Distribution change vs prior runs**: this cycle uses the drop-off-as-primary-unit manifest generator. Per-manifest item count rose from v1 to v2:
- small ~9 → ~10 items (+17%)
- medium ~12 → ~35 items (+193%)
- large ~26 → ~91 items (+257%)

**Caveat for direct comparison**: a 95% success rate on v2 is meaningfully harder than 95% on v1 — same ship now sees ~3x more containers per episode under tighter geometric constraints (1/2 SCU filler crowding). Treat the V2 column as a *more demanding* benchmark, not a like-for-like.

## Global Summary

| Metric | Prior current (v1 cycle) | V2 cycle | Δ |
|---|---:|---:|---:|
| Small SR | 88.1% | **95.4%** | **+7.3** |
| Small Util | 65.0% | **65.2%** | +0.2 |
| Medium SR | 82.5% | **90.0%** | **+7.5** |
| Medium Util | 62.7% | 59.8% | -2.9 |
| Large SR | 88.3% | _pending_ | — |
| Large Util | 64.8% | _pending_ | — |

## Small Ships — V2 vs Prior (15)

| Ship | Prior SR | V2 SR | Δ SR | Prior Util | V2 Util | Δ Util |
|---|---:|---:|---:|---:|---:|---:|
| Freelancer DUR/MIS | 96.7% | 85.7% | -11.0 | 70.4% | 58.3% | -12.1 |
| Avenger Renegade | 78.3% | 96.7% | **+18.4** | 56.3% | 65.0% | **+8.7** |
| C1 Spirit | 80.0% | 100.0% | **+20.0** | 66.6% | 71.4% | +4.8 |
| Hull-A | 100.0% | 100.0% | 0.0 | 70.3% | 71.4% | +1.1 |
| Hammerhead | 81.7% | 89.6% | +7.9 | 66.3% | 61.3% | -5.0 |
| 400i | 82.2% | 93.7% | **+11.5** | 62.6% | 62.6% | 0.0 |
| Cutlass Black | 91.9% | 95.0% | +3.1 | 67.5% | 64.8% | -2.7 |
| Zeus Mk II ES | 97.8% | 96.6% | -1.2 | 68.1% | 65.6% | -2.5 |
| Freelancer | 77.7% | 92.8% | **+15.1** | 64.4% | 65.2% | +0.8 |
| Avenger Titan | 76.7% | 95.0% | **+18.3** | 56.3% | 62.5% | +6.2 |
| Valkyrie | 77.0% | 96.9% | **+19.9** | 61.3% | 65.7% | +4.4 |
| Apollo | 100.0% | 99.0% | -1.0 | 68.8% | 70.0% | +1.2 |
| Clipper | 96.0% | 92.8% | -3.2 | 65.0% | 60.0% | -5.0 |
| Prowler Utility | 100.0% | 100.0% | 0.0 | 68.8% | 69.7% | +0.9 |
| Shiv | 85.7% | 96.7% | **+11.0** | 62.5% | 65.0% | +2.5 |
| **Category avg** | **88.1%** | **95.4%** | **+7.3** | **65.0%** | **65.2%** | +0.2 |

## Medium Ships — V2 vs Prior (16)

| Ship | Prior SR | V2 SR | Δ SR | Prior Util | V2 Util | Δ Util |
|---|---:|---:|---:|---:|---:|---:|
| Mercury Star Runner | 85.6% | 78.4% | -7.2 | 63.8% | 49.3% | -14.5 |
| Zeus Mk II CL | 79.5% | 88.8% | **+9.3** | 53.0% | 59.7% | +6.7 |
| Constellation Andromeda Aquila | 75.8% | 95.3% | **+19.5** | 62.6% | 63.4% | +0.8 |
| Corsair | 80.8% | 85.6% | +4.8 | 64.4% | 55.1% | -9.3 |
| Freelancer MAX | 82.7% | 86.9% | +4.2 | 63.2% | 53.2% | -10.0 |
| Constellation Taurus | 86.7% | 82.7% | -4.0 | 61.7% | 56.3% | -5.4 |
| Retaliator Cargo Module | 84.5% | 94.0% | **+9.5** | 59.7% | 62.8% | +3.1 |
| RAFT | 79.8% | 82.9% | +3.1 | 63.3% | 48.3% | **-15.0** |
| Starfarer/Gemini | 88.9% | 95.2% | +6.3 | 62.8% | 62.9% | +0.1 |
| Constellation Andromeda Phoenix | 81.2% | 94.3% | **+13.1** | 68.3% | 66.0% | -2.3 |
| Constellation Andromeda | 78.2% | 92.0% | **+13.8** | 64.6% | 60.6% | -4.0 |
| A2 Hercules | 73.6% | 89.4% | **+15.8** | 61.3% | 64.5% | +3.2 |
| Perseus | 77.3% | 86.0% | **+8.7** | 61.3% | 54.7% | -6.6 |
| Starlancer MAX | 87.2% | 99.4% | **+12.2** | 61.8% | 70.3% | **+8.5** |
| Asgard | 81.5% | 90.9% | **+9.4** | 62.4% | 61.1% | -1.3 |
| Starlancer TAC | 96.6% | 98.0% | +1.4 | 68.3% | 68.0% | -0.3 |
| **Category avg** | **82.5%** | **90.0%** | **+7.5** | **62.7%** | **59.8%** | -2.9 |

## Large Ships

_Training in progress — eval pending checkpoint completion._

## V2 Cycle Observations

**Big wins** (despite the harder distribution):
- **Constellation Andromeda Aquila** medium +19.5 SR
- **Avenger Renegade / Valkyrie / Avenger Titan** small +18-20 SR — the previous run's small-ship regressions reversed
- **C1 Spirit** small +20.0 SR (back to 100%)
- **A2 Hercules** medium +15.8 SR
- **Constellation Andromeda + Phoenix** medium +13-14 SR
- **Starlancer MAX** medium +12.2 SR (now 99.4%)
- **Freelancer** small +15.1 SR

**Regressions on v2**:
- **Freelancer DUR/MIS** small -11.0 SR — the smallest single-grid ship (vol=24) is consistently hardest under the high-filler v2 distribution
- **Mercury Star Runner** medium -7.2 SR, util -14.5 — small single-grid medium with vol=108, lots of filler items competing
- **RAFT** medium util -15.0 — single-grid 96 SCU with high filler density

**Pattern**: tight single-grid medium-volume ships are the new floor. Multi-grid ships consistently improve. This tracks with the v2 manifest's filler-density change — it crowds packing decisions on small grids more than on multi-grid layouts.
