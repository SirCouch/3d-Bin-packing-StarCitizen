"""Acceptance tests for ``generate_scu_manifest``.

These tests treat the generator as a black box and assert its statistical
and structural contract. Run with::

    python -m unittest tests/test_manifest_v2_distribution.py -v
"""

import os
import random
import sys
import unittest
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from scu_manifest_generator import (  # noqa: E402
    DROPOFF_RATIO_BOUNDS,
    DROPOFF_WEIGHTS,
    FILLER_CONTAINERS,
    GRID_CATEGORIES,
    SCU_DEFINITIONS,
    container_fits_any_grid,
    generate_scu_manifest,
    get_grid_category,
    sample_ratios,
)


def grids_in_category(category: str):
    """Return only those grids in GRID_CATEGORIES[category] that re-categorize
    via get_grid_category back to the same category. The GRID_CATEGORIES table
    has overlap at the boundaries (e.g. 12x6x8=576 listed under both medium
    and large but actually re-categorizes to large), so tests asserting
    "<category>-ship behavior" must filter."""
    return [
        g for g in GRID_CATEGORIES[category]["grids"]
        if get_grid_category([tuple(t) for t in g]) == category
    ]

RARE = {"24 SCU", "32 SCU"}
FILLER = set(FILLER_CONTAINERS)


def _pick_grids(category: str, rng: random.Random):
    """Pick one grids_list from a category. Returns list of (w,l,h) tuples."""
    raw = rng.choice(GRID_CATEGORIES[category]["grids"])
    # Normalize to list of tuples (config uses lists-of-tuples already, but
    # be defensive in case future configs nest lists).
    return [tuple(g) for g in raw]


def _location_totals(manifest):
    """Return dict {priority: total_scu_volume}."""
    totals = defaultdict(int)
    for entry in manifest:
        vol = SCU_DEFINITIONS[entry["scu_type"]]["volume"]
        totals[entry["priority"]] += vol * entry["quantity"]
    return dict(totals)


def _location_types(manifest):
    """Return dict {priority: set(scu_types)}."""
    by_loc = defaultdict(set)
    for entry in manifest:
        by_loc[entry["priority"]].add(entry["scu_type"])
    return dict(by_loc)


class ManifestV2Tests(unittest.TestCase):
    # --------------------------- helpers ---------------------------

    def _generate_many(self, n, category, seed, **kwargs):
        rng = random.Random(seed)
        np.random.seed(seed)
        random.seed(seed)
        manifests = []
        for i in range(n):
            grids = _pick_grids(category, rng)
            m = generate_scu_manifest(grids_list=grids, **kwargs)
            manifests.append((grids, m))
        return manifests

    # --------------------------- 1. filler presence per location ---------------------------

    def test_01_filler_presence_per_location(self):
        random.seed(101)
        np.random.seed(101)
        rng = random.Random(101)

        n_per_cat = 350  # ~1050 total; well over 1000 drop-off allocations
        total_locations = 0
        with_filler = 0
        for category in ("small", "medium", "large"):
            for _ in range(n_per_cat):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                types_by_loc = _location_types(m)
                for prio, types in types_by_loc.items():
                    total_locations += 1
                    if types & FILLER:
                        with_filler += 1

        rate = with_filler / max(total_locations, 1)
        self.assertGreaterEqual(
            rate,
            0.95,
            f"Filler (1/2 SCU) presence per location was {rate:.4f} "
            f"({with_filler}/{total_locations}); expected >= 0.95.",
        )

    # --------------------------- 2. modal distribution ---------------------------

    def test_02_modal_sizes_medium_large(self):
        random.seed(202)
        np.random.seed(202)
        rng = random.Random(202)

        n = 1000
        for category in ("medium", "large"):
            volume_by_type = Counter()
            for _ in range(n):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                for entry in m:
                    vol = SCU_DEFINITIONS[entry["scu_type"]]["volume"]
                    volume_by_type[entry["scu_type"]] += vol * entry["quantity"]

            ranked = volume_by_type.most_common()
            top_two = {ranked[0][0], ranked[1][0]} if len(ranked) >= 2 else set()
            self.assertEqual(
                top_two,
                {"8 SCU", "16 SCU"},
                f"Category={category}: top-two SCU types by total volume were "
                f"{ranked[:3]}; expected top two to be 8 SCU and 16 SCU.",
            )

    # --------------------------- 3. rare container isolation ---------------------------

    def test_03a_small_never_has_rare(self):
        random.seed(301)
        np.random.seed(301)
        rng = random.Random(301)

        n = 1000
        offenders = 0
        for _ in range(n):
            grids = _pick_grids("small", rng)
            m = generate_scu_manifest(grids_list=grids)
            if any(e["scu_type"] in RARE for e in m):
                offenders += 1
        self.assertEqual(
            offenders,
            0,
            f"Small ships produced {offenders}/{n} manifests with rare (24/32 SCU) "
            "containers; expected 0.",
        )

    def test_03b_medium_default_no_rare(self):
        random.seed(302)
        np.random.seed(302)
        rng = random.Random(302)

        n = 1000
        pool = grids_in_category("medium")
        offenders = 0
        for _ in range(n):
            grids = [tuple(t) for t in rng.choice(pool)]
            m = generate_scu_manifest(
                grids_list=grids, medium_mixed_probability=0.0
            )
            if any(e["scu_type"] in RARE for e in m):
                offenders += 1
        self.assertEqual(
            offenders,
            0,
            f"Medium ships with medium_mixed_probability=0.0 produced "
            f"{offenders}/{n} manifests with rare containers; expected 0.",
        )

    def test_03c_medium_mixed_prob_bounded(self):
        random.seed(303)
        np.random.seed(303)
        rng = random.Random(303)

        n = 1000
        prob = 0.05
        offenders = 0
        for _ in range(n):
            grids = _pick_grids("medium", rng)
            m = generate_scu_manifest(
                grids_list=grids, medium_mixed_probability=prob
            )
            if any(e["scu_type"] in RARE for e in m):
                offenders += 1
        rate = offenders / n
        self.assertLessEqual(
            rate,
            prob + 0.01,
            f"Medium with medium_mixed_probability={prob}: rare-rate was "
            f"{rate:.4f} ({offenders}/{n}); expected <= {prob + 0.01:.4f}.",
        )

    def test_03d_large_default_rare_rate(self):
        random.seed(304)
        np.random.seed(304)
        rng = random.Random(304)

        n = 1000
        pool = grids_in_category("large")
        offenders = 0
        for _ in range(n):
            grids = [tuple(t) for t in rng.choice(pool)]
            m = generate_scu_manifest(grids_list=grids)
            if any(e["scu_type"] in RARE for e in m):
                offenders += 1
        rate = offenders / n
        self.assertGreaterEqual(
            rate,
            0.20,
            f"Large default rare-rate was {rate:.4f} ({offenders}/{n}); "
            "expected >= 0.20.",
        )
        self.assertLessEqual(
            rate,
            0.40,
            f"Large default rare-rate was {rate:.4f} ({offenders}/{n}); "
            "expected <= 0.40.",
        )

    # --------------------------- 4. odd parity per location ---------------------------

    def test_04_odd_total_parity(self):
        random.seed(404)
        np.random.seed(404)
        rng = random.Random(404)

        n_per_cat = 350
        total_locations = 0
        odd_count = 0
        for category in ("small", "medium", "large"):
            for _ in range(n_per_cat):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                for prio, total in _location_totals(m).items():
                    total_locations += 1
                    if total % 2 == 1:
                        odd_count += 1

        rate = odd_count / max(total_locations, 1)
        self.assertGreaterEqual(
            rate,
            0.40,
            f"Odd-SCU-total per location was {rate:.4f} "
            f"({odd_count}/{total_locations}); expected >= 0.40.",
        )

    # --------------------------- 5. drop-off distribution ---------------------------

    def test_05_dropoff_distribution(self):
        random.seed(505)
        np.random.seed(505)
        rng = random.Random(505)

        n = 1000
        for category in ("small", "medium", "large"):
            pool = grids_in_category(category)
            counts = Counter()
            for _ in range(n):
                grids = [tuple(t) for t in rng.choice(pool)]
                m = generate_scu_manifest(grids_list=grids)
                num_dropoffs = len({e["priority"] for e in m})
                counts[num_dropoffs] += 1

            expected = DROPOFF_WEIGHTS[category]  # [p1, p2, p3, p4]
            for k in (1, 2, 3, 4):
                obs = counts.get(k, 0) / n
                exp = expected[k - 1]
                diff = abs(obs - exp)
                self.assertLessEqual(
                    diff,
                    0.05,
                    f"Category={category}, num_dropoffs={k}: observed freq "
                    f"{obs:.4f}, expected {exp:.4f}, |delta|={diff:.4f} > 0.05. "
                    f"Full counts: {dict(counts)}",
                )

    # --------------------------- 6. ratio bounds ---------------------------

    def test_06_ratio_bounds(self):
        # Per A1 the contract is on the helper `sample_ratios`: Dirichlet
        # (concentration=5.0) + rejection sampling, ≥99% in-bounds. Asserting
        # this on post-rounding integer SCU shares (as observed in manifests)
        # is downstream of the contract; integer rounding can shift ratios by
        # ~1/total_scu and push some out of bounds. We test the helper directly.
        random.seed(606)
        np.random.seed(606)

        trials = 1000
        for n in (1, 2, 3, 4):
            bounds = DROPOFF_RATIO_BOUNDS[n]
            in_bounds = 0
            violations = []
            for _ in range(trials):
                ratios = sample_ratios(n)
                ok = all(
                    lo - 1e-9 <= r <= hi + 1e-9
                    for r, (lo, hi) in zip(ratios, bounds)
                )
                if ok:
                    in_bounds += 1
                elif len(violations) < 5:
                    violations.append(
                        f"n={n} ratios={ratios} bounds={bounds}"
                    )

            rate = in_bounds / trials
            self.assertGreaterEqual(
                rate,
                0.99,
                f"sample_ratios({n}) in-bounds rate was {rate:.4f} "
                f"({in_bounds}/{trials}); expected >= 0.99. "
                f"First violations: {violations}",
            )

    # --------------------------- 7. physical fit ---------------------------

    def test_07_physical_fit(self):
        random.seed(707)
        np.random.seed(707)
        rng = random.Random(707)

        n_per_cat = 200
        bad = []
        for category in ("small", "medium", "large"):
            for _ in range(n_per_cat):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                for entry in m:
                    dims = SCU_DEFINITIONS[entry["scu_type"]]["dimensions"]
                    if not container_fits_any_grid(dims, grids):
                        bad.append(
                            f"cat={category} grids={grids} "
                            f"type={entry['scu_type']} dims={dims}"
                        )
                        break  # one fail per manifest is enough
                if bad and len(bad) >= 5:
                    break
            if bad and len(bad) >= 5:
                break

        self.assertEqual(
            bad,
            [],
            f"Found {len(bad)} containers that do not fit any grid "
            f"(showing first 5): {bad[:5]}",
        )

    # --------------------------- 8. force_num_dropoffs ---------------------------

    def test_08a_force_num_dropoffs_basic(self):
        random.seed(808)
        np.random.seed(808)
        rng = random.Random(808)

        n = 200
        for category in ("small", "medium", "large"):
            for forced in (1, 2, 3, 4):
                fails = []
                for _ in range(n // 4):  # 50 each
                    grids = _pick_grids(category, rng)
                    m = generate_scu_manifest(
                        grids_list=grids,
                        force_num_dropoffs=forced,
                    )
                    unique = len({e["priority"] for e in m})
                    if unique != forced:
                        fails.append(
                            f"cat={category} forced={forced} got={unique} "
                            f"priorities={sorted({e['priority'] for e in m})}"
                        )
                        if len(fails) >= 3:
                            break
                self.assertEqual(
                    fails,
                    [],
                    f"force_num_dropoffs not honored. First failures: {fails}",
                )

    def test_08b_force_num_dropoffs_skips_sub2_guard(self):
        # On a small ship at very-easy difficulty, total_scu can fall below
        # 2 * num_dropoffs. force_num_dropoffs must still produce 4 priority
        # groups regardless of that internal guard.
        random.seed(880)
        np.random.seed(880)
        rng = random.Random(880)

        n = 100
        fails = []
        for _ in range(n):
            grids = _pick_grids("small", rng)
            m = generate_scu_manifest(
                grids_list=grids,
                difficulty="very-easy",
                force_num_dropoffs=4,
            )
            unique = len({e["priority"] for e in m})
            if unique != 4:
                fails.append(
                    f"grids={grids} got_unique={unique} "
                    f"priorities={sorted({e['priority'] for e in m})} "
                    f"manifest_len={len(m)}"
                )
                if len(fails) >= 5:
                    break
        self.assertEqual(
            fails,
            [],
            "force_num_dropoffs=4 on small/very-easy did not produce 4 "
            f"priority groups in {len(fails)}/{n} manifests. "
            f"First failures: {fails}",
        )

    # --------------------------- 9. priority is contiguous from 1 ---------------------------

    def test_09_priority_contiguous_from_one(self):
        random.seed(909)
        np.random.seed(909)
        rng = random.Random(909)

        n_per_cat = 200
        bad = []
        for category in ("small", "medium", "large"):
            for _ in range(n_per_cat):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                prios = {e["priority"] for e in m}
                if not prios:
                    bad.append(f"cat={category} empty manifest")
                    continue
                expected = set(range(1, max(prios) + 1))
                if prios != expected:
                    bad.append(
                        f"cat={category} got={sorted(prios)} "
                        f"expected={sorted(expected)}"
                    )
                if len(bad) >= 5:
                    break
            if len(bad) >= 5:
                break

        self.assertEqual(
            bad,
            [],
            f"Priority sets are not contiguous starting at 1. First failures: {bad[:5]}",
        )

    # --------------------------- 10. sort stability ---------------------------

    def test_10_sort_order(self):
        random.seed(1010)
        np.random.seed(1010)
        rng = random.Random(1010)

        n_per_cat = 200
        bad = []
        for category in ("small", "medium", "large"):
            for _ in range(n_per_cat):
                grids = _pick_grids(category, rng)
                m = generate_scu_manifest(grids_list=grids)
                prev_prio = None
                prev_vol = None
                for idx, entry in enumerate(m):
                    vol = SCU_DEFINITIONS[entry["scu_type"]]["volume"]
                    prio = entry["priority"]
                    if prev_prio is None:
                        prev_prio, prev_vol = prio, vol
                        continue
                    if prio < prev_prio:
                        bad.append(
                            f"cat={category} idx={idx} priority dropped: "
                            f"prev={prev_prio} cur={prio}"
                        )
                        break
                    if prio == prev_prio and vol > prev_vol:
                        bad.append(
                            f"cat={category} idx={idx} priority={prio} "
                            f"volume rose within group: prev_vol={prev_vol} "
                            f"cur_vol={vol} type={entry['scu_type']}"
                        )
                        break
                    prev_prio, prev_vol = prio, vol
                if len(bad) >= 5:
                    break
            if len(bad) >= 5:
                break

        self.assertEqual(
            bad,
            [],
            f"Manifest not sorted by (priority ASC, volume DESC). First "
            f"failures: {bad[:5]}",
        )


if __name__ == "__main__":
    unittest.main()
