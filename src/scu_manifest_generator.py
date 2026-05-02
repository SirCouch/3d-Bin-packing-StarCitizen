import random
from typing import List, Tuple, Dict, Optional

import numpy as np

# SCU Container Definitions
SCU_DEFINITIONS = {
    "1 SCU": {"dimensions": [1, 1, 1], "volume": 1, "weight": 10},
    "2 SCU": {"dimensions": [2, 1, 1], "volume": 2, "weight": 20},
    "4 SCU": {"dimensions": [2, 2, 1], "volume": 4, "weight": 40},
    "8 SCU": {"dimensions": [2, 2, 2], "volume": 8, "weight": 80},
    "16 SCU": {"dimensions": [2, 4, 2], "volume": 16, "weight": 160},
    "24 SCU": {"dimensions": [2, 6, 2], "volume": 24, "weight": 240},
    "32 SCU": {"dimensions": [2, 8, 2], "volume": 32, "weight": 320}
}

# v2 generator constants — drop-off-as-primary-unit refactor.
# Filler containers are always available across every category.
FILLER_CONTAINERS = ["1 SCU", "2 SCU"]

# In-game frequency weights for the bulk-phase weighted selection.
# Values are starting points; recalibrate against logged contracts when available.
GAME_FREQUENCY = {
    "1 SCU": 0.7,
    "2 SCU": 0.9,
    "4 SCU": 1.0,
    "8 SCU": 1.5,
    "16 SCU": 1.3,
    "24 SCU": 0.4,
    "32 SCU": 0.3,
}

# Per-category drop-off-count probability (1, 2, 3, 4 stops).
# Small ships rarely run 4-stop contracts; large ships rarely run single-stop.
DROPOFF_WEIGHTS = {
    "small": [0.50, 0.35, 0.13, 0.02],
    "medium": [0.20, 0.40, 0.30, 0.10],
    "large": [0.10, 0.30, 0.40, 0.20],
}

# Per-num-dropoffs (lo, hi) ratio bounds. Each location's SCU share must fall
# within these bounds; ratios are sampled via Dirichlet + rejection (concentration=5.0).
DROPOFF_RATIO_BOUNDS = {
    1: [(1.0, 1.0)],
    2: [(0.35, 0.65), (0.35, 0.65)],
    3: [(0.20, 0.50), (0.20, 0.50), (0.20, 0.50)],
    4: [(0.10, 0.35)] * 4,
}

# Difficulty controls per-location bulk fraction (vs. 1/2-SCU remainder).
# Higher difficulty = more fragmentation = more filler items.
DIFFICULTY_BULK_RATIO = {
    "very-easy":   (0.88, 0.92),
    "easy":        (0.85, 0.90),
    "medium-low":  (0.82, 0.88),
    "medium":      (0.78, 0.85),
    "medium-high": (0.75, 0.82),
    "hard":        (0.70, 0.80),
    "very-hard":   (0.65, 0.78),
}

# Grid size categories for training
GRID_CATEGORIES = {
    "small": {
        "grids": [[(4,4,4)], [(6,6,3)], [(4,6,3)], [(3,3,3)], [(4,4,2)], [(5,5,2)], [(4,4,2), (2,2,2)]],
        "max_scu": 50,
        "bulk_containers": ["4 SCU", "8 SCU"],
        "rare_containers": [],
    },
    "medium": {
        "grids": [[(8,8,8)], [(12,6,8)], [(8,6,2)], [(5,7,5)], [(6,9,3)], [(4,10,2)], [(8,8,4), (4,4,4)], [(6,6,3), (6,6,3)]],
        "max_scu": 200,
        "bulk_containers": ["4 SCU", "8 SCU", "16 SCU"],
        # Rare is empty by default for medium and only enabled when
        # run_type="mixed" (rolled by medium_mixed_probability, default 0.0).
        "rare_containers": ["24 SCU", "32 SCU"],
    },
    "large": {
        "grids": [[(12,6,8)], [(8,15,4)], [(8,12,4)], [(6,18,2)], [(12,6,8), (8,8,4), (4,4,4)]],
        "max_scu": 400,
        "bulk_containers": ["8 SCU", "16 SCU"],
        "rare_containers": ["24 SCU", "32 SCU"],
    }
}

def get_grid_category(grids_list: List[Tuple[int, int, int]]) -> str:
    """Determine which category a ship belongs to based on its total volume."""
    volume = sum(g[0] * g[1] * g[2] for g in grids_list)

    if volume <= 64:  # 4x4x4 = 64
        return "small"
    elif volume <= 512:  # 8x8x8 = 512
        return "medium"
    else:
        return "large"


def container_fits_any_grid(container_dims: List[int], grids: List[Tuple[int, int, int]]) -> bool:
    """Check if a container can physically fit in at least one grid with Z-axis rotation."""
    sd = sorted(container_dims, reverse=True)
    for grid in grids:
        gd = sorted(grid, reverse=True)
        # Z-axis rotation: height locked, only swap X/Y
        rot0 = sd[0] <= gd[0] and sd[1] <= gd[1] and sd[2] <= gd[2]
        rot1 = sd[1] <= gd[0] and sd[0] <= gd[1] and sd[2] <= gd[2]
        if rot0 or rot1:
            return True
    return False


# ---------------------------------------------------------------------------
# v2 helpers (drop-off-as-primary-unit). Pure functions; tested in isolation.
# ---------------------------------------------------------------------------

_DIFFICULTY_MULTIPLIERS = {
    "very-easy": 0.3, "easy": 0.5, "medium-low": 0.6, "medium": 0.7,
    "medium-high": 0.8, "hard": 0.9, "very-hard": 1.0,
}


def sample_ratios(num_dropoffs: int, max_attempts: int = 200) -> List[float]:
    """Sample per-location SCU ratios that fall within DROPOFF_RATIO_BOUNDS.

    Dirichlet (concentration=5.0) + rejection sampling. Falls back to a
    uniform split only if `max_attempts` rejections occur in a row. Per A1,
    validation tolerates ≥99% in-bounds, not 100%.
    """
    if num_dropoffs == 1:
        return [1.0]
    bounds = DROPOFF_RATIO_BOUNDS[num_dropoffs]
    for _ in range(max_attempts):
        ratios = np.random.dirichlet([5.0] * num_dropoffs).tolist()
        if all(lo <= r <= hi for r, (lo, hi) in zip(ratios, bounds)):
            return ratios
    return [1.0 / num_dropoffs] * num_dropoffs


def _resolve_run_type(category: str, medium_mixed_probability: float) -> str:
    """Select normal/mixed/bulk per category. Large samples weighted; medium
    rolls medium_mixed_probability; small is always normal."""
    if category == "large":
        return random.choices(["normal", "mixed", "bulk"], weights=[0.7, 0.2, 0.1])[0]
    if category == "medium" and random.random() < medium_mixed_probability:
        return "mixed"
    return "normal"


def _split_total_scu(total_scu: int, num_dropoffs: int) -> List[int]:
    """Sample ratios and convert to per-location int SCU.

    Rounding remainder is pushed onto the largest location. When total_scu
    is large enough (>= num_dropoffs), every location is guaranteed >= 1 SCU
    so fill_location always emits at least one entry per priority. This
    matters for `force_num_dropoffs` correctness (A2): a forced 4-stop run
    on a tiny manifest must still surface 4 priorities.
    """
    ratios = sample_ratios(num_dropoffs)
    location_scu = [int(total_scu * r) for r in ratios]
    remainder = total_scu - sum(location_scu)
    while remainder > 0:
        largest_idx = location_scu.index(max(location_scu))
        location_scu[largest_idx] += 1
        remainder -= 1
    while remainder < 0:
        largest_idx = location_scu.index(max(location_scu))
        location_scu[largest_idx] -= 1
        remainder += 1
    # Ensure every location is non-empty when arithmetically possible.
    # Iteratively borrow 1 SCU from the largest into any zero location.
    while 0 in location_scu and total_scu >= num_dropoffs:
        zero_idx = location_scu.index(0)
        largest_idx = location_scu.index(max(location_scu))
        if location_scu[largest_idx] <= 1:
            break
        location_scu[largest_idx] -= 1
        location_scu[zero_idx] += 1
    return location_scu


def _resolve_dropoffs(
    category: str,
    total_scu: int,
    force_num_dropoffs: Optional[int],
    medium_mixed_probability: float,
) -> Tuple[int, str, List[int]]:
    """Pick drop-off count, run_type, and per-location SCU split.

    A2: when `force_num_dropoffs` is set, the sub-2-SCU guard is skipped —
    the caller has explicitly asked for that count.
    """
    if force_num_dropoffs is not None:
        num_dropoffs = force_num_dropoffs
    else:
        num_dropoffs = random.choices(
            [1, 2, 3, 4], weights=DROPOFF_WEIGHTS[category]
        )[0]
        # Guard: each location must hold at least 2 SCU on average.
        while num_dropoffs > 1 and total_scu / num_dropoffs < 2:
            num_dropoffs -= 1

    run_type = _resolve_run_type(category, medium_mixed_probability)
    location_scu = _split_total_scu(total_scu, num_dropoffs)
    return num_dropoffs, run_type, location_scu


def _compute_total_scu(grid_volume: int, target_fill_ratio: float, difficulty: str) -> int:
    """Apply fill ratio × difficulty multiplier × ±3 jitter, clamped at 4."""
    target = int(grid_volume * target_fill_ratio)
    if difficulty in _DIFFICULTY_MULTIPLIERS:
        target = int(target * _DIFFICULTY_MULTIPLIERS[difficulty])
    target += random.randint(-3, 3)
    return max(4, target)


def fill_location(
    target_scu: int,
    priority: int,
    available_containers: List[str],
    difficulty: str,
) -> List[Dict]:
    """Two-phase fill for a single drop-off location.

    Phase 1 (bulk): weighted-pick from GAME_FREQUENCY over `available_containers`
                    until `current_scu >= target * uniform(*DIFFICULTY_BULK_RATIO[diff])`.
    Phase 2 (remainder): 60% bias to 2 SCU when remainder >= 2, else 1 SCU.

    All entries share `priority`. Quantity per entry: fillers up to 12, bulk up to 5.
    """
    bulk_lo, bulk_hi = DIFFICULTY_BULK_RATIO.get(difficulty, (0.78, 0.85))
    bulk_target = int(target_scu * random.uniform(bulk_lo, bulk_hi))

    entries: List[Dict] = []
    current_scu = 0

    # Phase 1 — bulk fill. Cap quantity to leave at least 1 SCU for Phase 2,
    # so filler presence per location is guaranteed (spec validation req #1
    # and spec §5: "Phase 2 guarantees every location has 1/2 SCU presence").
    PHASE2_HEADROOM = 1
    while current_scu < bulk_target:
        # Available SCU for Phase 1 (reserve headroom for Phase 2 filler).
        phase1_capacity = max(0, target_scu - current_scu - PHASE2_HEADROOM)
        if phase1_capacity <= 0:
            break
        valid = [
            c for c in available_containers
            if SCU_DEFINITIONS[c]["volume"] <= phase1_capacity
        ]
        if not valid:
            break
        weights = [GAME_FREQUENCY.get(c, 1.0) for c in valid]
        container_type = random.choices(valid, weights=weights)[0]
        unit_volume = SCU_DEFINITIONS[container_type]["volume"]
        max_quantity = phase1_capacity // unit_volume
        if max_quantity <= 0:
            break
        if container_type in FILLER_CONTAINERS:
            quantity = random.randint(1, min(12, max_quantity))
        else:
            quantity = random.randint(1, min(5, max_quantity))
        entries.append({
            "scu_type": container_type,
            "quantity": quantity,
            "priority": priority,
        })
        current_scu += quantity * unit_volume

    # Phase 2 — remainder (1/2 SCU only). Guarantees filler presence per location.
    remainder = target_scu - current_scu
    while remainder > 0:
        if remainder >= 2 and random.random() < 0.6:
            container_type = "2 SCU"
            unit_volume = 2
        else:
            container_type = "1 SCU"
            unit_volume = 1
        entries.append({
            "scu_type": container_type,
            "quantity": 1,
            "priority": priority,
        })
        remainder -= unit_volume

    return entries


# ---------------------------------------------------------------------------
# Public generator. Drop-off-as-primary-unit.
# ---------------------------------------------------------------------------

def generate_scu_manifest(
    grid_dims: Optional[Tuple[int, int, int]] = None,
    grids_list: Optional[List[Tuple[int, int, int]]] = None,
    target_fill_ratio: float = 0.8,
    difficulty: str = "medium",
    run_type: Optional[str] = None,
    force_num_dropoffs: Optional[int] = None,
    medium_mixed_probability: float = 0.0,
) -> List[Dict]:
    """Generate a cargo manifest, drop-off-location-first.

    Args:
        grid_dims: Single grid dims (used only if grids_list is None).
        grids_list: Actual grid dimensions for physical-fit filtering.
        target_fill_ratio: Target ratio of grid volume to fill (pre-difficulty).
        difficulty: Curriculum difficulty ("very-easy" .. "very-hard").
        run_type: "normal" | "mixed" | "bulk". When None, sampled per category.
        force_num_dropoffs: If set, overrides random drop-off count and skips
            the sub-2-SCU guard (A2). For testing/override only.
        medium_mixed_probability: Probability of medium ships running a mixed
            (rare-container-permitted) contract. Default 0.0 matches game.

    Returns:
        Manifest list of {"scu_type", "quantity", "priority"} dicts.
        Priority equals 1-based drop-off index (1 = first stop).
    """
    # Resolve grids and category.
    if grids_list is None:
        if grid_dims is not None:
            grids_list = [grid_dims]
        else:
            category = random.choice(["small", "medium", "large"])
            grids_list = random.choice(GRID_CATEGORIES[category]["grids"])
    category = get_grid_category(grids_list)
    grid_volume = sum(g[0] * g[1] * g[2] for g in grids_list)

    # Total SCU with jitter; resolve drop-off count, run_type, per-loc split.
    total_scu = _compute_total_scu(grid_volume, target_fill_ratio, difficulty)
    num_dropoffs, sampled_run_type, location_scu = _resolve_dropoffs(
        category, total_scu, force_num_dropoffs, medium_mixed_probability,
    )
    if run_type is None:
        run_type = sampled_run_type

    # Build the available container pool:
    # - filler always
    # - bulk always (per category)
    # - rare iff run_type permits AND category permits
    cat_cfg = GRID_CATEGORIES[category]
    available = list(FILLER_CONTAINERS) + list(cat_cfg["bulk_containers"])
    if run_type in ("mixed", "bulk") and cat_cfg["rare_containers"]:
        available.extend(cat_cfg["rare_containers"])

    # Hard physical-fit constraint.
    available = [
        c for c in available
        if container_fits_any_grid(SCU_DEFINITIONS[c]["dimensions"], grids_list)
    ]
    if not available:
        available = ["1 SCU"]  # fallback — can always place a 1×1×1

    # Fill each drop-off independently. Priority = drop-off index (1-based).
    manifest: List[Dict] = []
    for idx, loc_scu in enumerate(location_scu):
        manifest.extend(fill_location(
            target_scu=loc_scu,
            priority=idx + 1,
            available_containers=available,
            difficulty=difficulty,
        ))

    # Sort by (priority, -volume) — preserves existing downstream contract.
    manifest.sort(key=lambda x: (x["priority"], -SCU_DEFINITIONS[x["scu_type"]]["volume"]))
    return manifest


def manifest_to_item_list(manifest: List[Dict]) -> List[Tuple[float, float, float, float, int]]:
    """
    Convert SCU manifest to item list format expected by the bin packing environment.
    
    Args:
        manifest: List of SCU container dictionaries
        
    Returns:
        List of tuples (width, length, height, weight, priority)
    """
    item_list = []
    
    for entry in manifest:
        scu_type = entry["scu_type"]
        quantity = entry["quantity"]
        priority = entry["priority"]
        
        dims = SCU_DEFINITIONS[scu_type]["dimensions"]
        weight = SCU_DEFINITIONS[scu_type]["weight"]
        
        # Add each container instance
        for _ in range(quantity):
            item_list.append((
                float(dims[0]),  # width
                float(dims[1]),  # length
                float(dims[2]),  # height
                float(weight),   # weight
                priority         # priority
            ))

    # Hybrid sort: items that need to reserve a whole grid go first (largest-first
    # within that tier); everything else respects priority order so the priority
    # constraint (higher priority must be at lower Y) never gets blocked by lower-
    # priority items consuming low-Y rows ahead of them.
    if item_list:
        # Threshold: 80% of the smallest grid we know about. We don't have grid info
        # in this function, so use absolute SCU thresholds aligned with category
        # breakpoints (smallest SCU grid in the project ≈ 24-32 SCU).
        big_threshold = 25  # any item ≥ 25 SCU is "reserve a grid" sized
        big = [x for x in item_list if (x[0] * x[1] * x[2]) >= big_threshold]
        small = [x for x in item_list if (x[0] * x[1] * x[2]) < big_threshold]
        big.sort(key=lambda x: (-(x[0] * x[1] * x[2]), x[4]))
        small.sort(key=lambda x: (x[4], -(x[0] * x[1] * x[2])))
        item_list = big + small

    return item_list

def generate_training_batch(
    batch_size: int = 10,
    category: str = "medium"
) -> List[Tuple[List[Tuple[int, int, int]], List[Tuple]]]:
    """
    Generate a batch of training examples for a specific grid category.
    
    Args:
        batch_size: Number of examples to generate
        category: Grid size category ("small", "medium", "large")
        
    Returns:
        List of (grids_list, item_list) tuples
    """
    training_batch = []
    
    for _ in range(batch_size):
        # Select random grid from category
        grids_list = random.choice(GRID_CATEGORIES[category]["grids"])
        
        # Vary difficulty and fill ratio
        difficulty = random.choice(["easy", "medium-low", "medium", "medium-high", "hard"])
        fill_ratio = random.uniform(0.6, 0.9)
        
        # Generate manifest
        manifest = generate_scu_manifest(
            grids_list=grids_list,
            target_fill_ratio=fill_ratio,
            difficulty=difficulty,
        )
        
        # Convert to item list
        item_list = manifest_to_item_list(manifest)
        
        training_batch.append((grids_list, item_list))
    
    return training_batch

def print_manifest_summary(manifest: List[Dict], grids_list: Optional[List[Tuple[int, int, int]]] = None):
    """Print a human-readable summary of a manifest."""
    total_scu = sum(
        entry["quantity"] * SCU_DEFINITIONS[entry["scu_type"]]["volume"] 
        for entry in manifest
    )
    
    print(f"\n=== Cargo Manifest Summary ===")
    if grids_list:
        grid_volume = sum(g[0] * g[1] * g[2] for g in grids_list)
        print(f"Grid Configurations: {grids_list} ({grid_volume} SCU capacity)")
        print(f"Total Cargo: {total_scu} SCU ({total_scu/grid_volume*100:.1f}% utilization)")
    else:
        print(f"Total Cargo: {total_scu} SCU")
    
    print(f"\nManifest by Priority:")
    
    # Group by priority
    by_priority = {}
    for entry in manifest:
        priority = entry["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(entry)
    
    for priority in sorted(by_priority.keys()):
        print(f"\nDrop-off {priority} (priority {priority}):")
        for entry in by_priority[priority]:
            scu_volume = entry["quantity"] * SCU_DEFINITIONS[entry["scu_type"]]["volume"]
            print(f"  - {entry['quantity']}x {entry['scu_type']} = {scu_volume} SCU")

# Example usage and testing
if __name__ == "__main__":
    # Test manifest generation for different grid sizes
    test_grids = [
        [(4, 4, 4)],   # Small
        [(8, 8, 8)],   # Medium
        [(12, 6, 8), (8, 8, 4)],  # Large Multi-grid
    ]

    for grids_list in test_grids:
        print(f"\n{'='*50}")
        manifest = generate_scu_manifest(
            grids_list=grids_list,
            target_fill_ratio=0.8,
            difficulty="medium",
        )
        
        print_manifest_summary(manifest, grids_list)
        
        # Convert to item list
        item_list = manifest_to_item_list(manifest)
        print(f"\nGenerated {len(item_list)} individual containers for packing")
    
    # Test training batch generation
    print(f"\n{'='*50}")
    print("Generating training batch...")
    batch = generate_training_batch(batch_size=3, category="medium")
    print(f"Generated {len(batch)} training examples")
    for i, (dims, items) in enumerate(batch):
        print(f"  Example {i+1}: Grids {dims}, {len(items)} items")


def validate_distribution(category: str, n: int = 1000) -> Dict:
    """Generate n manifests for `category` and report seven validation metrics.

    Filters the grid pool to only those that re-categorize to `category` via
    `get_grid_category`. Uses default `medium_mixed_probability=0.0` for the
    standard report. Prints a labeled summary and returns metrics as a dict.
    """
    # Filter grid pool so that only grids matching the requested category are used.
    pool = [
        g for g in GRID_CATEGORIES[category]["grids"]
        if get_grid_category(g) == category
    ]
    if not pool:
        pool = list(GRID_CATEGORIES[category]["grids"])

    rare_set = set(["24 SCU", "32 SCU"])

    # Counters / accumulators.
    total_locations = 0
    locations_with_filler = 0
    locations_odd = 0
    size_volume_totals: Dict[str, float] = {k: 0.0 for k in SCU_DEFINITIONS.keys()}
    manifests_with_rare = 0
    dropoff_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    ratio_groups_total = 0
    ratio_groups_in_bounds = 0
    manifests_all_fit = 0

    for _ in range(n):
        grids_list = random.choice(pool)
        manifest = generate_scu_manifest(
            grids_list=grids_list,
            target_fill_ratio=0.8,
            difficulty="medium",
            medium_mixed_probability=0.0,
        )
        if not manifest:
            continue

        # Group entries by priority (= drop-off index).
        by_priority: Dict[int, List[Dict]] = {}
        for entry in manifest:
            by_priority.setdefault(entry["priority"], []).append(entry)

        num_dropoffs = len(by_priority)
        if num_dropoffs in dropoff_counts:
            dropoff_counts[num_dropoffs] += 1

        # Per-location metrics.
        loc_totals: List[int] = []
        for prio in sorted(by_priority.keys()):
            entries = by_priority[prio]
            total_locations += 1
            loc_total = sum(
                e["quantity"] * SCU_DEFINITIONS[e["scu_type"]]["volume"]
                for e in entries
            )
            loc_totals.append(loc_total)
            if any(e["scu_type"] in FILLER_CONTAINERS for e in entries):
                locations_with_filler += 1
            if loc_total % 2 == 1:
                locations_odd += 1
            for e in entries:
                size_volume_totals[e["scu_type"]] += (
                    e["quantity"] * SCU_DEFINITIONS[e["scu_type"]]["volume"]
                )

        # Rare presence (per manifest).
        if any(e["scu_type"] in rare_set for e in manifest):
            manifests_with_rare += 1

        # Ratio bounds compliance.
        manifest_total = sum(loc_totals)
        if manifest_total > 0 and num_dropoffs in DROPOFF_RATIO_BOUNDS:
            bounds = DROPOFF_RATIO_BOUNDS[num_dropoffs]
            for loc_total, (lo, hi) in zip(loc_totals, bounds):
                ratio_groups_total += 1
                r = loc_total / manifest_total
                if lo <= r <= hi:
                    ratio_groups_in_bounds += 1

        # Physical fit.
        if all(
            container_fits_any_grid(SCU_DEFINITIONS[e["scu_type"]]["dimensions"], grids_list)
            for e in manifest
        ):
            manifests_all_fit += 1

    # Compute percentages and frequencies.
    filler_pct = (locations_with_filler / total_locations * 100.0) if total_locations else 0.0
    odd_pct = (locations_odd / total_locations * 100.0) if total_locations else 0.0
    rare_pct = (manifests_with_rare / n * 100.0) if n else 0.0
    fit_pct = (manifests_all_fit / n * 100.0) if n else 0.0
    ratio_pct = (
        ratio_groups_in_bounds / ratio_groups_total * 100.0
        if ratio_groups_total else 0.0
    )
    dropoff_observed = {k: (v / n) for k, v in dropoff_counts.items()}
    dropoff_expected = list(DROPOFF_WEIGHTS[category])

    # Top 3 by total volume.
    sorted_sizes = sorted(size_volume_totals.items(), key=lambda x: -x[1])
    top3 = sorted_sizes[:3]

    # Print summary.
    print(f"=== Distribution validation: {category} (n={n}) ===")
    print(f"Filler presence per location: {filler_pct:.2f}% (expected >= 95%)")
    top3_str = ", ".join(f"{k}={v:.0f}" for k, v in top3)
    print(f"Top 3 container sizes by total volume: {top3_str}")
    print(f"Rare container rate (24/32 SCU): {rare_pct:.2f}%")
    print(f"Per-location odd-total parity: {odd_pct:.2f}% (expected >= 40%)")
    obs_str = ", ".join(f"{k}={dropoff_observed[k]:.3f}" for k in sorted(dropoff_observed.keys()))
    exp_str = ", ".join(f"{i+1}={w:.3f}" for i, w in enumerate(dropoff_expected))
    print(f"Drop-off count observed: {obs_str}")
    print(f"Drop-off count expected: {exp_str}")
    print(f"Ratio bounds compliance: {ratio_pct:.2f}% (informational)")
    print(f"Physical fit: {fit_pct:.2f}% (expected 100%)")

    return {
        "category": category,
        "n": n,
        "filler_presence_pct": float(filler_pct),
        "size_distribution": {k: float(v) for k, v in size_volume_totals.items()},
        "rare_rate_pct": float(rare_pct),
        "odd_parity_pct": float(odd_pct),
        "dropoff_observed": {k: float(v) for k, v in dropoff_observed.items()},
        "dropoff_expected": [float(x) for x in dropoff_expected],
        "ratio_bound_compliance_pct": float(ratio_pct),
        "physical_fit_pct": float(fit_pct),
    }