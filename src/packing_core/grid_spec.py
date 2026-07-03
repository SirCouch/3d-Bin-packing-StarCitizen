"""Grid schema parsing and validation.

A grid entry denotes a cargo grid as its full bounding envelope plus a list
of blocked cuboids (hull structure / trimmed regions where cargo can never
be placed):

    {
        "dimensions": [4, 6, 2],          # envelope [x, y, z]
        "name": "Cargo Grid",
        "blocked": [                       # optional; omitted = plain cuboid
            {
                "position": [2, 0, 0],     # min corner, grid units
                "dimensions": [2, 2, 2],   # extent, grid units
                "supports": true           # top face acts as floor for cargo
            }
        ]
    }

`supports: true` means cargo may rest on the blocker's top face (a raised
floor / load-bearing step). `supports: false` (the default) means the face
is a wall or void — items above it are not considered supported by it.

In-code grid entries are `(dims, name)` or `(dims, name, blocked)` tuples;
`normalize_grid_entry` accepts both plus the JSON dict form.
"""


def normalize_grid_entry(entry):
    """Return (dims_tuple, name, blocked_list) from any accepted entry form.

    Accepted forms:
      - (dims, name)
      - (dims, name, blocked)
      - {"dimensions": [...], "name": ..., "blocked": [...]}
    """
    if isinstance(entry, dict):
        dims = tuple(float(d) for d in entry["dimensions"])
        name = entry.get("name", "Grid")
        blocked = entry.get("blocked", []) or []
    else:
        dims = tuple(float(d) for d in entry[0])
        name = entry[1]
        blocked = entry[2] if len(entry) > 2 else []
        blocked = blocked or []
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise ValueError(f"grid '{name}': dimensions must be 3 positive numbers, got {dims}")
    blockers = [_normalize_blocker(b, name) for b in blocked]
    validate_blockers(dims, blockers, name)
    return dims, name, blockers


def _normalize_blocker(raw, grid_name):
    """Return {'position': (x,y,z), 'dimensions': (w,l,h), 'supports': bool}."""
    pos = tuple(float(v) for v in raw["position"])
    dims = tuple(float(v) for v in raw["dimensions"])
    if len(pos) != 3:
        raise ValueError(f"grid '{grid_name}': blocker position must have 3 values, got {pos}")
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise ValueError(f"grid '{grid_name}': blocker dimensions must be 3 positive numbers, got {dims}")
    return {"position": pos, "dimensions": dims, "supports": bool(raw.get("supports", False))}


def validate_blockers(grid_dims, blockers, grid_name="Grid"):
    """Raise ValueError if any blocker is out of bounds, overlaps another,
    or the blockers fill the entire envelope."""
    for b in blockers:
        for axis in range(3):
            lo = b["position"][axis]
            hi = lo + b["dimensions"][axis]
            if lo < 0 or hi > grid_dims[axis]:
                raise ValueError(
                    f"grid '{grid_name}': blocker at {b['position']} size {b['dimensions']} "
                    f"exceeds envelope {tuple(grid_dims)} on axis {axis}")
    for i in range(len(blockers)):
        for j in range(i + 1, len(blockers)):
            if _cuboids_overlap(blockers[i], blockers[j]):
                raise ValueError(
                    f"grid '{grid_name}': blockers {i} and {j} overlap "
                    f"({blockers[i]['position']}/{blockers[i]['dimensions']} vs "
                    f"{blockers[j]['position']}/{blockers[j]['dimensions']})")
    if blockers and blocked_volume(blockers) >= grid_dims[0] * grid_dims[1] * grid_dims[2]:
        raise ValueError(f"grid '{grid_name}': blockers fill the entire envelope — no usable space")


def _cuboids_overlap(a, b):
    for axis in range(3):
        a_lo = a["position"][axis]
        a_hi = a_lo + a["dimensions"][axis]
        b_lo = b["position"][axis]
        b_hi = b_lo + b["dimensions"][axis]
        if a_hi <= b_lo or b_hi <= a_lo:
            return False
    return True


def blocked_volume(blockers):
    """Total volume of the blocked cuboids (assumes they are disjoint,
    which validate_blockers enforces)."""
    return sum(b["dimensions"][0] * b["dimensions"][1] * b["dimensions"][2] for b in blockers)


def usable_volume(dims, blockers):
    """Envelope volume minus blocked volume."""
    return dims[0] * dims[1] * dims[2] - blocked_volume(blockers)


def usable_volume_of_entry(entry):
    """Usable volume for any accepted grid-entry form (validates as a side effect)."""
    dims, _, blockers = normalize_grid_entry(entry)
    return usable_volume(dims, blockers)
