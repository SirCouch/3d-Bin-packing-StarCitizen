"""One-off: parse the sc-cargo.space JS bundle and extract ship cargo grid data
with spatial positions, then merge into ships_cargo_grids.json.

The bundle stores each ship as:
  X="Manufacturer",Y="ShipName",Z={capacity:N,groups:[{x,z,grids:[{x,y,z,width,height,length,...}]}],...}
"""
import json
import re
from pathlib import Path

BUNDLE = Path(__file__).parent / "sc_bundle.js"
SHIPS_JSON = Path(__file__).resolve().parent.parent / "ships_cargo_grids.json"
OUT_JSON = Path(__file__).resolve().parent.parent / "ships_cargo_grids_positioned.json"


def extract_balanced_object(src: str, start: int) -> str:
    """Given index of '{' in src, return the slice through the matching '}'."""
    assert src[start] == '{'
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(src)):
        c = src[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise ValueError("Unbalanced braces")


# Convert a JS object-literal (unquoted keys, no trailing commas) to JSON.
# The bundle uses simple keys like capacity, groups, grids, x, y, z, width, height,
# length, maxSize, labels, value, fontSize. No function expressions or templates inside
# the data objects we care about.
KEY_RE = re.compile(r'([{,]\s*)([a-zA-Z_$][\w$]*)\s*:')


BOOL_RE = re.compile(r'(?<![\w$])!0(?![\w$])')
BOOL_RE_F = re.compile(r'(?<![\w$])!1(?![\w$])')


def js_obj_to_json(s: str) -> str:
    # Substitute minified booleans
    s = BOOL_RE.sub('true', s)
    s = BOOL_RE_F.sub('false', s)
    # Quote bareword keys
    return KEY_RE.sub(r'\1"\2":', s)


def load_bundle():
    with open(BUNDLE, 'r', encoding='utf-8') as f:
        return f.read()


SHIP_HEAD = re.compile(
    r'([a-zA-Z_$][\w$]*)="([^"]{2,40})",([a-zA-Z_$][\w$]*)="([^"]{2,60})",([a-zA-Z_$][\w$]*)=\{capacity:(\d+),groups:\['
)


def extract_ships(src: str):
    ships = []
    for m in SHIP_HEAD.finditer(src):
        manufacturer = m.group(2)
        name = m.group(4)
        # The data object starts at m.end()-len('{capacity:<digits>,groups:[')
        # We know the '{' of the data object is right after the third '='
        data_brace = src.find('{', m.start(5))
        obj_str = extract_balanced_object(src, data_brace)
        try:
            data = json.loads(js_obj_to_json(obj_str))
        except json.JSONDecodeError as e:
            # Some objects may contain values we didn't anticipate (e.g. template strings).
            # Skip silently and report.
            print(f"[warn] Could not parse {manufacturer} {name}: {e}")
            continue
        ships.append({"manufacturer": manufacturer, "name": name, "data": data})
    return ships


def flatten_ship(ship_entry):
    """Convert {groups:[{x,z,grids:[{x,y,z,w,h,l}...]}]} → flat grids with world positions."""
    data = ship_entry["data"]
    flat = []
    for gi, group in enumerate(data.get("groups", [])):
        gx = group.get("x", 0)
        gy = group.get("y", 0)
        gz = group.get("z", 0)
        for ci, cell in enumerate(group.get("grids", [])):
            # Skip cells that are obstacles / non-cargo (they have maxSize sometimes)
            w = cell.get("width")
            h = cell.get("height")
            l = cell.get("length")
            if w is None or h is None or l is None:
                continue
            flat.append({
                "name": f"G{gi+1}-{ci+1}",
                "dimensions": [w, l, h],   # project convention: [x,y,z]-dims
                "position": [
                    gx + cell.get("x", 0),
                    gz + cell.get("z", 0),   # sc-cargo z ≈ depth → our y
                    gy + cell.get("y", 0),
                ],
            })
    return flat


NAME_ALIASES = {
    # project-name -> sc-cargo-name
    "C2 Hercules": "C2 Hercules",
    "M2 Hercules": "M2 Hercules",
    "Hull-A": "Hull-A",
    "Hull-B": None,  # not in sc-cargo bundle (see warning)
    "Hull-c": "Hull-C",
    "890 Jump": "890 Jump",
    "Carrack": "Carrack",
    "Caterpillar": "Caterpillar",
    "Hermes": "Hermes",
    "Polaris": "Polaris",
    "Perseus": "Perseus",
    "Apollo": "Apollo Medivac",
    "Clipper": "Clipper",
    "Corsair": "Corsair",
    "Cutlass Black": "Cutlass Black",
    "Cutlass Red": "Cutlass Red",
    "Cutter": "Cutter",
    "Freelancer": "Freelancer",
    "Freelancer MAX": "Freelancer MAX",
    "Mercury Star Runner": "Mercury Star Runner",
    "MOLE": "MOLE",
    "MPUV-C": "MPUV-C",
    "MPUV-T": "MPUV-T",
    "Nomad": "Nomad",
    "Phoenix": "Constellation Phoenix",
    "Andromeda": "Constellation Andromeda",
    "Aquila": "Constellation Aquila",
    "RAFT": "Raft",
    "Raft": "Raft",
    "Reliant Kore": "Reliant Kore",
    "Retaliator": "Retaliator",
    "Valkyrie": None,
    "Shiv": None,
    "Avenger Titan": "Avenger Titan",
    "Avenger Renegade": None,
    "Prowler Utility": "Prowler Utility",
    "Intrepid": "Intrepid",
    "Fortune": "Fortune",
    "Cyclone": "Cyclone",
    "Mule": "Mule",
    "Golem Ox": "Golem Ox",
    "Cutter Scout": "Cutter Scout",
}


def merge():
    src = load_bundle()
    sc_ships = extract_ships(src)
    by_name = {s["name"]: s for s in sc_ships}
    print(f"Parsed {len(sc_ships)} ships from bundle.")

    with open(SHIPS_JSON, "r") as f:
        project = json.load(f)

    matched = 0
    unmatched = []
    for ship in project["ships"]:
        pname = ship["name"]
        sc_name = NAME_ALIASES.get(pname, pname)
        if sc_name is None:
            unmatched.append(f"{pname} (no sc-cargo equivalent)")
            continue
        sc_ship = by_name.get(sc_name)
        if sc_ship is None:
            unmatched.append(f"{pname} -> {sc_name} (not in bundle)")
            continue
        flat = flatten_ship(sc_ship)
        # Keep project's existing grid dims as ground truth; only import positions.
        # Strategy: for each project grid, try to find best-matching sc-cargo cell by dims.
        # Fall back to index order if counts match.
        existing = ship["grids"]
        if len(existing) == len(flat):
            # Try to reorder flat to match by best dimension match
            used = set()
            mapped = []
            for eg in existing:
                ed = sorted(eg["dimensions"])
                best_i = None
                best_score = float('inf')
                for i, fg in enumerate(flat):
                    if i in used:
                        continue
                    fd = sorted(fg["dimensions"])
                    score = sum(abs(a - b) for a, b in zip(ed, fd))
                    if score < best_score:
                        best_score = score
                        best_i = i
                used.add(best_i)
                mapped.append(flat[best_i] if best_i is not None else None)
            for eg, fg in zip(existing, mapped):
                if fg is not None:
                    eg["position"] = fg["position"]
            matched += 1
        else:
            # Count mismatch — record but still try a best-effort positional import
            # using the first N cells from sc-cargo
            n = min(len(existing), len(flat))
            for eg, fg in zip(existing[:n], flat[:n]):
                eg["position"] = fg["position"]
            matched += 1
            unmatched.append(f"{pname}: grid count {len(existing)} vs sc-cargo {len(flat)} (partial position import)")

    with open(OUT_JSON, "w") as f:
        json.dump(project, f, indent=2)

    print(f"\nMatched {matched}/{len(project['ships'])} project ships.")
    if unmatched:
        print("\nUnmatched / issues:")
        for u in unmatched:
            print(f"  - {u}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    merge()
