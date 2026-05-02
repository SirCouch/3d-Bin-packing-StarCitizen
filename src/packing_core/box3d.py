import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Box3D:
    """Float-primary 3D box. Position and dimensions stored as Python floats
    for fast Python-level math (overlap/contains/fit checks). Tensor views of
    `position` and `dimensions` are materialized lazily on first access — most
    Box3D instances created during MER expansion are filtered out before any
    tensor view is needed, so the cost stays off the hot path.

    Properties `x1, x2, y1, y2, z1, z2, width, length, height, volume` all
    return Python floats. Callers that previously did `box.width.item()` should
    drop the `.item()`; the value is already a float.
    """

    __slots__ = ("_fx1", "_fy1", "_fz1", "_fw", "_fl", "_fh", "_fvol",
                 "_pos_t", "_dim_t")

    def __init__(self, position, dimensions):
        if isinstance(position, torch.Tensor):
            x1, y1, z1 = position.tolist()
            self._pos_t = position if position.device == device else position.to(device)
        else:
            x1, y1, z1 = float(position[0]), float(position[1]), float(position[2])
            self._pos_t = None
        if isinstance(dimensions, torch.Tensor):
            w, l, h = dimensions.tolist()
            self._dim_t = dimensions if dimensions.device == device else dimensions.to(device)
        else:
            w, l, h = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
            self._dim_t = None
        self._fx1 = x1
        self._fy1 = y1
        self._fz1 = z1
        self._fw = w
        self._fl = l
        self._fh = h
        self._fvol = w * l * h

    @property
    def position(self):
        if self._pos_t is None:
            self._pos_t = torch.tensor(
                [self._fx1, self._fy1, self._fz1], dtype=torch.float, device=device,
            )
        return self._pos_t

    @property
    def dimensions(self):
        if self._dim_t is None:
            self._dim_t = torch.tensor(
                [self._fw, self._fl, self._fh], dtype=torch.float, device=device,
            )
        return self._dim_t

    @property
    def x1(self): return self._fx1
    @property
    def y1(self): return self._fy1
    @property
    def z1(self): return self._fz1
    @property
    def x2(self): return self._fx1 + self._fw
    @property
    def y2(self): return self._fy1 + self._fl
    @property
    def z2(self): return self._fz1 + self._fh
    @property
    def width(self): return self._fw
    @property
    def length(self): return self._fl
    @property
    def height(self): return self._fh
    @property
    def volume(self): return self._fvol

    def overlaps(self, other):
        """Float-only overlap test. Strict inequality matches the original
        tensor-based behavior (touching faces do NOT count as overlapping)."""
        return not (
            self._fx1 + self._fw <= other._fx1 or other._fx1 + other._fw <= self._fx1 or
            self._fy1 + self._fl <= other._fy1 or other._fy1 + other._fl <= self._fy1 or
            self._fz1 + self._fh <= other._fz1 or other._fz1 + other._fh <= self._fz1
        )

    def contains(self, other):
        """Float-only containment test."""
        return (
            self._fx1 <= other._fx1 and other._fx1 + other._fw <= self._fx1 + self._fw and
            self._fy1 <= other._fy1 and other._fy1 + other._fl <= self._fy1 + self._fl and
            self._fz1 <= other._fz1 and other._fz1 + other._fh <= self._fz1 + self._fh
        )

    def get_intersection(self, other):
        """Float-only intersection. Returns Box3D constructed from float lists
        (no tensor allocation unless caller reads .position/.dimensions later)."""
        sx2 = self._fx1 + self._fw
        sy2 = self._fy1 + self._fl
        sz2 = self._fz1 + self._fh
        ox2 = other._fx1 + other._fw
        oy2 = other._fy1 + other._fl
        oz2 = other._fz1 + other._fh
        if sx2 <= other._fx1 or ox2 <= self._fx1:
            return None
        if sy2 <= other._fy1 or oy2 <= self._fy1:
            return None
        if sz2 <= other._fz1 or oz2 <= self._fz1:
            return None
        x1 = self._fx1 if self._fx1 > other._fx1 else other._fx1
        y1 = self._fy1 if self._fy1 > other._fy1 else other._fy1
        z1 = self._fz1 if self._fz1 > other._fz1 else other._fz1
        x2 = sx2 if sx2 < ox2 else ox2
        y2 = sy2 if sy2 < oy2 else oy2
        z2 = sz2 if sz2 < oz2 else oz2
        return Box3D((x1, y1, z1), (x2 - x1, y2 - y1, z2 - z1))

    def can_fit(self, item_dimensions):
        """Float-only fit check. Accepts tuple/list of floats or tensor."""
        if isinstance(item_dimensions, torch.Tensor):
            iw, il, ih = item_dimensions.tolist()
        else:
            iw = float(item_dimensions[0])
            il = float(item_dimensions[1])
            ih = float(item_dimensions[2])
        return self._fw >= iw and self._fl >= il and self._fh >= ih

    def to_dict(self):
        return {
            "position": self.position.clone(),
            "dimensions": self.dimensions.clone(),
        }

    def __repr__(self):
        return (f"Box3D(pos=[{self._fx1:.1f},{self._fy1:.1f},{self._fz1:.1f}], "
                f"dims=[{self._fw:.1f},{self._fl:.1f},{self._fh:.1f}])")
