import torch
from .box3d import Box3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MERManager:
    """Manages Maximal Empty Rectangles. All hot-path math runs in pure Python
    floats via Box3D's float-primary properties — no tensor allocations during
    update/filter/get_feasible_mers."""

    def __init__(self, container_dimensions):
        self.container = Box3D(
            position=torch.zeros(3, dtype=torch.float),
            dimensions=container_dimensions,
        )
        self.mers = [
            Box3D(torch.zeros(3, dtype=torch.float), container_dimensions.clone())
        ]

    def update(self, placed_item):
        """Update MERs after placing a new item. Child MERs are constructed
        from float tuples — no per-call tensor allocation."""
        # Snapshot placed-item floats once.
        ix1 = placed_item._fx1
        iy1 = placed_item._fy1
        iz1 = placed_item._fz1
        ix2 = ix1 + placed_item._fw
        iy2 = iy1 + placed_item._fl
        iz2 = iz1 + placed_item._fh

        overlapping = []
        non_overlapping = []
        for mer in self.mers:
            if mer.overlaps(placed_item):
                overlapping.append(mer)
            else:
                non_overlapping.append(mer)

        new_mers = []
        for mer in overlapping:
            mx1 = mer._fx1
            my1 = mer._fy1
            mz1 = mer._fz1
            mw = mer._fw
            ml = mer._fl
            mh = mer._fh
            mx2 = mx1 + mw
            my2 = my1 + ml
            mz2 = mz1 + mh
            # Up to 6 child MERs (one per face of the placed item, where there's room).
            if ix2 < mx2:
                new_mers.append(Box3D((ix2, my1, mz1), (mx2 - ix2, ml, mh)))
            if ix1 > mx1:
                new_mers.append(Box3D((mx1, my1, mz1), (ix1 - mx1, ml, mh)))
            if iy2 < my2:
                new_mers.append(Box3D((mx1, iy2, mz1), (mw, my2 - iy2, mh)))
            if iy1 > my1:
                new_mers.append(Box3D((mx1, my1, mz1), (mw, iy1 - my1, mh)))
            if iz2 < mz2:
                new_mers.append(Box3D((mx1, my1, iz2), (mw, ml, mz2 - iz2)))
            if iz1 > mz1:
                new_mers.append(Box3D((mx1, my1, mz1), (mw, ml, iz1 - mz1)))

        candidates = new_mers + non_overlapping
        self.mers = self._filter_redundant(candidates)
        return self.mers

    def _filter_redundant(self, boxes):
        """Filter out redundant MERs (contained within others or too small).
        Float-only volume + intersection math; no tensor ops."""
        min_vol = 1.0  # smallest packable SCU volume
        kept = []
        for i, box_i in enumerate(boxes):
            if box_i._fvol < min_vol:
                continue
            is_maximal = True
            cutoff = 0.9 * box_i._fvol
            for j, box_j in enumerate(boxes):
                if i == j:
                    continue
                intersection = box_i.get_intersection(box_j)
                if intersection is not None and intersection._fvol > cutoff:
                    is_maximal = False
                    break
            if is_maximal:
                kept.append(box_i)
        return kept

    def get_feasible_mers(self, item_dimensions):
        """Indices of MERs that can fit the item in either Z-axis orientation.
        Accepts a tensor or sequence of 3 floats."""
        if isinstance(item_dimensions, torch.Tensor):
            iw, il, ih = item_dimensions.tolist()
        else:
            iw = float(item_dimensions[0])
            il = float(item_dimensions[1])
            ih = float(item_dimensions[2])
        feasible = []
        for i, mer in enumerate(self.mers):
            mw = mer._fw
            ml = mer._fl
            mh = mer._fh
            if mh < ih:
                continue  # height is locked under Z-axis rotation
            # Default orientation OR X/Y-swap rotation.
            if (mw >= iw and ml >= il) or (mw >= il and ml >= iw):
                feasible.append(i)
        return feasible

    def __len__(self):
        return len(self.mers)
