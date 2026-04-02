import torch
from .box3d import Box3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MERManager:
    """Manages the set of Maximal Empty Rectangles in the container.
    Stores MERs as Box3D objects directly to avoid repeated dict→Box3D conversion."""

    def __init__(self, container_dimensions):
        self.container = Box3D(
            position=torch.zeros(3, dtype=torch.float),
            dimensions=container_dimensions
        )
        # Store as Box3D objects directly
        self.mers = [Box3D(torch.zeros(3, dtype=torch.float), container_dimensions.clone())]

    def update(self, placed_item):
        """Update MERs after placing a new item."""
        overlapping = []
        non_overlapping = []

        for mer in self.mers:
            if mer.overlaps(placed_item):
                overlapping.append(mer)
            else:
                non_overlapping.append(mer)

        new_mers = []
        for mer in overlapping:
            # Up to 6 new MERs (one per face of the placed item)
            if placed_item.x2 < mer.x2:
                new_mers.append(Box3D(
                    torch.tensor([placed_item.x2, mer.y1, mer.z1], dtype=torch.float),
                    torch.tensor([mer.x2 - placed_item.x2, mer.length, mer.height], dtype=torch.float)
                ))
            if placed_item.x1 > mer.x1:
                new_mers.append(Box3D(
                    torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    torch.tensor([placed_item.x1 - mer.x1, mer.length, mer.height], dtype=torch.float)
                ))
            if placed_item.y2 < mer.y2:
                new_mers.append(Box3D(
                    torch.tensor([mer.x1, placed_item.y2, mer.z1], dtype=torch.float),
                    torch.tensor([mer.width, mer.y2 - placed_item.y2, mer.height], dtype=torch.float)
                ))
            if placed_item.y1 > mer.y1:
                new_mers.append(Box3D(
                    torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    torch.tensor([mer.width, placed_item.y1 - mer.y1, mer.height], dtype=torch.float)
                ))
            if placed_item.z2 < mer.z2:
                new_mers.append(Box3D(
                    torch.tensor([mer.x1, mer.y1, placed_item.z2], dtype=torch.float),
                    torch.tensor([mer.width, mer.length, mer.z2 - placed_item.z2], dtype=torch.float)
                ))
            if placed_item.z1 > mer.z1:
                new_mers.append(Box3D(
                    torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    torch.tensor([mer.width, mer.length, placed_item.z1 - mer.z1], dtype=torch.float)
                ))

        candidates = new_mers + non_overlapping
        self.mers = self._filter_redundant(candidates)
        return self.mers

    def _filter_redundant(self, boxes):
        """Filter out redundant MERs (contained within others or too small)."""
        min_vol = 0.01 * self.container.volume

        kept = []
        for i, box_i in enumerate(boxes):
            if box_i.volume < min_vol:
                continue
            is_maximal = True
            for j, box_j in enumerate(boxes):
                if i != j:
                    intersection = box_i.get_intersection(box_j)
                    if intersection and intersection.volume > 0.9 * box_i.volume:
                        is_maximal = False
                        break
            if is_maximal:
                kept.append(box_i)
        return kept

    def get_feasible_mers(self, item_dimensions):
        """Get indices of MERs that can fit the item in either Z-axis orientation."""
        feasible = []
        orientations = [
            item_dimensions,
            torch.tensor([item_dimensions[1], item_dimensions[0], item_dimensions[2]], dtype=torch.float)
        ]
        for i, mer in enumerate(self.mers):
            for orientation in orientations:
                if mer.can_fit(orientation):
                    feasible.append(i)
                    break
        return feasible

    def __len__(self):
        return len(self.mers)
