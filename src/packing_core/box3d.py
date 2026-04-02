import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Box3D:
    """Represents a 3D box with position and dimensions"""

    def __init__(self, position, dimensions):
        """
        Args:
            position (torch.Tensor): [x, y, z] coordinates of the box's bottom-left-front corner
            dimensions (torch.Tensor): [width, length, height] of the box
        """
        self.position = position.to(device) if isinstance(position, torch.Tensor) else torch.tensor(position, dtype=torch.float, device=device)
        self.dimensions = dimensions.to(device) if isinstance(dimensions, torch.Tensor) else torch.tensor(dimensions,
                                                                                               dtype=torch.float, device=device)

    @property
    def x1(self):
        return self.position[0]

    @property
    def y1(self):
        return self.position[1]

    @property
    def z1(self):
        return self.position[2]

    @property
    def x2(self):
        return self.position[0] + self.dimensions[0]

    @property
    def y2(self):
        return self.position[1] + self.dimensions[1]

    @property
    def z2(self):
        return self.position[2] + self.dimensions[2]

    @property
    def width(self):
        return self.dimensions[0]

    @property
    def length(self):
        return self.dimensions[1]

    @property
    def height(self):
        return self.dimensions[2]

    @property
    def volume(self):
        return self.width * self.length * self.height

    def overlaps(self, other):
        """Check if this box overlaps with another"""
        return not (
                self.x2 <= other.x1 or other.x2 <= self.x1 or
                self.y2 <= other.y1 or other.y2 <= self.y1 or
                self.z2 <= other.z1 or other.z2 <= self.z1
        )

    def contains(self, other):
        """Check if this box fully contains another"""
        return (
                self.x1 <= other.x1 and other.x2 <= self.x2 and
                self.y1 <= other.y1 and other.y2 <= self.y2 and
                self.z1 <= other.z1 and other.z2 <= self.z2
        )

    def get_intersection(self, other):
        """Get the intersection box between this box and another"""
        if not self.overlaps(other):
            return None

        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        z1 = max(self.z1, other.z1)

        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        z2 = min(self.z2, other.z2)

        position = torch.tensor([x1, y1, z1], dtype=torch.float)
        dimensions = torch.tensor([x2 - x1, y2 - y1, z2 - z1], dtype=torch.float)

        return Box3D(position, dimensions)

    def can_fit(self, item_dimensions):
        """Check if an item with given dimensions can fit in this box"""
        return (
                self.width >= item_dimensions[0] and
                self.length >= item_dimensions[1] and
                self.height >= item_dimensions[2]
        )

    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'position': self.position.clone(),
            'dimensions': self.dimensions.clone()
        }

    def __repr__(self):
        return f"Box3D(pos=[{self.x1:.1f},{self.y1:.1f},{self.z1:.1f}], dims=[{self.width:.1f},{self.length:.1f},{self.height:.1f}])"


