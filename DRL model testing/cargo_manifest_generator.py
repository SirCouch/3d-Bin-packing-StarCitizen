import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class CargoItem:
    width: float
    length: float
    height: float
    weight: float
    unload_priority: int
    item_id: str
    item_type: str = "standard"
    location: str = "main"

    def __str__(self):
        return f"ID: {self.item_id}, Type: {self.item_type}, Dims: {self.width}x{self.length}x{self.height}, Weight: {self.weight}, Priority: {self.unload_priority}"

    def to_tuple(self):
        """Convert to the format expected by the DRL environment"""
        return (self.width, self.length, self.height, self.weight, self.unload_priority)


# Standard container definitions
STANDARD_CONTAINERS = {
    "1 SCU": (1, 1, 1, 1),  # (width, length, height, weight)
    "2 SCU": (2, 1, 1, 2),
    "4 SCU": (2, 2, 1, 4),
    "8 SCU": (2, 2, 2, 8),
    "16 SCU": (2, 4, 2, 16),
    "32 SCU": (2, 8, 2, 32)
}

# Parse cargo items from the provided data
def parse_cargo_data(data_lines):
    """Parse the cargo data from the provided lines"""
    standard_containers = {}
    cargo_items = {}

    # Parse standard containers
    for line in data_lines[:6]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            container_id = parts[0]
            container_name = parts[1]
            width = float(parts[2])
            length = float(parts[3])
            height = float(parts[4])
            weight = float(parts[5])
            standard_containers[container_name] = (width, length, height, weight)

    # Parse cargo items
    for line in data_lines[6:]:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            item_id = parts[0]
            ship_id = parts[1]
            location_type = parts[2]
            location_name = parts[3]
            width = float(parts[4])
            length = float(parts[5])
            height = float(parts[6])
            weight = float(parts[7])

            # Create a unique identifier
            unique_id = f"{item_id}_{location_type}_{location_name}"

            cargo_items[unique_id] = CargoItem(
                width=width,
                length=length,
                height=height,
                weight=weight,
                unload_priority=random.randint(1, 5),  # Assign random unload priority for now
                item_id=unique_id,
                item_type="cargo",
                location=f"{location_type}_{location_name}"
            )

    return standard_containers, cargo_items

# Generate sample cargo manifests
def generate_cargo_manifest(
        num_items: int = 10,
        standard_containers: Optional[Dict] = None,
        cargo_items: Optional[Dict] = None,
        prioritize_custom: bool = True,
        seed: Optional[int] = None
) -> List[CargoItem]:
    """
    Generate a cargo manifest with a mix of standard and custom containers

    Args:
        num_items: Number of items to include in the manifest
        standard_containers: Dictionary of standard container definitions
        cargo_items: Dictionary of custom cargo items
        prioritize_custom: Whether to prioritize using custom cargo items
        seed: Random seed for reproducibility

    Returns:
        List of CargoItem objects
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if standard_containers is None:
        standard_containers = STANDARD_CONTAINERS

    manifest = []

    # If we have custom cargo items and want to prioritize them
    if cargo_items and prioritize_custom:
        # Determine how many custom items to use (up to num_items)
        num_custom = min(len(cargo_items), num_items)
        selected_custom = random.sample(list(cargo_items.values()), num_custom)

        # Add custom items to manifest
        manifest.extend(selected_custom)

        # Fill remaining slots with standard containers if needed
        remaining = num_items - num_custom
        if remaining > 0 and standard_containers:
            for _ in range(remaining):
                container_type = random.choice(list(standard_containers.keys()))
                w, l, h, weight = standard_containers[container_type]

                manifest.append(CargoItem(
                    width=w,
                    length=l,
                    height=h,
                    weight=weight,
                    unload_priority=random.randint(1, 5),
                    item_id=f"std_{container_type}_{len(manifest)}",
                    item_type="standard",
                    location="cargo_bay"
                ))
    # If we're not prioritizing custom or don't have custom items
    else:
        # Generate manifest from standard containers
        for i in range(num_items):
            if cargo_items and random.random() < 0.3:  # 30% chance of using custom item
                item_id = random.choice(list(cargo_items.keys()))
                manifest.append(cargo_items[item_id])
            else:
                container_type = random.choice(list(standard_containers.keys()))
                w, l, h, weight = standard_containers[container_type]

                manifest.append(CargoItem(
                    width=w,
                    length=l,
                    height=h,
                    weight=weight,
                    unload_priority=random.randint(1, 5),
                    item_id=f"std_{container_type}_{i}",
                    item_type="standard",
                    location="cargo_bay"
                ))

    # Adjust unload priorities to ensure they are distinct and follow a logical order
    # Sort items by their current unload priority
    manifest.sort(key=lambda x: x.unload_priority)

    # Assign new priorities based on their position in the sorted list
    for i, item in enumerate(manifest):
        item.unload_priority = i + 1

    # Shuffle the manifest to ensure random order for training
    random.shuffle(manifest)

    return manifest

# Function to use in the DRL training loop
def sample_cargo_manifest(
        min_items: int = 5,
        max_items: int = 15,
        standard_containers: Optional[Dict] = None,
        cargo_items: Optional[Dict] = None,
        seed: Optional[int] = None
) -> List[Tuple]:
    """
    Sample a cargo manifest for training the DRL model

    Args:
        min_items: Minimum number of items in the manifest
        max_items: Maximum number of items in the manifest
        standard_containers: Dictionary of standard container definitions
        cargo_items: Dictionary of custom cargo items
        seed: Random seed

    Returns:
        List of tuples in the format (width, length, height, weight, unload_priority)
    """
    # Generate a random number of items
    num_items = random.randint(min_items, max_items)

    # Generate the cargo manifest
    manifest = generate_cargo_manifest(
        num_items=num_items,
        standard_containers=standard_containers,
        cargo_items=cargo_items,
        prioritize_custom=random.random() < 0.7,  # 70% chance of prioritizing custom items
        seed=seed
    )

    # Convert to the format expected by the DRL environment
    return [item.to_tuple() for item in manifest]

# Parse the provided data
def parse_provided_data(raw_data):
    """Parse the raw data string into standard containers and cargo items"""
    lines = raw_data.strip().split('\n')
    return parse_cargo_data(lines)

# Example usage
if __name__ == "__main__":
    # Example raw data string (replace with actual data)
    raw_data = """
1,1 SCU,1,1,1,1,Standard 1 SCU cargo container
3,4 SCU,2,2,1,4,Standard 4 SCU cargo container
5,16 SCU,2,4,2,16,Standard 16 SCU cargo container
6,32 SCU,2,8,2,32,Standard 32 SCU cargo container
4,8 SCU,2,2,2,8,Standard 8 SCU cargo container
2,2 SCU,2,1,1,2,Standard 2 SCU cargo container

72,1,main,Main,2,4,3,2
73,16,main,Main,6,6,3,32
74,17,auxiliary,Left,1,2,2,2
(truncated for brevity)
    """

    # Parse data
    std_containers, cargo_items = parse_provided_data(raw_data)

    # Generate a sample manifest
    manifest = generate_cargo_manifest(
        num_items=10,
        standard_containers=std_containers,
        cargo_items=cargo_items,
        seed=42
    )

    # Print the manifest
    print(f"Sample manifest with {len(manifest)} items:")
    for i, item in enumerate(manifest, 1):
        print(f"{i}. {item}")

    # Generate a training sample
    training_manifest = sample_cargo_manifest(
        standard_containers=std_containers,
        cargo_items=cargo_items,
        seed=123
    )

    print("\nSample training manifest:")
    for i, (w, l, h, wt, p) in enumerate(training_manifest, 1):
        print(f"{i}. Width: {w}, Length: {l}, Height: {h}, Weight: {wt}, Priority: {p}")