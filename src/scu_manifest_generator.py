import random
from typing import List, Tuple, Dict, Optional

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

# Grid size categories for training
GRID_CATEGORIES = {
    "small": {
        "grids": [(4,4,4), (6,6,3), (4,6,3), (3,3,3), (4,4,2), (5,5,2)],
        "max_scu": 50,
        "preferred_containers": ["1 SCU", "2 SCU", "4 SCU", "8 SCU"]
    },
    "medium": {
        "grids": [(8,8,8), (12,6,8), (8,6,2), (5,7,5), (6,9,3), (4,10,2)],
        "max_scu": 200,
        "preferred_containers": ["4 SCU", "8 SCU", "16 SCU", "24 SCU"]
    },
    "large": {
        "grids": [(12,6,8), (8,15,4), (8,12,4), (6,18,2), (12,6,8)],
        "max_scu": 400,
        "preferred_containers": ["8 SCU", "16 SCU", "24 SCU", "32 SCU"]
    }
}

def get_grid_category(grid_dims: Tuple[int, int, int]) -> str:
    """Determine which category a grid belongs to based on its volume."""
    volume = grid_dims[0] * grid_dims[1] * grid_dims[2]
    
    if volume <= 64:  # 4x4x4 = 64
        return "small"
    elif volume <= 512:  # 8x8x8 = 512
        return "medium"
    else:
        return "large"

def generate_scu_manifest(
    grid_dims: Optional[Tuple[int, int, int]] = None,
    target_fill_ratio: float = 0.8,
    difficulty: str = "medium",
    priority_groups: int = 3
) -> List[Dict]:
    """
    Generate a cargo manifest with SCU containers.
    
    Args:
        grid_dims: Grid dimensions to generate manifest for. If None, uses random grid.
        target_fill_ratio: Target ratio of grid volume to fill (0.0 to 1.0)
        difficulty: Difficulty level affecting container variety and packing complexity
        priority_groups: Number of priority groups (1-5)
        
    Returns:
        List of dictionaries with SCU container info
    """
    # Determine grid and category
    if grid_dims is None:
        category = random.choice(["small", "medium", "large"])
        grid_dims = random.choice(GRID_CATEGORIES[category]["grids"])
    else:
        category = get_grid_category(grid_dims)
    
    grid_volume = grid_dims[0] * grid_dims[1] * grid_dims[2]
    target_scu = int(grid_volume * target_fill_ratio)
    
    # Adjust target based on difficulty
    difficulty_multipliers = {
        "very-easy": 0.3,
        "easy": 0.5,
        "medium-low": 0.6,
        "medium": 0.7,
        "medium-high": 0.8,
        "hard": 0.9,
        "very-hard": 1.0
    }
    
    if difficulty in difficulty_multipliers:
        target_scu = int(target_scu * difficulty_multipliers[difficulty])
    
    # Get preferred containers for this category
    preferred_containers = GRID_CATEGORIES[category]["preferred_containers"]
    
    # Generate manifest
    manifest = []
    current_scu = 0
    
    # Difficulty affects container variety
    if difficulty in ["very-easy", "easy"]:
        # Use fewer container types
        available_containers = preferred_containers[:2]
    elif difficulty in ["medium-low", "medium", "medium-high"]:
        # Use moderate variety
        available_containers = preferred_containers[:3]
    else:
        # Use all container types
        available_containers = list(SCU_DEFINITIONS.keys())
    
    # Generate containers
    while current_scu < target_scu:
        remaining_scu = target_scu - current_scu
        
        # Choose container type based on remaining space
        valid_containers = [
            c for c in available_containers 
            if SCU_DEFINITIONS[c]["volume"] <= remaining_scu
        ]
        
        if not valid_containers:
            # Use smallest container if no valid options
            container_type = "1 SCU"
        else:
            # Weighted selection favoring larger containers
            weights = [SCU_DEFINITIONS[c]["volume"] for c in valid_containers]
            container_type = random.choices(valid_containers, weights=weights)[0]
        
        # Determine quantity
        max_quantity = remaining_scu // SCU_DEFINITIONS[container_type]["volume"]
        if max_quantity > 0:
            # For easier difficulties, use more of the same container
            if difficulty in ["very-easy", "easy"]:
                quantity = random.randint(max(1, max_quantity // 2), max_quantity)
            else:
                quantity = random.randint(1, min(5, max_quantity))
            
            # Assign priority (1 = unload first, higher = unload later)
            priority = random.randint(1, priority_groups)
            
            manifest.append({
                "scu_type": container_type,
                "quantity": quantity,
                "priority": priority
            })
            
            current_scu += quantity * SCU_DEFINITIONS[container_type]["volume"]
    
    # Sort by priority for better organization
    manifest.sort(key=lambda x: x["priority"])
    
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
    
    return item_list

def generate_training_batch(
    batch_size: int = 10,
    category: str = "medium"
) -> List[Tuple[Tuple[int, int, int], List[Tuple]]]:
    """
    Generate a batch of training examples for a specific grid category.
    
    Args:
        batch_size: Number of examples to generate
        category: Grid size category ("small", "medium", "large")
        
    Returns:
        List of (grid_dims, item_list) tuples
    """
    training_batch = []
    
    for _ in range(batch_size):
        # Select random grid from category
        grid_dims = random.choice(GRID_CATEGORIES[category]["grids"])
        
        # Vary difficulty and fill ratio
        difficulty = random.choice(["easy", "medium-low", "medium", "medium-high", "hard"])
        fill_ratio = random.uniform(0.6, 0.9)
        
        # Generate manifest
        manifest = generate_scu_manifest(
            grid_dims=grid_dims,
            target_fill_ratio=fill_ratio,
            difficulty=difficulty,
            priority_groups=random.randint(2, 4)
        )
        
        # Convert to item list
        item_list = manifest_to_item_list(manifest)
        
        training_batch.append((grid_dims, item_list))
    
    return training_batch

def print_manifest_summary(manifest: List[Dict], grid_dims: Optional[Tuple[int, int, int]] = None):
    """Print a human-readable summary of a manifest."""
    total_scu = sum(
        entry["quantity"] * SCU_DEFINITIONS[entry["scu_type"]]["volume"] 
        for entry in manifest
    )
    
    print(f"\n=== Cargo Manifest Summary ===")
    if grid_dims:
        grid_volume = grid_dims[0] * grid_dims[1] * grid_dims[2]
        print(f"Grid Dimensions: {grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]} ({grid_volume} SCU capacity)")
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
        print(f"\nPriority {priority} (unload {'first' if priority == 1 else 'later'}):")
        for entry in by_priority[priority]:
            scu_volume = entry["quantity"] * SCU_DEFINITIONS[entry["scu_type"]]["volume"]
            print(f"  - {entry['quantity']}x {entry['scu_type']} = {scu_volume} SCU")

# Example usage and testing
if __name__ == "__main__":
    # Test manifest generation for different grid sizes
    test_grids = [
        (4, 4, 4),   # Small
        (8, 8, 8),   # Medium
        (12, 6, 8),  # Large
    ]
    
    for grid_dims in test_grids:
        print(f"\n{'='*50}")
        manifest = generate_scu_manifest(
            grid_dims=grid_dims,
            target_fill_ratio=0.8,
            difficulty="medium",
            priority_groups=3
        )
        
        print_manifest_summary(manifest, grid_dims)
        
        # Convert to item list
        item_list = manifest_to_item_list(manifest)
        print(f"\nGenerated {len(item_list)} individual containers for packing")
    
    # Test training batch generation
    print(f"\n{'='*50}")
    print("Generating training batch...")
    batch = generate_training_batch(batch_size=3, category="medium")
    print(f"Generated {len(batch)} training examples")
    for i, (dims, items) in enumerate(batch):
        print(f"  Example {i+1}: Grid {dims}, {len(items)} items")