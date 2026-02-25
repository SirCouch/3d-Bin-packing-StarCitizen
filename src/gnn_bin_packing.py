import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import random

# PyTorch Geometric imports
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, GATConv
import torch.nn.functional as F

# Import from existing bin packing modules (adjust as needed)
from scu_manifest_generator import (
    generate_scu_manifest,
    manifest_to_item_list,
    generate_training_batch,
    GRID_CATEGORIES,
    get_grid_category,
    print_manifest_summary
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done, log_prob=None, feasibility_mask=None):
        """
        Store a transition in the buffer

        Args:
            state: PyG Data object representing the state
            action: Action taken
            reward: Reward received
            next_state: Next state (PyG Data object)
            done: Boolean indicating if episode is done
            log_prob: Log probability of the action (for PPO)
            feasibility_mask: Mask of feasible actions
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob, feasibility_mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, log_probs, feasibility_masks = zip(*batch)

        # Convert to tensor where appropriate
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        actions = torch.LongTensor(actions).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        if log_probs[0] is not None:
            log_probs = torch.cat([lp.unsqueeze(0) for lp in log_probs if lp is not None])
        else:
            log_probs = None

        # Keep states and next_states as PyG Data objects
        # Note: For batched processing, you would use PyG's Batch.from_data_list()

        return states, actions, rewards, next_states, dones, log_probs, feasibility_masks

    def __len__(self):
        return len(self.buffer)


# Helper class for 3D geometry operations (MER handling)
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


class MERManager:
    """Manages the set of Maximal Empty Rectangles in the container"""

    def __init__(self, container_dimensions):
        """
        Args:
            container_dimensions (torch.Tensor): [width, length, height] of the container
        """
        self.container = Box3D(
            position=torch.zeros(3, dtype=torch.float),
            dimensions=container_dimensions
        )
        # Initialize with one MER covering the entire container
        self.mers = [self.container.to_dict()]

    def update(self, placed_item):
        """
        Update MERs after placing a new item

        Args:
            placed_item (Box3D): The newly placed item

        Returns:
            List of updated MERs
        """
        # Find MERs that overlap with the new item
        overlapping_mers = []
        non_overlapping_mers = []

        for mer_dict in self.mers:
            mer = Box3D(mer_dict['position'], mer_dict['dimensions'])
            if mer.overlaps(placed_item):
                overlapping_mers.append(mer)
            else:
                non_overlapping_mers.append(mer_dict)

        # Generate new MERs from the overlapping regions
        new_mers = []
        for mer in overlapping_mers:
            # For each overlapping MER, generate up to 6 new MERs (one for each face)

            # 1. MER to the right of placed_item (X+)
            if placed_item.x2 < mer.x2:
                new_mer = Box3D(
                    position=torch.tensor([placed_item.x2, mer.y1, mer.z1], dtype=torch.float),
                    dimensions=torch.tensor([mer.x2 - placed_item.x2, mer.length, mer.height], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

            # 2. MER to the left of placed_item (X-)
            if placed_item.x1 > mer.x1:
                new_mer = Box3D(
                    position=torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    dimensions=torch.tensor([placed_item.x1 - mer.x1, mer.length, mer.height], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

            # 3. MER behind placed_item (Y+)
            if placed_item.y2 < mer.y2:
                new_mer = Box3D(
                    position=torch.tensor([mer.x1, placed_item.y2, mer.z1], dtype=torch.float),
                    dimensions=torch.tensor([mer.width, mer.y2 - placed_item.y2, mer.height], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

            # 4. MER in front of placed_item (Y-)
            if placed_item.y1 > mer.y1:
                new_mer = Box3D(
                    position=torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    dimensions=torch.tensor([mer.width, placed_item.y1 - mer.y1, mer.height], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

            # 5. MER above placed_item (Z+)
            if placed_item.z2 < mer.z2:
                new_mer = Box3D(
                    position=torch.tensor([mer.x1, mer.y1, placed_item.z2], dtype=torch.float),
                    dimensions=torch.tensor([mer.width, mer.length, mer.z2 - placed_item.z2], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

            # 6. MER below placed_item (Z-)
            if placed_item.z1 > mer.z1:
                new_mer = Box3D(
                    position=torch.tensor([mer.x1, mer.y1, mer.z1], dtype=torch.float),
                    dimensions=torch.tensor([mer.width, mer.length, placed_item.z1 - mer.z1], dtype=torch.float)
                )
                new_mers.append(new_mer.to_dict())

        # Combine new and non-overlapping MERs
        candidate_mers = new_mers + non_overlapping_mers

        # Filter out redundant MERs (those completely contained within others)
        filtered_mers = self._filter_redundant_mers(candidate_mers)

        # Update MERs list
        self.mers = filtered_mers
        return self.mers

    def _filter_redundant_mers(self, mer_dicts):
        """Filter out redundant MERs but ensure at least one remains"""
        min_vol_threshold = 0.01 * self.container.volume

        # Convert dicts to Box3D
        boxes = [Box3D(m['position'], m['dimensions']) for m in mer_dicts]

        non_redundant_indices = []
        for i, box_i in enumerate(boxes):
            # Skip boxes with negligible volume
            if box_i.volume < min_vol_threshold:
                continue

            # Check for near-redundancy
            is_maximal = True
            for j, box_j in enumerate(boxes):
                if i != j:
                    intersection = box_i.get_intersection(box_j)
                    if intersection and intersection.volume > 0.9 * box_i.volume:
                        is_maximal = False
                        break

            if is_maximal:
                non_redundant_indices.append(i)

        # CRITICAL FIX: Ensure at least one MER remains
        if len(non_redundant_indices) == 0 and len(mer_dicts) > 0:
            # Keep the largest MER
            largest_idx = max(range(len(boxes)), key=lambda i: boxes[i].volume)
            non_redundant_indices = [largest_idx]

        return [mer_dicts[i] for i in non_redundant_indices]

    def get_feasible_mers(self, item_dimensions):
        """
        Get MERs that can fit an item with given dimensions

        Args:
            item_dimensions (torch.Tensor): [width, length, height] of the item

        Returns:
            List of feasible MERs indices
        """
        feasible_indices = []

        for i, mer_dict in enumerate(self.mers):
            mer = Box3D(mer_dict['position'], mer_dict['dimensions'])

            # Check all 6 possible orientations
            orientations = [
                [item_dimensions[0], item_dimensions[1], item_dimensions[2]],  # Original
                [item_dimensions[0], item_dimensions[2], item_dimensions[1]],  # Rotate around X
                [item_dimensions[1], item_dimensions[0], item_dimensions[2]],  # Rotate around Y
                [item_dimensions[1], item_dimensions[2], item_dimensions[0]],  # Rotate around X & Y
                [item_dimensions[2], item_dimensions[0], item_dimensions[1]],  # Rotate around X & Z
                [item_dimensions[2], item_dimensions[1], item_dimensions[0]],  # Rotate around Y & Z
            ]

            for orientation in orientations:
                if mer.can_fit(torch.tensor(orientation, dtype=torch.float)):
                    feasible_indices.append(i)
                    break

        return feasible_indices

    def __len__(self):
        return len(self.mers)


# Modified DRL Bin Packing Environment using Graph State
class DRLBinPackingEnv:
    def __init__(self, container_dims=(10, 10, 10), max_stack_weight=100.0):
        self.container_dims = torch.tensor(container_dims, dtype=torch.float, device=device)
        self.max_stack_weight = max_stack_weight

        # Node type constants
        self.NODE_TYPE_CONTAINER = 0  # Container/global info
        self.NODE_TYPE_PLACED = 1  # Placed items
        self.NODE_TYPE_ITEM = 2  # Current item to place
        self.NODE_TYPE_MER = 3  # Maximal Empty Rectangles

        # State representation components
        self.placed_items = []  # List of placed items as Box3D objects with additional info
        self.mer_manager = MERManager(self.container_dims)

        # Cargo information
        self.cargo_manifest = []
        self.current_item_idx = 0
        self.current_item = None

        # Episode tracking
        self.successful_placements = 0
        self.total_items = 0

    def reset(self, cargo_manifest=None, difficulty=None):
        """Reset the environment with an optional cargo manifest and difficulty level"""
        # Reset step counter
        self.steps_in_episode = 0

        # Rest of the original reset logic...
        self.placed_items = []
        self.mer_manager = MERManager(self.container_dims)

        # Generate cargo manifest
        if cargo_manifest is None:
            # Use SCU manifest generator instead of the old one
            from scu_manifest_generator import generate_scu_manifest, manifest_to_item_list

            # Generate manifest based on difficulty
            if difficulty is not None:
                fill_ratio = 0.7  # Default fill ratio
                manifest = generate_scu_manifest(
                    grid_dims=tuple(self.container_dims.tolist()),
                    target_fill_ratio=fill_ratio,
                    difficulty=difficulty,
                    priority_groups=3
                )
                self.cargo_manifest = manifest_to_item_list(manifest)
            else:
                # Generate a medium difficulty manifest by default
                manifest = generate_scu_manifest(
                    grid_dims=tuple(self.container_dims.tolist()),
                    target_fill_ratio=0.7,
                    difficulty="medium",
                    priority_groups=3
                )
                self.cargo_manifest = manifest_to_item_list(manifest)
        else:
            self.cargo_manifest = cargo_manifest

        # Reset counters
        self.current_item_idx = 0
        self.successful_placements = 0
        self.total_items = len(self.cargo_manifest)

        # Set first item
        if self.cargo_manifest:
            self.current_item = self._get_item_features(self.cargo_manifest[0])
        else:
            self.current_item = None

        return self._build_graph_state()

    def _get_item_features(self, item_tuple):
        """Convert manifest tuple to item features"""
        w, l, h, weight, priority = item_tuple
        return {
            'dimensions': torch.tensor([w, l, h], dtype=torch.float),
            'weight': weight,
            'priority': priority
        }

    def step(self, action):
        """
        Take a step by placing the current item in a selected MER

        Args:
            action: Index of the MER to place the item in

        Returns:
            next_state, reward, done, info
        """
        # Add timeout check to prevent infinite episodes
        MAX_STEPS_PER_EPISODE = 1000  # Adjust as needed
        if not hasattr(self, 'steps_in_episode'):
            self.steps_in_episode = 0

        self.steps_in_episode += 1
        if self.steps_in_episode >= MAX_STEPS_PER_EPISODE:
            print(f"Episode timeout after {MAX_STEPS_PER_EPISODE} steps")
            return self._build_graph_state(), -10, True, {"error": "Episode timeout"}

        if self.current_item is None:
            return self._build_graph_state(), 0, True, {"error": "No item to place"}

        # Check if there are any MERs available
        if len(self.mer_manager.mers) == 0:
            print("No MERs available - forcing episode end")
            return self._build_graph_state(), -10, True, {"error": "No MERs available"}

        # Check if action is valid
        if action >= len(self.mer_manager.mers):
            # Instead of just returning, move to next item
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), -5, done, {"error": "Invalid MER index"}

        # Rest of the original step logic...
        # Get selected MER and item dimensions
        selected_mer = Box3D(
            self.mer_manager.mers[action]['position'],
            self.mer_manager.mers[action]['dimensions']
        )
        item_dims = self.current_item['dimensions']

        # Check if placement is feasible
        if not selected_mer.can_fit(item_dims):
            reward = -5.0
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), reward, done, {"feasible": False}

        # Place item at MER origin (lower corner)
        placement_position = selected_mer.position.clone()

        # Check additional constraints
        feasible = self._check_additional_constraints(
            placement_position,
            item_dims,
            self.current_item['weight'],
            self.current_item['priority']
        )

        if not feasible:
            reward = -5.0
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), reward, done, {"feasible": False}

        # Successful placement
        placed_item = Box3D(placement_position, item_dims)
        self.placed_items.append({
            'box': placed_item,
            'weight': self.current_item['weight'],
            'priority': self.current_item['priority']
        })
        self.successful_placements += 1

        # Update MERs
        self.mer_manager.update(placed_item)

        # Calculate reward
        reward = self._calculate_reward(placement_position, item_dims)

        # Move to next item
        self._move_to_next_item()
        done = self.current_item is None

        # Bonus for completing all items
        if done and self.successful_placements == self.total_items:
            reward += 50.0

        return self._build_graph_state(), reward, done, {"feasible": True}

    def _move_to_next_item(self):
        """Move to the next item in the manifest"""
        self.current_item_idx += 1
        if self.current_item_idx < len(self.cargo_manifest):
            self.current_item = self._get_item_features(self.cargo_manifest[self.current_item_idx])
        else:
            self.current_item = None

    def _check_additional_constraints(self, position, dimensions, weight, priority):
        """Check additional constraints beyond simple fitting"""
        # Create a temporary box for the item
        item_box = Box3D(position, dimensions)

        # 1. Check container bounds (redundant with MER but kept for safety)
        container_box = Box3D(torch.zeros(3), self.container_dims)
        if not container_box.contains(item_box):
            return False

        # 2. Check overlap with existing items (redundant with MER but kept for safety)
        for placed in self.placed_items:
            if placed['box'].overlaps(item_box):
                return False

        # 3. Check stacking weight
        if not self._check_stacking_weight(item_box, weight):
            return False

        # 4. Check priority/route constraint
        if not self._check_priority_constraint(item_box, priority):
            return False

        # All checks passed
        return True

    def _check_stacking_weight(self, item_box, item_weight):
        """Check if weight constraints are satisfied"""
        # Calculate weight stacked on top of this item
        weight_above = 0.0

        for placed in self.placed_items:
            placed_box = placed['box']

            # Check if placed item is above the new item
            if (placed_box.z1 >= item_box.z2 and  # Placed item is above
                    placed_box.x1 < item_box.x2 and placed_box.x2 > item_box.x1 and  # X overlap
                    placed_box.y1 < item_box.y2 and placed_box.y2 > item_box.y1):  # Y overlap
                weight_above += placed['weight']

        # Check if any items are below this item and would have too much weight
        for placed in self.placed_items:
            placed_box = placed['box']

            # Check if placed item is below the new item
            if (placed_box.z2 <= item_box.z1 and  # Placed item is below
                    placed_box.x1 < item_box.x2 and placed_box.x2 > item_box.x1 and  # X overlap
                    placed_box.y1 < item_box.y2 and placed_box.y2 > item_box.y1):  # Y overlap

                # Check weight constraint
                if placed['weight'] + item_weight > self.max_stack_weight:
                    return False

        return True

    def _check_priority_constraint(self, item_box, item_priority):
        """
        Check if priority/route constraints are satisfied
        Lower priority = unload earlier = closer to door (lower y-value)
        """
        # Check if items with higher priority are behind this item
        for placed in self.placed_items:
            placed_box = placed['box']
            placed_priority = placed['priority']

            # If this item has lower priority (unload earlier), it should be closer to door
            if item_priority < placed_priority and item_box.y1 >= placed_box.y1:
                return False

            # If this item has higher priority (unload later), it should be further from door
            if item_priority > placed_priority and item_box.y1 <= placed_box.y1:
                return False

        return True

    def _calculate_reward(self, position, dimensions):
        """Calculate reward for a successful placement"""
        # Increase base reward for successful placement
        reward = 15.0

        # Calculate volume utilization component
        item_volume = torch.prod(dimensions)
        container_volume = torch.prod(self.container_dims)
        volume_ratio = item_volume / container_volume
        reward += 5.0 * volume_ratio

        # Efficiency reward: encourage placing items against walls or other items
        touching_reward = 0.0

        # Check if item touches the container walls
        if position[0] < 0.001:  # X = 0 wall
            touching_reward += 1.0
        if position[1] < 0.001:  # Y = 0 wall
            touching_reward += 1.0
        if position[2] < 0.001:  # Z = 0 wall
            touching_reward += 1.0
        if abs(position[0] + dimensions[0] - self.container_dims[0]) < 0.001:  # X = W wall
            touching_reward += 1.0
        if abs(position[1] + dimensions[1] - self.container_dims[1]) < 0.001:  # Y = L wall
            touching_reward += 1.0
        if abs(position[2] + dimensions[2] - self.container_dims[2]) < 0.001:  # Z = H wall
            touching_reward += 1.0

        # Check if item touches other items
        item_box = Box3D(position, dimensions)
        for placed in self.placed_items[:-1]:  # Exclude the item just placed
            placed_box = placed['box']

            # Check if boxes touch on any face
            if (abs(item_box.x1 - placed_box.x2) < 0.001 or
                    abs(item_box.x2 - placed_box.x1) < 0.001 or
                    abs(item_box.y1 - placed_box.y2) < 0.001 or
                    abs(item_box.y2 - placed_box.y1) < 0.001 or
                    abs(item_box.z1 - placed_box.z2) < 0.001 or
                    abs(item_box.z2 - placed_box.z1) < 0.001):
                touching_reward += 0.5

            # Bonus for items with similar priority being close to each other
            if abs(placed['priority'] - self.current_item['priority']) <= 1:
                touching_reward += 0.5

        # Cap the touching reward and add to base reward
        reward += min(touching_reward, 5.0)

        return reward

    def _build_graph_state(self):
        """Build the graph representation of the current state"""
        node_features = []
        node_types = []
        node_positions = []

        # 1. Container node (global state information)
        container_features = [
            self.NODE_TYPE_CONTAINER,  # Node type
            self.container_dims[0],  # Width
            self.container_dims[1],  # Length
            self.container_dims[2],  # Height
            float(self.successful_placements),  # Num items placed
            float(self.total_items),  # Total items
            0.0,  # Placeholder
            0.0,  # Placeholder
        ]
        node_features.append(container_features)
        node_types.append(self.NODE_TYPE_CONTAINER)
        node_positions.append([0.0, 0.0, 0.0])  # Container origin

        # Container node index is 0
        container_node_idx = 0

        # 2. Current item to place node
        if self.current_item is not None:
            item_features = [
                self.NODE_TYPE_ITEM,  # Node type
                self.current_item['dimensions'][0],  # Width
                self.current_item['dimensions'][1],  # Length
                self.current_item['dimensions'][2],  # Height
                float(self.current_item['weight']),  # Weight
                float(self.current_item['priority']),  # Priority
                0.0,  # Placeholder
                0.0,  # Placeholder
            ]
            node_features.append(item_features)
            node_types.append(self.NODE_TYPE_ITEM)
            node_positions.append([0.0, 0.0, 0.0])  # Placeholder position

            # Current item node index is 1 (if it exists)
            current_item_node_idx = 1
        else:
            current_item_node_idx = -1

        # 3. Placed item nodes
        placed_item_indices = {}
        for i, placed in enumerate(self.placed_items):
            placed_box = placed['box']
            placed_features = [
                self.NODE_TYPE_PLACED,  # Node type
                placed_box.width,  # Width
                placed_box.length,  # Length
                placed_box.height,  # Height
                float(placed['weight']),  # Weight
                float(placed['priority']),  # Priority
                0.0,  # Placeholder
                0.0,  # Placeholder
            ]
            node_features.append(placed_features)
            node_types.append(self.NODE_TYPE_PLACED)
            node_positions.append([
                placed_box.x1,
                placed_box.y1,
                placed_box.z1
            ])

            # Map placed item index to node index
            placed_item_indices[i] = len(node_features) - 1

        # 4. MER nodes
        mer_indices = {}
        for i, mer_dict in enumerate(self.mer_manager.mers):
            mer = Box3D(mer_dict['position'], mer_dict['dimensions'])
            mer_features = [
                self.NODE_TYPE_MER,  # Node type
                mer.width,  # Width
                mer.length,  # Length
                mer.height,  # Height
                mer.volume,  # Volume (useful feature)
                0.0,  # Placeholder for priority
                0.0,  # Placeholder
                0.0,  # Placeholder
            ]
            node_features.append(mer_features)
            node_types.append(self.NODE_TYPE_MER)
            node_positions.append([
                mer.x1,
                mer.y1,
                mer.z1
            ])

            # Map MER index to node index
            mer_indices[i] = len(node_features) - 1

        # Handle empty graph case
        if not node_features:
            # Return minimal graph with just container
            x = torch.tensor([[self.NODE_TYPE_CONTAINER, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index,
                        node_type=torch.tensor([self.NODE_TYPE_CONTAINER], dtype=torch.long),
                        pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float))

        # Convert lists to tensors
        x = torch.tensor(node_features, dtype=torch.float, device=device)
        node_type = torch.tensor(node_types, dtype=torch.long, device=device)
        pos = torch.tensor(node_positions, dtype=torch.float, device=device)

        # 5. Define edges (connect nodes based on geometric relationships)
        edge_index_list = []

        # Connect container to all nodes
        for i in range(1, len(node_features)):
            edge_index_list.append((container_node_idx, i))
            edge_index_list.append((i, container_node_idx))

        # Connect current item to all MER
            if current_item_node_idx != -1:
                for mer_idx, node_idx in mer_indices.items():
                    if mer_idx in self.mer_manager.get_feasible_mers(self.current_item['dimensions']):
                        edge_index_list.append((current_item_node_idx, node_idx))
                        edge_index_list.append((node_idx, current_item_node_idx))

                # Connect placed items to nearby placed items (optional, based on distance threshold)
            distance_threshold = 2.0  # Items within this distance are considered "nearby"
            for i, placed_i in enumerate(self.placed_items):
                box_i = placed_i['box']
                for j, placed_j in enumerate(self.placed_items):
                    if i != j:
                        box_j = placed_j['box']
                        # Compute center-to-center distance
                        center_i = torch.tensor([
                            box_i.x1 + box_i.width / 2,
                            box_i.y1 + box_i.length / 2,
                            box_i.z1 + box_i.height / 2
                        ])
                        center_j = torch.tensor([
                            box_j.x1 + box_j.width / 2,
                            box_j.y1 + box_j.length / 2,
                            box_j.z1 + box_j.height / 2
                        ])
                        distance = torch.norm(center_i - center_j)

                        if distance < distance_threshold:
                            edge_index_list.append((placed_item_indices[i], placed_item_indices[j]))
                            edge_index_list.append((placed_item_indices[j], placed_item_indices[i]))

            # Connect MERs to adjacent MERs and items
            for i, mer_dict in enumerate(self.mer_manager.mers):
                mer_i = Box3D(mer_dict['position'], mer_dict['dimensions'])

                # Connect to placed items that are adjacent
                for j, placed in enumerate(self.placed_items):
                    box_j = placed['box']

                    # Check if MER and item are adjacent (sharing a face)
                    if (abs(mer_i.x1 - box_j.x2) < 0.001 or
                            abs(mer_i.x2 - box_j.x1) < 0.001 or
                            abs(mer_i.y1 - box_j.y2) < 0.001 or
                            abs(mer_i.y2 - box_j.y1) < 0.001 or
                            abs(mer_i.z1 - box_j.z2) < 0.001 or
                            abs(mer_i.z2 - box_j.z1) < 0.001):
                        edge_index_list.append((mer_indices[i], placed_item_indices[j]))
                        edge_index_list.append((placed_item_indices[j], mer_indices[i]))

                # Connect to other MERs that are adjacent
                for j, mer_dict_j in enumerate(self.mer_manager.mers):
                    if i != j:
                        mer_j = Box3D(mer_dict_j['position'], mer_dict_j['dimensions'])

                        # Check if MERs are adjacent
                        if (abs(mer_i.x1 - mer_j.x2) < 0.001 or
                                abs(mer_i.x2 - mer_j.x1) < 0.001 or
                                abs(mer_i.y1 - mer_j.y2) < 0.001 or
                                abs(mer_i.y2 - mer_j.y1) < 0.001 or
                                abs(mer_i.z1 - mer_j.z2) < 0.001 or
                                abs(mer_i.z2 - mer_j.z1) < 0.001):
                            edge_index_list.append((mer_indices[i], mer_indices[j]))
                            edge_index_list.append((mer_indices[j], mer_indices[i]))

            # Create edge index tensor
            if not edge_index_list:  # Handle case with no edges
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=device).t().contiguous()

            # Create PyG Data object
            graph_data = Data(x=x, edge_index=edge_index, node_type=node_type, pos=pos)

            # Add additional info for accessing nodes
            graph_data.mer_node_indices = mer_indices
            graph_data.current_item_node_idx = current_item_node_idx

            return graph_data

    def get_feasibility_mask(self):
            """
            Returns a binary mask over MERs indicating which ones can fit the current item
            """
            if self.current_item is None:
                return np.zeros(len(self.mer_manager.mers), dtype=np.float32)

            feasible_mer_indices = self.mer_manager.get_feasible_mers(self.current_item['dimensions'])
            mask = np.zeros(len(self.mer_manager.mers), dtype=np.float32)

            for idx in feasible_mer_indices:
                mask[idx] = 1.0

            return mask

        # GNN-based Actor-Critic Networks with Batch Normalization & Regularization
class ActorGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
                # Graph Attention Network layers with multi-head attention
        self.conv1 = GATConv(node_feature_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)

                # Attention mechanism for MER selection
        self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data, feasibility_mask=None):
                """
                Forward pass to select a MER for item placement

                Args:
                    data: PyG Data object containing the graph state
                    feasibility_mask: Binary mask indicating which MERs can fit the current item

                Returns:
                    Categorical distribution over valid MER actions
                """
                if hasattr(data, 'x') and data.x.device != device:
                    data = data.to(device)

                x, edge_index = data.x, data.edge_index
                node_type = data.node_type

                # Graph convolutions with batch normalization and dropout
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.dropout1(x)

                x = self.conv2(x, edge_index)
                x = self.bn2(x)
                x = F.relu(x)
                x = self.dropout2(x)

                x = self.conv3(x, edge_index)
                x = self.bn3(x)
                x = F.relu(x)
                x = self.dropout3(x)

                # Get embeddings of MER nodes
                mer_mask = (node_type == DRLBinPackingEnv().NODE_TYPE_MER)

                if not torch.any(mer_mask):
                    # No MERs in the graph, return dummy distribution
                    return Categorical(logits=torch.zeros(1, device=x.device))

                mer_embeddings = x[mer_mask]

                # Calculate attention scores for each MER
                mer_scores = self.attention(mer_embeddings).squeeze(-1)

                # Apply feasibility mask if provided
                if feasibility_mask is not None:
                    # Convert mask to tensor and ensure it's on the same device
                    mask_tensor = torch.FloatTensor(feasibility_mask).to(device)
                    # Apply mask: set scores of infeasible MERs to a very low value
                    mer_scores = mer_scores + torch.log(mask_tensor + 1e-10)

                # Return categorical distribution over MER nodes
                return Categorical(logits=mer_scores)

class CriticGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
                # Graph Attention Network layers with multi-head attention
        self.conv1 = GATConv(node_feature_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)

                # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        """
        Forward pass to estimate the value of the current state

        Args:
            data: PyG Data object containing the graph state

        Returns:
            Estimated state value
        """
        x, edge_index = data.x, data.edge_index

        # Graph convolutions with batch normalization and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Global mean pooling to get graph-level representation
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_features = global_mean_pool(x, batch)

        # Predict value
        value = self.value_head(global_features)

        return value

        # Enhanced training function with experience replay and curriculum learning

def train_agent_with_scu(
        num_episodes=1000,
        gamma=0.95,
        lr_actor=5e-5,
        lr_critic=1e-5,
        print_interval=10,
        save_interval=10,
        checkpoint_path="scu_gnn_model_checkpoint.pt",
        batch_size=32,
        replay_buffer_size=1000,
        weight_decay=1e-4,
        grid_category_rotation=True
):
    """
    Train the GNN-based DRL agent using SCU manifests and grid categories

    Args:
        num_episodes: Number of episodes to train for
        gamma: Discount factor
        lr_actor: Learning rate for actor network
        lr_critic: Learning rate for critic network
        print_interval: How often to print training progress
        save_interval: How often to save model checkpoints
        checkpoint_path: Path to save checkpoints
        batch_size: Batch size for updates from replay buffer
        replay_buffer_size: Capacity of experience replay buffer
        weight_decay: L2 regularization strength
        grid_category_rotation: Whether to rotate between grid categories during training

    Returns:
        Trained actor and critic networks, and training statistics
    """
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

    # Initialize networks
    # Use a medium-sized grid to determine feature dimensions
    temp_env = DRLBinPackingEnv(container_dims=(8, 8, 8))
    test_state = temp_env.reset()
    node_feature_dim = test_state.x.size(1)
    del temp_env

    actor = ActorGNN(node_feature_dim=node_feature_dim).to(device)
    critic = CriticGNN(node_feature_dim=node_feature_dim).to(device)

    # Optimizers
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    # Learning rate schedulers
    scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(optimizer_actor, mode='max', factor=0.5, patience=50)
    scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_critic, mode='max', factor=0.5, patience=50)

    # Training metrics
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    actor_losses = []
    critic_losses = []
    success_rates = deque(maxlen=100)
    category_performance = {cat: deque(maxlen=50) for cat in ["small", "medium", "large"]}

    # Curriculum learning: Start with easier scenarios
    curriculum_phases = [
        {"episodes": num_episodes * 0.2, "categories": ["small"], "difficulties": ["very-easy", "easy"]},
        {"episodes": num_episodes * 0.3, "categories": ["small", "medium"], "difficulties": ["easy", "medium-low"]},
        {"episodes": num_episodes * 0.3, "categories": ["medium", "large"], "difficulties": ["medium", "medium-high"]},
        {"episodes": num_episodes * 0.2, "categories": ["small", "medium", "large"],
         "difficulties": ["hard", "very-hard"]}
    ]

    phase_idx = 0
    phase_episode_count = 0

    # Training loop
    for episode in range(num_episodes):
        # Determine current curriculum phase
        if phase_idx < len(curriculum_phases) - 1:
            if phase_episode_count >= curriculum_phases[phase_idx]["episodes"]:
                phase_idx += 1
                phase_episode_count = 0
                print(f"\n=== Moving to curriculum phase {phase_idx + 1} ===")
        phase_episode_count += 1

        current_phase = curriculum_phases[phase_idx]

        # Select grid category and size
        category = random.choice(current_phase["categories"])
        grid_dims = random.choice(GRID_CATEGORIES[category]["grids"])
        difficulty = random.choice(current_phase["difficulties"])

        # Create environment with selected grid
        env = DRLBinPackingEnv(container_dims=grid_dims)

        # Generate SCU manifest
        fill_ratio = random.uniform(0.6, 0.9)
        priority_groups = random.randint(2, 4)
        manifest = generate_scu_manifest(
            grid_dims=grid_dims,
            target_fill_ratio=fill_ratio,
            difficulty=difficulty,
            priority_groups=priority_groups
        )

        # Convert to item list format
        cargo_manifest = manifest_to_item_list(manifest)

        # Reset environment with SCU manifest
        state = env.reset(cargo_manifest=cargo_manifest)
        episode_reward = 0
        episode_transitions = []
        steps_in_episode = 0
        max_steps = 500

        # Episode loop
        while steps_in_episode < max_steps:
            steps_in_episode += 1

            # Get feasibility mask
            feasibility_mask = env.get_feasibility_mask()

            # Check if current item exists
            if env.current_item is None:
                break

            # Check if there are any feasible actions
            if np.sum(feasibility_mask) == 0:
                # No feasible placements - skip this item
                env._move_to_next_item()

                if env.current_item is None:
                    break

                state = env._build_graph_state()
                continue

            # Get action from actor
            try:
                with torch.no_grad():
                    action_dist = actor(state, feasibility_mask)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
            except Exception as e:
                print(f"Error in actor forward pass: {e}")
                break

            # Take step in environment
            try:
                next_state, reward, done, info = env.step(action.item())
            except Exception as e:
                print(f"Error in environment step: {e}")
                break

            episode_reward += reward

            # Store transition
            replay_buffer.push(
                state, action.item(), reward,
                next_state if not done else None,
                done, log_prob, feasibility_mask
            )

            episode_transitions.append((
                state, action, reward,
                next_state if not done else None,
                log_prob, done, feasibility_mask
            ))

            # Update state
            state = next_state

            if done:
                break

        # Skip update if episode was too short
        if len(episode_transitions) < 2:
            continue

        # Perform batch updates from replay buffer
        if len(replay_buffer) >= batch_size:
            # Sample batch of transitions
            states, actions, rewards, next_states, dones, _, _ = replay_buffer.sample(batch_size)

            # Process states in batches
            actor_loss_total = 0
            critic_loss_total = 0

            for i in range(len(states)):
                state = states[i]
                action_from_buffer = actions[i]
                reward = rewards[i].item()
                next_state = next_states[i]
                done = dones[i].item()

                try:
                    # Calculate current state's value and target value
                    current_value_q = critic(state).squeeze()
                    current_device = current_value_q.device

                    with torch.no_grad():
                        if next_state is not None:
                            next_value_q = critic(next_state).squeeze()
                            reward_t = torch.tensor(reward, dtype=torch.float, device=current_device)
                            done_t = torch.tensor(float(done), dtype=torch.float, device=current_device)
                            target_q_value = reward_t + gamma * next_value_q * (1.0 - done_t)
                        else:
                            target_q_value = torch.tensor(reward, dtype=torch.float, device=current_device)

                    # Update Critic
                    critic_loss = F.mse_loss(current_value_q, target_q_value)

                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    optimizer_critic.step()

                    # Calculate Advantage
                    with torch.no_grad():
                        advantage = target_q_value - current_value_q

                    # Actor Update
                    action_dist = actor(state, None)
                    action_tensor = torch.tensor(action_from_buffer.item() if isinstance(action_from_buffer,
                                                                                         torch.Tensor) else action_from_buffer,
                                                 device=current_device)

                    log_prob = action_dist.log_prob(action_tensor)
                    actor_loss = -log_prob * advantage.detach()

                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    optimizer_actor.step()

                    actor_loss_total += actor_loss.item()
                    critic_loss_total += critic_loss.item()

                except Exception as e:
                    print(f"ERROR during replay buffer update: {e}")
                    continue

            # Average losses over batch
            if len(states) > 0:
                actor_loss_avg = actor_loss_total / len(states)
                critic_loss_avg = critic_loss_total / len(states)
            else:
                actor_loss_avg = 0
                critic_loss_avg = 0

        # Track metrics
        episode_rewards.append(float(episode_reward))
        avg_rewards.append(float(episode_reward))

        # Calculate success rate
        success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
        success_rates.append(success_rate)
        category_performance[category].append(success_rate)

        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(list(avg_rewards))
            avg_success = np.mean(list(success_rates))

            print(f"\nEpisode {episode + 1}/{num_episodes} | Phase {phase_idx + 1}")
            print(f"Grid: {grid_dims} ({category}) | Difficulty: {difficulty}")
            print(f"Avg Reward: {avg_reward:.2f} | Avg Success Rate: {avg_success:.2f}")
            print(f"Items: {env.successful_placements}/{env.total_items} | Steps: {steps_in_episode}")

            # Print category-specific performance
            print("Category Performance:")
            for cat in category_performance:
                if len(category_performance[cat]) > 0:
                    cat_success = np.mean(list(category_performance[cat]))
                    print(f"  {cat}: {cat_success:.2f}")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            torch.save({
                'episode': episode,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                'category_performance': dict(category_performance)
            }, checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")

    return actor, critic, {
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'category_performance': dict(category_performance)
    }


def evaluate_agent_with_scu(actor, num_episodes=10, category="medium", visualize=False):
    """
    Evaluate the trained GNN agent with SCU manifests

    Args:
        actor: Trained actor network
        num_episodes: Number of episodes to evaluate
        category: Grid category to evaluate on
        visualize: Whether to visualize the final packing

    Returns:
        Average reward, success rate, and volume utilization
    """
    episode_rewards = []
    success_rates = []
    volume_utilizations = []

    for episode in range(num_episodes):
        # Select random grid from category
        grid_dims = random.choice(GRID_CATEGORIES[category]["grids"])
        env = DRLBinPackingEnv(container_dims=grid_dims)

        # Generate SCU manifest
        manifest = generate_scu_manifest(
            grid_dims=grid_dims,
            target_fill_ratio=0.8,
            difficulty="hard",
            priority_groups=3
        )
        cargo_manifest = manifest_to_item_list(manifest)

        # Print manifest summary for first episode
        if episode == 0:
            print_manifest_summary(manifest, grid_dims)

        state = env.reset(cargo_manifest=cargo_manifest)
        episode_reward = 0
        total_item_volume = 0
        container_volume = torch.prod(env.container_dims).item()

        # Episode loop
        steps = 0
        max_steps = 500

        while steps < max_steps:
            steps += 1

            # Check if current item exists
            if env.current_item is None:
                break

            # Get feasibility mask
            feasibility_mask = env.get_feasibility_mask()

            # Check if there are any feasible actions
            if np.sum(feasibility_mask) == 0:
                # No feasible placements, move to next item
                env._move_to_next_item()

                if env.current_item is None:
                    break

                state = env._build_graph_state()
                continue

            # Get action from actor (greedy)
            with torch.no_grad():
                try:
                    action_dist = actor(state, feasibility_mask)
                    action = torch.argmax(action_dist.probs)
                except Exception as e:
                    print(f"Error in actor forward pass during evaluation: {e}")
                    break

            # Take step in environment
            try:
                next_state, reward, done, info = env.step(action.item())

                # Convert reward to scalar if it's a tensor
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()

                episode_reward += reward

                # Track volume if successful placement
                if info.get("feasible", False) and env.current_item_idx > 0:
                    prev_item_idx = env.current_item_idx - 1
                    if prev_item_idx < len(cargo_manifest):
                        prev_item = cargo_manifest[prev_item_idx]
                        total_item_volume += prev_item[0] * prev_item[1] * prev_item[2]

                # Update state
                state = next_state

                if done:
                    break

            except Exception as e:
                print(f"Error during environment step in evaluation: {e}")
                break

        # Calculate metrics
        success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
        volume_utilization = total_item_volume / container_volume

        # Ensure we're appending scalar values
        episode_rewards.append(float(episode_reward))
        success_rates.append(float(success_rate))
        volume_utilizations.append(float(volume_utilization))

        # Visualize first episode if requested
        if visualize and episode == 0:
            try:
                visualize_packing(env)
            except Exception as e:
                print(f"Visualization error: {e}")

    # Calculate averages
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_success_rate = np.mean(success_rates) if success_rates else 0
    avg_volume_utilization = np.mean(volume_utilizations) if volume_utilizations else 0

    print(f"\nEvaluation Results ({num_episodes} episodes on {category} grids):")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Success Rate: {avg_success_rate:.2f}")
    print(f"Average Volume Utilization: {avg_volume_utilization:.2f}")

    return avg_reward, avg_success_rate, avg_volume_utilization


# Also create a function to load a saved model for inference
def load_trained_model(checkpoint_path="scu_gnn_model_checkpoint.pt", device=None):
    """
    Load a trained model from checkpoint

    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device to load model on (defaults to cuda if available)

    Returns:
        actor: Loaded actor network
        checkpoint: Full checkpoint data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize actor network
    # We need to determine the feature dimension - use a small test environment
    temp_env = DRLBinPackingEnv(container_dims=(4, 4, 4))
    test_state = temp_env.reset()
    node_feature_dim = test_state.x.size(1)
    del temp_env

    # Create and load actor
    actor = ActorGNN(node_feature_dim=node_feature_dim).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # Set to evaluation mode

    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['episode']} episodes")

    return actor, checkpoint


# Function for single inference (for API use)
def pack_single_manifest(actor, grid_dims, manifest, device=None):
    """
    Pack a single manifest using the trained model

    Args:
        actor: Trained actor network
        grid_dims: Tuple of (width, length, height) for the grid
        manifest: List of dictionaries with SCU manifest data
        device: Device to run on

    Returns:
        Dictionary with packing results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = DRLBinPackingEnv(container_dims=grid_dims)

    # Convert manifest to item list
    cargo_manifest = manifest_to_item_list(manifest)

    # Reset environment with manifest
    state = env.reset(cargo_manifest=cargo_manifest)

    # Run packing
    steps = 0
    max_steps = 500

    while steps < max_steps and env.current_item is not None:
        steps += 1

        # Get feasibility mask
        feasibility_mask = env.get_feasibility_mask()

        # Check if there are any feasible actions
        if np.sum(feasibility_mask) == 0:
            env._move_to_next_item()
            if env.current_item is None:
                break
            state = env._build_graph_state()
            continue

        # Get action from actor (greedy)
        with torch.no_grad():
            action_dist = actor(state, feasibility_mask)
            action = torch.argmax(action_dist.probs)

        # Take step
        next_state, reward, done, info = env.step(action.item())
        state = next_state

        if done:
            break

    # Prepare results
    placements = []
    for i, placed in enumerate(env.placed_items):
        box = placed['box']

        # Find which SCU type this corresponds to
        dims = [box.width.item(), box.length.item(), box.height.item()]
        scu_type = None
        for scu, scu_info in SCU_DEFINITIONS.items():
            if dims == scu_info["dimensions"]:
                scu_type = scu
                break

        placement = {
            "item_id": f"{scu_type}_{i + 1:03d}",
            "scu_type": scu_type or "Unknown",
            "position": [box.x1.item(), box.y1.item(), box.z1.item()],
            "dimensions": dims,
            "priority": placed['priority']
        }
        placements.append(placement)

    # Calculate metrics
    success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
    total_volume = sum(p['box'].volume.item() for p in env.placed_items)
    container_volume = torch.prod(env.container_dims).item()
    utilization = total_volume / container_volume

    # Identify unplaced items
    unplaced_items = []
    if env.successful_placements < env.total_items:
        # Track which items were not placed
        placed_count = env.successful_placements
        for i in range(placed_count, env.total_items):
            if i < len(cargo_manifest):
                item = cargo_manifest[i]
                # Find SCU type
                for scu, scu_info in SCU_DEFINITIONS.items():
                    if list(item[:3]) == scu_info["dimensions"]:
                        unplaced_items.append({
                            "scu_type": scu,
                            "priority": item[4]
                        })
                        break

    result = {
        "success": True,
        "grid_dimensions": list(grid_dims),
        "placements": placements,
        "unplaced_items": unplaced_items,
        "metrics": {
            "success_rate": float(success_rate),
            "volume_utilization": float(utilization),
            "items_placed": env.successful_placements,
            "total_items": env.total_items
        }
    }

    return result

# Main execution
if __name__ == "__main__":
    # Train agent with SCU manifests
    print("Starting training with SCU manifests...")
    actor, critic, stats = train_agent_with_scu(
        num_episodes=1000,
        gamma=0.95,
        lr_actor=5e-5,
        lr_critic=1e-5,
        print_interval=10,
        save_interval=50,
        batch_size=32,
        replay_buffer_size=1000,
        weight_decay=1e-4
    )


    # Evaluate on different grid categories
    print("\n=== Evaluating on different grid categories ===")
    for category in ["small", "medium", "large"]:
        print(f"\nEvaluating on {category} grids:")
        evaluate_agent_with_scu(actor, num_episodes=10, category=category, visualize=(category == "medium"))

    print("\nTraining complete!")