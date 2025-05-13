import gym
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
from cargo_manifest_generator import sample_cargo_manifest

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
        """Filter out redundant MERs (contained within others or too small)"""
        # Improved filtering method: use a minimum volume threshold
        min_vol_threshold = 0.01 * self.container.volume

        # Convert dicts to Box3D for easier manipulation
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
                    # If more than 90% contained, consider redundant
                    intersection = box_i.get_intersection(box_j)
                    if intersection and intersection.volume > 0.9 * box_i.volume:
                        is_maximal = False
                        break

            if is_maximal:
                non_redundant_indices.append(i)

        # Return filtered MERs as dictionaries
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
        """
        Reset the environment with an optional cargo manifest and difficulty level

        Args:
            cargo_manifest: Optional predefined cargo manifest
            difficulty: Optional difficulty level (controls number of items)

        Returns:
            Initial state as PyG Data object
        """
        # Reset state components
        self.placed_items = []
        self.mer_manager = MERManager(self.container_dims)

        # Generate cargo manifest based on difficulty if not provided
        if cargo_manifest is None:
            if difficulty is not None:
                # Simple curriculum: control number of items based on difficulty
                if difficulty == "very-easy":
                    num_items = np.random.randint(1, 3)
                elif difficulty == "easy":
                    num_items = np.random.randint(3, 6)
                elif difficulty == "medium-low":
                    num_items = np.random.randint(5, 8)
                elif difficulty == "medium-high":
                    num_items = np.random.randint(7, 12)
                elif difficulty == "hard":
                    num_items = np.random.randint(10, 15)
                else:  # "very-hard"
                    num_items = np.random.randint(12, 20)

                # Generate manifest with controlled size
                self.cargo_manifest = sample_cargo_manifest(
                    min_items=num_items,
                    max_items=num_items
                )
            else:
                # Default behavior
                self.cargo_manifest = sample_cargo_manifest()
        else:
            self.cargo_manifest = cargo_manifest

        # Reset counters
        self.current_item_idx = 0
        self.successful_placements = 0
        self.total_items = len(self.cargo_manifest)

        # Set first item as current
        if self.cargo_manifest:
            self.current_item = self._get_item_features(self.cargo_manifest[0])
        else:
            self.current_item = None

        # Build and return initial graph state
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
        if self.current_item is None:
            return self._build_graph_state(), 0, True, {"error": "No item to place"}

        # Check if action is valid
        if action >= len(self.mer_manager.mers):
            return self._build_graph_state(), -5, False, {"error": "Invalid MER index"}

        # Get selected MER and item dimensions
        selected_mer = Box3D(
            self.mer_manager.mers[action]['position'],
            self.mer_manager.mers[action]['dimensions']
        )
        item_dims = self.current_item['dimensions']

        # Check if placement is feasible
        if not selected_mer.can_fit(item_dims):
            reward = -5.0  # Reduced penalty for impossible placement
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), reward, done, {"feasible": False}

        # Place item at MER origin (lower corner)
        placement_position = selected_mer.position.clone()

        # Check additional constraints (access path, stacking, etc.)
        feasible = self._check_additional_constraints(
            placement_position,
            item_dims,
            self.current_item['weight'],
            self.current_item['priority']
        )

        if not feasible:
            reward = -5.0  # Penalty for constraint violation
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), reward, done, {"feasible": False}

        # Successful placement - create placed item
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
def train_agent(num_episodes=10000, gamma=0.95, lr_actor=1e-5, lr_critic=1e-5,
                print_interval=100, save_interval=10, checkpoint_path="enhanced_gnn_model_checkpoint.pt",
                batch_size=32, replay_buffer_size=10000, weight_decay=1e-4,
                possible_container_dims=None):
            """
            Train the GNN-based DRL agent using advanced techniques

            Args:
                env: DRLBinPackingEnv instance
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

            Returns:
                Trained actor and critic networks, and training statistics
            """
            if possible_container_dims is None:
                # Default to fixed size if none provided
                possible_container_dims = [(10, 10, 10)]

            # Initialize replay buffer
            replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

            # Initialize networks
            # Node feature dimension from environment (assumes consistent feature size)
            temp_env = DRLBinPackingEnv(container_dims=possible_container_dims[0])
            test_state = temp_env.reset()
            node_feature_dim = test_state.x.size(1)
            del temp_env

            actor = ActorGNN(node_feature_dim=node_feature_dim).to(device)
            critic = CriticGNN(node_feature_dim=node_feature_dim).to(device)

            # Optimizers with weight decay for L2 regularization
            optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
            optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

            # Learning rate schedulers
            scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(optimizer_actor, mode='max', factor=0.5,
                                                                   patience=50)
            scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(optimizer_critic, mode='max', factor=0.5,
                                                                    patience=50)

            # Training metrics
            episode_rewards = []
            avg_rewards = deque(maxlen=100)
            actor_losses = []
            critic_losses = []
            success_rates = deque(maxlen=100)

            # Training loop with curriculum learning
            for episode in range(num_episodes):
                # Determine difficulty based on episode number
                selected_dims = random.choice(possible_container_dims)
                env = DRLBinPackingEnv(container_dims=selected_dims)
                if episode < num_episodes * 0.2:  # First 20% - easy
                    difficulty = "Very Easy"
                elif episode < num_episodes * 0.3:  # Next 10% - easy
                    difficulty = "easy"
                elif episode < num_episodes * 0.4:  # Next 30% - medium
                    difficulty = "medium-low"
                elif episode < num_episodes * 0.5:  # Next 20% - medium
                    difficulty = "medium-high"
                elif episode < num_episodes * 0.8:  # Next 30% - medium
                    difficulty = "Hard"
                else:  # Last 50% - hard
                    difficulty = "very-hard"

                # Reset environment with appropriate difficulty
                state = env.reset(difficulty=difficulty)

                episode_reward = 0
                episode_transitions = []

                # Episode loop
                while True:
                    # Get feasibility mask for current item and MERs
                    feasibility_mask = env.get_feasibility_mask()

                    # Check if there are any feasible actions
                    if np.sum(feasibility_mask) == 0:
                        # No feasible placements, move to next item
                        _, reward, done, _ = env.step(0)  # Dummy action, will be rejected
                        episode_reward += reward

                        if done:
                            break

                        # Continue with next item
                        continue

                    # Get action from actor
                    with torch.no_grad():
                        action_dist = actor(state, feasibility_mask)
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action)

                    # Take step in environment
                    next_state, reward, done, _ = env.step(action.item())
                    episode_reward += reward

                    # Store transition in replay buffer
                    replay_buffer.push(
                        state, action.item(), reward,
                        next_state if not done else None,
                        done, log_prob, feasibility_mask
                    )

                    # Store transition for episode-based updates
                    episode_transitions.append((
                        state,
                        action,
                        reward,
                        next_state if not done else None,
                        log_prob,
                        done,
                        feasibility_mask
                    ))

                    # Update state
                    state = next_state

                    # Break if episode is done
                    if done:
                        break

                # Skip update if episode was too short
                if len(episode_transitions) < 2:
                    continue

                # Perform batch updates from replay buffer
                if len(replay_buffer) >= batch_size:
                    # Sample batch of transitions
                    states, actions, rewards, next_states, dones, _, _ = replay_buffer.sample(batch_size)

                    # Process states in batches (handle variable sized graphs)
                    # For simplicity in this implementation, we'll update one by one
                    actor_loss_total = 0
                    critic_loss_total = 0

                    for i in range(len(states)):
                        state = states[i]
                        action_from_buffer = actions[i]  # Renaming to avoid confusion with action_dist.sample()
                        reward = rewards[i].item()  # Get scalar
                        next_state = next_states[i]  # PyG Data or None
                        done = dones[i].item()  # Get scalar

                        # --- Calculate current state's value and target value ---
                        try:
                            # 1. Get value of current state
                            current_value_q = critic(state).squeeze()  # Calculate current state value, ensure scalar
                            current_device = current_value_q.device

                            # 2. Calculate target value (Q-target)
                            with torch.no_grad():
                                if next_state is not None:
                                    next_value_q = critic(next_state).squeeze()  # Ensure scalar
                                    reward_t = torch.tensor(reward, dtype=torch.float, device=current_device)
                                    done_t = torch.tensor(float(done), dtype=torch.float, device=current_device)
                                    target_q_value = reward_t + gamma * next_value_q * (1.0 - done_t)  # Scalar
                                else:  # Terminal state
                                    target_q_value = torch.tensor(reward, dtype=torch.float,
                                                                  device=current_device)  # Scalar

                            # --- Line 1217 (or close) context ---
                            # Update Critic:
                            # 'current_value_q' is the output of critic(state) for the loss
                            # 'target_q_value' is the target it should predict
                            critic_loss = F.mse_loss(current_value_q, target_q_value)  # Both should be scalar

                            optimizer_critic.zero_grad()
                            critic_loss.backward()
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                            optimizer_critic.step()

                            # --- Calculate Advantage (for Actor update) ---
                            with torch.no_grad():
                                advantage = target_q_value - current_value_q  # Both scalar

                            # --- Actor Update ---
                            # Re-evaluate action distribution for the current state for PPO/A2C (if log_prob from buffer is not used)
                            # If using log_prob from buffer, you'd typically use an importance sampling ratio.
                            # For simple A2C-style from replay:
                            action_dist = actor(state, None)  # Pass original feasibility_mask if stored & needed
                            # action_from_buffer is the action taken at that time
                            action_tensor = torch.tensor(action_from_buffer.item(),
                                                         device=current_device) if isinstance(action_from_buffer,
                                                                                              torch.Tensor) else torch.tensor(
                                action_from_buffer, device=current_device)

                            log_prob = action_dist.log_prob(action_tensor)
                            actor_loss = -log_prob * advantage.detach()  # advantage is scalar

                            optimizer_actor.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                            optimizer_actor.step()

                            actor_loss_total += actor_loss.item()
                            critic_loss_total += critic_loss.item()

                        except Exception as e:
                            print(f"ERROR during replay buffer update for transition {i}: {e}")
                            # Potentially skip this problematic transition or batch
                            # For now, let's just print and continue to avoid halting training
                            # but you'll lose this update.
                            # You might want to re-calculate actor_loss_avg and critic_loss_avg more carefully if continuing.
                            if len(states) > 0:  # Avoid division by zero if error on first item
                                actor_loss_avg = actor_loss_total / (i + 1) if (i + 1) <= len(
                                    states) else actor_loss_total / len(states)
                                critic_loss_avg = critic_loss_total / (i + 1) if (i + 1) <= len(
                                    states) else critic_loss_total / len(states)
                            else:
                                actor_loss_avg = 0
                                critic_loss_avg = 0
                            continue  # Skip to the next transition in the batch

                    # Average losses over batch (after the loop)
                    if len(states) > 0:  # Ensure states is not empty before division
                        actor_loss_avg = actor_loss_total / len(states)
                        critic_loss_avg = critic_loss_total / len(states)
                    else:
                        actor_loss_avg = 0
                        critic_loss_avg = 0
                else:
                    # If not enough samples in buffer, use episode-based updates
                    # Calculate returns and advantages
                    returns = []
                    advantages = []

                    # Compute critic values for all states
                    states = [t[0] for t in episode_transitions]
                    with torch.no_grad():
                        values = [critic(s).item() for s in states]

                    # Compute returns and advantages
                    G = 0
                    for i in reversed(range(len(episode_transitions))):
                        _, _, reward, _, _, done, _ = episode_transitions[i]
                        G = reward + gamma * G * (1 - int(done))
                        returns.insert(0, G)
                        advantages.insert(0, G - values[i])

                    returns = torch.tensor(returns, dtype=torch.float, device=device)
                    advantages = torch.tensor(advantages, dtype=torch.float, device=device)

                    # Normalize advantages
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Update actor and critic
                    actor_loss_total = 0
                    critic_loss_total = 0

                    for i, (state, action, _, _, log_prob, _, _) in enumerate(episode_transitions):
                        # Actor update
                        action_dist = actor(state, None)  # No feasibility mask during update
                        new_log_prob = action_dist.log_prob(action)
                        actor_loss = -new_log_prob * advantages[i]

                        optimizer_actor.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                        optimizer_actor.step()

                        # Critic update
                        value = critic(state).squeeze()
                        critic_loss = F.mse_loss(value, returns[i])

                        optimizer_critic.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                        optimizer_critic.step()

                        actor_loss_total += actor_loss.item()
                        critic_loss_total += critic_loss.item()

                    # Average losses over episode
                    if len(episode_transitions) > 0:
                        actor_loss_avg = actor_loss_total / len(episode_transitions)
                        critic_loss_avg = critic_loss_total / len(episode_transitions)
                    else:
                        actor_loss_avg = 0
                        critic_loss_avg = 0

                # Track metrics
                if isinstance(episode_reward, torch.Tensor):
                    episode_reward = episode_reward.detach().cpu().item()
                episode_rewards.append(episode_reward)
                avg_rewards.append(episode_reward)
                actor_losses.append(actor_loss_avg)
                critic_losses.append(critic_loss_avg)

                # Calculate success rate (successful placements / total items)
                success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
                success_rates.append(success_rate)

                # Update learning rate schedulers
                if episode % 10 == 0:
                    # Make sure we're working with CPU data when calling numpy
                    rewards_cpu = [r.cpu().item() if isinstance(r, torch.Tensor) else r for r in avg_rewards]
                    mean_reward = np.mean(rewards_cpu)
                    scheduler_actor.step(mean_reward)
                    scheduler_critic.step(mean_reward)

                # Print progress
                if (episode + 1) % print_interval == 0:
                    avg_reward = np.mean(list(avg_rewards))
                    avg_success = np.mean(list(success_rates))
                    print(f"Episode {episode + 1}/{num_episodes} | "
                          f"Dims: {selected_dims} | "
                          f"Difficulty: {difficulty} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Success Rate: {avg_success:.2f} | "
                          f"Actor Loss: {actor_loss_avg:.4f} | "
                          f"Critic Loss: {critic_loss_avg:.4f}")

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
                        'critic_losses': critic_losses
                    }, checkpoint_path)

            # Plot training curves
            plot_training_curves(episode_rewards, actor_losses, critic_losses)

            return actor, critic, {
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses
            }

def plot_training_curves(rewards, actor_losses, critic_losses):
            """Plot training metrics"""
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            plt.subplot(1, 3, 2)
            plt.plot(actor_losses)
            plt.title('Actor Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')

            plt.subplot(1, 3, 3)
            plt.plot(critic_losses)
            plt.title('Critic Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')

            plt.tight_layout()
            plt.savefig('enhanced_gnn_training_curves.png')
            plt.show()

def evaluate_agent(env, actor, num_episodes=10, visualize=False):
            """
            Evaluate the trained GNN agent

            Args:
                env: DRLBinPackingEnv instance
                actor: Trained actor network
                num_episodes: Number of episodes to evaluate
                visualize: Whether to visualize the final packing

            Returns:
                Average reward and success rate
            """
            episode_rewards = []
            success_rates = []
            volume_utilizations = []

            for episode in range(num_episodes):
                state = env.reset(difficulty="hard")  # Evaluate on hard difficulty
                episode_reward = 0
                total_item_volume = 0
                container_volume = torch.prod(env.container_dims).item()

                # Episode loop
                while True:
                    # Get feasibility mask
                    feasibility_mask = env.get_feasibility_mask()

                    # Check if there are any feasible actions
                    if np.sum(feasibility_mask) == 0:
                        # No feasible placements, move to next item
                        _, reward, done, _ = env.step(0)  # Dummy action
                        episode_reward += reward

                        if done:
                            break

                        # Continue with next item
                        continue

                    # Get action from actor (greedy)
                    with torch.no_grad():
                        action_dist = actor(state, feasibility_mask)
                        # Select action with highest probability
                        action = torch.argmax(action_dist.probs)

                    # Take step in environment
                    next_state, reward, done, _ = env.step(action.item())
                    episode_reward += reward

                    # If successful placement, add volume
                    if env.successful_placements > 0 and env.current_item_idx > 0:
                        prev_item = env.cargo_manifest[env.current_item_idx - 1]
                        w, l, h, _, _ = prev_item
                        total_item_volume += w * l * h

                    # Update state
                    state = next_state

                    # Break if episode is done
                    if done:
                        break

                # Calculate metrics
                success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
                volume_utilization = total_item_volume / container_volume

                episode_rewards.append(episode_reward)
                success_rates.append(success_rate)
                volume_utilizations.append(volume_utilization)

                # Optional: Visualize final packing
                if visualize and episode == 0:  # Visualize first episode
                    visualize_packing(env)

            avg_reward = np.mean(episode_rewards)
            avg_success_rate = np.mean(success_rates)
            avg_volume_utilization = np.mean(volume_utilizations)

            print(f"Evaluation over {num_episodes} episodes:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Success Rate: {avg_success_rate:.2f}")
            print(f"Average Volume Utilization: {avg_volume_utilization:.2f}")

            return avg_reward, avg_success_rate, avg_volume_utilization

def visualize_packing(env):
    """
    Visualize the 3D bin packing result
    Requires matplotlib and possibly mplot3d
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot container boundaries
        container_width, container_length, container_height = [x.item() for x in env.container_dims]

        # Define the vertices of the container
        vertices = [
            [0, 0, 0],
            [container_width, 0, 0],
            [container_width, container_length, 0],
            [0, container_length, 0],
            [0, 0, container_height],
            [container_width, 0, container_height],
            [container_width, container_length, container_height],
            [0, container_length, container_height]
        ]

        # Define the faces of the container
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]]  # Right
        ]

        # Plot container as wireframe
        for face in faces:
            face_array = np.array(face)
            ax.plot3D(face_array[:, 0], face_array[:, 1], face_array[:, 2], 'k-', alpha=0.2)

        # Plot placed items
        for i, placed in enumerate(env.placed_items):
            box = placed['box']
            priority = placed['priority']

            # Define the vertices of the box
            x, y, z = box.position
            w, l, h = box.dimensions

            box_vertices = [
                [x, y, z],
                [x + w, y, z],
                [x + w, y + l, z],
                [x, y + l, z],
                [x, y, z + h],
                [x + w, y, z + h],
                [x + w, y + l, z + h],
                [x, y + l, z + h]
            ]

        # Define the faces of the box
            box_faces = [
                [box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]],  # Bottom
                [box_vertices[4], box_vertices[5], box_vertices[6], box_vertices[7]],  # Top
                [box_vertices[0], box_vertices[1], box_vertices[5], box_vertices[4]],  # Front
                [box_vertices[2], box_vertices[3], box_vertices[7], box_vertices[6]],  # Back
                [box_vertices[0], box_vertices[3], box_vertices[7], box_vertices[4]],  # Left
                [box_vertices[1], box_vertices[2], box_vertices[6], box_vertices[5]]  # Right
            ]

        # Color based on priority (1-5)
            cmap = plt.cm.get_cmap('viridis')
            color = cmap((priority - 1) / 4)  # normalize to [0, 1]

            # Create a 3D collection of polygons
            box_collection = Poly3DCollection(box_faces, alpha=0.7)
            box_collection.set_facecolor(color)
            box_collection.set_edgecolor('black')

            # Add the collection to the plot
            ax.add_collection3d(box_collection)

            # Add text label in the center of the box
            box_center = [x + w / 2, y + l / 2, z + h / 2]
            ax.text(box_center[0], box_center[1], box_center[2], str(i + 1),
                    color='white', ha='center', va='center', fontsize=8)

        # Set labels and title
        ax.set_xlabel('Width (X)')
        ax.set_ylabel('Length (Y)')
        ax.set_zlabel('Height (Z)')
        ax.set_title('3D Bin Packing Result')

        # Set axis limits
        ax.set_xlim([0, container_width])
        ax.set_ylim([0, container_length])
        ax.set_zlim([0, container_height])

        # Add a colorbar to show priority mapping
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Unload Priority')

        # Add volume utilization information
        total_volume = 0
        container_volume = float(torch.prod(env.container_dims).item())
        for placed in env.placed_items:
            box = placed['box']
            total_volume += float(box.volume)

        utilization = total_volume / container_volume * 100
        plt.figtext(0.02, 0.02, f"Volume Utilization: {utilization:.1f}%", fontsize=10)
        plt.figtext(0.02, 0.05, f"Success Rate: {env.successful_placements}/{env.total_items}", fontsize=10)

        plt.tight_layout()
        plt.savefig('enhanced_3d_packing_visualization.png')
        plt.show()

    except ImportError as e:
        print(f"Visualization requires additional libraries: {e}")

        # Example usage
if __name__ == "__main__":
            # Create environment
    POSSIBLE_DIMS = [
        (10, 10, 10), (8, 8, 8), (12, 8, 6),
        (6, 6, 3), (5, 8, 3), (4, 6, 3), (2,4,3), (6,6,3), (1,2,2),(5,8,3),
        (1,2,2), (4,8,3), (4,6,3), (4,9,3), (3,2,3), (3,3,3), (3,3,3), (3,2,3),
        (3,2,3), (3,3,3), (3,3,3), (3,2,3), (5,7,2), (5,5,2), (5,7,2), (5,5,2), (2,4,1),
        (8,15,4), (6,9,4), (2,8,2), (2,8,2), (2,4,2), (2,4,2), (2,4,2), (2,4,2), (8,12,4), (2,3,1),
        (4,14,3), (3,7,2), (3,6,2), (4,5,2), (4,4,6), (4,4,6), (4,4,6), (4,4,6), (6,5,2), (3,7,2),
        (4,5,2), (1,1,2), (4,4,2), (8,6,2), (8,15,3), (6,9,3), (4,4,4), (4,4,4), (4,4,4), (4,4,4), (4,4,4), (4,4,4),
        (4,3,2), (4,3,2), (4,3,2), (5,7,5), (4,10,2), (4,8,3), (12,6,8), (12,6,8), (12,6,8), (12,6,8), (12,6,8),
        (12,6,8), (12,6,8), (12,6,8), (2,9,3), (2,4,1), (5,6,1), (6,18,2)# Examples
                # Add more relevant sizes
    ]

            # Train agent with enhanced techniques
    actor, critic, stats = train_agent(
                possible_container_dims=POSSIBLE_DIMS,
                num_episodes=10000,
                gamma=0.95,
                lr_actor=1e-5,
                lr_critic=1e-5,
                print_interval=100,
                save_interval=10,
                batch_size=32,
                replay_buffer_size=10000,
                weight_decay=1e-4
            )

    # Evaluate trained agent
    eval_env = DRLBinPackingEnv(container_dims=(10, 10, 10))  # Evaluate on a standard size
    evaluate_agent(eval_env, actor, num_episodes=10, visualize=True)

    eval_env_small = DRLBinPackingEnv(container_dims=(6, 6, 3))  # Evaluate on a small size
    evaluate_agent(eval_env_small, actor, num_episodes=10, visualize=False)