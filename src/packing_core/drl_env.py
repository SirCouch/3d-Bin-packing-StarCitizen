import torch
import numpy as np
from torch_geometric.data import Data
from scu_manifest_generator import generate_scu_manifest, manifest_to_item_list, SCU_DEFINITIONS
from .box3d import Box3D
from .mer_manager import MERManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRLBinPackingEnv:
    def __init__(self, grids_list=[((10, 10, 10), "Main")], max_stack_weight=100000.0):
        self.grids_list = grids_list
        self.grids = [{'dims': torch.tensor(g[0], dtype=torch.float, device=device), 'name': g[1]} for g in grids_list]
        self.max_stack_weight = max_stack_weight

        self.NODE_TYPE_CONTAINER = 0
        self.NODE_TYPE_PLACED = 1
        self.NODE_TYPE_ITEM = 2
        self.NODE_TYPE_MER = 3

        self.placed_items = []
        self.mer_managers = [MERManager(g['dims']) for g in self.grids]
        self._cached_mers = None  # #5: invalidated on placement
        self._grid_placed_vol = [0.0] * len(self.grids)  # #3: precomputed
        self._grid_placed_count = [0] * len(self.grids)   # #3: precomputed

        self.cargo_manifest = []
        self.current_item_idx = 0
        self.current_item = None
        self.successful_placements = 0
        self.total_items = 0

    def reset(self, cargo_manifest=None, difficulty=None):
        self.steps_in_episode = 0
        self.placed_items = []
        self.mer_managers = [MERManager(g['dims']) for g in self.grids]
        self._cached_mers = None
        self._grid_placed_vol = [0.0] * len(self.grids)
        self._grid_placed_count = [0] * len(self.grids)

        # Combined volume for target SCU calculation
        total_vol = int(sum(torch.prod(g['dims']).item() for g in self.grids))
        dummy_dims = (total_vol, 1, 1)
        # Actual grid dimensions for physical fit filtering
        actual_grids = [tuple(int(d) for d in g['dims'].tolist()) for g in self.grids]

        if cargo_manifest is None:
            diff = difficulty if difficulty is not None else "medium"
            manifest = generate_scu_manifest(
                grid_dims=dummy_dims, grids_list=actual_grids,
                target_fill_ratio=0.7, difficulty=diff, priority_groups=3)
            self.cargo_manifest = manifest_to_item_list(manifest)
        else:
            self.cargo_manifest = cargo_manifest

        self.current_item_idx = 0
        self.successful_placements = 0
        self.total_items = len(self.cargo_manifest)

        if self.cargo_manifest:
            self.current_item = self._get_item_features(self.cargo_manifest[0])
        else:
            self.current_item = None

        return self._build_graph_state()

    def _get_item_features(self, item_tuple):
        w, l, h, weight, priority = item_tuple
        dims = sorted([w, l, h], reverse=True)
        return {
            'dimensions': torch.tensor(dims, dtype=torch.float),
            'weight': weight,
            'priority': priority
        }

    def _get_all_mers(self):
        """Returns list of dicts with grid_idx, mer_idx, and the Box3D object. Cached until placement."""
        if self._cached_mers is not None:
            return self._cached_mers
        all_mers = []
        for grid_idx, manager in enumerate(self.mer_managers):
            for mer_idx, mer_box in enumerate(manager.mers):
                all_mers.append({'grid_idx': grid_idx, 'mer_idx': mer_idx, 'box': mer_box})
        self._cached_mers = all_mers
        return all_mers

    def step(self, action):
        MAX_STEPS_PER_EPISODE = 1000
        if not hasattr(self, 'steps_in_episode'):
            self.steps_in_episode = 0

        self.steps_in_episode += 1
        if self.steps_in_episode >= MAX_STEPS_PER_EPISODE:
            return self._build_graph_state(), -10, True, {"error": "Episode timeout"}

        if self.current_item is None:
            return self._build_graph_state(), 0, True, {"error": "No item to place"}

        all_mers = self._get_all_mers()
        if len(all_mers) == 0:
            return self._build_graph_state(), -10, True, {"error": "No MERs available"}

        mer_action = action // 2
        rot_action = action % 2

        if mer_action >= len(all_mers):
            self._move_to_next_item()
            done = self.current_item is None
            return self._build_graph_state(), -5, done, {"error": "Invalid MER index"}

        selected = all_mers[mer_action]
        grid_idx = selected['grid_idx']
        mer_idx = selected['mer_idx']
        manager = self.mer_managers[grid_idx]
        grid_dims = self.grids[grid_idx]['dims']

        selected_mer = selected['box']
        base_dims = self.current_item['dimensions']
        if rot_action == 0:
            item_dims = base_dims
        else:
            item_dims = torch.tensor([base_dims[1], base_dims[0], base_dims[2]], dtype=torch.float)

        if not selected_mer.can_fit(item_dims):
            self._move_to_next_item()
            return self._build_graph_state(), -5.0, self.current_item is None, {"feasible": False}

        placement_position = selected_mer.position.clone()

        feasible = self._check_additional_constraints(placement_position, item_dims, self.current_item['weight'], self.current_item['priority'], grid_idx, grid_dims)

        if not feasible:
            self._move_to_next_item()
            return self._build_graph_state(), -5.0, self.current_item is None, {"feasible": False}

        placed_item = Box3D(placement_position, item_dims)
        self.placed_items.append({
            'box': placed_item,
            'weight': self.current_item['weight'],
            'priority': self.current_item['priority'],
            'grid_idx': grid_idx
        })
        self.successful_placements += 1
        self._grid_placed_vol[grid_idx] += float(placed_item.volume.item())
        self._grid_placed_count[grid_idx] += 1
        self._cached_mers = None  # invalidate MER cache

        manager.update(placed_item)

        reward = self._calculate_reward(placement_position, item_dims, grid_idx, grid_dims)

        self._move_to_next_item()
        done = self.current_item is None

        if done and self.successful_placements == self.total_items:
            # Scale bonus by manifest size, capped to prevent overwhelming per-step rewards
            reward += min(5.0 * self.total_items, 100.0)

        return self._build_graph_state(), reward, done, {"feasible": True}

    def _move_to_next_item(self):
        self.current_item_idx += 1
        if self.current_item_idx < len(self.cargo_manifest):
            self.current_item = self._get_item_features(self.cargo_manifest[self.current_item_idx])
        else:
            self.current_item = None

    def _check_additional_constraints(self, position, dimensions, weight, priority, grid_idx, grid_dims):
        item_box = Box3D(position, dimensions)
        container_box = Box3D(torch.zeros(3), grid_dims)
        
        if not container_box.contains(item_box):
            return False

        grid_items = [p for p in self.placed_items if p['grid_idx'] == grid_idx]
        for placed in grid_items:
            if placed['box'].overlaps(item_box):
                return False

        if not self._check_stacking_weight(item_box, weight, grid_items):
            return False

        if not self._check_priority_constraint(item_box, priority, grid_items):
            return False

        return True

    def _check_stacking_weight(self, item_box, item_weight, grid_items):
        for placed in grid_items:
            placed_box = placed['box']
            # Check items below the current placement (placed item's top <= our bottom)
            if (placed_box.z2 <= item_box.z1 and
                    placed_box.x1 < item_box.x2 and placed_box.x2 > item_box.x1 and
                    placed_box.y1 < item_box.y2 and placed_box.y2 > item_box.y1):
                if placed['weight'] + item_weight > self.max_stack_weight:
                    return False
                # Heavier items cannot rest on lighter items
                if item_weight > placed['weight']:
                    return False
        return True

    def _check_priority_constraint(self, item_box, item_priority, grid_items):
        for placed in grid_items:
            placed_box = placed['box']
            placed_priority = placed['priority']
            # Higher-priority items (lower number) must be closer to the front (lower Y)
            # Items at the same Y position are allowed regardless of priority
            if item_priority < placed_priority and item_box.y1 > placed_box.y1:
                return False
            if item_priority > placed_priority and item_box.y1 < placed_box.y1:
                return False
        return True

    def _calculate_reward(self, position, dimensions, grid_idx, grid_dims):
        item_volume = torch.prod(dimensions)
        # #1: Normalize reward per-grid volume, not total ship volume.
        # This keeps the reward signal strength consistent regardless of ship size.
        grid_volume = torch.prod(grid_dims)
        volume_ratio = item_volume / grid_volume
        reward = float(volume_ratio * 10.0)

        # Wall/floor touching bonus
        touching_reward = 0.0
        if position[0] < 0.001: touching_reward += 0.5
        if position[1] < 0.001: touching_reward += 0.5
        if position[2] < 0.001: touching_reward += 0.5
        if abs(position[0] + dimensions[0] - grid_dims[0]) < 0.001: touching_reward += 0.5
        if abs(position[1] + dimensions[1] - grid_dims[1]) < 0.001: touching_reward += 0.5
        if abs(position[2] + dimensions[2] - grid_dims[2]) < 0.001: touching_reward += 0.5

        # Adjacent item touching bonus
        item_box = Box3D(position, dimensions)
        grid_items = [p for p in self.placed_items if p['grid_idx'] == grid_idx]

        for placed in grid_items[:-1]:
            placed_box = placed['box']
            if (abs(item_box.x1 - placed_box.x2) < 0.001 or
                    abs(item_box.x2 - placed_box.x1) < 0.001 or
                    abs(item_box.y1 - placed_box.y2) < 0.001 or
                    abs(item_box.y2 - placed_box.y1) < 0.001 or
                    abs(item_box.z1 - placed_box.z2) < 0.001 or
                    abs(item_box.z2 - placed_box.z1) < 0.001):
                touching_reward += 0.5

            if abs(placed['priority'] - self.current_item['priority']) <= 1:
                touching_reward += 0.25

        reward += 2.0 + min(touching_reward, 2.0)

        # Grid balance reward — encourage even distribution across grids
        if len(self.grids) > 1:
            fill_ratios = [
                self._grid_placed_vol[gidx] / torch.prod(g['dims']).item()
                for gidx, g in enumerate(self.grids)
            ]
            balance = 1.0 - float(np.std(fill_ratios))
            reward += max(balance, 0.0) * 2.0

        return reward

    def _build_graph_state(self):
        node_features = []
        node_types = []
        node_positions = []

        container_indices = {}
        for idx, g in enumerate(self.grids):
            grid_vol = torch.prod(g['dims']).item()
            fill_ratio = self._grid_placed_vol[idx] / grid_vol if grid_vol > 0 else 0.0
            num_mers = len(self.mer_managers[idx].mers)
            container_features = [
                self.NODE_TYPE_CONTAINER,
                g['dims'][0], g['dims'][1], g['dims'][2],
                float(self._grid_placed_count[idx]),
                float(self.total_items),
                fill_ratio,
                float(num_mers),
            ]
            node_features.append(container_features)
            node_types.append(self.NODE_TYPE_CONTAINER)
            node_positions.append([0.0, 0.0, 0.0])
            container_indices[idx] = len(node_features) - 1

        if self.current_item is not None:
            item_features = [
                self.NODE_TYPE_ITEM,
                self.current_item['dimensions'][0], self.current_item['dimensions'][1], self.current_item['dimensions'][2],
                float(self.current_item['weight']), float(self.current_item['priority']), 0.0, 0.0
            ]
            node_features.append(item_features)
            node_types.append(self.NODE_TYPE_ITEM)
            node_positions.append([0.0, 0.0, 0.0])
            current_item_node_idx = len(node_features) - 1
        else:
            current_item_node_idx = -1

        placed_item_indices = {}
        for i, placed in enumerate(self.placed_items):
            placed_box = placed['box']
            placed_features = [
                self.NODE_TYPE_PLACED,
                placed_box.width, placed_box.length, placed_box.height,
                float(placed['weight']), float(placed['priority']), float(placed['grid_idx']), 0.0
            ]
            node_features.append(placed_features)
            node_types.append(self.NODE_TYPE_PLACED)
            node_positions.append([placed_box.x1, placed_box.y1, placed_box.z1])
            placed_item_indices[i] = len(node_features) - 1

        mer_indices = {}
        all_mers = self._get_all_mers()
        for i, mer_info in enumerate(all_mers):
            mer = mer_info['box']
            grid_vol = torch.prod(self.grids[mer_info['grid_idx']]['dims']).item()
            mer_fill_ratio = float(mer.volume.item() / grid_vol) if grid_vol > 0 else 0.0
            mer_features = [
                self.NODE_TYPE_MER,
                mer.width, mer.length, mer.height,
                mer.volume, float(mer_info['grid_idx']),
                mer_fill_ratio,         # #9: MER size relative to its grid
                0.0,
            ]
            node_features.append(mer_features)
            node_types.append(self.NODE_TYPE_MER)
            node_positions.append([mer.x1, mer.y1, mer.z1])
            mer_indices[i] = len(node_features) - 1

        if not node_features:
            x = torch.tensor([[self.NODE_TYPE_CONTAINER, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, node_type=torch.tensor([self.NODE_TYPE_CONTAINER], dtype=torch.long), pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float))

        x = torch.tensor(node_features, dtype=torch.float, device=device)
        node_type = torch.tensor(node_types, dtype=torch.long, device=device)
        pos = torch.tensor(node_positions, dtype=torch.float, device=device)

        edge_index_list = []

        for i, placed in enumerate(self.placed_items):
            edge_index_list.append((container_indices[placed['grid_idx']], placed_item_indices[i]))
            edge_index_list.append((placed_item_indices[i], container_indices[placed['grid_idx']]))
            
        for i, mer_info in enumerate(all_mers):
            edge_index_list.append((container_indices[mer_info['grid_idx']], mer_indices[i]))
            edge_index_list.append((mer_indices[i], container_indices[mer_info['grid_idx']]))

        if current_item_node_idx != -1:
            # #8: Connect current item to ALL container nodes (grid-level routing attention)
            for grid_idx, container_node_idx in container_indices.items():
                edge_index_list.append((current_item_node_idx, container_node_idx))
                edge_index_list.append((container_node_idx, current_item_node_idx))

            # Connect current item to feasible MER nodes
            for mer_idx, node_idx in mer_indices.items():
                mer_info = all_mers[mer_idx]
                feasible = self.mer_managers[mer_info['grid_idx']].get_feasible_mers(
                    self.current_item['dimensions'])
                if mer_info['mer_idx'] in feasible:
                    edge_index_list.append((current_item_node_idx, node_idx))
                    edge_index_list.append((node_idx, current_item_node_idx))

        # Connect placed items that share a face (touching/adjacent)
        # Uses face-adjacency instead of O(n²) distance thresholding
        eps = 0.001
        for i in range(len(self.placed_items)):
            if i not in placed_item_indices:
                continue
            pi = self.placed_items[i]
            box_i = pi['box']
            for j in range(i + 1, len(self.placed_items)):
                if j not in placed_item_indices:
                    continue
                pj = self.placed_items[j]
                if pi['grid_idx'] != pj['grid_idx']:
                    continue
                box_j = pj['box']
                # Check if boxes share a face (touching on any axis)
                x_overlap = (box_i.x1 < box_j.x2) and (box_i.x2 > box_j.x1)
                y_overlap = (box_i.y1 < box_j.y2) and (box_i.y2 > box_j.y1)
                z_overlap = (box_i.z1 < box_j.z2) and (box_i.z2 > box_j.z1)
                x_touch = abs(box_i.x1 - box_j.x2) < eps or abs(box_i.x2 - box_j.x1) < eps
                y_touch = abs(box_i.y1 - box_j.y2) < eps or abs(box_i.y2 - box_j.y1) < eps
                z_touch = abs(box_i.z1 - box_j.z2) < eps or abs(box_i.z2 - box_j.z1) < eps
                if ((x_touch and y_overlap and z_overlap) or
                        (y_touch and x_overlap and z_overlap) or
                        (z_touch and x_overlap and y_overlap)):
                    edge_index_list.append((placed_item_indices[i], placed_item_indices[j]))
                    edge_index_list.append((placed_item_indices[j], placed_item_indices[i]))

        if not edge_index_list:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=device).t().contiguous()

        graph_data = Data(x=x, edge_index=edge_index, node_type=node_type, pos=pos)
        graph_data.mer_node_indices = mer_indices
        graph_data.current_item_node_idx = current_item_node_idx
        return graph_data

    def get_feasibility_mask(self):
        all_mers = self._get_all_mers()
        if self.current_item is None or len(all_mers) == 0:
            return np.zeros(max(1, len(all_mers) * 2), dtype=np.float32)

        mask = np.zeros(len(all_mers) * 2, dtype=np.float32)
        item_dims = self.current_item['dimensions']
        item_weight = self.current_item['weight']
        item_priority = self.current_item['priority']

        orientations = [
            item_dims,
            torch.tensor([item_dims[1], item_dims[0], item_dims[2]], dtype=torch.float)
        ]

        # #4: Precompute per-grid placed items once (avoids refiltering per MER)
        grid_items_cache = {}
        for action_idx, mer_info in enumerate(all_mers):
            gidx = mer_info['grid_idx']
            mer_box = mer_info['box']
            mer_pos = mer_box.position
            grid_dims = self.grids[gidx]['dims']

            for rot_idx, rot_dims in enumerate(orientations):
                if not mer_box.can_fit(rot_dims):
                    continue
                # Lazy-compute grid items
                if gidx not in grid_items_cache:
                    grid_items_cache[gidx] = [p for p in self.placed_items if p['grid_idx'] == gidx]
                if self._check_constraints_fast(mer_pos, rot_dims, item_weight, item_priority, gidx, grid_dims, grid_items_cache[gidx]):
                    mask[action_idx * 2 + rot_idx] = 1.0

        return mask

    def _check_constraints_fast(self, position, dimensions, weight, priority, grid_idx, grid_dims, grid_items):
        """Like _check_additional_constraints but takes precomputed grid_items."""
        item_box = Box3D(position, dimensions)
        container_box = Box3D(torch.zeros(3), grid_dims)

        if not container_box.contains(item_box):
            return False
        for placed in grid_items:
            if placed['box'].overlaps(item_box):
                return False
        if not self._check_stacking_weight(item_box, weight, grid_items):
            return False
        if not self._check_priority_constraint(item_box, priority, grid_items):
            return False
        return True

        # GNN-based Actor-Critic Networks
