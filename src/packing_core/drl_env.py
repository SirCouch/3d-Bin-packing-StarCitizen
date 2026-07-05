import torch
import numpy as np
from torch_geometric.data import Data
try:
    from scu_manifest_generator import (
        BULK_ONLY_RUN_TYPE,
        SCU_DEFINITIONS,
        generate_scu_manifest,
        get_grid_blockers,
        get_grid_dimensions,
        is_bulk_only_manifest_target,
        manifest_to_item_list,
    )
except ModuleNotFoundError:
    from src.scu_manifest_generator import (
        BULK_ONLY_RUN_TYPE,
        SCU_DEFINITIONS,
        generate_scu_manifest,
        get_grid_blockers,
        get_grid_dimensions,
        is_bulk_only_manifest_target,
        manifest_to_item_list,
    )
from .box3d import Box3D
from .mer_manager import MERManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _as_float_triplet(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain 3 numbers.")
    triplet = tuple(float(x) for x in value)
    if any(x < 0 for x in triplet):
        raise ValueError(f"{label} values must be non-negative.")
    return triplet


def _boxes_overlap(a_pos, a_dims, b_pos, b_dims):
    ax, ay, az = a_pos
    aw, al, ah = a_dims
    bx, by, bz = b_pos
    bw, bl, bh = b_dims
    return not (
        ax + aw <= bx or bx + bw <= ax or
        ay + al <= by or by + bl <= ay or
        az + ah <= bz or bz + bh <= az
    )


class DRLBinPackingEnv:
    def __init__(
            self,
            grids_list=[((10, 10, 10), "Main")],
            max_stack_weight=100000.0,
            ship_name=None,
    ):
        self.grid_specs = [self._normalize_grid_spec(g, idx) for idx, g in enumerate(grids_list)]
        self.grids_list = [
            (spec["dims"], spec["name"], spec["blocked"]) if spec["blocked"]
            else (spec["dims"], spec["name"])
            for spec in self.grid_specs
        ]
        self.grids = [
            {
                'dims': torch.tensor(spec["dims"], dtype=torch.float, device=device),
                'name': spec["name"],
                'blocked': spec["blocked"],
            }
            for spec in self.grid_specs
        ]
        # Precomputed scalar grid volumes — avoids torch.prod().item() on the hot path.
        self.bounding_grid_volumes = [
            spec["dims"][0] * spec["dims"][1] * spec["dims"][2]
            for spec in self.grid_specs
        ]
        self.blocked_grid_volumes = [
            sum(b["dimensions"][0] * b["dimensions"][1] * b["dimensions"][2] for b in spec["blocked"])
            for spec in self.grid_specs
        ]
        self.grid_volumes = [
            max(0.0, bbox - blocked)
            for bbox, blocked in zip(self.bounding_grid_volumes, self.blocked_grid_volumes)
        ]
        self.total_ship_volume = sum(self.grid_volumes)
        self.max_stack_weight = max_stack_weight
        self.ship_name = ship_name

        self.NODE_TYPE_CONTAINER = 0
        self.NODE_TYPE_PLACED = 1
        self.NODE_TYPE_ITEM = 2
        self.NODE_TYPE_MER = 3
        self.NODE_TYPE_SHIP = 4

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

    def _normalize_grid_spec(self, grid, idx):
        dims = get_grid_dimensions(grid)
        if any(x <= 0 for x in dims):
            raise ValueError(f"Grid {idx} dimensions must be positive.")

        if isinstance(grid, dict):
            name = grid.get("name") or f"Grid {idx + 1}"
        elif (
            isinstance(grid, (list, tuple))
            and len(grid) >= 2
            and isinstance(grid[0], (list, tuple))
        ):
            name = grid[1]
        else:
            name = f"Grid {idx + 1}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"Grid {idx} name must be a non-empty string.")

        blockers = self._normalize_blockers(get_grid_blockers(grid), dims, name)
        return {"dims": dims, "name": name, "blocked": blockers}

    def _normalize_blockers(self, raw_blockers, grid_dims, grid_name):
        blockers = []
        gw, gl, gh = grid_dims
        for idx, raw in enumerate(raw_blockers):
            if isinstance(raw, dict):
                position = raw.get("position")
                dimensions = raw.get("dimensions")
                supports = raw.get(
                    "supports",
                    raw.get("support", raw.get("counts_as_support", True)),
                )
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                position = raw[0]
                dimensions = raw[1]
                supports = raw[2] if len(raw) >= 3 else True
            else:
                raise ValueError(f"Invalid blocker {idx} on grid '{grid_name}'.")

            pos = _as_float_triplet(position, f"Blocker {idx} position")
            dims = _as_float_triplet(dimensions, f"Blocker {idx} dimensions")
            if any(x <= 0 for x in dims):
                raise ValueError(f"Blocker {idx} dimensions must be positive.")
            if not isinstance(supports, bool):
                raise ValueError(f"Blocker {idx} supports must be a boolean.")

            x, y, z = pos
            w, l, h = dims
            if x + w > gw or y + l > gl or z + h > gh:
                raise ValueError(f"Blocker {idx} exceeds grid '{grid_name}' bounds.")
            for prior_idx, prior in enumerate(blockers):
                if _boxes_overlap(pos, dims, prior["position"], prior["dimensions"]):
                    raise ValueError(
                        f"Blocker {idx} overlaps blocker {prior_idx} on grid '{grid_name}'."
                    )
            blockers.append({
                "position": pos,
                "dimensions": dims,
                "supports": bool(supports),
            })
        return blockers

    def _placed_record(
            self, box, weight, priority, grid_idx, is_blocker=False, supports=True,
    ):
        return {
            'box': box,
            'weight': float(weight),
            'priority': int(priority),
            'grid_idx': grid_idx,
            'is_blocker': bool(is_blocker),
            'supports': bool(supports),
            'x1': box.x1, 'y1': box.y1, 'z1': box.z1,
            'x2': box.x2, 'y2': box.y2, 'z2': box.z2,
        }

    def _seed_blockers(self):
        for grid_idx, grid in enumerate(self.grids):
            manager = self.mer_managers[grid_idx]
            for blocker in grid['blocked']:
                box = Box3D(blocker["position"], blocker["dimensions"])
                self.placed_items.append(self._placed_record(
                    box=box,
                    weight=0.0,
                    priority=0,
                    grid_idx=grid_idx,
                    is_blocker=True,
                    supports=blocker["supports"],
                ))
                manager.update(box)
        self._cached_mers = None

    def reset(self, cargo_manifest=None, difficulty=None):
        self.steps_in_episode = 0
        self.placed_items = []
        self.mer_managers = [MERManager(g['dims']) for g in self.grids]
        self._cached_mers = None
        self._grid_placed_vol = [0.0] * len(self.grids)
        self._grid_placed_count = [0] * len(self.grids)
        self._seed_blockers()

        # Combined usable volume for target SCU calculation
        total_vol = int(self.total_ship_volume)
        dummy_dims = (total_vol, 1, 1)
        # Actual grid specs for physical fit filtering and usable-volume targeting
        actual_grids = self.grids_list

        if cargo_manifest is None:
            diff = difficulty if difficulty is not None else "medium"
            run_type = (
                BULK_ONLY_RUN_TYPE
                if is_bulk_only_manifest_target(actual_grids, ship_name=self.ship_name)
                else None
            )
            manifest = generate_scu_manifest(
                grid_dims=dummy_dims, grids_list=actual_grids,
                target_fill_ratio=0.7, difficulty=diff, run_type=run_type)
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
            grid_items = [p for p in self.placed_items if p['grid_idx'] == grid_idx]
            anchor = self._find_valid_anchor_in_mer(
                selected_mer, item_dims, self.current_item['weight'],
                self.current_item['priority'], grid_idx, grid_dims, grid_items,
            )
            if anchor is not None:
                placement_position = anchor
                feasible = True

        if not feasible:
            self._move_to_next_item()
            return self._build_graph_state(), -5.0, self.current_item is None, {"feasible": False}

        placed_item = Box3D(placement_position, item_dims)
        # Cache coords as plain Python floats to avoid CPU↔GPU sync per constraint check.
        px = float(placement_position[0].item() if hasattr(placement_position[0], 'item') else placement_position[0])
        py = float(placement_position[1].item() if hasattr(placement_position[1], 'item') else placement_position[1])
        pz = float(placement_position[2].item() if hasattr(placement_position[2], 'item') else placement_position[2])
        dw = float(item_dims[0].item()); dl = float(item_dims[1].item()); dh = float(item_dims[2].item())
        self.placed_items.append(self._placed_record(
            box=placed_item,
            weight=self.current_item['weight'],
            priority=self.current_item['priority'],
            grid_idx=grid_idx,
        ))
        self.successful_placements += 1
        self._grid_placed_vol[grid_idx] += placed_item.volume
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
        grid_items = [p for p in self.placed_items if p['grid_idx'] == grid_idx]
        return self._check_constraints_fast(
            position, dimensions, weight, priority, grid_idx, grid_dims, grid_items,
        )

    def _check_support(self, item_box, grid_items, min_ratio=0.99):
        """Require item to be fully supported (floor or items below) to prevent hover."""
        if item_box.z1 < 0.001:
            return True
        bottom_area = (item_box.x2 - item_box.x1) * (item_box.y2 - item_box.y1)
        if bottom_area <= 0:
            return True
        supported = 0.0
        z1 = item_box.z1
        for placed in grid_items:
            if placed.get('is_blocker') and not placed.get('supports', True):
                continue
            pb = placed['box']
            if abs(pb.z2 - z1) < 0.001:
                ox = max(0.0, min(item_box.x2, pb.x2) - max(item_box.x1, pb.x1))
                oy = max(0.0, min(item_box.y2, pb.y2) - max(item_box.y1, pb.y1))
                supported += ox * oy
        return (supported / bottom_area) >= min_ratio

    def _check_stacking_weight(self, item_box, item_weight, grid_items):
        for placed in grid_items:
            if placed.get('is_blocker'):
                continue
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
            if placed.get('is_blocker'):
                continue
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
        item_volume_value = float(item_volume.item() if hasattr(item_volume, 'item') else item_volume)
        # #1: Normalize reward per-grid volume, not total ship volume.
        # This keeps the reward signal strength consistent regardless of ship size.
        grid_volume = self.grid_volumes[grid_idx]
        volume_ratio = item_volume_value / grid_volume if grid_volume > 0 else 0.0
        reward = float(volume_ratio * 10.0)

        # Pull position/dims/grid_dims as floats once; no more tensor ops in the hot path.
        px = float(position[0].item() if hasattr(position[0], 'item') else position[0])
        py = float(position[1].item() if hasattr(position[1], 'item') else position[1])
        pz = float(position[2].item() if hasattr(position[2], 'item') else position[2])
        dw = float(dimensions[0].item() if hasattr(dimensions[0], 'item') else dimensions[0])
        dl = float(dimensions[1].item() if hasattr(dimensions[1], 'item') else dimensions[1])
        dh = float(dimensions[2].item() if hasattr(dimensions[2], 'item') else dimensions[2])
        gw = float(grid_dims[0].item()); gl = float(grid_dims[1].item()); gh = float(grid_dims[2].item())
        ix2, iy2, iz2 = px + dw, py + dl, pz + dh

        # Wall/floor touching bonus
        touching_reward = 0.0
        if px < 0.001: touching_reward += 0.5
        if py < 0.001: touching_reward += 0.5
        if pz < 0.001: touching_reward += 0.5
        if abs(ix2 - gw) < 0.001: touching_reward += 0.5
        if abs(iy2 - gl) < 0.001: touching_reward += 0.5
        if abs(iz2 - gh) < 0.001: touching_reward += 0.5

        # grid_items (cached floats) — exclude the just-placed item via [:-1]
        grid_items = [p for p in self.placed_items if p['grid_idx'] == grid_idx]
        item_box = Box3D(position, dimensions)

        # Priority clustering bonus
        item_priority = self.current_item['priority']
        same_prio_count = sum(
            1 for p in grid_items[:-1]
            if not p.get('is_blocker') and p['priority'] == item_priority
        )
        touching_reward += min(3.0, 0.4 * same_prio_count)

        # Support (stacking) bonus — uses cached floats
        bottom_area = dw * dl
        if bottom_area > 0:
            if pz < 0.001:
                supported_area = bottom_area
            else:
                supported_area = 0.0
                for p in grid_items[:-1]:
                    if p.get('is_blocker') and not p.get('supports', True):
                        continue
                    if abs(p['z2'] - pz) < 0.001:
                        ox = min(ix2, p['x2']) - max(px, p['x1'])
                        oy = min(iy2, p['y2']) - max(py, p['y1'])
                        if ox > 0 and oy > 0:
                            supported_area += ox * oy
            support_ratio = min(1.0, supported_area / bottom_area)
            touching_reward += 2.0 * support_ratio

        # Adjacent item touching bonus — pure Python coords
        for p in grid_items[:-1]:
            if p.get('is_blocker'):
                continue
            if (abs(px - p['x2']) < 0.001 or
                    abs(ix2 - p['x1']) < 0.001 or
                    abs(py - p['y2']) < 0.001 or
                    abs(iy2 - p['y1']) < 0.001 or
                    abs(pz - p['z2']) < 0.001 or
                    abs(iz2 - p['z1']) < 0.001):
                touching_reward += 0.5

            if abs(p['priority'] - self.current_item['priority']) <= 1:
                touching_reward += 0.25

        reward += 2.0 + min(touching_reward, 2.0)

        # Grid balance reward — encourage even distribution across grids
        reward += self._grid_balance_bonus()

        return reward

    def _grid_balance_bonus(self):
        """Encourage grid use without over-spreading low-grid-count ships."""
        grid_count = len(self.grids)
        if grid_count <= 1:
            return 0.0

        fill_ratios = [
            self._grid_placed_vol[gidx] / self.grid_volumes[gidx]
            if self.grid_volumes[gidx] > 0 else 0.0
            for gidx in range(grid_count)
        ]
        balance = max(1.0 - float(np.std(fill_ratios)), 0.0)
        return balance * self._grid_balance_weight()

    def _grid_balance_weight(self):
        grid_count = len(self.grids)
        if grid_count <= 1:
            return 0.0
        if grid_count == 2:
            return 0.75
        if grid_count == 3:
            return 1.25
        return 2.0

    def _build_graph_state(self):
        node_features = []
        node_types = []
        node_positions = []

        total_ship_volume = self.total_ship_volume

        # Add master SHIP node
        ship_features = [self.NODE_TYPE_SHIP] + [0.0] * 12
        node_features.append(ship_features)
        node_types.append(self.NODE_TYPE_SHIP)
        node_positions.append([0.0, 0.0, 0.0])
        ship_node_idx = len(node_features) - 1

        container_indices = {}
        for idx, g in enumerate(self.grids):
            grid_vol = self.grid_volumes[idx]
            fill_ratio = self._grid_placed_vol[idx] / grid_vol if grid_vol > 0 else 0.0
            num_mers = len(self.mer_managers[idx].mers)

            # Raw grid dims (cached at construction as python floats via grids_list)
            gw, gl, gh = self.grids_list[idx][0]
            aspect_w_l = float(gw / gl) if gl > 0 else 0.0
            aspect_w_h = float(gw / gh) if gh > 0 else 0.0
            aspect_l_h = float(gl / gh) if gh > 0 else 0.0
            relative_volume = (grid_vol / total_ship_volume) if total_ship_volume > 0 else 0.0
            positional_rank = float(idx)

            container_features = [
                self.NODE_TYPE_CONTAINER,
                float(gw), float(gl), float(gh),
                float(self._grid_placed_count[idx]),
                float(self.total_items),
                fill_ratio,
                float(num_mers),
                aspect_w_l, aspect_w_h, aspect_l_h, relative_volume, positional_rank
            ]
            node_features.append(container_features)
            node_types.append(self.NODE_TYPE_CONTAINER)
            node_positions.append([0.0, 0.0, 0.0])
            container_indices[idx] = len(node_features) - 1

        if self.current_item is not None:
            item_features = [
                self.NODE_TYPE_ITEM,
                self.current_item['dimensions'][0], self.current_item['dimensions'][1], self.current_item['dimensions'][2],
                float(self.current_item['weight']), float(self.current_item['priority']), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features.append(item_features)
            node_types.append(self.NODE_TYPE_ITEM)
            node_positions.append([0.0, 0.0, 0.0])
            current_item_node_idx = len(node_features) - 1
        else:
            current_item_node_idx = -1

        placed_item_indices = {}
        for i, placed in enumerate(self.placed_items):
            w = placed['x2'] - placed['x1']
            l = placed['y2'] - placed['y1']
            h = placed['z2'] - placed['z1']
            placed_features = [
                self.NODE_TYPE_PLACED,
                w, l, h,
                float(placed['weight']), float(placed['priority']), float(placed['grid_idx']), 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features.append(placed_features)
            node_types.append(self.NODE_TYPE_PLACED)
            node_positions.append([placed['x1'], placed['y1'], placed['z1']])
            placed_item_indices[i] = len(node_features) - 1

        mer_indices = {}
        all_mers = self._get_all_mers()
        # Pre-extract MER coords to plain floats once (avoids O(N) .item() calls per feature)
        for i, mer_info in enumerate(all_mers):
            mer = mer_info['box']
            mx1 = mer.x1; my1 = mer.y1; mz1 = mer.z1
            mw = mer.width; ml = mer.length; mh = mer.height
            mvol = mw * ml * mh
            grid_vol = self.grid_volumes[mer_info['grid_idx']]
            mer_fill_ratio = mvol / grid_vol if grid_vol > 0 else 0.0
            mer_features = [
                self.NODE_TYPE_MER,
                mw, ml, mh,
                mvol, float(mer_info['grid_idx']),
                mer_fill_ratio,         # #9: MER size relative to its grid
                0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features.append(mer_features)
            node_types.append(self.NODE_TYPE_MER)
            node_positions.append([mx1, my1, mz1])
            mer_indices[i] = len(node_features) - 1

        if not node_features:
            x = torch.tensor([[self.NODE_TYPE_CONTAINER] + [0.0]*12], dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, node_type=torch.tensor([self.NODE_TYPE_CONTAINER], dtype=torch.long), pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float))

        x = torch.tensor(node_features, dtype=torch.float, device=device)
        node_type = torch.tensor(node_types, dtype=torch.long, device=device)
        pos = torch.tensor(node_positions, dtype=torch.float, device=device)

        edge_index_list = []

        # Connect SHIP node to all CONTAINER nodes
        for c_idx in container_indices.values():
            edge_index_list.append((ship_node_idx, c_idx))
            edge_index_list.append((c_idx, ship_node_idx))

        # Connect all CONTAINER nodes to each other
        c_idx_list = list(container_indices.values())
        for i in range(len(c_idx_list)):
            for j in range(i + 1, len(c_idx_list)):
                edge_index_list.append((c_idx_list[i], c_idx_list[j]))
                edge_index_list.append((c_idx_list[j], c_idx_list[i]))

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

            # Connect current item to feasible MER nodes.
            # Cache per-grid feasibility once; `get_feasible_mers` was being called
            # per-MER-node and re-scanning the whole grid's MER list each time (O(N²)).
            feasibility_by_grid = {}
            for mer_idx, node_idx in mer_indices.items():
                mer_info = all_mers[mer_idx]
                gidx = mer_info['grid_idx']
                if gidx not in feasibility_by_grid:
                    feasibility_by_grid[gidx] = set(
                        self.mer_managers[gidx].get_feasible_mers(self.current_item['dimensions'])
                    )
                if mer_info['mer_idx'] in feasibility_by_grid[gidx]:
                    edge_index_list.append((current_item_node_idx, node_idx))
                    edge_index_list.append((node_idx, current_item_node_idx))

        # Connect placed items that share a face (touching/adjacent)
        # Uses cached float coords (no GPU sync). Group by grid first to skip cross-grid pairs.
        eps = 0.001
        by_grid = {}
        for i, p in enumerate(self.placed_items):
            by_grid.setdefault(p['grid_idx'], []).append(i)
        for idxs in by_grid.values():
            n = len(idxs)
            for ii in range(n):
                i = idxs[ii]
                pi = self.placed_items[i]
                ix1, iy1, iz1 = pi['x1'], pi['y1'], pi['z1']
                ix2, iy2, iz2 = pi['x2'], pi['y2'], pi['z2']
                for jj in range(ii + 1, n):
                    j = idxs[jj]
                    pj = self.placed_items[j]
                    jx1, jy1, jz1 = pj['x1'], pj['y1'], pj['z1']
                    jx2, jy2, jz2 = pj['x2'], pj['y2'], pj['z2']
                    x_overlap = ix1 < jx2 and ix2 > jx1
                    y_overlap = iy1 < jy2 and iy2 > jy1
                    z_overlap = iz1 < jz2 and iz2 > jz1
                    x_touch = abs(ix1 - jx2) < eps or abs(ix2 - jx1) < eps
                    y_touch = abs(iy1 - jy2) < eps or abs(iy2 - jy1) < eps
                    z_touch = abs(iz1 - jz2) < eps or abs(iz2 - jz1) < eps
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
        """Returns a bool torch.Tensor on CPU of shape `(n_mers * 2,)`.
        True at index `2*mer_idx + rot_idx` if (MER, rotation) is feasible.
        CPU placement keeps the rollout-time `.any()` check off the GPU
        (avoids a CUDA→CPU sync per env step). The model uploads to GPU
        once when the mask is consumed."""
        all_mers = self._get_all_mers()
        n_actions = max(1, len(all_mers) * 2)
        if self.current_item is None or len(all_mers) == 0:
            return torch.zeros(n_actions, dtype=torch.bool)

        mask_cpu = [False] * (len(all_mers) * 2)
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
                if gidx not in grid_items_cache:
                    grid_items_cache[gidx] = [p for p in self.placed_items if p['grid_idx'] == gidx]
                if self._check_constraints_fast(mer_pos, rot_dims, item_weight, item_priority,
                                                gidx, grid_dims, grid_items_cache[gidx]):
                    mask_cpu[action_idx * 2 + rot_idx] = True
                elif self._find_valid_anchor_in_mer(
                        mer_box, rot_dims, item_weight, item_priority,
                        gidx, grid_dims, grid_items_cache[gidx]) is not None:
                    # Corner failed a constraint (overlap, support, priority) but the item
                    # still fits at an interior anchor — keep the MER in the action set.
                    # Interior search is bounded by MAX_INTERIOR_PROBES.
                    mask_cpu[action_idx * 2 + rot_idx] = True

        return torch.tensor(mask_cpu, dtype=torch.bool)

    def _check_constraints_fast(self, position, dimensions, weight, priority, grid_idx, grid_dims, grid_items):
        """Pure-Python constraint check using cached coords on grid_items to avoid GPU sync.
        `position` and `dimensions` may be tensors or sequences."""
        px = float(position[0].item() if hasattr(position[0], 'item') else position[0])
        py = float(position[1].item() if hasattr(position[1], 'item') else position[1])
        pz = float(position[2].item() if hasattr(position[2], 'item') else position[2])
        dw = float(dimensions[0].item() if hasattr(dimensions[0], 'item') else dimensions[0])
        dl = float(dimensions[1].item() if hasattr(dimensions[1], 'item') else dimensions[1])
        dh = float(dimensions[2].item() if hasattr(dimensions[2], 'item') else dimensions[2])
        ix2, iy2, iz2 = px + dw, py + dl, pz + dh
        gw = float(grid_dims[0].item() if hasattr(grid_dims[0], 'item') else grid_dims[0])
        gl = float(grid_dims[1].item() if hasattr(grid_dims[1], 'item') else grid_dims[1])
        gh = float(grid_dims[2].item() if hasattr(grid_dims[2], 'item') else grid_dims[2])

        # Container containment
        if px < 0 or py < 0 or pz < 0 or ix2 > gw or iy2 > gl or iz2 > gh:
            return False

        # Overlap, weight stacking, priority — single pass
        bottom_area = dw * dl
        supported = 0.0
        for p in grid_items:
            px1 = p['x1']; py1 = p['y1']; pz1 = p['z1']
            px2 = p['x2']; py2 = p['y2']; pz2 = p['z2']
            # Overlap (strict: touching faces don't count)
            if not (ix2 <= px1 or px2 <= px or iy2 <= py1 or py2 <= py or iz2 <= pz1 or pz2 <= pz):
                return False
            is_blocker = p.get('is_blocker', False)
            # Stacking weight: placed is directly below this item
            if (not is_blocker and
                    pz2 <= pz and px1 < ix2 and px2 > px and py1 < iy2 and py2 > py):
                if p['weight'] + weight > self.max_stack_weight:
                    return False
                if weight > p['weight']:
                    return False
            # Priority: higher-priority (lower number) must be at lower or equal y
            if not is_blocker:
                if priority < p['priority'] and py > py1:
                    return False
                if priority > p['priority'] and py < py1:
                    return False
            # Support accumulation (only when our bottom face is at pz > 0)
            if (
                    pz > 0.001
                    and (not is_blocker or p.get('supports', True))
                    and abs(pz2 - pz) < 0.001
            ):
                ox = min(ix2, px2) - max(px, px1)
                oy = min(iy2, py2) - max(py, py1)
                if ox > 0 and oy > 0:
                    supported += ox * oy

        if pz > 0.001 and bottom_area > 0 and (supported / bottom_area) < 0.99:
            return False
        return True

    # Cap on interior anchor probes per call. Each probe is O(grid_items)
    # constraint checks; with ~90 placed items and ~20 MERs × 2 rotations
    # being probed per env step, an uncapped exhaustive search on a large
    # empty MER (e.g. 12×6×8 = 576 positions) can stall an episode for
    # 20+ minutes of pure-Python work. Capping bounds the worst case at
    # ~MAX_INTERIOR_PROBES × |grid_items| checks per call, with the cost
    # that some legal-but-deep anchors are missed (the (MER, rot) gets
    # marked infeasible). The corner is always checked first, so MERs
    # whose corner fits keep working unchanged.
    # Cap of 32 chosen empirically: smoke tests show <1% action-mask
    # divergence vs uncapped, while bounding per-step worst case to
    # ~32 × 91 × 5 = 15k Python ops per call (≈8ms), keeping a worst
    # case env step under ~500ms even with 40 cap-bound search calls.
    MAX_INTERIOR_PROBES = 32

    def _find_valid_anchor_in_mer(self, mer_box, item_dims, weight, priority, grid_idx, grid_dims, grid_items):
        """Return the first integer position within mer_box where item satisfies all constraints."""
        mx1 = int(mer_box.x1); my1 = int(mer_box.y1); mz1 = int(mer_box.z1)
        mx2 = int(mer_box.x2); my2 = int(mer_box.y2); mz2 = int(mer_box.z2)
        iw = int(item_dims[0].item()); il = int(item_dims[1].item()); ih = int(item_dims[2].item())

        # Fast path: corner (pass plain floats)
        corner = (float(mx1), float(my1), float(mz1))
        if self._check_constraints_fast(corner, item_dims, weight, priority, grid_idx, grid_dims, grid_items):
            return torch.tensor([corner[0], corner[1], corner[2]], dtype=torch.float)

        probes = 0
        for x in range(mx1, mx2 - iw + 1):
            for y in range(my1, my2 - il + 1):
                for z in range(mz1, mz2 - ih + 1):
                    if x == mx1 and y == my1 and z == mz1:
                        continue
                    if probes >= self.MAX_INTERIOR_PROBES:
                        return None
                    probes += 1
                    pos = (float(x), float(y), float(z))
                    if self._check_constraints_fast(pos, item_dims, weight, priority, grid_idx, grid_dims, grid_items):
                        return torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float)
        return None

        # GNN-based Actor-Critic Networks
