import pytest
import torch
from src.packing_core.drl_env import DRLBinPackingEnv

@pytest.fixture
def multi_grid_env():
    grids = [
        ((10.0, 20.0, 5.0), "Grid1"),
        ((15.0, 10.0, 10.0), "Grid2"),
        ((5.0, 5.0, 5.0), "Grid3")
    ]
    env = DRLBinPackingEnv(grids_list=grids)
    # Give it a dummy manifest to ensure reset works and returns a valid graph
    env.reset(cargo_manifest=[(2, 2, 2, 10.0, 1)])
    return env

def test_node_type_ship_exists(multi_grid_env):
    """1. NODE_TYPE_SHIP = 4 must exist."""
    assert hasattr(multi_grid_env, "NODE_TYPE_SHIP"), "NODE_TYPE_SHIP must be defined in the environment."
    assert multi_grid_env.NODE_TYPE_SHIP == 4, "NODE_TYPE_SHIP must equal 4."

def test_exactly_one_ship_node(multi_grid_env):
    """2. The graph state should contain exactly one node of type NODE_TYPE_SHIP."""
    if not hasattr(multi_grid_env, "NODE_TYPE_SHIP"):
        pytest.skip("NODE_TYPE_SHIP not implemented yet")
        
    graph = multi_grid_env.reset(cargo_manifest=[(2, 2, 2, 10.0, 1)])
    
    # Node types are at index 0 of x
    node_types = graph.x[:, 0]
    ship_nodes = (node_types == multi_grid_env.NODE_TYPE_SHIP).sum().item()
    
    assert ship_nodes == 1, f"Expected exactly 1 ship node, found {ship_nodes}"

def test_ship_to_container_edges(multi_grid_env):
    """3. The NODE_TYPE_SHIP node must have an edge connecting it to every NODE_TYPE_CONTAINER node (and vice-versa)."""
    if not hasattr(multi_grid_env, "NODE_TYPE_SHIP"):
        pytest.skip("NODE_TYPE_SHIP not implemented yet")
        
    graph = multi_grid_env.reset(cargo_manifest=[(2, 2, 2, 10.0, 1)])
    
    # Find node indices
    node_types = graph.x[:, 0]
    ship_idx = (node_types == multi_grid_env.NODE_TYPE_SHIP).nonzero(as_tuple=True)[0]
    container_indices = (node_types == multi_grid_env.NODE_TYPE_CONTAINER).nonzero(as_tuple=True)[0]
    
    assert len(ship_idx) == 1, "There must be exactly one ship node."
    ship_idx = ship_idx[0].item()
    
    # Convert edge_index to a set of tuples for easy checking
    edges = set(tuple(e) for e in graph.edge_index.t().tolist())
    
    for c_idx in container_indices.tolist():
        assert (ship_idx, c_idx) in edges, f"Missing edge from ship ({ship_idx}) to container ({c_idx})"
        assert (c_idx, ship_idx) in edges, f"Missing edge from container ({c_idx}) to ship ({ship_idx})"

def test_container_fully_connected(multi_grid_env):
    """4. All NODE_TYPE_CONTAINER nodes must be fully connected to each other."""
    graph = multi_grid_env.reset(cargo_manifest=[(2, 2, 2, 10.0, 1)])
    
    node_types = graph.x[:, 0]
    container_indices = (node_types == multi_grid_env.NODE_TYPE_CONTAINER).nonzero(as_tuple=True)[0].tolist()
    
    edges = set(tuple(e) for e in graph.edge_index.t().tolist())
    
    for c1 in container_indices:
        for c2 in container_indices:
            if c1 != c2:
                assert (c1, c2) in edges, f"Missing edge between container nodes: ({c1}, {c2})"

def test_container_expanded_features(multi_grid_env):
    """5. The feature vector for NODE_TYPE_CONTAINER nodes must be expanded to include new geometric and positional features."""
    graph = multi_grid_env.reset(cargo_manifest=[(2, 2, 2, 10.0, 1)])
    
    node_types = graph.x[:, 0]
    container_indices = (node_types == multi_grid_env.NODE_TYPE_CONTAINER).nonzero(as_tuple=True)[0].tolist()
    
    # Calculate expected ship volume
    grids = multi_grid_env.grids
    total_vol = sum((g['dims'][0] * g['dims'][1] * g['dims'][2]).item() for g in grids)
    
    # original features are 8. 5 new ones => 13
    num_features = graph.x.shape[1]
    assert num_features >= 13, f"Expected container node to have at least 13 features, but got {num_features}"
    
    for idx, c_node_idx in enumerate(container_indices):
        features = graph.x[c_node_idx].tolist()
        
        # Dimensions are features[1], features[2], features[3] based on original code
        w, l, h = features[1], features[2], features[3]
        
        # Calculate expected new features
        expected_w_l = w / l if l > 0 else 0
        expected_w_h = w / h if h > 0 else 0
        expected_l_h = l / h if h > 0 else 0
        
        grid_vol = w * l * h
        expected_rel_vol = grid_vol / total_vol if total_vol > 0 else 0
        expected_rank = float(idx)
        
        # We assume the new features are appended at the end: W/L, W/H, L/H, RelVol, Rank
        w_l_feat = features[-5]
        w_h_feat = features[-4]
        l_h_feat = features[-3]
        rel_vol_feat = features[-2]
        rank_feat = features[-1]
        
        assert w_l_feat == pytest.approx(expected_w_l, rel=1e-4), "Aspect ratio W/L mismatch"
        assert w_h_feat == pytest.approx(expected_w_h, rel=1e-4), "Aspect ratio W/H mismatch"
        assert l_h_feat == pytest.approx(expected_l_h, rel=1e-4), "Aspect ratio L/H mismatch"
        assert rel_vol_feat == pytest.approx(expected_rel_vol, rel=1e-4), "Relative volume mismatch"
        assert rank_feat == pytest.approx(expected_rank, rel=1e-4), "Positional rank mismatch"
