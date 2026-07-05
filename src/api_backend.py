import os
from flask import Flask, request, jsonify
from ensemble_inference import EnsembleRouter
from scu_manifest_generator import SCU_DEFINITIONS

app = Flask(__name__)

# Global variable for the ensemble router
router = None

def init_models():
    global router
    print("Booting up Multi-Model Ensemble...")
    try:
        router = EnsembleRouter()
    except Exception as e:
        print(f"Critical Ensemble Boot Failure: {e}")

# Initialize models before serving requests
init_models()

def _validate_ship_grids(ship_grids):
    if not ship_grids or not isinstance(ship_grids, list):
        return "Invalid or missing 'ship_grids'. Must be a list of grids."

    for idx, grid in enumerate(ship_grids):
        if isinstance(grid, dict):
            dims = grid.get("dimensions")
            name = grid.get("name")
            blocked = grid.get("blocked", [])
        elif isinstance(grid, (list, tuple)) and len(grid) in {2, 3}:
            dims, name = grid[0], grid[1]
            blocked = grid[2] if len(grid) == 3 else []
        else:
            return f"Invalid ship_grids[{idx}]. Expected [dimensions, name] or [dimensions, name, blocked]."

        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            return f"Invalid ship_grids[{idx}][0]. Dimensions must contain 3 numbers."
        if any(not isinstance(x, (int, float)) or x <= 0 for x in dims):
            return f"Invalid ship_grids[{idx}][0]. Dimensions must be positive numbers."
        if not isinstance(name, str) or not name:
            return f"Invalid ship_grids[{idx}][1]. Grid name must be a non-empty string."
        if blocked is None:
            blocked = []
        if not isinstance(blocked, list):
            return f"Invalid ship_grids[{idx}] blocked regions. Expected a list."
        for b_idx, blocker in enumerate(blocked):
            if isinstance(blocker, dict):
                pos = blocker.get("position")
                b_dims = blocker.get("dimensions")
                supports = blocker.get(
                    "supports",
                    blocker.get("support", blocker.get("counts_as_support", True)),
                )
            elif isinstance(blocker, (list, tuple)) and len(blocker) in {2, 3}:
                pos, b_dims = blocker[0], blocker[1]
                supports = blocker[2] if len(blocker) == 3 else True
            else:
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Expected position and dimensions."
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Position must contain 3 numbers."
            if not isinstance(b_dims, (list, tuple)) or len(b_dims) != 3:
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Dimensions must contain 3 numbers."
            if any(not isinstance(x, (int, float)) or x < 0 for x in pos):
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Position values must be non-negative numbers."
            if any(not isinstance(x, (int, float)) or x <= 0 for x in b_dims):
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Dimensions must be positive numbers."
            if any(pos[i] + b_dims[i] > dims[i] for i in range(3)):
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. Blocker exceeds grid bounds."
            if not isinstance(supports, bool):
                return f"Invalid blocker ship_grids[{idx}][{b_idx}]. supports must be a boolean."

    return None

def _validate_manifest(manifest):
    if not manifest or not isinstance(manifest, list):
        return "Invalid or missing 'manifest'. Must be a list of SCU dictionaries."

    for idx, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            return f"Invalid manifest[{idx}]. Each entry must be an object."

        scu_type = entry.get("scu_type")
        if scu_type not in SCU_DEFINITIONS:
            valid = ", ".join(sorted(SCU_DEFINITIONS))
            return f"Unknown scu_type at manifest[{idx}]: {scu_type!r}. Valid values: {valid}."

        quantity = entry.get("quantity")
        if not isinstance(quantity, int) or quantity <= 0:
            return f"Invalid quantity at manifest[{idx}]. Must be a positive integer."

        priority = entry.get("priority")
        if not isinstance(priority, int) or priority <= 0:
            return f"Invalid priority at manifest[{idx}]. Must be a positive integer."

    return None

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Expects a JSON payload with 'ship_grids' and 'manifest'.
    Example:
    {
      "ship_grids": [
        [[4, 10, 2], "Cargo Module 1"],
        [[4, 6, 2], "Front Module"]
      ],
      "manifest": [
        {"scu_type": "4 SCU", "quantity": 2, "priority": 1},
        {"scu_type": "1 SCU", "quantity": 5, "priority": 2}
      ]
    }
    """
    if router is None:
        return jsonify({"error": "GNN ensemble router is offline."}), 500
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
        
    ship_grids = data.get("ship_grids")
    grid_error = _validate_ship_grids(ship_grids)
    if grid_error:
        return jsonify({"error": grid_error}), 400
        
    manifest = data.get("manifest")
    manifest_error = _validate_manifest(manifest)
    if manifest_error:
        return jsonify({"error": manifest_error}), 400

    try:
        # Pass payload to the ensemble router
        result = router.route_manifest(ship_grids, manifest)
        return jsonify(result)
    except Exception:
        app.logger.exception("Optimization failed")
        return jsonify({"error": "Internal optimization error"}), 500

if __name__ == '__main__':
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    host = os.environ.get("PACKING_API_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(debug=debug, host=host, port=port)
