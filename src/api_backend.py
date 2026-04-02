import os
import json
from flask import Flask, request, jsonify
from ensemble_inference import EnsembleRouter

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
    if not ship_grids or not isinstance(ship_grids, list):
        return jsonify({"error": "Invalid or missing 'ship_grids'. Must be a list of lists: [[dim_tuple, name], ...]"}), 400
        
    manifest = data.get("manifest")
    if not manifest or not isinstance(manifest, list):
        return jsonify({"error": "Invalid or missing 'manifest'. Must be a list of SCU dictionaries."}), 400

    try:
        # Pass payload to the ensemble router
        result = router.route_manifest(ship_grids, manifest)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
