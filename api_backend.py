import json
from flask import Flask, request, jsonify
from numba import cuda, jit
import cupy as cp
import math
import psycopg2
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus, PULP_CBC_CMD
)
import sys
from typing import List, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)



@cuda.jit
def validate_placements_gpu(placements_gpu, dims_gpu, priorities_gpu, results_gpu):
    # CUDA kernel for validating box placements in parallel
    idx = cuda.grid(1)
    if idx < placements_gpu.shape[0]:
        x, y, z = placements_gpu[idx, 0:3]
        w, l, h = placements_gpu[idx, 3:6]
        priority = priorities_gpu[idx]

        # Check container bounds
        if (x + w > dims_gpu[0] or
                y + l > dims_gpu[1] or
                z + h > dims_gpu[2]):
            results_gpu[idx] = 0
            return

class CargoBox:
    def __init__(self, box_id: int, width: float, length: float, height: float,
                 weight: float, name: str):
        self.id = box_id
        self.width = width
        self.length = length
        self.height = height
        self.weight = weight
        self.name = name

def validate_access_path(placements: List[Tuple], container_dims: Tuple[float, float, float]) -> bool:
    # Validates that boxes have a clear path to the container door (assumes door is at y=0)
    W, L, H = container_dims
    sorted_placements = sorted(placements, key=lambda p: p[-1])  # Sort by priority

    for i, p1 in enumerate(sorted_placements):
        x1, y1, z1 = p1[2:5]
        w1, l1, h1 = p1[5:8]

        # Check for blocking boxes with lower priority
        for p2 in sorted_placements[i+1:]:
            x2, y2, z2 = p2[2:5]
            w2, l2, h2 = p2[5:8]

            # Check if box2 blocks the path to the door
            if (y2 < y1 and  # Box2 is closer to door
                    x2 < x1 + w1 and x2 + w2 > x1 and  # X overlap
                    z2 < z1 + h1 and z2 + h2 > z1):    # Z overlap
                return False
    return True

def validate_stacking_weights(placements: List[Tuple], cargo_boxes: List[CargoBox],
                              max_stack_weight: float) -> bool:
    # Validates weight constraints for stacked boxes.
    for p1 in placements:
        total_weight = 0
        box1 = next(box for box in cargo_boxes if box.id == p1[0])
        for p2 in placements:
            if p1 == p2:
                continue
            box2 = next(box for box in cargo_boxes if box.id == p2[0])
            # Check if box2 is above box1
            if (p2[2] >= p1[2] + p1[5] and  # z overlap
                    p2[3] < p1[3] + p1[6] and    # y overlap
                    p2[4] < p1[4] + p1[7]):      # x overlap
                total_weight += box2.weight
        if total_weight > max_stack_weight:
            return False
    return True

def add_no_overlap_constraints(model, n, g, cargo_list, grid_assign, x_vars, y_vars, z_vars, bigM):
    # Create binary variables for the six possible separating planes
    A_vars = {}
    for i in range(n):
        for k in range(i+1, n):
            for p in range(6):
                A_vars[(i, k, p)] = LpVariable(f"A_{i}_{k}_{p}", cat="Binary")
    # Add no-overlap constraints
    for i in range(n):
        w_i, l_i, h_i = cargo_list[i][1:4]
        for k in range(i+1, n):
            w_k, l_k, h_k = cargo_list[k][1:4]
            # Only enforce no-overlap when boxes are in the same grid
            for j in range(g):
                # Indicator that both boxes are in grid j
                same_grid = grid_assign[(i, j)] + grid_assign[(k, j)] == 2

                # Six separating plane constraints
                model.addConstraint(
                    x_vars[i] + w_i <= x_vars[k] + bigM * (1 - A_vars[(i, k, 0)] + (1 - same_grid)),
                    name=f"no_overlap_x1_{i}_{k}_{j}"
                )
                model.addConstraint(
                    x_vars[k] + w_k <= x_vars[i] + bigM * (1 - A_vars[(i, k, 1)] + (1 - same_grid)),
                    name=f"no_overlap_x2_{i}_{k}_{j}"
                )
                model.addConstraint(
                    y_vars[i] + l_i <= y_vars[k] + bigM * (1 - A_vars[(i, k, 2)] + (1 - same_grid)),
                    name=f"no_overlap_y1_{i}_{k}_{j}"
                )
                model.addConstraint(
                    y_vars[k] + l_k <= y_vars[i] + bigM * (1 - A_vars[(i, k, 3)] + (1 - same_grid)),
                    name=f"no_overlap_y2_{i}_{k}_{j}"
                )
                model.addConstraint(
                    z_vars[i] + h_i <= z_vars[k] + bigM * (1 - A_vars[(i, k, 4)] + (1 - same_grid)),
                    name=f"no_overlap_z1_{i}_{k}_{j}"
                )
                model.addConstraint(
                    z_vars[k] + h_k <= z_vars[i] + bigM * (1 - A_vars[(i, k, 5)] + (1 - same_grid)),
                    name=f"no_overlap_z2_{i}_{k}_{j}"
                )
                # At least one separating plane must exist between boxes in the same grid
                model.addConstraint(
                    lpSum(A_vars[(i, k, p)] for p in range(6)) >= same_grid,
                    name=f"at_least_one_separation_{i}_{k}_{j}"
                )
    return A_vars

def add_route_constraints(model, n, g, route_order, grid_assign, y_vars, bigM):
    min_spacing = 0.1  # Minimum spacing between boxes
    for i in range(n):
        for k in range(n):
            if i != k and route_order[i] < route_order[k]:
                for j in range(g):
                    # Only apply if both boxes are in the same grid
                    model.addConstraint(
                        y_vars[i] >= y_vars[k] + min_spacing - bigM * (2 - grid_assign[(i, j)] - grid_assign[(k, j)]),
                        name=f"route_order_{i}_{k}_{j}"
                    )

def build_model(n, g, cargo_list, grids, route_order):
    model = LpProblem(name="3D_Stacking_Multiple_Grids", sense=LpMinimize)
    # Calculate a tighter bigM based on maximum possible coordinate
    bigM = max(max(grid[1:4]) for grid in grids) * 2
    # Create variables
    grid_assign = {(i, j): LpVariable(f"grid_{i}_{j}", cat="Binary")
                   for i in range(n) for j in range(g)}
    x_vars = {i: LpVariable(f"x_{i}", lowBound=0) for i in range(n)}
    y_vars = {i: LpVariable(f"y_{i}", lowBound=0) for i in range(n)}
    z_vars = {i: LpVariable(f"z_{i}", lowBound=0) for i in range(n)}
    # Add grid assignment constraints
    for i in range(n):
        model.addConstraint(
            lpSum(grid_assign[(i, j)] for j in range(g)) == 1,
            name=f"box_{i}_grid_assignment"
        )
    # Add dimension constraints
    for i in range(n):
        for j in range(g):
            W_j, L_j, H_j = grids[j][1:4]
            _, w_i, l_i, h_i, _ = cargo_list[i]
            model.addConstraint(
                x_vars[i] + w_i <= W_j + bigM * (1 - grid_assign[(i, j)]),
                name=f"bound_x_{i}_{j}"
            )
            model.addConstraint(
                y_vars[i] + l_i <= L_j + bigM * (1 - grid_assign[(i, j)]),
                name=f"bound_y_{i}_{j}"
            )
            model.addConstraint(
                z_vars[i] + h_i <= H_j + bigM * (1 - grid_assign[(i, j)]),
                name=f"bound_z_{i}_{j}"
            )
    # Add no-overlap constraints
    A_vars = add_no_overlap_constraints(model, n, g, cargo_list, grid_assign,
                                        x_vars, y_vars, z_vars, bigM)
    # Add route constraints
    add_route_constraints(model, n, g, route_order, grid_assign, y_vars, bigM)
    # Add symmetry-breaking constraints
    for i in range(1, n):
        model.addConstraint(
            x_vars[i-1] <= x_vars[i],
            name=f"symmetry_breaking_x_{i}"
        )
        model.addConstraint(
            y_vars[i-1] <= y_vars[i],
            name=f"symmetry_breaking_y_{i}"
        )
        model.addConstraint(
            z_vars[i-1] <= z_vars[i],
            name=f"symmetry_breaking_z_{i}"
        )
    return model, x_vars, y_vars, z_vars, grid_assign


def run_optimization(data):
    """
    Expects a JSON object with the following structure:
    {
        "ship_name": "Some Ship",
        "cargo_selection": [
            { "cargo_item_id": 1, "quantity": 3, "unload_priorities": [1, 2, 3] },
            { "cargo_item_id": 2, "quantity": 2, "unload_priorities": [4, 5] }
        ]
    }
    """
    if "ship_name" not in data or "cargo_selection" not in data:
        return {"error": "Missing required fields: ship_name and cargo_selection"}

    ship_name = data["ship_name"]
    cargo_selection = data["cargo_selection"]

    # Connect to the database
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
    except Exception as e:
        return {"error": f"Could not connect to the database: {str(e)}"}

    cursor = conn.cursor()

    # Get all grids for the selected ship
    cursor.execute("""
        SELECT grid_name, dimension_x, dimension_y, dimension_z, grid_size 
        FROM temp_grid_import 
        WHERE ship_name = %s AND grid_name IS NOT NULL 
        ORDER BY grid_name;
    """, (ship_name,))
    grids = cursor.fetchall()
    if not grids:
        cursor.close()
        conn.close()
        return {"error": f"No grids found for ship {ship_name}"}

    # Build cargo_list and route_order based on the user's cargo selections
    cargo_list = []
    route_order = {}
    item_index = 0
    for item in cargo_selection:
        # Each item should have cargo_item_id and quantity; unload_priorities is optional.
        if "cargo_item_id" not in item or "quantity" not in item:
            continue
        cargo_item_id = item["cargo_item_id"]
        quantity = item["quantity"]
        unload_priorities = item.get("unload_priorities", None)
        # Fetch cargo item details from the database
        cursor.execute(
            "SELECT name, width_x, length_y, height_z FROM cargo_items WHERE cargo_item_id = %s",
            (cargo_item_id,)
        )
        cargo_row = cursor.fetchone()
        if not cargo_row:
            # Skip if not found
            continue
        c_name, w_i, l_i, h_i = cargo_row
        for q in range(quantity):
            cargo_list.append((item_index, w_i, l_i, h_i, c_name))
            # Use provided unload priority if available; otherwise, use default value 9999.
            if unload_priorities and isinstance(unload_priorities, list) and len(unload_priorities) > q:
                route_order[item_index] = int(unload_priorities[q])
            else:
                route_order[item_index] = 9999
            item_index += 1

    if not cargo_list:
        cursor.close()
        conn.close()
        return {"error": "No valid cargo items selected."}

    n = len(cargo_list)

    # Build the MILP model (Stacking + Route constraints)
    model, x_vars, y_vars, z_vars, grid_assign = build_model(
        n=n,
        g=len(grids),
        cargo_list=cargo_list,
        grids=grids,
        route_order=route_order
    )
    # Solve the model
    solver_status = model.solve(PULP_CBC_CMD(msg=0))
    gridOutput = []  # This will hold our JSON-friendly output

    if LpStatus[model.status] == "Optimal":
        num_grids = len(grids)
        grid_placements = {j: [] for j in range(num_grids)}
        for i in range(n):
            x_val = x_vars[i].varValue
            y_val = y_vars[i].varValue
            z_val = z_vars[i].varValue
            # Find which grid this box was assigned to
            assigned_grid = next(j for j in range(num_grids) if grid_assign[(i, j)].varValue > 0.5)
            grid_placements[assigned_grid].append((
                i,                   # Box ID
                cargo_list[i][4],    # Cargo Name
                x_val, y_val, z_val, # Coordinates
                cargo_list[i][1],    # Width
                cargo_list[i][2],    # Length
                cargo_list[i][3],    # Height
                route_order[i]       # Unload Priority
            ))
        # Build the JSON-friendly dictionary for each grid
        for j in range(num_grids):
            grid_data = {
                "gridName": grids[j][0],
                "dimensions": {
                    "width": grids[j][1],
                    "length": grids[j][2],
                    "height": grids[j][3]
                },
                "boxes": []
            }
            for p in grid_placements[j]:
                box_data = {
                    "boxId": p[0],
                    "cargoName": p[1],
                    "x": p[2],
                    "y": p[3],
                    "z": p[4],
                    "width": p[5],
                    "length": p[6],
                    "height": p[7],
                    "priority": p[8]
                }
                grid_data["boxes"].append(box_data)
            gridOutput.append(grid_data)
    else:
        cursor.close()
        conn.close()
        return {"error": "No optimal solution found."}

    cursor.close()
    conn.close()
    return gridOutput


@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Expects a JSON payload with keys 'ship_name' and 'cargo_selection'.
    Returns the computed grid placement as JSON.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    result = run_optimization(data)
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
