"""
Generate the dataset: 20 small + 20 large instances with reference solutions.

Outputs:
  dataset/small/instances/circuit_NNN.json
  dataset/small/solutions/circuit_NNN_solution.json
  dataset/small/index.json
  dataset/large/  (same structure)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset_gen.circuit_topologies import SMALL_TOPOLOGIES, LARGE_TOPOLOGIES
from core.reference_solver import solve
from core.placement_objective import compute_cost, compute_hpwl

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def build_split(topologies: list, out_dir: str, label: str):
    inst_dir = os.path.join(out_dir, "instances")
    sol_dir  = os.path.join(out_dir, "solutions")
    os.makedirs(inst_dir, exist_ok=True)
    os.makedirs(sol_dir,  exist_ok=True)

    index = []

    for i, topo in enumerate(topologies):
        circuit_id = f"circuit_{i+1:03d}"
        n_transistors = len(topo["transistors"])
        rows, cols    = topo["grid"]

        # Instance file
        instance = {
            "id":          circuit_id,
            "name":        topo["name"],
            "description": topo["description"],
            "grid":        {"rows": rows, "cols": cols},
            "transistors": topo["transistors"],
            "sizing_constraints": topo["sizing_constraints"],
        }
        inst_path = os.path.join(inst_dir, f"{circuit_id}.json")
        with open(inst_path, "w") as f:
            json.dump(instance, f, indent=2)

        # Reference solution
        solution = solve(topo)
        placement_list = {tid: list(rc) for tid, rc in solution["placement"].items()}
        sizing         = solution["sizing"]

        metrics = compute_cost(solution["placement"], sizing, topo)
        hpwl    = metrics["hpwl"]

        sol_data = {
            "id":        circuit_id,
            "placement": placement_list,
            "sizing":    sizing,
            "metrics":   metrics,
        }
        sol_path = os.path.join(sol_dir, f"{circuit_id}_solution.json")
        with open(sol_path, "w") as f:
            json.dump(sol_data, f, indent=2)

        sz_v = metrics["sizing_violations"]
        mt_v = metrics["matching_violations"]
        status = "OK" if sz_v == 0 and mt_v == 0 else "PARTIAL"

        print(f"[{label.upper()}] {circuit_id}  {topo['name']:<30s}  "
              f"{rows}x{cols}  T={n_transistors}  HPWL={hpwl:5.1f}  {status}")

        index.append({
            "id":            circuit_id,
            "name":          topo["name"],
            "instance_path": inst_path,
            "solution_path": sol_path,
            "grid":          {"rows": rows, "cols": cols},
            "n_transistors": n_transistors,
            "ref_hpwl":      hpwl,
            "ref_cost":      metrics["cost"],
        })

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[{label.upper()}] {len(topologies)} instances saved → {out_dir}")
    print(f"[{label.upper()}] Index → {index_path}\n")
    return index


if __name__ == "__main__":
    build_split(SMALL_TOPOLOGIES, os.path.join(DATASET_ROOT, "small"), "small")
    build_split(LARGE_TOPOLOGIES, os.path.join(DATASET_ROOT, "large"), "large")
