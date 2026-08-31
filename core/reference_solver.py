"""
Reference solver for the joint placement + sizing problem.

Placement: greedy HPWL minimization via random restart hill-climbing.
Sizing:    assign reference W/L from topology's reference_sizing field
           (Sky130 presets chosen by circuit designers).
"""

import random
from itertools import permutations
from typing import Dict, Tuple

from core.placement_objective import compute_hpwl


def _all_cells(rows: int, cols: int) -> list:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _hpwl_for_assignment(assignment: Dict[str, Tuple[int, int]],
                          transistors: list) -> float:
    return compute_hpwl(assignment, transistors)


def solve_placement(topology: dict, n_restarts: int = 200, seed: int = 42) -> Dict[str, Tuple[int, int]]:
    """
    Find a low-HPWL placement via random-restart hill-climbing.
    Returns {transistor_id: (row, col)}.
    """
    rng = random.Random(seed)
    transistors = topology["transistors"]
    rows, cols  = topology["grid"]
    ids         = [t["id"] for t in transistors]
    n           = len(ids)
    cells       = _all_cells(rows, cols)

    if n > len(cells):
        raise ValueError(f"More transistors ({n}) than grid cells ({len(cells)})")

    best_cost     = float("inf")
    best_placement = {}

    for _ in range(n_restarts):
        chosen = rng.sample(cells, n)
        assignment = dict(zip(ids, chosen))
        cost = _hpwl_for_assignment(assignment, transistors)

        # Simple swap hill-climb
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = ids[i], ids[j]
                    assignment[a], assignment[b] = assignment[b], assignment[a]
                    new_cost = _hpwl_for_assignment(assignment, transistors)
                    if new_cost < cost:
                        cost     = new_cost
                        improved = True
                    else:
                        assignment[a], assignment[b] = assignment[b], assignment[a]

        if cost < best_cost:
            best_cost      = cost
            best_placement = dict(assignment)

    return best_placement


def solve_sizing(topology: dict) -> Dict[str, dict]:
    """
    Return the reference sizing directly from the topology definition.
    These are Sky130 analog design presets.
    """
    return {tid: dict(sz) for tid, sz in topology["reference_sizing"].items()}


def solve(topology: dict) -> dict:
    """Full reference solution: placement + sizing."""
    placement = solve_placement(topology)
    sizing    = solve_sizing(topology)
    return {"placement": placement, "sizing": sizing}
