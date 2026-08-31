"""
Objective function for the joint placement + sizing problem.

Cost = α·HPWL_norm + β·SizingViolations + γ·MatchingViolations

Where:
  HPWL_norm      = HPWL / max_possible_HPWL  (normalized 0–1)
  SizingViolations = number of transistors with W or L outside their allowed bounds
  MatchingViolations = number of matched pairs with different W or L

ρ = C_LLM / C_reference  (lower is better, 1.0 = reference quality)
"""

from typing import Dict, Tuple

ALPHA = 1.0   # HPWL weight
BETA  = 10.0  # sizing violation weight
GAMMA = 20.0  # matching violation weight


# ── HPWL ─────────────────────────────────────────────────────────────────────

def compute_hpwl(placement: Dict[str, Tuple[int, int]],
                 transistors: list) -> float:
    """
    Half-Perimeter Wire Length.
    For each net, collect all transistor pins connected to it and compute
    the bounding box half-perimeter over grid cell coordinates.
    """
    net_to_cells: Dict[str, list] = {}
    for t in transistors:
        tid = t["id"]
        if tid not in placement:
            continue
        r, c = placement[tid]
        for pin_name, net_name in t["nets"].items():
            net_to_cells.setdefault(net_name, []).append((r, c))

    hpwl = 0.0
    for net, cells in net_to_cells.items():
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        hpwl += (max(rows) - min(rows)) + (max(cols) - min(cols))
    return hpwl


def max_hpwl(grid: Tuple[int, int]) -> float:
    rows, cols = grid
    return float((rows - 1) + (cols - 1))


# ── Sizing violations ─────────────────────────────────────────────────────────

def compute_sizing_violations(sizing: Dict[str, dict],
                               constraints: dict) -> int:
    """Count transistors whose W or L fall outside their allowed bounds."""
    bounds = constraints.get("bounds", {})
    violations = 0
    for tid, sz in sizing.items():
        if tid not in bounds:
            continue
        b = bounds[tid]
        W, L = sz.get("W", 0), sz.get("L", 0)
        if not (b["W_min"] <= W <= b["W_max"]):
            violations += 1
        if not (b["L_min"] <= L <= b["L_max"]):
            violations += 1
    return violations


# ── Matching violations ───────────────────────────────────────────────────────

def compute_matching_violations(sizing: Dict[str, dict],
                                 constraints: dict) -> int:
    """Count matched-group pairs where W or L differ."""
    violations = 0
    for group in constraints.get("match_groups", []):
        sizes = [sizing.get(tid) for tid in group if sizing.get(tid) is not None]
        if len(sizes) < 2:
            continue
        ref = sizes[0]
        for sz in sizes[1:]:
            if abs(sz.get("W", -1) - ref.get("W", -2)) > 1e-9:
                violations += 1
            if abs(sz.get("L", -1) - ref.get("L", -2)) > 1e-9:
                violations += 1
    return violations


# ── Combined cost ─────────────────────────────────────────────────────────────

def compute_cost(placement: Dict[str, Tuple[int, int]],
                 sizing: Dict[str, dict],
                 topology: dict,
                 alpha: float = ALPHA,
                 beta: float  = BETA,
                 gamma: float = GAMMA) -> dict:
    transistors  = topology["transistors"]
    constraints  = topology["sizing_constraints"]
    grid         = topology["grid"]

    hpwl      = compute_hpwl(placement, transistors)
    hpwl_n    = hpwl / max(max_hpwl(grid), 1.0)
    sz_viols  = compute_sizing_violations(sizing, constraints)
    mt_viols  = compute_matching_violations(sizing, constraints)
    cost      = alpha * hpwl_n + beta * sz_viols + gamma * mt_viols

    return {
        "hpwl":                hpwl,
        "hpwl_normalized":     hpwl_n,
        "sizing_violations":   sz_viols,
        "matching_violations": mt_viols,
        "cost":                cost,
    }


def compute_all_metrics(placement: Dict[str, Tuple[int, int]],
                        sizing: Dict[str, dict],
                        topology: dict,
                        ref_placement: Dict[str, Tuple[int, int]],
                        ref_sizing: Dict[str, dict]) -> dict:
    """Full metrics including ρ and exact_match."""
    llm = compute_cost(placement, sizing, topology)
    ref = compute_cost(ref_placement, ref_sizing, topology)

    rho = llm["cost"] / ref["cost"] if ref["cost"] > 0 else float("inf")

    # Exact match: same grid positions AND same W/L for every transistor
    placement_match = all(
        placement.get(tid) == ref_placement.get(tid)
        for tid in ref_placement
    )
    sizing_match = all(
        sizing.get(tid, {}).get("W") == ref_sizing.get(tid, {}).get("W") and
        sizing.get(tid, {}).get("L") == ref_sizing.get(tid, {}).get("L")
        for tid in ref_sizing
    )
    exact_match = placement_match and sizing_match and llm["sizing_violations"] == 0 and llm["matching_violations"] == 0

    return {
        **llm,
        "ref_cost":            ref["cost"],
        "ref_hpwl":            ref["hpwl"],
        "rho":                 rho,
        "placement_match":     placement_match,
        "sizing_match":        sizing_match,
        "exact_match":         exact_match,
    }
