"""
Parse and validate LLM outputs for the joint placement + sizing task.

Expected LLM output format:
{
  "M1":    {"row": 0, "col": 2, "W": 2.10, "L": 0.50},
  "M2":    {"row": 1, "col": 0, "W": 2.10, "L": 0.50},
  "MTAIL": {"row": 1, "col": 1, "W": 4.20, "L": 0.50}
}
"""

import json
import re
from typing import Optional, Dict, Tuple

from technology.sky130_lib import is_valid_size


def extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object from a model response."""
    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Fall back to first bare { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def parse_output(text: str) -> Optional[Dict[str, dict]]:
    """
    Parse LLM text into {transistor_id: {row, col, W, L}}.
    Returns None if parsing fails.
    """
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    result = {}
    for tid, val in data.items():
        if not isinstance(val, dict):
            return None
        if not all(k in val for k in ("row", "col", "W", "L")):
            return None
        try:
            result[tid] = {
                "row": int(val["row"]),
                "col": int(val["col"]),
                "W":   float(val["W"]),
                "L":   float(val["L"]),
            }
        except (TypeError, ValueError):
            return None
    return result


def split_output(parsed: Dict[str, dict]):
    """
    Separate a parsed output into:
      placement: {tid: (row, col)}
      sizing:    {tid: {W, L}}
    """
    placement = {tid: (v["row"], v["col"]) for tid, v in parsed.items()}
    sizing    = {tid: {"W": v["W"], "L": v["L"]} for tid, v in parsed.items()}
    return placement, sizing


def validate_output(parsed: Optional[Dict[str, dict]], topology: dict) -> dict:
    """
    Full validation of a parsed LLM output against the topology.
    Returns a validation report.
    """
    transistors = topology["transistors"]
    grid_rows, grid_cols = topology["grid"]
    required_ids = {t["id"] for t in transistors}

    if parsed is None:
        return {
            "valid": False,
            "parse_ok": False,
            "missing_transistors": list(required_ids),
            "unknown_transistors": [],
            "out_of_bounds": [],
            "placement_conflicts": [],
            "invalid_sizes": [],
            "errors": ["Failed to parse JSON output"],
        }

    present_ids  = set(parsed.keys())
    missing      = list(required_ids - present_ids)
    unknown      = list(present_ids - required_ids)
    out_of_bounds = []
    invalid_sizes = []

    for tid, val in parsed.items():
        r, c = val["row"], val["col"]
        if not (0 <= r < grid_rows and 0 <= c < grid_cols):
            out_of_bounds.append(tid)

    # Check for duplicate placements
    positions = {}
    conflicts = []
    for tid, val in parsed.items():
        pos = (val["row"], val["col"])
        if pos in positions:
            conflicts.append((positions[pos], tid))
        else:
            positions[pos] = tid

    # Check valid Sky130 sizes
    t_type_map = {t["id"]: t["type"] for t in transistors}
    for tid, val in parsed.items():
        if tid not in t_type_map:
            continue
        if not is_valid_size(val["W"], val["L"], t_type_map[tid]):
            invalid_sizes.append(tid)

    errors = []
    if missing:
        errors.append(f"Missing transistors: {missing}")
    if unknown:
        errors.append(f"Unknown transistors: {unknown}")
    if out_of_bounds:
        errors.append(f"Out-of-bounds positions: {out_of_bounds}")
    if conflicts:
        errors.append(f"Placement conflicts: {conflicts}")
    if invalid_sizes:
        errors.append(f"Invalid Sky130 sizes: {invalid_sizes}")

    valid = not (missing or out_of_bounds or conflicts or invalid_sizes)

    return {
        "valid": valid,
        "parse_ok": True,
        "missing_transistors": missing,
        "unknown_transistors": unknown,
        "out_of_bounds": out_of_bounds,
        "placement_conflicts": [list(p) for p in conflicts],
        "invalid_sizes": invalid_sizes,
        "errors": errors,
    }
