"""
Shared utilities: prompt builders, result I/O, retry wrapper.
"""

import json
import os
import time
import datetime
from typing import Optional

from core.validate_output import parse_output, validate_output, split_output
from core.placement_objective import compute_all_metrics
from technology.sky130_lib import NFET_W_OPTIONS, PFET_W_OPTIONS, L_OPTIONS


# ── Sky130 option strings ─────────────────────────────────────────────────────

_NW = ", ".join(str(w) for w in NFET_W_OPTIONS)
_PW = ", ".join(str(w) for w in PFET_W_OPTIONS)
_LL = ", ".join(str(l) for l in L_OPTIONS)


# ── Grid ASCII visualisation ──────────────────────────────────────────────────

def _grid_ascii(rows: int, cols: int) -> str:
    row_labels = [" ".join(f"({r},{c})" for c in range(cols)) for r in range(rows)]
    header = "     " + "  ".join(f" c{c} " for c in range(cols))
    lines  = [header]
    for r, row in enumerate(row_labels):
        lines.append(f" r{r}  " + "  ".join(f"( . )" for _ in range(cols)))
    return "\n".join(lines)


# ── Zero-shot prompt ──────────────────────────────────────────────────────────

def build_zero_shot_prompt(instance: dict) -> str:
    rows  = instance["grid"]["rows"]
    cols  = instance["grid"]["cols"]
    trans = instance["transistors"]
    const = instance["sizing_constraints"]

    t_lines = []
    for t in trans:
        nets_str = ", ".join(f"{p}={n}" for p, n in t["nets"].items())
        bounds   = const.get("bounds", {}).get(t["id"], {})
        bstr     = (f"W∈[{bounds.get('W_min','?')}–{bounds.get('W_max','?')}] μm, "
                    f"L∈[{bounds.get('L_min','?')}–{bounds.get('L_max','?')}] μm")
        t_lines.append(f"  {t['id']} ({t['type'].upper()}): {nets_str} | allowed {bstr}")

    match_lines = []
    for grp in const.get("match_groups", []):
        match_lines.append(f"  MATCH: {' = '.join(grp)}  (must have identical W and L)")

    t_block  = "\n".join(t_lines)
    m_block  = "\n".join(match_lines) if match_lines else "  (none)"
    grid_vis = _grid_ascii(rows, cols)

    first_id = trans[0]["id"]

    return f"""You are an analog IC layout expert solving a TOY placement + sizing problem.
This is NOT industrial layout — no DRC, no parasitic extraction, no PDK rules beyond sizing bounds.

TECHNOLOGY: Sky130 PDK (1.8V)
  NMOS width options (μm): {_NW}
  PMOS width options (μm): {_PW}
  Length options    (μm): {_LL}

GRID: {rows} rows × {cols} columns (zero-indexed, row 0 = top, col 0 = left)
{grid_vis}

TRANSISTORS TO PLACE AND SIZE:
{t_block}

MATCHING CONSTRAINTS (identical W and L required):
{m_block}

OBJECTIVES (in order of priority):
  1. Satisfy ALL matching constraints (same W and L within each group)
  2. Use only valid Sky130 discrete W/L values
  3. Keep W and L within each transistor's allowed bounds
  4. Minimize HPWL (half-perimeter wire length) of the placement
  5. Each transistor must occupy a unique grid cell

Return ONLY a JSON object — no explanation, no markdown, no comments:
{{
  "{first_id}": {{"row": <int>, "col": <int>, "W": <float>, "L": <float>}},
  ...
}}
"""


# ── Iterative prompt ──────────────────────────────────────────────────────────

def build_iterative_prompt(instance: dict,
                            prev_parsed: dict,
                            prev_metrics: dict,
                            validation: dict,
                            round_num: int) -> str:
    rows  = instance["grid"]["rows"]
    cols  = instance["grid"]["cols"]
    trans = instance["transistors"]
    const = instance["sizing_constraints"]

    prev_json = json.dumps(
        {tid: {"row": v["row"], "col": v["col"], "W": v["W"], "L": v["L"]}
         for tid, v in prev_parsed.items()},
        indent=2
    )

    feedback_lines = []
    if validation.get("placement_conflicts"):
        feedback_lines.append(f"  - Placement conflicts (two transistors in same cell): {validation['placement_conflicts']}")
    if validation.get("out_of_bounds"):
        feedback_lines.append(f"  - Out-of-bounds positions: {validation['out_of_bounds']}")
    if validation.get("invalid_sizes"):
        feedback_lines.append(f"  - Invalid Sky130 W/L values: {validation['invalid_sizes']}")
    if validation.get("missing_transistors"):
        feedback_lines.append(f"  - Missing transistors: {validation['missing_transistors']}")

    sz_v = prev_metrics.get("sizing_violations", "?")
    mt_v = prev_metrics.get("matching_violations", "?")
    rho  = prev_metrics.get("rho", "?")
    if isinstance(rho, float):
        rho = f"{rho:.3f}"

    feedback_lines += [
        f"  - Sizing violations (W/L out of bounds): {sz_v}",
        f"  - Matching violations (W or L differ in matched group): {mt_v}",
        f"  - Cost ratio ρ = {rho} (1.00 = reference quality; lower is better)",
    ]

    match_lines = []
    for grp in const.get("match_groups", []):
        match_lines.append(f"  MATCH: {' = '.join(grp)}")

    first_id = trans[0]["id"]
    feedback = "\n".join(feedback_lines)
    m_block  = "\n".join(match_lines) if match_lines else "  (none)"

    return f"""This is refinement round {round_num}.

Your previous solution:
{prev_json}

FEEDBACK:
{feedback}

MATCHING CONSTRAINTS (must all be satisfied):
{m_block}

GRID: {rows}×{cols} — each transistor needs a unique cell (row 0–{rows-1}, col 0–{cols-1}).
Valid Sky130 W (μm): {_NW}
Valid Sky130 L (μm): {_LL}

Fix ALL issues above and return an improved JSON object only:
{{
  "{first_id}": {{"row": <int>, "col": <int>, "W": <float>, "L": <float>}},
  ...
}}
"""


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_instance(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_solution(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_index(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def save_results(results: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def summarize(results: list, label: str = ""):
    parsed   = [r for r in results if r.get("parse_ok")]
    valid    = [r for r in results if r.get("valid")]
    exact    = [r for r in results if r.get("exact_match")]
    rhos     = [r["rho"] for r in results if isinstance(r.get("rho"), float)]
    sz_viols = [r for r in results if r.get("sizing_violations", 1) == 0]
    mt_viols = [r for r in results if r.get("matching_violations", 1) == 0]
    n        = len(results)

    print(f"\n{'='*55}")
    if label:
        print(f"  {label}")
    print(f"  Circuits        : {n}")
    print(f"  Parse OK        : {len(parsed)}/{n}  ({100*len(parsed)/max(n,1):.1f}%)")
    print(f"  Valid           : {len(valid)}/{n}  ({100*len(valid)/max(n,1):.1f}%)")
    print(f"  Sizing OK       : {len(sz_viols)}/{n}  ({100*len(sz_viols)/max(n,1):.1f}%)")
    print(f"  Matching OK     : {len(mt_viols)}/{n}  ({100*len(mt_viols)/max(n,1):.1f}%)")
    print(f"  Exact match     : {len(exact)}/{n}  ({100*len(exact)/max(n,1):.1f}%)")
    if rhos:
        print(f"  Mean ρ          : {sum(rhos)/len(rhos):.3f}")
        print(f"  Best ρ          : {min(rhos):.3f}")
    print(f"{'='*55}\n")


def with_backoff(fn, max_retries: int = 5):
    """Call fn() with exponential backoff on rate-limit errors."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg:
                print(f"  Rate limit hit, waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("Max retries exceeded")
