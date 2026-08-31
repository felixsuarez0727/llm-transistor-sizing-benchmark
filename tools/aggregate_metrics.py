"""
Aggregate all results/*.json files into a summary table for the paper.
Usage:
  python3 tools/aggregate_metrics.py --results-dir results --output results/paper_metrics.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def discover_results(results_dir: str) -> list:
    return sorted(glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True))


def parse_run_label(path: str) -> dict:
    parts = os.path.basename(os.path.dirname(path)).split("_")
    # format: TIMESTAMP_MODEL_sizing_SPLIT_MODE[_rN]
    # find "sizing" separator
    try:
        si = parts.index("sizing")
        model = "_".join(parts[1:si])
        rest  = parts[si+1:]
        split = rest[0] if rest else "?"
        mode  = "_".join(rest[1:]) if len(rest) > 1 else "zero_shot"
    except (ValueError, IndexError):
        model, split, mode = "unknown", "?", "?"
    return {"model": model, "split": split, "mode": mode}


def aggregate(path: str) -> dict:
    with open(path) as f:
        records = json.load(f)

    n         = len(records)
    parsed    = sum(1 for r in records if r.get("parse_ok"))
    valid     = sum(1 for r in records if r.get("valid"))
    exact     = sum(1 for r in records if r.get("exact_match"))
    sz_ok     = sum(1 for r in records if r.get("sizing_violations", 1) == 0)
    mt_ok     = sum(1 for r in records if r.get("matching_violations", 1) == 0)
    rhos      = [r["rho"] for r in records if isinstance(r.get("rho"), (int, float)) and r["rho"] < 1e9]

    return {
        "n":                    n,
        "parse_pct":            100 * parsed / max(n, 1),
        "valid_pct":            100 * valid  / max(n, 1),
        "sizing_ok_pct":        100 * sz_ok  / max(n, 1),
        "matching_ok_pct":      100 * mt_ok  / max(n, 1),
        "exact_match_pct":      100 * exact  / max(n, 1),
        "mean_rho":             sum(rhos) / len(rhos) if rhos else None,
        "best_rho":             min(rhos) if rhos else None,
        "n_rho_computed":       len(rhos),
    }


def run(args):
    paths = discover_results(args.results_dir)
    if not paths:
        print(f"No results.json found under {args.results_dir}")
        return

    rows = []
    header = f"{'Model':<35} {'Split':<7} {'Mode':<25} {'Parse%':>7} {'Valid%':>7} {'SzOK%':>7} {'MtOK%':>7} {'Exact%':>7} {'MeanRho':>8} {'BestRho':>8}"
    print(header)
    print("-" * len(header))

    for path in paths:
        label = parse_run_label(path)
        agg   = aggregate(path)
        row   = {**label, **agg, "path": path}
        rows.append(row)

        mean_rho = f"{agg['mean_rho']:.3f}" if agg["mean_rho"] is not None else "  -  "
        best_rho = f"{agg['best_rho']:.3f}" if agg["best_rho"] is not None else "  -  "
        print(f"{label['model']:<35} {label['split']:<7} {label['mode']:<25} "
              f"{agg['parse_pct']:>7.1f} {agg['valid_pct']:>7.1f} "
              f"{agg['sizing_ok_pct']:>7.1f} {agg['matching_ok_pct']:>7.1f} "
              f"{agg['exact_match_pct']:>7.1f} {mean_rho:>8} {best_rho:>8}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output",      default="results/paper_metrics.json")
    run(p.parse_args())
