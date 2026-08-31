"""
Iterative evaluation — OpenAI GPT models.
Usage:
  python3 evaluation/improved/evaluate_llm_chatgpt_iterative.py \
      --index dataset/small/index.json --split small --rounds 3 [--model gpt-4o]
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import openai
from dotenv import load_dotenv

from evaluation.eval_utils import (
    load_index, load_instance, load_solution,
    build_zero_shot_prompt, build_iterative_prompt,
    save_results, summarize, with_backoff,
)
from core.validate_output import parse_output, validate_output, split_output
from core.placement_objective import compute_all_metrics

load_dotenv()

DEFAULT_MODEL = "gpt-4o"


def run(args):
    client   = openai.OpenAI(api_key=os.environ["CHATGPT_API_KEY"])
    index    = load_index(args.index)
    model    = args.model
    n_rounds = args.rounds

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"{ts}_{model}_sizing_{args.split}_iterative_r{n_rounds}")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for entry in index:
        inst = load_instance(entry["instance_path"])
        sol  = load_solution(entry["solution_path"])
        topo = {
            "name": inst["name"], "grid": tuple(inst["grid"].values()),
            "transistors": inst["transistors"],
            "sizing_constraints": inst["sizing_constraints"],
            "reference_sizing": sol["sizing"],
        }
        ref_placement = {tid: tuple(rc) for tid, rc in sol["placement"].items()}
        ref_sizing    = sol["sizing"]

        print(f"  {entry['id']} {inst['name']:<30s}", end=" ", flush=True)

        best_record = None
        best_rho    = float("inf")
        prompt      = build_zero_shot_prompt(inst)

        for rnd in range(1, n_rounds + 1):
            def call(p=prompt):
                return client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": p}],
                )

            try:
                resp   = with_backoff(call)
                text   = resp.choices[0].message.content
                parsed = parse_output(text)
                val    = validate_output(parsed, inst)

                if parsed:
                    placement, sizing = split_output(parsed)
                    metrics = compute_all_metrics(placement, sizing, topo, ref_placement, ref_sizing)
                else:
                    metrics = {"cost": float("inf"), "rho": float("inf"),
                               "sizing_violations": -1, "matching_violations": -1,
                               "exact_match": False}

                record = {
                    "id": entry["id"], "name": inst["name"],
                    "round": rnd,
                    "parse_ok": parsed is not None,
                    "valid": val["valid"],
                    **metrics, "validation": val,
                }

                rho = metrics.get("rho", float("inf"))
                if isinstance(rho, float) and rho < best_rho:
                    best_rho    = rho
                    best_record = record

                if metrics.get("exact_match"):
                    break
                if parsed and rnd < n_rounds:
                    prompt = build_iterative_prompt(inst, parsed, metrics, val, rnd + 1)

            except Exception as e:
                record = {"id": entry["id"], "name": inst["name"],
                          "round": rnd, "parse_ok": False, "valid": False, "error": str(e)}
                if best_record is None:
                    best_record = record
                break

        if best_record is None:
            best_record = {"id": entry["id"], "name": inst["name"],
                           "parse_ok": False, "valid": False, "error": "no response"}

        rho_str = f"{best_rho:.3f}" if best_rho < float("inf") else "inf"
        print(f"ρ={rho_str}  exact={best_record.get('exact_match', False)}  rounds={best_record.get('round', '?')}")
        results.append(best_record)

    out_path = os.path.join(out_dir, "results.json")
    save_results(results, out_path)
    summarize(results, f"{model} | {args.split} | iterative r{n_rounds}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", required=True, choices=["small", "large"])
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--rounds", type=int, default=3)
    run(p.parse_args())
