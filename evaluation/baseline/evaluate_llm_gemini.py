"""
Zero-shot evaluation — Google Gemini models.
Usage:
  python3 evaluation/baseline/evaluate_llm_gemini.py \
      --index dataset/small/index.json --split small [--model gemini-2.5-flash]
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import google.generativeai as genai
from dotenv import load_dotenv

from evaluation.eval_utils import (
    load_index, load_instance, load_solution,
    build_zero_shot_prompt, save_results, summarize, with_backoff,
)
from core.validate_output import parse_output, validate_output, split_output
from core.placement_objective import compute_all_metrics

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def run(args):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_obj = genai.GenerativeModel(args.model)
    index     = load_index(args.index)
    model     = args.model

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"{ts}_{model}_sizing_{args.split}_zero_shot")
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

        prompt = build_zero_shot_prompt(inst)
        print(f"  {entry['id']} {inst['name']:<30s}", end=" ", flush=True)

        def call():
            return model_obj.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0),
            )

        try:
            resp   = with_backoff(call)
            text   = resp.text
            parsed = parse_output(text)
            val    = validate_output(parsed, inst)

            if parsed:
                placement, sizing = split_output(parsed)
                ref_placement = {tid: tuple(rc) for tid, rc in sol["placement"].items()}
                ref_sizing    = sol["sizing"]
                metrics = compute_all_metrics(placement, sizing, topo, ref_placement, ref_sizing)
            else:
                metrics = {"cost": float("inf"), "rho": float("inf"),
                           "sizing_violations": -1, "matching_violations": -1,
                           "exact_match": False}

            record = {
                "id": entry["id"], "name": inst["name"],
                "parse_ok": parsed is not None,
                "valid": val["valid"],
                **metrics,
                "validation": val,
                "raw_response": text,
            }
        except Exception as e:
            record = {"id": entry["id"], "name": inst["name"],
                      "parse_ok": False, "valid": False, "error": str(e)}

        rho = record.get("rho", "ERR")
        rho_str = f"{rho:.3f}" if isinstance(rho, float) else str(rho)
        print(f"ρ={rho_str}  exact={record.get('exact_match', False)}")
        results.append(record)

    out_path = os.path.join(out_dir, "results.json")
    save_results(results, out_path)
    summarize(results, f"{model} | {args.split} | zero-shot")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", required=True, choices=["small", "large"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    run(p.parse_args())
