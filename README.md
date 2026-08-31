# LLM Transistor Sizing

**IMSE-CNM (CSIC / Universidad de Sevilla)**  
Félix D. Suárez Bonilla · Jose M. de la Rosa · G. Liñán-Cembrano

---

## What This Is

A systematic benchmark evaluating whether frontier LLMs can jointly solve **transistor placement + sizing** on small discrete grids, using real Sky130 PDK discrete W/L values.

> **This is NOT industrial layout.** No DRC, no parasitic extraction, no PDK rule checks.  
> The task is: given a circuit netlist, assign each transistor a grid cell AND a valid W/L size from the Sky130 library, while satisfying matching constraints.

This project is a **companion to our placement benchmark** (`llm-transistor-placement`), extending the problem by adding the sizing dimension.

### Key difference from the placement-only benchmark

| | Placement only | Placement + Sizing (this project) |
|---|---|---|
| **LLM decides** | Where to put each transistor | Where + what W/L size |
| **Transistor sizes** | All identical (unit cell) | Discrete Sky130 W/L from PDK |
| **Extra constraints** | None | Matching groups, size bounds per device |
| **LLM output** | `{"M1": [r,c], ...}` | `{"M1": {"row":r, "col":c, "W":2.10, "L":0.50}, ...}` |

---

## Technology: Sky130 PDK

Discrete options available (1.8V devices):

| Parameter | Options (μm) |
|---|---|
| NMOS width W | 0.42, 0.84, 1.26, 1.68, 2.10, 4.20 |
| PMOS width W | 0.42, 0.84, 1.26, 1.68, 2.10, 4.20 |
| Length L | 0.15, 0.18, 0.35, 0.50, 1.00 |

---

## Problem Formulation

Given:
- A grid of R×C cells
- A list of transistors (NMOS/PMOS) with their netlist connections
- Sizing bounds per transistor (W_min, W_max, L_min, L_max)
- Matching groups (transistors that must have identical W and L)

Find for each transistor:
1. A unique grid cell `(row, col)`
2. A valid discrete Sky130 size `{W, L}` within its bounds

Minimizing:

```
C = α·HPWL_norm + β·SizingViolations + γ·MatchingViolations
    (1.0)          (10.0)               (20.0 per pair)
```

**Cost ratio:** ρ = C_LLM / C_reference (lower is better, 1.0 = reference quality)

---

## Benchmark

### Small (2×3 grid, 2–6 transistors) — 20 instances
Inverter, NAND2, NOR2, current mirrors, differential pairs, cascodes, latches, buffers, Schmitt trigger, transmission gate, and more.

### Large (3×4 grid, 5–9 transistors) — 20 instances
OTA 5T, two-stage Miller OTA, folded-cascode OTA, telescopic OTA, recycling FC-OTA, StrongARM comparator, bandgap core, LDO core, charge pump, ring VCO, CMFB amp, super mirror, and more.

---

## Repository Structure

```
llm-transistor-sizing/
├── technology/
│   └── sky130_lib.py              # Sky130 discrete W/L options + role presets
├── dataset_gen/
│   ├── circuit_topologies.py      # 40 circuit definitions (20 small + 20 large)
│   └── build_dataset.py           # Generate instances + reference solutions
├── dataset/
│   ├── small/  instances/ + solutions/ + index.json
│   └── large/  instances/ + solutions/ + index.json
├── core/
│   ├── placement_objective.py     # Cost function: HPWL + sizing + matching violations
│   ├── validate_output.py         # Parse + validate LLM JSON output
│   └── reference_solver.py        # Hill-climbing placement + Sky130 preset sizing
├── evaluation/
│   ├── eval_utils.py              # Prompt builders + shared utilities
│   ├── baseline/                  # Zero-shot (one call per circuit)
│   │   ├── evaluate_llm_claude.py
│   │   ├── evaluate_llm_chatgpt.py
│   │   ├── evaluate_llm_gemini.py
│   │   └── evaluate_llm_deepseek.py
│   └── improved/                  # Iterative (up to 3 rounds with feedback)
│       ├── evaluate_llm_claude_iterative.py
│       ├── evaluate_llm_chatgpt_iterative.py
│       ├── evaluate_llm_gemini_iterative.py
│       └── evaluate_llm_deepseek_iterative.py
├── tools/
│   └── aggregate_metrics.py       # Aggregate results → paper table
├── results/                       # Evaluation outputs (auto-generated)
├── requirements.txt
└── .env                           # API keys (not committed)
```

---

## Setup

```bash
cd llm-transistor-sizing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Fill in `.env` with your API keys:
```
ANTHROPIC_API_KEY=...
CHATGPT_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

---

## Workflow

### 1. Generate the dataset
```bash
python3 dataset_gen/build_dataset.py
```

### 2. Zero-shot evaluation
```bash
python3 evaluation/baseline/evaluate_llm_claude.py   --index dataset/small/index.json --split small
python3 evaluation/baseline/evaluate_llm_chatgpt.py  --index dataset/small/index.json --split small
python3 evaluation/baseline/evaluate_llm_gemini.py   --index dataset/small/index.json --split small
python3 evaluation/baseline/evaluate_llm_deepseek.py --index dataset/small/index.json --split small

# Repeat with --index dataset/large/index.json --split large
```

### 3. Iterative evaluation (3 rounds)
```bash
python3 evaluation/improved/evaluate_llm_claude_iterative.py \
    --index dataset/small/index.json --split small --rounds 3
```

### 4. Aggregate results
```bash
python3 tools/aggregate_metrics.py --results-dir results --output results/paper_metrics.json
```

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| **Parse (%)** | Fraction of circuits with valid JSON output |
| **Valid (%)** | Parse OK + in-bounds positions + no cell conflicts + valid Sky130 sizes |
| **SizingOK (%)** | All transistors within their W/L bounds |
| **MatchingOK (%)** | All matched groups have identical W and L |
| **Exact match (%)** | Same placement AND same sizing as reference, zero violations |
| **ρ (cost ratio)** | C_LLM / C_reference (lower = better, 1.0 = reference quality) |

---

## Models Evaluated

- **Anthropic:** Claude Sonnet 4.6, Claude Opus 4.5
- **OpenAI:** GPT-4o, GPT-4.1, GPT-4.1-mini
- **Google:** Gemini 2.5-Flash, Gemini 2.5-Pro
- **DeepSeek:** DeepSeek-Chat, DeepSeek-Reasoner

All at temperature=0 for reproducibility.

---

## Related Projects

- **LLM Transistor Placement** — `felixsuarez0727/llm-transistor-placement`  
  Placement only (uniform transistor sizes, 2×3 and 3×4 grids)

- **LLM Toy Routing** — `felixsuarez0727/llm-routing-research`  
  Wire routing on pre-placed pin positions

---

## Funding

This work was supported by:
- European Union NextGenerationEU / Red.es
- PID2022-138078OB-I00 (MICIU/AEI)
- PDC2023-145808-I00
- USECHIP (TSI-069100-2023-001)
