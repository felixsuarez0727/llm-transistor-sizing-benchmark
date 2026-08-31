"""
Sky130 PDK transistor sizing library.

Discrete W/L options available in the Sky130 process node (1.8V devices).
All dimensions in micrometers (μm).
"""

# ── Discrete width options (μm) ───────────────────────────────────────────────
NFET_W_OPTIONS = [0.42, 0.84, 1.26, 1.68, 2.10, 4.20]
PFET_W_OPTIONS = [0.42, 0.84, 1.26, 1.68, 2.10, 4.20]
L_OPTIONS      = [0.15, 0.18, 0.35, 0.50, 1.00]

# ── Minimum sizes ─────────────────────────────────────────────────────────────
NFET_W_MIN = 0.42
PFET_W_MIN = 0.42
L_MIN      = 0.15
L_MAX      = 1.00

# ── Device models ─────────────────────────────────────────────────────────────
NFET_MODEL = "sky130_fd_pr__nfet_01v8"
PFET_MODEL = "sky130_fd_pr__pfet_01v8"

# ── Recommended sizing presets for analog roles ───────────────────────────────
# (role) -> {W, L} in μm — used by the reference solver
ROLE_PRESETS = {
    # Differential pair: long L for matching and low 1/f noise
    "diff_pair_nmos":   {"W": 2.10, "L": 0.50},
    "diff_pair_pmos":   {"W": 2.10, "L": 0.50},
    # Current mirror: same L as matched device
    "mirror_nmos":      {"W": 2.10, "L": 0.50},
    "mirror_pmos":      {"W": 2.10, "L": 0.50},
    # Tail current source
    "tail_nmos":        {"W": 4.20, "L": 0.50},
    "tail_pmos":        {"W": 4.20, "L": 0.50},
    # Load devices
    "load_pmos":        {"W": 2.10, "L": 0.35},
    "load_nmos":        {"W": 2.10, "L": 0.35},
    # Output stage / driver
    "output_nmos":      {"W": 4.20, "L": 0.18},
    "output_pmos":      {"W": 4.20, "L": 0.18},
    # Bias / cascode
    "bias_nmos":        {"W": 1.68, "L": 0.35},
    "bias_pmos":        {"W": 1.68, "L": 0.35},
    # Simple inverter pull-up / pull-down
    "inv_pmos":         {"W": 1.68, "L": 0.18},
    "inv_nmos":         {"W": 0.84, "L": 0.18},
    # Logic (NAND, NOR)
    "logic_pmos":       {"W": 1.26, "L": 0.15},
    "logic_nmos":       {"W": 0.84, "L": 0.15},
}


def nearest_valid(W: float, L: float, fet_type: str = "nmos") -> dict:
    """Snap W and L to the nearest discrete Sky130 values."""
    w_opts = NFET_W_OPTIONS if fet_type == "nmos" else PFET_W_OPTIONS
    best_W = min(w_opts, key=lambda x: abs(x - W))
    best_L = min(L_OPTIONS, key=lambda x: abs(x - L))
    return {"W": best_W, "L": best_L}


def is_valid_size(W: float, L: float, fet_type: str = "nmos") -> bool:
    w_opts = NFET_W_OPTIONS if fet_type == "nmos" else PFET_W_OPTIONS
    return W in w_opts and L in L_OPTIONS


def all_sizes(fet_type: str = "nmos") -> list:
    """Return all valid (W, L) pairs for a given FET type."""
    w_opts = NFET_W_OPTIONS if fet_type == "nmos" else PFET_W_OPTIONS
    return [{"W": w, "L": l} for w in w_opts for l in L_OPTIONS]
