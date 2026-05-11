"""
src/constants.py
~~~~~~~~~~~~~~~~
Centralised constants shared across the F1 2026 Race Outcome Predictor pipeline.
Import from here — never redefine in individual modules.
"""

# ── Sample weights (2026 regulation reset design decision) ────────────────
WEIGHT_2026     = 1.0
WEIGHT_PRE_2026 = 0.4

# ── Target leakage columns — NEVER use as model features ─────────────────
# Source of truth; also written to model_metadata.json by 01_EDA.ipynb
LEAKAGE_COLS = [
    "final_position",   # direct source of top3 label
    "points",           # awarded after race result
    "top3",             # the target variable itself
    "is_dnf",           # race outcome, determined during race
    "fastest_lap_s",    # measured during race
]

# ── Paths ─────────────────────────────────────────────────────────────────
from pathlib import Path

ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
MODEL_DIR      = ROOT_DIR
META_PATH      = ROOT_DIR / "model_metadata.json"
MODEL_PKL_PATH = ROOT_DIR / "model.pkl"