"""
src/features.py
~~~~~~~~~~~~~~~
Feature engineering pipeline for the F1 2026 Race Outcome Predictor.

Input  : Raw combined DataFrame (all seasons, from data_loader.collect_season)
Output : X (features), y (target), weights (sample_weight)

Design decisions
----------------
- LEAKAGE_COLS imported from constants.py — not redefined here
- rolling_avg_finish_3 / rolling_dnf_rate_3 use .shift(1) (no target leakage)
- NaN rolling values filled with per-driver season average
- Final feature list written back to model_metadata.json["safe_feature_cols"]
- adaptation_score computed for 2026 only; 0.0 for pre-2026 rows
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from constants import (
    DATA_DIR,
    LEAKAGE_COLS,
    META_PATH,
    WEIGHT_2026,
    WEIGHT_PRE_2026,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT TYPE LOOKUP
# 0 = street circuit  |  1 = technical  |  2 = high-speed
# ─────────────────────────────────────────────────────────────────────────────
CIRCUIT_TYPE: dict[str, int] = {
    # Street circuits (0)
    "Monaco Grand Prix"            : 0,
    "Azerbaijan Grand Prix"        : 0,
    "Singapore Grand Prix"         : 0,
    "Saudi Arabian Grand Prix"     : 0,
    "Las Vegas Grand Prix"         : 0,
    "Miami Grand Prix"             : 0,

    # Technical circuits (1)
    "Hungarian Grand Prix"         : 1,
    "Spanish Grand Prix"           : 1,
    "Japanese Grand Prix"          : 1,
    "Abu Dhabi Grand Prix"         : 1,
    "Australian Grand Prix"        : 1,
    "Canadian Grand Prix"          : 1,
    "United States Grand Prix"     : 1,
    "São Paulo Grand Prix"         : 1,
    "Mexican Grand Prix"           : 1,
    "Chinese Grand Prix"           : 1,
    "Bahrain Grand Prix"           : 1,
    "Qatar Grand Prix"             : 1,

    # High-speed circuits (2)
    "British Grand Prix"           : 2,
    "Italian Grand Prix"           : 2,
    "Belgian Grand Prix"           : 2,
    "Austrian Grand Prix"          : 2,
    "Dutch Grand Prix"             : 2,
    "Emilia Romagna Grand Prix"    : 2,
    "French Grand Prix"            : 2,
}

CIRCUIT_TYPE_DEFAULT = 1  # fallback: treat unknown circuits as technical


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL FEATURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _add_grid_and_quali(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carry forward grid_position and quali_gap_to_pole (= gap_to_pole_s).

    gap_to_pole_s is renamed to quali_gap_to_pole for clarity.
    Pole sitter always has gap = 0.0 (enforced here in case of float drift).
    """
    df = df.copy()

    if "gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole"] = df["gap_to_pole_s"].clip(lower=0.0)
    else:
        df["quali_gap_to_pole"] = np.nan
        logger.warning("gap_to_pole_s column missing — quali_gap_to_pole set to NaN.")

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling_avg_finish_3 and rolling_dnf_rate_3 per driver.

    Both use .shift(1) so the current race result is NEVER in the window.
    NaN values (first races in a season) are filled with the driver's own
    season average for that metric.

    Rolling signal rationale (from EDA)
    ------------------------------------
    - rolling_avg_finish_3 : top correlated safe feature with top3_finish
    - rolling_dnf_rate_3   : DNF rate 2026 ≈ 12% vs 2025 ≈ 8%;
                             meaningful signal, especially for 2026 season
    """
    df = df.copy()
    df = df.sort_values(["driver_abbr", "year", "round"]).reset_index(drop=True)

    def _rolling_mean_shifted(series: pd.Series, window: int = 3) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    # ── rolling_avg_finish_3 ──────────────────────────────────────────────
    df["rolling_avg_finish_3"] = (
        df.groupby(["driver_abbr", "year"])["final_position"]
        .transform(_rolling_mean_shifted)
    )

    # ── rolling_dnf_rate_3 ───────────────────────────────────────────────
    # is_dnf must be numeric (ensured by data_loader, but coerce defensively)
    df["is_dnf"] = pd.to_numeric(df.get("is_dnf", 0), errors="coerce").fillna(0)
    df["rolling_dnf_rate_3"] = (
        df.groupby(["driver_abbr", "year"])["is_dnf"]
        .transform(_rolling_mean_shifted)
    )

    # ── Fill NaN with per-driver season average ───────────────────────────
    for feat in ["rolling_avg_finish_3", "rolling_dnf_rate_3"]:
        season_avg = df.groupby(["driver_abbr", "year"])[feat].transform("mean")
        df[feat] = df[feat].fillna(season_avg)
        # Secondary fallback: if whole season is NaN, fill with global median
        global_fallback = df[feat].median()
        df[feat] = df[feat].fillna(global_fallback)

    return df


def _add_constructor_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add constructor_rank: team's cumulative championship rank *before*
    each race (using points accumulated in prior rounds of the same season).

    Rank 1 = most points. Uses shift(1) per team per season to exclude
    the current race's points contribution.
    """
    df = df.copy()
    df = df.sort_values(["year", "round", "team"]).reset_index(drop=True)

    # Cumulative team points up to (but not including) current round
    team_pts = (
        df.groupby(["year", "team", "round"])["points"]
        .sum()
        .reset_index(name="round_team_pts")
    )
    team_pts = team_pts.sort_values(["year", "team", "round"])
    team_pts["cum_pts"] = (
        team_pts.groupby(["year", "team"])["round_team_pts"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    # Rank teams by cumulative points within each (year, round)
    team_pts["constructor_rank"] = (
        team_pts.groupby(["year", "round"])["cum_pts"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    df = df.merge(
        team_pts[["year", "team", "round", "constructor_rank"]],
        on=["year", "team", "round"],
        how="left",
    )

    # Round 1 always has rank NaN (no prior data) → fill with midpoint
    n_teams = df["team"].nunique()
    df["constructor_rank"] = df["constructor_rank"].fillna((n_teams + 1) / 2)

    return df


def _add_circuit_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map event_name → circuit_type integer encoding.

    0 = street circuit
    1 = technical circuit (default for unknowns)
    2 = high-speed circuit
    """
    df = df.copy()
    df["circuit_type"] = (
        df["event_name"]
        .map(CIRCUIT_TYPE)
        .fillna(CIRCUIT_TYPE_DEFAULT)
        .astype(int)
    )

    unmapped = df.loc[df["circuit_type"] == CIRCUIT_TYPE_DEFAULT, "event_name"].unique()
    if len(unmapped) > 0:
        logger.info(
            "circuit_type: %d unmapped event(s) defaulted to %d (technical): %s",
            len(unmapped), CIRCUIT_TYPE_DEFAULT, list(unmapped),
        )

    return df


def _add_adaptation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add adaptation_score for 2026 rows only (0.0 for pre-2026).

    Definition: team's average improvement in finishing position
    from Race 1 to the current race within 2026.

    Formula (per driver row at round R):
        avg_pos_race1  = team's mean final_position at round 1 (2026)
        avg_pos_recent = team's rolling mean final_position over last 3 rounds
        adaptation_score = avg_pos_race1 - avg_pos_recent

    Positive score → team has improved (lower position = better).
    Uses shift(1) data already computed in rolling_avg_finish_3.
    Set to 0.0 for pre-2026 (no regulation reset context).
    """
    df = df.copy()
    df["adaptation_score"] = 0.0

    if df_2026_mask := (df["year"] == 2026).any():
        mask_2026 = df["year"] == 2026

        # Team avg finishing position at round 1 (2026)
        round1_avg = (
            df[mask_2026 & (df["round"] == df.loc[mask_2026, "round"].min())]
            .groupby("team")["final_position"]
            .mean()
            .rename("team_pos_round1")
        )

        df = df.merge(round1_avg, on="team", how="left")

        # adaptation_score = baseline - current rolling avg (already shift(1))
        df.loc[mask_2026, "adaptation_score"] = (
            df.loc[mask_2026, "team_pos_round1"]
            - df.loc[mask_2026, "rolling_avg_finish_3"]
        ).fillna(0.0)

        df = df.drop(columns=["team_pos_round1"], errors="ignore")

    return df


def _add_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign sample_weight per row based on season year.

    2026 → 1.0 (primary signal, regulation reset year)
    pre-2026 → 0.4 (historical context, reduced trust)
    """
    df = df.copy()
    df["sample_weight"] = df["year"].apply(
        lambda y: WEIGHT_2026 if y == 2026 else WEIGHT_PRE_2026
    )
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add top3_finish binary target: 1 if final_position <= 3, else 0.
    NaN final_position (unclassified) → 0.
    """
    df = df.copy()
    df["final_position"] = pd.to_numeric(df["final_position"], errors="coerce")
    df["top3_finish"] = (df["final_position"] <= 3).fillna(False).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

#: Ordered list of model input features — excludes all LEAKAGE_COLS
FEATURE_COLS: list[str] = [
    "grid_position",
    "quali_gap_to_pole",
    "rolling_avg_finish_3",
    "rolling_dnf_rate_3",
    "constructor_rank",
    "circuit_type",
    "adaptation_score",
    "pit_stop_count",
]


def build_features(
    df_raw: pd.DataFrame,
    write_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the full feature engineering pipeline on a raw combined DataFrame.

    Steps
    -----
    1. Sort by driver + year + round
    2. Add grid_position + quali_gap_to_pole
    3. Add rolling_avg_finish_3 + rolling_dnf_rate_3 (shift=1, NaN-filled)
    4. Add constructor_rank (cumulative, shift=1)
    5. Add circuit_type (hardcoded lookup)
    6. Add adaptation_score (2026 only)
    7. Add sample_weight
    8. Add top3_finish target
    9. Drop rows where core features are all NaN (unrunnable races)
    10. Optionally write final feature list to model_metadata.json

    Parameters
    ----------
    df_raw : pd.DataFrame
        Combined raw DataFrame from data_loader.collect_season().
        Must contain: driver_abbr, team, year, round, event_name,
        final_position, grid_position, gap_to_pole_s, is_dnf,
        pit_stop_count, points.
    write_metadata : bool, default True
        If True, write FEATURE_COLS to model_metadata.json["safe_feature_cols"].

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (shape: n_samples × len(FEATURE_COLS)).
    y : pd.Series
        Binary target — top3_finish (0 or 1).
    weights : pd.Series
        Sample weights aligned with X and y.
    """
    logger.info("━━━ Feature Engineering Pipeline ━━━")
    logger.info("Input shape: %s", df_raw.shape)

    df = df_raw.copy()
    df = df.sort_values(["driver_abbr", "year", "round"]).reset_index(drop=True)

    # ── Step-by-step feature construction ────────────────────────────────
    df = _add_grid_and_quali(df)
    logger.info("✅ Step 1/7 — grid_position + quali_gap_to_pole")

    df = _add_rolling_features(df)
    logger.info("✅ Step 2/7 — rolling_avg_finish_3 + rolling_dnf_rate_3")

    df = _add_constructor_rank(df)
    logger.info("✅ Step 3/7 — constructor_rank")

    df = _add_circuit_type(df)
    logger.info("✅ Step 4/7 — circuit_type")

    df = _add_adaptation_score(df)
    logger.info("✅ Step 5/7 — adaptation_score")

    df = _add_sample_weight(df)
    logger.info("✅ Step 6/7 — sample_weight")

    df = _add_target(df)
    logger.info("✅ Step 7/7 — top3_finish target")

    # ── Confirm no LEAKAGE_COLS sneak into features ───────────────────────
    leaked = [c for c in FEATURE_COLS if c in LEAKAGE_COLS]
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED — remove these from FEATURE_COLS: {leaked}"
        )

    # ── Build X, y, weights ───────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing_features:
        logger.warning("Missing features (will be excluded): %s", missing_features)

    # Drop rows where ALL core features are NaN
    df_model = df.dropna(subset=available_features, how="all").copy()

    # Impute remaining NaN with column median (XGBoost handles NaN natively,
    # but explicit imputation improves SHAP stability)
    for col in available_features:
        null_n = df_model[col].isna().sum()
        if null_n > 0:
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val)
            logger.info(
                "  Imputed %d NaN in %-25s → median=%.3f",
                null_n, col, median_val,
            )

    X       = df_model[available_features].reset_index(drop=True)
    y       = df_model["top3_finish"].reset_index(drop=True)
    weights = df_model["sample_weight"].reset_index(drop=True)

    # ── Verify class balance ──────────────────────────────────────────────
    top3_rate = y.mean()
    logger.info(
        "Target distribution — top3=1: %.1f%% | top3=0: %.1f%%",
        top3_rate * 100, (1 - top3_rate) * 100,
    )
    if top3_rate < 0.10 or top3_rate > 0.25:
        logger.warning(
            "Unexpected top3 rate %.1f%% — check final_position parsing.",
            top3_rate * 100,
        )

    logger.info("Output  X : %s", X.shape)
    logger.info("Output  y : %s (dtype=%s)", y.shape, y.dtype)
    logger.info("Weights   : 2026=%.1f  pre-2026=%.1f", WEIGHT_2026, WEIGHT_PRE_2026)

    # ── Write final feature list to model_metadata.json ──────────────────
    if write_metadata:
        _update_metadata(available_features, X)

    return X, y, weights


def _update_metadata(feature_cols: list[str], X: pd.DataFrame) -> None:
    """
    Read existing model_metadata.json (if any), append feature engineering
    metadata, and write back.

    Written fields
    --------------
    safe_feature_cols      : ordered list of features used in X
    feature_dtypes         : dtype per feature
    feature_null_pct       : % null before imputation (informational)
    leakage_cols           : echoed from constants.LEAKAGE_COLS
    """
    metadata = {}
    if META_PATH.exists():
        try:
            metadata = json.loads(META_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("model_metadata.json is malformed — overwriting.")

    metadata["safe_feature_cols"] = feature_cols
    metadata["feature_dtypes"]    = {c: str(X[c].dtype) for c in feature_cols}
    metadata["leakage_cols"]      = LEAKAGE_COLS

    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("💾 Feature metadata written → %s", META_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    End-to-end feature engineering pipeline.

    Loads all available raw_YYYY.csv files, runs build_features(),
    and prints output shapes and feature statistics.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix for F1 predictor.")
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=[2022, 2023, 2024, 2025, 2026],
        help="Seasons to include (e.g. --years 2023 2024 2025 2026).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save X, y, weights to data/features.csv.",
    )
    args = parser.parse_args()

    # Load raw CSVs
    frames = []
    for yr in args.years:
        path = DATA_DIR / f"raw_{yr}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            logger.info("Loaded %s (%d rows)", path.name, len(frames[-1]))
        else:
            logger.warning("Missing %s — skipping.", path)

    if not frames:
        logger.error("No raw data found. Run data_loader.py first.")
        return

    df_raw = pd.concat(frames, ignore_index=True)

    X, y, weights = build_features(df_raw, write_metadata=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "━" * 55)
    print(f"{'FEATURE ENGINEERING SUMMARY':^55}")
    print("━" * 55)
    print(f"  Samples          : {len(X):,}")
    print(f"  Features         : {X.shape[1]}")
    print(f"  Top-3 rate       : {y.mean():.1%}")
    print(f"  2026 rows        : {(weights == WEIGHT_2026).sum():,}  (weight=1.0)")
    print(f"  Pre-2026 rows    : {(weights == WEIGHT_PRE_2026).sum():,}  (weight=0.4)")
    print("━" * 55)
    print("\n  Feature stats:")
    print(X.describe().round(3).to_string())

    if args.save:
        out_path = DATA_DIR / "features.csv"
        save_df  = X.copy()
        save_df["top3_finish"]   = y
        save_df["sample_weight"] = weights
        save_df.to_csv(out_path, index=False)
        logger.info("💾 Saved features → %s", out_path)


if __name__ == "__main__":
    main()