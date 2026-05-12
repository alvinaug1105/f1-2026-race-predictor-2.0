"""
src/predict.py  (Final)
~~~~~~~~~~~~~~~~~~~~~~~~
Inference module for the F1 2026 Race Outcome Predictor.

Key design decisions
--------------------
win_probability
    model.predict_proba()[:, 1] represents raw podium probability per driver.
    Normalised across all drivers (sum=1) to create an interpretable
    "share of podium probability" — useful for Streamlit display and
    comparison across drivers.

rolling features for future races
    No .shift(1) needed — we ARE predicting the next race, so using the
    last 3 completed races directly is correct (no leakage possible).

Feature alignment
    FEATURE_COLS loaded from model_metadata.json["safe_feature_cols"] to
    guarantee exact column order match with training matrix.

grid_csv_path  (Bug #1 fix)
    Optional path to an uploaded qualifying CSV from app.py.
    When supplied, FastF1 qualifying fetch is skipped entirely.
    Column alias "gap_to_pole" is normalised to "gap_to_pole_s" automatically.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import DATA_DIR, META_PATH, MODEL_PKL_PATH
from features import CIRCUIT_TYPE, CIRCUIT_TYPE_DEFAULT

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── FastF1 optional import + cache setup ──────────────────────────────────────
try:
    import fastf1  # type: ignore
    _CACHE_DIR = DATA_DIR / "cache"
    if _CACHE_DIR.exists():
        fastf1.Cache.enable_cache(str(_CACHE_DIR))
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    logger.warning("fastf1 not installed — live quali fetch unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# ARTEFACT LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load serialised XGBClassifier from model.pkl."""
    if not MODEL_PKL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PKL_PATH}. "
            "Run: python src/model.py"
        )
    model = joblib.load(MODEL_PKL_PATH)
    logger.info("Model loaded from %s", MODEL_PKL_PATH)
    return model


def _load_feature_cols() -> list:
    """
    Read exact feature list from model_metadata.json["safe_feature_cols"].
    Guarantees column order matches training matrix.
    Falls back to a sensible default list if metadata is unavailable.
    """
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            cols = meta.get("safe_feature_cols")
            if cols:
                logger.info("Feature cols from metadata: %s", cols)
                return cols
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read metadata: %s", exc)

    fallback = [
        "grid_position",
        "quali_gap_to_pole",
        "rolling_avg_finish_3",
        "rolling_dnf_rate_3",
        "constructor_rank",
        "circuit_type",
        "adaptation_score",
        "pit_stop_count",
    ]
    logger.warning("safe_feature_cols not in metadata — using fallback: %s", fallback)
    return fallback


def _load_history(years=None) -> pd.DataFrame:
    """Load all completed race CSVs for rolling feature computation."""
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]
    frames = []
    for yr in years:
        p = DATA_DIR / f"raw_{yr}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        logger.warning("No history CSVs found — rolling features will be NaN.")
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    for col in ["final_position", "is_dnf", "points", "grid_position",
                "gap_to_pole_s", "pit_stop_count"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

_COL_ALIASES = {
    "gap_to_pole"       : "gap_to_pole_s",
    "gap_s"             : "gap_to_pole_s",
    "qual_gap"          : "gap_to_pole_s",
    "quali_gap"         : "gap_to_pole_s",
    "abbreviation"      : "driver_abbr",
    "driver_code"       : "driver_abbr",
    "driver"            : "driver_abbr",
    "teamname"          : "team",
    "team_name"         : "team",
    "constructor"       : "team",
    "starting_position" : "grid_position",
    "start_pos"         : "grid_position",
    "position"          : "grid_position",
}


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column name variants to canonical names."""
    return df.rename(columns={k: v for k, v in _COL_ALIASES.items()
                               if k in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# FASTF1 QUALIFYING FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_quali_data(year: int, race_name: str) -> pd.DataFrame:
    """
    Fetch qualifying session for an upcoming race via FastF1.

    Extracts per-driver: driver_abbr, team, grid_position,
    gap_to_pole_s (qualifying gap).
    """
    if not FASTF1_AVAILABLE:
        raise RuntimeError(
            "fastf1 is not installed. "
            "Install with: pip install fastf1==3.4.0 "
            "or upload a Grid CSV manually."
        )

    try:
        session = fastf1.get_session(year, race_name, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        logger.info("Quali loaded: %s %d", race_name, year)
    except Exception as exc:
        logger.error("Failed to load quali for %s %d: %s", race_name, year, exc)
        raise

    results = session.results.copy()
    results.columns = [c.strip() for c in results.columns]

    rename = {
        "DriverNumber" : "driver_number",
        "Abbreviation" : "driver_abbr",
        "TeamName"     : "team",
        "GridPosition" : "grid_position",
    }
    results = results.rename(columns={k: v for k, v in rename.items()
                                       if k in results.columns})

    results["grid_position"] = pd.to_numeric(
        results["grid_position"], errors="coerce"
    ).replace(0, np.nan)
    results["driver_number"] = results["driver_number"].astype(str)

    laps = session.laps.copy()
    best_laps = (
        laps.groupby("DriverNumber")["LapTime"]
        .min().dt.total_seconds()
        .reset_index()
        .rename(columns={"DriverNumber": "driver_number", "LapTime": "quali_best_lap_s"})
    )
    best_laps["driver_number"] = best_laps["driver_number"].astype(str)
    pole_time = best_laps["quali_best_lap_s"].min()

    if pd.isna(pole_time):
        logger.warning("No valid pole time — gap_to_pole_s = NaN for all drivers.")
        best_laps["gap_to_pole_s"] = np.nan
    else:
        best_laps["gap_to_pole_s"] = (
            best_laps["quali_best_lap_s"] - pole_time
        ).clip(lower=0.0).round(4)

    df   = results.merge(best_laps, on="driver_number", how="left")
    keep = [c for c in ["driver_number", "driver_abbr", "team",
                         "grid_position", "gap_to_pole_s", "quali_best_lap_s"]
            if c in df.columns]
    return df[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV QUALIFYING LOADER  (Bug #1 fix — grid_csv_path)
# ─────────────────────────────────────────────────────────────────────────────

def _load_grid_csv(grid_csv_path: str) -> pd.DataFrame:
    """
    Load qualifying/grid data from an uploaded CSV file.

    Accepts both "gap_to_pole" and "gap_to_pole_s" column names.
    Minimum required columns: driver_abbr (or alias), team, grid_position.
    """
    try:
        df = pd.read_csv(grid_csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read grid CSV at '{grid_csv_path}': {exc}") from exc

    if df.empty:
        raise ValueError("Uploaded grid CSV is empty.")

    df = _normalise_cols(df)

    required = {"driver_abbr", "team", "grid_position"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Grid CSV missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")

    if "gap_to_pole_s" not in df.columns:
        logger.warning("gap_to_pole_s not in CSV — will be imputed from history median.")
        df["gap_to_pole_s"] = np.nan

    logger.info("Grid CSV loaded: %d drivers, columns: %s", len(df), list(df.columns))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING FEATURES  (no shift — predicting future race)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_features(
    history: pd.DataFrame,
    target_year: int,
    target_round: int,
    window: int = 3,
) -> pd.DataFrame:
    """
    Compute rolling_avg_finish_3 and rolling_dnf_rate_3 from last `window`
    completed races before the target round.

    No .shift(1) required — the target race has not happened yet,
    so there is no current-race result to leak.
    """
    if history.empty:
        return pd.DataFrame(columns=[
            "driver_abbr", "rolling_avg_finish_3", "rolling_dnf_rate_3"
        ])

    prior = history[
        (history["year"] < target_year) |
        ((history["year"] == target_year) & (history["round"] < target_round))
    ].sort_values(["driver_abbr", "year", "round"])

    if prior.empty:
        logger.warning("No prior races found — rolling features will be NaN.")
        return pd.DataFrame(columns=[
            "driver_abbr", "rolling_avg_finish_3", "rolling_dnf_rate_3"
        ])

    def _last_n_mean(grp: pd.DataFrame, col: str) -> float:
        vals = grp.sort_values(["year", "round"])[col].dropna().tail(window)
        return float(vals.mean()) if len(vals) > 0 else np.nan

    roll = (
        prior.groupby("driver_abbr", group_keys=False)
        .apply(lambda g: pd.Series({
            "rolling_avg_finish_3": _last_n_mean(g, "final_position"),
            "rolling_dnf_rate_3"  : _last_n_mean(
                g.assign(is_dnf=pd.to_numeric(g["is_dnf"], errors="coerce").fillna(0))
                if "is_dnf" in g.columns else g.assign(is_dnf=0),
                "is_dnf",
            ),
        }))
        .reset_index()
    )

    med_pos = prior["final_position"].median() if "final_position" in prior.columns else 10.0
    med_dnf = (pd.to_numeric(prior["is_dnf"], errors="coerce").median()
               if "is_dnf" in prior.columns else 0.0)
    roll["rolling_avg_finish_3"] = roll["rolling_avg_finish_3"].fillna(med_pos)
    roll["rolling_dnf_rate_3"]   = roll["rolling_dnf_rate_3"].fillna(med_dnf)
    return roll


def _constructor_rank(
    history: pd.DataFrame,
    target_year: int,
    target_round: int,
) -> pd.DataFrame:
    """Team constructor rank based on cumulative points before target round."""
    prior = history[
        (history["year"] == target_year) &
        (history["round"] < target_round)
    ] if not history.empty else pd.DataFrame()

    if prior.empty or "team" not in (prior.columns if not prior.empty else []):
        teams    = history["team"].dropna().unique() if not history.empty else []
        mid_rank = (len(teams) + 1) / 2 if len(teams) > 0 else 5.5
        return pd.DataFrame({"team": teams, "constructor_rank": mid_rank})

    team_pts = (
        prior.groupby("team")["points"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    team_pts["constructor_rank"] = range(1, len(team_pts) + 1)
    return team_pts[["team", "constructor_rank"]]


def _adaptation_score(
    history: pd.DataFrame,
    target_year: int,
    target_round: int,
) -> pd.DataFrame:
    """
    Carry forward latest adaptation_score per team for 2026.
    Pre-2026 teams default to 0.0.
    """
    if history.empty or "team" not in history.columns:
        return pd.DataFrame(columns=["team", "adaptation_score"])

    if target_year != 2026 or "adaptation_score" not in history.columns:
        teams = history["team"].dropna().unique()
        return pd.DataFrame({"team": teams, "adaptation_score": 0.0})

    prior = history[
        (history["year"] == target_year) &
        (history["round"] < target_round)
    ]
    if prior.empty:
        teams = history["team"].dropna().unique()
        return pd.DataFrame({"team": teams, "adaptation_score": 0.0})

    latest = (
        prior.sort_values("round")
        .groupby("team")["adaptation_score"]
        .last()
        .reset_index()
    )
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_pred_features(
    quali_df: pd.DataFrame,
    history: pd.DataFrame,
    target_year: int,
    target_round: int,
    event_name: str,
    feature_cols: list,
) -> pd.DataFrame:
    """Assemble prediction feature matrix aligned to training FEATURE_COLS."""
    df = quali_df.copy()

    # quali_gap_to_pole — canonical feature name used in training
    if "gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole"] = df["gap_to_pole_s"].clip(lower=0.0)
    elif "quali_gap_to_pole" not in df.columns:
        df["quali_gap_to_pole"] = np.nan

    # Rolling driver features
    if not history.empty and "driver_abbr" in df.columns:
        roll = _rolling_features(history, target_year, target_round)
        df   = df.merge(roll, on="driver_abbr", how="left")

    # Constructor rank
    if not history.empty and "team" in df.columns:
        cr = _constructor_rank(history, target_year, target_round)
        df = df.merge(cr, on="team", how="left")

    # Circuit type
    df["circuit_type"] = CIRCUIT_TYPE.get(event_name, CIRCUIT_TYPE_DEFAULT)

    # Adaptation score
    if not history.empty and "team" in df.columns:
        adapt = _adaptation_score(history, target_year, target_round)
        df    = df.merge(adapt, on="team", how="left")
    if "adaptation_score" not in df.columns:
        df["adaptation_score"] = 0.0
    df["adaptation_score"] = df["adaptation_score"].fillna(0.0)

    # Impute remaining NaN with history median (or 0 if no history)
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature '%s' missing — filling 0.", col)
            df[col] = 0.0
        null_n = int(df[col].isna().sum())
        if null_n > 0:
            fill = float(history[col].median()) if (
                not history.empty and col in history.columns
            ) else 0.0
            df[col] = df[col].fillna(fill)
            logger.info("Imputed %d NaN in %-25s with %.3f", null_n, col, fill)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTED POSITION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _assign_predicted_positions(podium_prob: np.ndarray) -> list:
    """
    Convert podium probabilities to ordinal predicted finishing positions.
    Highest probability → position 1. Ties broken by stable sort.
    """
    order     = np.argsort(-podium_prob, kind="stable")
    positions = np.empty(len(podium_prob), dtype=int)
    positions[order] = np.arange(1, len(podium_prob) + 1)
    return positions.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict_race(
    year: int,
    race_name: str,
    grid_csv_path: Optional[str] = None,
    history_years=None,
) -> pd.DataFrame:
    """
    Predict podium probabilities for all drivers in an upcoming race.

    Parameters
    ----------
    year : int
        Season year (e.g. 2026).
    race_name : str
        FastF1 event name or round number (e.g. "British Grand Prix" or 12).
    grid_csv_path : str, optional
        Path to an uploaded qualifying/grid CSV (temp file from app.py).
        When supplied, FastF1 qualifying fetch is skipped entirely.
        Minimum columns: driver_abbr, team, grid_position, gap_to_pole_s
        Column alias "gap_to_pole" is accepted and auto-renamed.
    history_years : list[int], optional
        Seasons to use for rolling feature computation.
        Defaults to [2022, 2023, 2024, 2025, 2026].

    Returns
    -------
    pd.DataFrame
        One row per driver, sorted by podium_probability descending.
        Columns: driver, team, grid_position, podium_probability,
                 win_probability, predicted_position

    Example
    -------
    >>> df = predict_race(2026, "British Grand Prix")
    >>> df = predict_race(2026, "Canadian Grand Prix", grid_csv_path="/tmp/grid.csv")
    """
    model        = _load_model()
    feature_cols = _load_feature_cols()
    history      = _load_history(history_years)

    # ── Source qualifying / grid data ─────────────────────────────────────────
    if grid_csv_path is not None:
        logger.info("Using uploaded grid CSV: %s", grid_csv_path)
        quali_df = _load_grid_csv(grid_csv_path)
    else:
        logger.info("Fetching qualifying data from FastF1: %s %d", race_name, year)
        quali_df = _fetch_quali_data(year, race_name)

    # ── Determine target round for rolling features ───────────────────────────
    if history.empty or "year" not in history.columns or "round" not in history.columns:
        target_round = 1
    else:
        year_hist    = history[history["year"] == year]
        target_round = int(year_hist["round"].max()) + 1 if not year_hist.empty else 1

    event_name = race_name if isinstance(race_name, str) else str(race_name)

    # ── Build feature matrix ──────────────────────────────────────────────────
    pred_df = _build_pred_features(
        quali_df, history, year, target_round, event_name, feature_cols
    )

    X_pred = pred_df[feature_cols].copy()

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        podium_prob = model.predict_proba(X_pred)[:, 1]
    except Exception as exc:
        logger.warning("predict_proba failed (%s) — falling back to predict()", exc)
        raw         = model.predict(X_pred).astype(float)
        podium_prob = raw / raw.max() if raw.max() > 0 else raw

    prob_sum = float(podium_prob.sum())
    win_prob = (
        podium_prob / prob_sum if prob_sum > 0
        else np.ones(len(podium_prob)) / len(podium_prob)
    )

    predicted_pos = _assign_predicted_positions(podium_prob)

    # ── Assemble output ───────────────────────────────────────────────────────
    output = pd.DataFrame({
        "driver"             : pred_df["driver_abbr"].values,
        "team"               : pred_df["team"].values,
        "grid_position"      : pred_df["grid_position"].values,
        "podium_probability" : podium_prob.round(4),
        "win_probability"    : win_prob.round(4),
        "predicted_position" : predicted_pos,
    })
    output = output.sort_values("podium_probability", ascending=False).reset_index(drop=True)

    logger.info("Prediction complete: %s %d", race_name, year)
    for _, row in output.head(5).iterrows():
        logger.info(
            "  P%-2d  %-4s  %-22s  podium=%.3f  win_share=%.3f",
            int(row["predicted_position"]), str(row["driver"]),
            str(row["team"]), float(row["podium_probability"]),
            float(row["win_probability"]),
        )
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict F1 race podium.")
    parser.add_argument("--year",          type=int, required=True)
    parser.add_argument("--race",          type=str, required=True,
                        help='e.g. "British Grand Prix" or round number')
    parser.add_argument("--grid-csv",      type=str, default=None,
                        help="Optional path to uploaded qualifying CSV")
    parser.add_argument("--history-years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()

    result = predict_race(
        year          = args.year,
        race_name     = args.race,
        grid_csv_path = args.grid_csv,
        history_years = args.history_years,
    )
    print("\n" + result.to_string(index=False))
