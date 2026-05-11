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
"""

import json
import logging
from pathlib import Path
from typing import Optional

import fastf1
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

# FastF1 cache
# FastF1 cache — disabled on Streamlit Cloud (no persistent disk)
import os
if os.path.exists(str(DATA_DIR / "cache")):
    fastf1.Cache.enable_cache(str(DATA_DIR / "cache"))
# else: no cache — FastF1 still works, just slower


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
    logger.info("✅ Model loaded from %s", MODEL_PKL_PATH)
    return model


def _load_feature_cols() -> list[str]:
    """
    Read exact feature list from model_metadata.json["safe_feature_cols"].
    Guarantees column order matches training matrix.
    """
    if not META_PATH.exists():
        raise FileNotFoundError(
            "model_metadata.json not found. Run features.py + model.py first."
        )
    meta = json.loads(META_PATH.read_text())
    cols = meta.get("safe_feature_cols")
    if not cols:
        raise KeyError("safe_feature_cols missing from model_metadata.json.")
    logger.info("Feature cols from metadata: %s", cols)
    return cols


def _load_history(years: list[int] | None = None) -> pd.DataFrame:
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
# FASTF1 QUALIFYING FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_quali_data(year: int, race_name: str) -> pd.DataFrame:
    """
    Fetch qualifying session for an upcoming race via FastF1.

    Extracts per-driver: driver_abbr, team, grid_position,
    gap_to_pole_s (qualifying gap).

    Parameters
    ----------
    year : int
    race_name : str
        Event name or round number accepted by FastF1.

    Returns
    -------
    pd.DataFrame
        One row per driver with qualifying features.
    """
    try:
        session = fastf1.get_session(year, race_name, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        logger.info("✅ Quali loaded: %s %d", race_name, year)
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

    # Qualifying best lap times → gap_to_pole_s
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

    df = results.merge(best_laps, on="driver_number", how="left")

    keep = [c for c in ["driver_number", "driver_abbr", "team",
                         "grid_position", "gap_to_pole_s", "quali_best_lap_s"]
            if c in df.columns]
    return df[keep].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING FEATURES (no shift — predicting future race)
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

    No .shift(1) required — the target race hasn't happened yet,
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
        return vals.mean() if len(vals) > 0 else np.nan

    roll = (
        prior.groupby("driver_abbr")
        .apply(lambda g: pd.Series({
            "rolling_avg_finish_3": _last_n_mean(g, "final_position"),
            "rolling_dnf_rate_3"  : _last_n_mean(
                g.assign(is_dnf=pd.to_numeric(g["is_dnf"], errors="coerce").fillna(0)),
                "is_dnf",
            ),
        }))
        .reset_index()
    )

    # Fallback: drivers with no history → season median
    med_pos = prior["final_position"].median()
    med_dnf = pd.to_numeric(prior.get("is_dnf", pd.Series([0])),
                             errors="coerce").median()
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
    ]
    if prior.empty:
        teams    = history["team"].dropna().unique()
        mid_rank = (len(teams) + 1) / 2
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
    feature_cols: list[str],
) -> pd.DataFrame:
    """Assemble prediction feature matrix aligned to training FEATURE_COLS."""
    df = quali_df.copy()

    # quali_gap_to_pole
    if "gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole"] = df["gap_to_pole_s"].clip(lower=0.0)
    else:
        df["quali_gap_to_pole"] = np.nan

    # Rolling features
    roll = _rolling_features(history, target_year, target_round)
    df   = df.merge(roll, on="driver_abbr", how="left")

    # Constructor rank
    cr = _constructor_rank(history, target_year, target_round)
    df = df.merge(cr, on="team", how="left")

    # Circuit type
    df["circuit_type"] = CIRCUIT_TYPE.get(event_name, CIRCUIT_TYPE_DEFAULT)

    # Adaptation score
    adapt = _adaptation_score(history, target_year, target_round)
    df    = df.merge(adapt, on="team", how="left")
    df["adaptation_score"] = df.get(
        "adaptation_score", pd.Series(0.0, index=df.index)
    ).fillna(0.0)

    # Impute remaining NaN with history median
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature '%s' missing from pred input — filling 0.", col)
            df[col] = 0.0
        null_n = df[col].isna().sum()
        if null_n > 0:
            fill = (history[col].median()
                    if col in history.columns else 0.0)
            df[col] = df[col].fillna(fill)
            logger.info("Imputed %d NaN in %-25s → %.3f", null_n, col, fill)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTED POSITION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _assign_predicted_positions(podium_prob: np.ndarray) -> list[int]:
    """
    Convert podium probabilities to ordinal predicted finishing positions.

    Highest probability → position 1, and so on.
    Ties broken by original array order (stable sort).
    """
    order = np.argsort(-podium_prob, kind="stable")
    positions = np.empty(len(podium_prob), dtype=int)
    positions[order] = np.arange(1, len(podium_prob) + 1)
    return positions.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict_race(
    year: int,
    race_name: str,
    history_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Predict podium probabilities for all drivers in an upcoming race.

    Parameters
    ----------
    year : int
        Season year (e.g. 2026).
    race_name : str
        FastF1 event name or round number (e.g. "British Grand Prix" or 12).
    history_years : list[int], optional
        Seasons to use for rolling feature computation.
        Defaults to [2022, 2023, 2024, 2025, 2026].

    Returns
    -------
    pd.DataFrame
        One row per driver, sorted by podium_probability descending.

        Columns
        -------
        driver          : three-letter abbreviation
        team            : constructor name
        grid_position   : qualifying grid slot
        podium_probability : raw model.predict_proba()[:, 1]
        win_probability : podium_prob normalised to sum=1 across all drivers
                          (interpretable "share of podium probability")
        predicted_position : ordinal rank by podium_probability (1 = most likely)

    Example
    -------
    >>> df = predict_race(2026, "British Grand Prix")
    >>> df.head(3)[["driver", "team", "podium_probability", "win_probability"]]
    #    driver    team       podium_probability  win_probability
    # 0  NOR     McLaren           0.721            0.142
    # 1  VER     Red Bull          0.683            0.134
    # 2  LEC     Ferrari           0.541            0.106
    """
    model        = _load_model()
    feature_cols = _load_feature_cols()
    history      = _load_history(history_years)

    # ── Fetch qualifying ──────────────────────────────────────────────────
    quali_df = _fetch_quali_data(year, race_name)

    # Determine round number for rolling feature computation
    target_round: int
    if history.empty:
        target_round = 1
    else:
        year_hist = history[history["year"] == year]
        target_round = (
            int(year_hist["round"].max()) + 1
            if not year_hist.empty else 1
        )

    # Resolve event name string for circuit_type lookup
    event_name = race_name if isinstance(race_name, str) else str(race_name)

    # ── Build feature matrix ──────────────────────────────────────────────
    pred_df = _build_pred_features(
        quali_df, history, year, target_round, event_name, feature_cols
    )

    X_pred = pred_df[feature_cols]

    # ── Inference ─────────────────────────────────────────────────────────
    podium_prob = model.predict_proba(X_pred)[:, 1]

    # win_probability: normalise raw probs to sum=1 for interpretability
    prob_sum = podium_prob.sum()
    win_prob = (podium_prob / prob_sum) if prob_sum > 0 else np.ones(len(podium_prob)) / len(podium_prob)

    predicted_pos = _assign_predicted_positions(podium_prob)

    # ── Assemble output ───────────────────────────────────────────────────
    output = pd.DataFrame({
        "driver"             : pred_df["driver_abbr"].values,
        "team"               : pred_df["team"].values,
        "grid_position"      : pred_df["grid_position"].values,
        "podium_probability" : podium_prob.round(4),
        "win_probability"    : win_prob.round(4),
        "predicted_position" : predicted_pos,
    })
    output = output.sort_values("podium_probability", ascending=False).reset_index(drop=True)

    logger.info("━━━ Prediction: %s %d ━━━", race_name, year)
    logger.info("Top 5 predicted podium candidates:")
    for _, row in output.head(5).iterrows():
        logger.info(
            "  P%-2d  %-4s  %-22s  podium=%.3f  win_share=%.3f",
            row["predicted_position"], row["driver"],
            row["team"], row["podium_probability"], row["win_probability"],
        )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict F1 race podium.")
    parser.add_argument("--year",       type=int, required=True)
    parser.add_argument("--race",       type=str, required=True,
                        help='e.g. "British Grand Prix" or round number')
    parser.add_argument("--history-years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()

    result = predict_race(
        year          = args.year,
        race_name     = args.race,
        history_years = args.history_years,
    )
    print("\n" + result.to_string(index=False))
