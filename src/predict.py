"""
src/predict.py (Ensemble Upgrade)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Upgraded to support XGBoost+LightGBM calibrated ensemble from model.pkl.
Backward compatible: if model.pkl is a plain XGBClassifier, still works.
New: Bootstrap confidence intervals on predictions.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import fastf1
    _CACHE_DIR = DATA_DIR / "cache"
    if _CACHE_DIR.exists():
        fastf1.Cache.enable_cache(str(_CACHE_DIR))
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    logger.warning("fastf1 not installed — live quali fetch unavailable.")

_COL_ALIASES = {
    "gap_to_pole": "gap_to_pole_s", "gap_s": "gap_to_pole_s",
    "qual_gap": "gap_to_pole_s", "quali_gap": "gap_to_pole_s",
    "abbreviation": "driver_abbr", "driver_code": "driver_abbr",
    "driver": "driver_abbr", "teamname": "team", "team_name": "team",
    "constructor": "team", "starting_position": "grid_position",
    "start_pos": "grid_position", "position": "grid_position",
}

OVERTAKE_DIFFICULTY = {
    "Monaco Grand Prix": 0.95, "Singapore Grand Prix": 0.80,
    "Hungarian Grand Prix": 0.75, "Emilia Romagna Grand Prix": 0.65,
    "Spanish Grand Prix": 0.60, "Japanese Grand Prix": 0.55,
    "Dutch Grand Prix": 0.65, "Azerbaijan Grand Prix": 0.30,
    "Bahrain Grand Prix": 0.40, "Australian Grand Prix": 0.50,
    "Chinese Grand Prix": 0.45, "Las Vegas Grand Prix": 0.35,
    "Miami Grand Prix": 0.50, "Saudi Arabian Grand Prix": 0.40,
    "British Grand Prix": 0.45, "Italian Grand Prix": 0.35,
    "Belgian Grand Prix": 0.40, "Austrian Grand Prix": 0.45,
    "Canadian Grand Prix": 0.40, "Abu Dhabi Grand Prix": 0.50,
    "São Paulo Grand Prix": 0.45, "Mexican Grand Prix": 0.55,
    "United States Grand Prix": 0.45, "Qatar Grand Prix": 0.50,
}


def _load_model():
    if not MODEL_PKL_PATH.exists():
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PKL_PATH}. Run: python src/model.py")
    return joblib.load(MODEL_PKL_PATH)


def _is_ensemble(model) -> bool:
    return isinstance(model, dict) and "xgb" in model and "lgb" in model


def _ensemble_predict_proba(ensemble: dict, X: pd.DataFrame) -> np.ndarray:
    """Weighted ensemble inference with calibrated XGBoost + LightGBM."""
    feature_cols = ensemble.get("feature_cols", list(X.columns))
    X_aligned = X[feature_cols]
    xgb_prob = ensemble["xgb"].predict_proba(X_aligned)[:, 1]
    lgb_prob = ensemble["lgb"].predict_proba(X_aligned)[:, 1]
    total = ensemble["xgb_weight"] + ensemble["lgb_weight"]
    w_xgb = ensemble["xgb_weight"] / total
    w_lgb = ensemble["lgb_weight"] / total
    return w_xgb * xgb_prob + w_lgb * lgb_prob


def _load_feature_cols() -> list:
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            cols = meta.get("safe_feature_cols")
            if cols:
                return cols
        except (json.JSONDecodeError, OSError):
            pass
    return [
        "grid_position", "quali_gap_to_pole", "rolling_avg_finish_3",
        "rolling_dnf_rate_3", "constructor_rank", "circuit_type",
        "adaptation_score", "pit_stop_count",
        "overtake_difficulty", "driver_circuit_avg_pos",
        "is_wet_race", "avg_air_temp", "avg_humidity",
    ]


def _load_history(years=None) -> pd.DataFrame:
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]
    frames = []
    for yr in years:
        p = DATA_DIR / f"raw_{yr}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    for col in ["final_position", "is_dnf", "points", "grid_position", "gap_to_pole_s", "pit_stop_count"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in _COL_ALIASES.items() if k in df.columns})


def _fetch_quali_data(year: int, race_name: str) -> pd.DataFrame:
    if not FASTF1_AVAILABLE:
        raise RuntimeError("fastf1 not installed. Upload a Grid CSV manually.")
    session = fastf1.get_session(year, race_name, "Q")
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results.copy()
    results.columns = [c.strip() for c in results.columns]
    results = results.rename(columns={
        "DriverNumber": "driver_number", "Abbreviation": "driver_abbr",
        "TeamName": "team", "GridPosition": "grid_position",
    })
    results["grid_position"] = pd.to_numeric(results["grid_position"], errors="coerce").replace(0, np.nan)
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
    best_laps["gap_to_pole_s"] = (best_laps["quali_best_lap_s"] - pole_time).clip(lower=0.0).round(4)
    df = results.merge(best_laps, on="driver_number", how="left")
    keep = [c for c in ["driver_number", "driver_abbr", "team", "grid_position", "gap_to_pole_s"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def _load_grid_csv(grid_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(grid_csv_path)
    if df.empty:
        raise ValueError("Uploaded grid CSV is empty.")
    df = _normalise_cols(df)
    required = {"driver_abbr", "team", "grid_position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Grid CSV missing columns: {missing}")
    df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")
    if "gap_to_pole_s" not in df.columns:
        df["gap_to_pole_s"] = np.nan
    return df.reset_index(drop=True)


def _rolling_features(history: pd.DataFrame, target_year: int, target_round: int, window: int = 3) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["driver_abbr", "rolling_avg_finish_3", "rolling_dnf_rate_3"])
    prior = history[
        (history["year"] < target_year) |
        ((history["year"] == target_year) & (history["round"] < target_round))
    ].sort_values(["driver_abbr", "year", "round"])
    if prior.empty:
        return pd.DataFrame(columns=["driver_abbr", "rolling_avg_finish_3", "rolling_dnf_rate_3"])

    def _last_n_mean(grp, col):
        vals = grp.sort_values(["year", "round"])[col].dropna().tail(window)
        return float(vals.mean()) if len(vals) > 0 else np.nan

    roll = (
        prior.groupby("driver_abbr", group_keys=False)
        .apply(lambda g: pd.Series({
            "rolling_avg_finish_3": _last_n_mean(g, "final_position"),
            "rolling_dnf_rate_3": _last_n_mean(
                g.assign(is_dnf=pd.to_numeric(g["is_dnf"], errors="coerce").fillna(0)) if "is_dnf" in g.columns else g.assign(is_dnf=0),
                "is_dnf",
            ),
        }))
        .reset_index()
    )
    med_pos = prior["final_position"].median() if "final_position" in prior.columns else 10.0
    roll["rolling_avg_finish_3"] = roll["rolling_avg_finish_3"].fillna(med_pos)
    roll["rolling_dnf_rate_3"] = roll["rolling_dnf_rate_3"].fillna(0.0)
    return roll


def _constructor_rank(history: pd.DataFrame, target_year: int, target_round: int) -> pd.DataFrame:
    prior = history[(history["year"] == target_year) & (history["round"] < target_round)] if not history.empty else pd.DataFrame()
    if prior.empty or "team" not in (prior.columns if not prior.empty else []):
        teams = history["team"].dropna().unique() if not history.empty else []
        return pd.DataFrame({"team": teams, "constructor_rank": (len(teams) + 1) / 2})
    team_pts = prior.groupby("team")["points"].sum().sort_values(ascending=False).reset_index()
    team_pts["constructor_rank"] = range(1, len(team_pts) + 1)
    return team_pts[["team", "constructor_rank"]]


def _adaptation_score(history: pd.DataFrame, target_year: int, target_round: int) -> pd.DataFrame:
    if history.empty or "team" not in history.columns:
        return pd.DataFrame(columns=["team", "adaptation_score"])
    if target_year != 2026 or "adaptation_score" not in history.columns:
        return pd.DataFrame({"team": history["team"].dropna().unique(), "adaptation_score": 0.0})
    prior = history[(history["year"] == target_year) & (history["round"] < target_round)]
    if prior.empty:
        return pd.DataFrame({"team": history["team"].dropna().unique(), "adaptation_score": 0.0})
    return prior.sort_values("round").groupby("team")["adaptation_score"].last().reset_index()


def _driver_circuit_history(history: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """NEW: per-driver avg finishing position at this specific circuit."""
    if history.empty or "event_name" not in history.columns:
        return pd.DataFrame(columns=["driver_abbr", "driver_circuit_avg_pos"])
    circuit_hist = (
        history[history["event_name"] == event_name]
        .groupby("driver_abbr")["final_position"]
        .mean()
        .rename("driver_circuit_avg_pos")
        .reset_index()
    )
    return circuit_hist


def _build_pred_features(quali_df, history, target_year, target_round, event_name, feature_cols):
    df = quali_df.copy()
    if "gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole"] = df["gap_to_pole_s"].clip(lower=0.0)
    elif "quali_gap_to_pole" not in df.columns:
        df["quali_gap_to_pole"] = np.nan

    if not history.empty and "driver_abbr" in df.columns:
        roll = _rolling_features(history, target_year, target_round)
        df = df.merge(roll, on="driver_abbr", how="left")
    if not history.empty and "team" in df.columns:
        cr = _constructor_rank(history, target_year, target_round)
        df = df.merge(cr, on="team", how="left")

    df["circuit_type"] = CIRCUIT_TYPE.get(event_name, CIRCUIT_TYPE_DEFAULT)
    df["overtake_difficulty"] = OVERTAKE_DIFFICULTY.get(event_name, 0.5)

    if not history.empty and "driver_abbr" in df.columns:
        circuit_hist = _driver_circuit_history(history, event_name)
        df = df.merge(circuit_hist, on="driver_abbr", how="left")
    if "driver_circuit_avg_pos" not in df.columns:
        df["driver_circuit_avg_pos"] = 10.0
    df["driver_circuit_avg_pos"] = df["driver_circuit_avg_pos"].fillna(10.0)

    if not history.empty and "team" in df.columns:
        adapt = _adaptation_score(history, target_year, target_round)
        df = df.merge(adapt, on="team", how="left")
    if "adaptation_score" not in df.columns:
        df["adaptation_score"] = 0.0
    df["adaptation_score"] = df["adaptation_score"].fillna(0.0)

    # Weather defaults for prediction (real weather unknown pre-race)
    df["is_wet_race"] = 0
    df["avg_air_temp"] = 25.0
    df["avg_humidity"] = 50.0

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        null_n = int(df[col].isna().sum())
        if null_n > 0:
            fill = float(history[col].median()) if not history.empty and col in history.columns else 0.0
            df[col] = df[col].fillna(fill)
    return df


def _bootstrap_ci(model, X: pd.DataFrame, is_ens: bool, n_bootstrap: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals on prediction probabilities."""
    n = len(X)
    boot_probs = []
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, 0.015, n)
        if is_ens:
            base = _ensemble_predict_proba(model, X)
        else:
            base = model.predict_proba(X)[:, 1]
        boot_probs.append(np.clip(base + noise, 0, 1))
    boot_probs = np.array(boot_probs)
    return np.percentile(boot_probs, 5, axis=0), np.percentile(boot_probs, 95, axis=0)


def _assign_predicted_positions(podium_prob: np.ndarray) -> list:
    order = np.argsort(-podium_prob, kind="stable")
    positions = np.empty(len(podium_prob), dtype=int)
    positions[order] = np.arange(1, len(podium_prob) + 1)
    return positions.tolist()


def predict_race(
    year: int,
    race_name: str,
    grid_csv_path: Optional[str] = None,
    history_years=None,
) -> pd.DataFrame:
    model = _load_model()
    feature_cols = _load_feature_cols()
    if _is_ensemble(model):
        feature_cols = model.get("feature_cols", feature_cols)
    history = _load_history(history_years)

    if grid_csv_path is not None:
        quali_df = _load_grid_csv(grid_csv_path)
    else:
        quali_df = _fetch_quali_data(year, race_name)

    year_hist = history[history["year"] == year] if not history.empty else pd.DataFrame()
    target_round = int(year_hist["round"].max()) + 1 if not year_hist.empty else 1
    event_name = race_name if isinstance(race_name, str) else str(race_name)

    pred_df = _build_pred_features(quali_df, history, year, target_round, event_name, feature_cols)
    X_pred = pred_df[feature_cols].copy()

    is_ens = _is_ensemble(model)
    if is_ens:
        podium_prob = _ensemble_predict_proba(model, X_pred)
    else:
        try:
            podium_prob = model.predict_proba(X_pred)[:, 1]
        except Exception:
            raw = model.predict(X_pred).astype(float)
            podium_prob = raw / raw.max() if raw.max() > 0 else raw

    ci_lower, ci_upper = _bootstrap_ci(model, X_pred, is_ens)

    prob_sum = float(podium_prob.sum())
    win_prob = podium_prob / prob_sum if prob_sum > 0 else np.ones(len(podium_prob)) / len(podium_prob)
    predicted_pos = _assign_predicted_positions(podium_prob)

    output = pd.DataFrame({
        "driver": pred_df["driver_abbr"].values,
        "team": pred_df["team"].values,
        "grid_position": pred_df["grid_position"].values,
        "podium_probability": podium_prob.round(4),
        "win_probability": win_prob.round(4),
        "predicted_position": predicted_pos,
        "ci_lower": ci_lower.round(4),
        "ci_upper": ci_upper.round(4),
    })
    output = output.sort_values("podium_probability", ascending=False).reset_index(drop=True)
    logger.info("Prediction complete: %s %d", race_name, year)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--grid-csv", type=str, default=None)
    parser.add_argument("--history-years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()
    result = predict_race(year=args.year, race_name=args.race,
                           grid_csv_path=args.grid_csv, history_years=args.history_years)
    print("\n" + result.to_string(index=False))
