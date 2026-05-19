"""
src/features.py
~~~~~~~~~~~~~~~
Feature engineering pipeline for the F1 2026 Race Outcome Predictor.
Upgrades: exponential decay sample weights, weather features, driver circuit history.
"""
import json
import logging
import fastf1
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Circuit type lookup ────────────────────────────────────────────────────────
# 0 = street | 1 = technical | 2 = high-speed
CIRCUIT_TYPE: dict[str, int] = {
    "Monaco Grand Prix": 0,
    "Azerbaijan Grand Prix": 0,
    "Singapore Grand Prix": 0,
    "Saudi Arabian Grand Prix": 0,
    "Las Vegas Grand Prix": 0,
    "Miami Grand Prix": 0,
    "Hungarian Grand Prix": 1,
    "Spanish Grand Prix": 1,
    "Japanese Grand Prix": 1,
    "Abu Dhabi Grand Prix": 1,
    "Australian Grand Prix": 1,
    "Canadian Grand Prix": 1,
    "United States Grand Prix": 1,
    "São Paulo Grand Prix": 1,
    "Mexican Grand Prix": 1,
    "Chinese Grand Prix": 1,
    "Bahrain Grand Prix": 1,
    "Qatar Grand Prix": 1,
    "British Grand Prix": 2,
    "Italian Grand Prix": 2,
    "Belgian Grand Prix": 2,
    "Austrian Grand Prix": 2,
    "Dutch Grand Prix": 2,
    "Emilia Romagna Grand Prix": 2,
    "French Grand Prix": 2,
}
CIRCUIT_TYPE_DEFAULT = 1

# ── Overtake difficulty per circuit (0–1, higher = harder to overtake) ─────────
OVERTAKE_DIFFICULTY: dict[str, float] = {
    "Monaco Grand Prix": 0.95,
    "Singapore Grand Prix": 0.80,
    "Hungarian Grand Prix": 0.75,
    "Emilia Romagna Grand Prix": 0.65,
    "Spanish Grand Prix": 0.60,
    "Japanese Grand Prix": 0.55,
    "Dutch Grand Prix": 0.65,
    "Azerbaijan Grand Prix": 0.30,
    "Bahrain Grand Prix": 0.40,
    "Australian Grand Prix": 0.50,
    "Chinese Grand Prix": 0.45,
    "Las Vegas Grand Prix": 0.35,
    "Miami Grand Prix": 0.50,
    "Saudi Arabian Grand Prix": 0.40,
    "British Grand Prix": 0.45,
    "Italian Grand Prix": 0.35,
    "Belgian Grand Prix": 0.40,
    "Austrian Grand Prix": 0.45,
    "Canadian Grand Prix": 0.40,
    "Abu Dhabi Grand Prix": 0.50,
    "São Paulo Grand Prix": 0.45,
    "Mexican Grand Prix": 0.55,
    "United States Grand Prix": 0.45,
    "Qatar Grand Prix": 0.50,
}

# ── Exponential decay sample weights (upgraded from flat 0.4×) ─────────────────
YEAR_WEIGHTS: dict[int, float] = {
    2022: 0.20,
    2023: 0.30,
    2024: 0.50,
    2025: 0.80,
    2026: 1.00,
}


def _add_grid_and_quali(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gap_to_pole_s" in df.columns:
        df["quali_gap_to_pole"] = df["gap_to_pole_s"].clip(lower=0.0)
    else:
        df["quali_gap_to_pole"] = np.nan
        logger.warning("gap_to_pole_s column missing — quali_gap_to_pole set to NaN.")
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["driver_abbr", "year", "round"]).reset_index(drop=True)

    def _rolling_mean_shifted(series: pd.Series, window: int = 3) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    df["rolling_avg_finish_3"] = (
        df.groupby(["driver_abbr", "year"])["final_position"]
        .transform(_rolling_mean_shifted)
    )
    df["is_dnf"] = pd.to_numeric(df.get("is_dnf", 0), errors="coerce").fillna(0)
    df["rolling_dnf_rate_3"] = (
        df.groupby(["driver_abbr", "year"])["is_dnf"]
        .transform(_rolling_mean_shifted)
    )
    for feat in ["rolling_avg_finish_3", "rolling_dnf_rate_3"]:
        season_avg = df.groupby(["driver_abbr", "year"])[feat].transform("mean")
        df[feat] = df[feat].fillna(season_avg)
        df[feat] = df[feat].fillna(df[feat].median())
    return df


def _add_constructor_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["year", "round", "team"]).reset_index(drop=True)
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
    n_teams = df["team"].nunique()
    df["constructor_rank"] = df["constructor_rank"].fillna((n_teams + 1) / 2)
    return df


def _add_circuit_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["circuit_type"] = (
        df["event_name"]
        .map(CIRCUIT_TYPE)
        .fillna(CIRCUIT_TYPE_DEFAULT)
        .astype(int)
    )
    return df


def _add_overtake_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """NEW: add overtake_difficulty feature per circuit."""
    df = df.copy()
    df["overtake_difficulty"] = (
        df["event_name"]
        .map(OVERTAKE_DIFFICULTY)
        .fillna(0.5)
    )
    return df


def _add_driver_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """NEW: each driver's historical avg finishing position at this specific circuit."""
    df = df.copy()
    circuit_history = (
        df.groupby(["driver_abbr", "event_name"])["final_position"]
        .mean()
        .rename("driver_circuit_avg_pos")
        .reset_index()
    )
    df = df.merge(circuit_history, on=["driver_abbr", "event_name"], how="left")
    df["driver_circuit_avg_pos"] = df["driver_circuit_avg_pos"].fillna(10.0)
    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """NEW: attempt to load weather from FastF1; fallback to dry defaults."""
    df = df.copy()
    df["is_wet_race"] = 0
    df["avg_air_temp"] = 25.0
    df["avg_humidity"] = 50.0

    for (year, round_num), group_idx in df.groupby(["year", "round"]).groups.items():
        try:
            session = fastf1.get_session(int(year), int(round_num), "R")
            session.load(weather=True, laps=False, telemetry=False)
            weather = session.weather_data
            is_wet = int(weather["Rainfall"].sum() > 0)
            avg_temp = float(weather["AirTemp"].mean())
            avg_humidity = float(weather["Humidity"].mean())
            df.loc[group_idx, "is_wet_race"] = is_wet
            df.loc[group_idx, "avg_air_temp"] = avg_temp
            df.loc[group_idx, "avg_humidity"] = avg_humidity
        except Exception:
            pass  # keep defaults for this race
    return df


def _add_adaptation_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adaptation_score"] = 0.0
    if (df["year"] == 2026).any():
        mask_2026 = df["year"] == 2026
        round1_avg = (
            df[mask_2026 & (df["round"] == df.loc[mask_2026, "round"].min())]
            .groupby("team")["final_position"]
            .mean()
            .rename("team_pos_round1")
        )
        df = df.merge(round1_avg, on="team", how="left")
        df.loc[mask_2026, "adaptation_score"] = (
            df.loc[mask_2026, "team_pos_round1"]
            - df.loc[mask_2026, "rolling_avg_finish_3"]
        ).fillna(0.0)
        df = df.drop(columns=["team_pos_round1"], errors="ignore")
    return df


def _add_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """UPGRADED: exponential decay weights instead of flat 0.4×."""
    df = df.copy()
    df["sample_weight"] = df["year"].map(YEAR_WEIGHTS).fillna(WEIGHT_PRE_2026)
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["final_position"] = pd.to_numeric(df["final_position"], errors="coerce")
    df["top3_finish"] = (df["final_position"] <= 3).fillna(False).astype(int)
    return df


# ── Feature columns ────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "grid_position",
    "quali_gap_to_pole",
    "rolling_avg_finish_3",
    "rolling_dnf_rate_3",
    "constructor_rank",
    "circuit_type",
    "adaptation_score",
    "pit_stop_count",
    # NEW features
    "overtake_difficulty",
    "driver_circuit_avg_pos",
    "is_wet_race",
    "avg_air_temp",
    "avg_humidity",
]


def build_features(
    df_raw: pd.DataFrame,
    write_metadata: bool = True,
    include_weather: bool = False,  # set True when retraining with full data
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    logger.info("━━━ Feature Engineering Pipeline ━━━")
    logger.info("Input shape: %s", df_raw.shape)

    df = df_raw.copy()
    df = df.sort_values(["driver_abbr", "year", "round"]).reset_index(drop=True)

    df = _add_grid_and_quali(df)
    logger.info("✅ Step 1/9 — grid_position + quali_gap_to_pole")
    df = _add_rolling_features(df)
    logger.info("✅ Step 2/9 — rolling_avg_finish_3 + rolling_dnf_rate_3")
    df = _add_constructor_rank(df)
    logger.info("✅ Step 3/9 — constructor_rank")
    df = _add_circuit_type(df)
    logger.info("✅ Step 4/9 — circuit_type")
    df = _add_overtake_difficulty(df)
    logger.info("✅ Step 5/9 — overtake_difficulty (NEW)")
    df = _add_driver_circuit_history(df)
    logger.info("✅ Step 6/9 — driver_circuit_avg_pos (NEW)")
    if include_weather:
        df = _add_weather_features(df)
        logger.info("✅ Step 7/9 — weather features (is_wet_race, avg_air_temp, avg_humidity)")
    else:
        df["is_wet_race"] = 0
        df["avg_air_temp"] = 25.0
        df["avg_humidity"] = 50.0
        logger.info("⏭️  Step 7/9 — weather features skipped (include_weather=False)")
    df = _add_adaptation_score(df)
    logger.info("✅ Step 8/9 — adaptation_score")
    df = _add_sample_weight(df)
    logger.info("✅ Step 9/9 — exponential decay sample_weight (NEW)")
    df = _add_target(df)

    leaked = [c for c in FEATURE_COLS if c in LEAKAGE_COLS]
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED — remove from FEATURE_COLS: {leaked}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        logger.warning("Missing features (excluded): %s", missing_features)

    df_model = df.dropna(subset=available_features, how="all").copy()
    for col in available_features:
        null_n = df_model[col].isna().sum()
        if null_n > 0:
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val)
            logger.info(" Imputed %d NaN in %-25s → median=%.3f", null_n, col, median_val)

    X = df_model[available_features].reset_index(drop=True)
    y = df_model["top3_finish"].reset_index(drop=True)
    weights = df_model["sample_weight"].reset_index(drop=True)

    top3_rate = y.mean()
    logger.info("Target: top3=1: %.1f%% | top3=0: %.1f%%", top3_rate * 100, (1 - top3_rate) * 100)
    logger.info("Output X: %s | y: %s", X.shape, y.shape)

    if write_metadata:
        _update_metadata(available_features, X)
    return X, y, weights


def _update_metadata(feature_cols: list[str], X: pd.DataFrame) -> None:
    metadata = {}
    if META_PATH.exists():
        try:
            metadata = json.loads(META_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("model_metadata.json malformed — overwriting.")
    metadata["safe_feature_cols"] = feature_cols
    metadata["feature_dtypes"] = {c: str(X[c].dtype) for c in feature_cols}
    metadata["leakage_cols"] = LEAKAGE_COLS
    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("💾 Feature metadata written → %s", META_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--weather", action="store_true")
    args = parser.parse_args()
    frames = []
    for yr in args.years:
        path = DATA_DIR / f"raw_{yr}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    df_raw = pd.concat(frames, ignore_index=True)
    X, y, weights = build_features(df_raw, write_metadata=True, include_weather=args.weather)
    print(f"\nSamples: {len(X)} | Features: {X.shape[1]} | Top-3 rate: {y.mean():.1%}")
    print(X.describe().round(3).to_string())
