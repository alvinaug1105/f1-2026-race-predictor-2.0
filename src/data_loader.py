"""
src/data_loader.py  (v3 — Final)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastF1-based data ingestion layer for the F1 2026 Race Outcome Predictor.

Changelog
---------
v1 → Initial implementation
v2 → Fix final_position coerce, grid_position pit-lane NaN,
      future event filtering, sample_weight early binding,
      fastest_lap_s null warning
v3 → Fix is_dnf keyword whitelist (exclude lapped cars),
      extract_quali_features pole_time NaN guard
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths & Cache ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/cache")
RAW_DIR   = Path("data")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))
logger.info("FastF1 cache enabled at: %s", CACHE_DIR.resolve())

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHT_2026     = 1.0   # 2026 regulation reset — primary signal
WEIGHT_PRE_2026 = 0.4   # Historical seasons — reduced trust

# Retirement causes that constitute a genuine DNF.
# Excludes "+1 LAP", "+2 LAPS" etc. — those are classified finishers.
DNF_KEYWORDS: tuple[str, ...] = (
    "ACCIDENT",
    "BRAKES",
    "COLLISION",
    "ELECTRICAL",   # covers 2026 ERS / hybrid failures
    "ENGINE",
    "GEARBOX",
    "HYDRAULICS",
    "MECHANICAL",
    "OVERHEATING",  # covers 2026 active aero cooling edge cases
    "POWER UNIT",
    "RETIRED",
    "SUSPENSION",
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_race_session(year: int, round_number: int) -> Optional[fastf1.core.Session]:
    """
    Load a Race session (type='R') for a given year and round.

    Parameters
    ----------
    year : int
        Championship season year (e.g. 2026).
    round_number : int
        Round number within the season (1-indexed).

    Returns
    -------
    fastf1.core.Session or None
        Loaded session, or None if unavailable / not yet completed.
    """
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)
        logger.info(
            "✅ Loaded RACE  | %d Round %-2d — %s",
            year, round_number, session.event["EventName"],
        )
        return session
    except Exception as exc:
        logger.warning(
            "⚠️  RACE unavailable  | %d Round %d — %s",
            year, round_number, exc,
        )
        return None


def load_quali_session(year: int, round_number: int) -> Optional[fastf1.core.Session]:
    """
    Load a Qualifying session (type='Q') for a given year and round.

    Parameters
    ----------
    year : int
        Championship season year (e.g. 2026).
    round_number : int
        Round number within the season (1-indexed).

    Returns
    -------
    fastf1.core.Session or None
        Loaded session, or None if unavailable.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        logger.info(
            "✅ Loaded QUALI | %d Round %-2d — %s",
            year, round_number, session.event["EventName"],
        )
        return session
    except Exception as exc:
        logger.warning(
            "⚠️  QUALI unavailable | %d Round %d — %s",
            year, round_number, exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def extract_race_features(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract driver-level features from a completed race session.

    Features extracted
    ------------------
    driver_number    – FastF1 driver number (str)
    driver_abbr      – Three-letter abbreviation (e.g. "VER")
    team             – Constructor name
    grid_position    – Starting grid slot; NaN for pit-lane starts
    final_position   – Classified finishing position; NaN if unclassified
    points           – Championship points scored
    is_dnf           – 1 if genuine retirement, 0 otherwise
                       (lapped finishers are NOT flagged as DNF)
    fastest_lap_s    – Personal best lap time in seconds
    pit_stop_count   – Number of pit stops made
    event_name       – Grand Prix name
    year             – Season year
    round            – Round number

    Parameters
    ----------
    session : fastf1.core.Session
        A fully loaded race session.

    Returns
    -------
    pd.DataFrame
        One row per driver.
    """
    results: pd.DataFrame = session.results.copy()
    results.columns = [c.strip() for c in results.columns]

    rename_map = {
        "DriverNumber" : "driver_number",
        "Abbreviation" : "driver_abbr",
        "TeamName"     : "team",
        "GridPosition" : "grid_position",
        "Position"     : "final_position",
        "Points"       : "points",
        "Status"       : "status",
    }
    results = results.rename(
        columns={k: v for k, v in rename_map.items() if k in results.columns}
    )

    # ── [FIX 1] final_position: coerce non-numeric ("NC", NaN, etc.) ──────────
    results["final_position"] = pd.to_numeric(
        results["final_position"], errors="coerce"
    )

    # ── [FIX 2] grid_position: coerce + replace 0 (pit-lane start) → NaN ─────
    results["grid_position"] = pd.to_numeric(
        results["grid_position"], errors="coerce"
    )
    pit_lane_starts = (results["grid_position"] == 0).sum()
    if pit_lane_starts > 0:
        logger.info(
            "Round %s — %d pit-lane start(s) detected; grid_position → NaN.",
            session.event["RoundNumber"], pit_lane_starts,
        )
    results["grid_position"] = results["grid_position"].replace(0, np.nan)

    # ── [FIX 5] is_dnf: keyword whitelist — excludes lapped finishers ─────────
    if "status" in results.columns:
        results["is_dnf"] = (
            results["status"]
            .str.upper()
            .str.contains("|".join(DNF_KEYWORDS), na=False)
            .astype(int)
        )
    else:
        results["is_dnf"] = 0

    # ── Lap-level features ────────────────────────────────────────────────────
    laps: pd.DataFrame = session.laps.copy()

    # Fastest lap per driver (seconds)
    fastest_laps = (
        laps.groupby("DriverNumber")["LapTime"]
        .min()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={
            "DriverNumber" : "driver_number",
            "LapTime"      : "fastest_lap_s",
        })
    )

    # ── [FIX 4b] Warn if lap timing data is incomplete ────────────────────────
    null_count = fastest_laps["fastest_lap_s"].isna().sum()
    if null_count > 0:
        logger.warning(
            "Round %s — %d driver(s) missing fastest_lap_s "
            "(red-flag / incomplete session data).",
            session.event["RoundNumber"], null_count,
        )

    # Pit stop count: rows where a driver exited the pit lane
    pit_counts = (
        laps[laps["PitOutTime"].notna()]
        .groupby("DriverNumber")
        .size()
        .reset_index(name="pit_stop_count")
        .rename(columns={"DriverNumber": "driver_number"})
    )

    # Normalise driver_number dtype for safe merging
    for frame in (results, fastest_laps, pit_counts):
        frame["driver_number"] = frame["driver_number"].astype(str)

    df = (
        results
        .merge(fastest_laps, on="driver_number", how="left")
        .merge(pit_counts,   on="driver_number", how="left")
    )
    df["pit_stop_count"] = df["pit_stop_count"].fillna(0).astype(int)

    # ── Session metadata ──────────────────────────────────────────────────────
    df["event_name"] = session.event["EventName"]
    df["year"]       = session.event["EventDate"].year
    df["round"]      = session.event["RoundNumber"]

    keep_cols = [
        "driver_number", "driver_abbr", "team",
        "grid_position", "final_position", "points",
        "is_dnf", "fastest_lap_s", "pit_stop_count",
        "event_name", "year", "round",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def extract_quali_features(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract driver-level qualifying features.

    Features extracted
    ------------------
    driver_number     – FastF1 driver number (str)
    quali_best_lap_s  – Best Q lap in seconds; NaN if no timed lap set
    gap_to_pole_s     – Delta to pole time in seconds;
                        NaN if session fully aborted (no valid pole time)

    Parameters
    ----------
    session : fastf1.core.Session
        A fully loaded qualifying session.

    Returns
    -------
    pd.DataFrame
        One row per driver.
    """
    laps: pd.DataFrame = session.laps.copy()

    best_laps = (
        laps.groupby("DriverNumber")["LapTime"]
        .min()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={
            "DriverNumber" : "driver_number",
            "LapTime"      : "quali_best_lap_s",
        })
    )
    best_laps["driver_number"] = best_laps["driver_number"].astype(str)

    pole_time = best_laps["quali_best_lap_s"].min()

    # ── [FIX 6] Guard: fully aborted quali → pole_time is NaN ────────────────
    if pd.isna(pole_time):
        logger.warning(
            "Quali session (Round %s) — no valid lap times found; "
            "gap_to_pole_s will be NaN for all drivers.",
            session.event["RoundNumber"],
        )
        best_laps["gap_to_pole_s"] = np.nan
    else:
        best_laps["gap_to_pole_s"] = (
            best_laps["quali_best_lap_s"] - pole_time
        ).round(4)

    return best_laps.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUND COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

def collect_round(year: int, round_number: int) -> Optional[pd.DataFrame]:
    """
    Collect and merge race + qualifying features for a single round.

    Binds both the top3 label and sample_weight at this layer so all
    downstream modules (features.py, model.py) can consume them directly.

    Parameters
    ----------
    year : int
        Season year.
    round_number : int
        Round number within the season.

    Returns
    -------
    pd.DataFrame or None
        Merged DataFrame with top3 label and sample_weight per driver,
        or None if the race session is unavailable.
    """
    race_session  = load_race_session(year, round_number)
    quali_session = load_quali_session(year, round_number)

    if race_session is None:
        logger.warning("Skipping Round %d — race session unavailable.", round_number)
        return None

    race_df = extract_race_features(race_session)

    if quali_session is not None:
        quali_df = extract_quali_features(quali_session)
        df = race_df.merge(quali_df, on="driver_number", how="left")
    else:
        logger.warning(
            "Round %d — quali session missing; quali columns set to NaN.",
            round_number,
        )
        df = race_df.copy()
        df["quali_best_lap_s"] = np.nan
        df["gap_to_pole_s"]    = np.nan

    # ── [FIX 1] Safe top3 label ───────────────────────────────────────────────
    # NaN final_position (unclassified) → 0, not silent False
    df["top3"] = (df["final_position"] <= 3).fillna(False).astype(int)

    # ── [FIX 3] sample_weight early binding ──────────────────────────────────
    df["sample_weight"] = WEIGHT_2026 if year == 2026 else WEIGHT_PRE_2026

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SEASON COLLECTOR  (main public API)
# ─────────────────────────────────────────────────────────────────────────────

def collect_season(year: int, save: bool = True) -> pd.DataFrame:
    """
    Collect race + qualifying data for all *completed* rounds in a season.

    Filters out future events using today's date before any FastF1 loads,
    preventing noisy warning logs for races that have not yet taken place.

    Parameters
    ----------
    year : int
        Season year (e.g. 2026).
    save : bool, default True
        If True, saves output to data/raw_{year}.csv.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame — one row per driver per completed race.
        Columns: driver_number, driver_abbr, team, grid_position,
                 final_position, points, is_dnf, fastest_lap_s,
                 pit_stop_count, event_name, year, round,
                 quali_best_lap_s, gap_to_pole_s, top3, sample_weight

    Examples
    --------
    >>> df_2026 = collect_season(2026)
    >>> df_all  = pd.concat([collect_season(y, save=False)
    ...                      for y in range(2023, 2027)],
    ...                     ignore_index=True)
    """
    logger.info("━━━ Collecting season %d ━━━", year)

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    # ── [FIX 3] Only process events that have already occurred ───────────────
    today = pd.Timestamp(date.today(), tz="UTC")

    # EventDate tz-awareness varies across FastF1 versions — normalise
    if schedule["EventDate"].dt.tz is None:
        schedule["EventDate"] = schedule["EventDate"].dt.tz_localize("UTC")

    completed = schedule[schedule["EventDate"] <= today].copy()
    rounds    = completed["RoundNumber"].tolist()

    logger.info(
        "Season %d — %d completed round(s) of %d scheduled.",
        year, len(rounds), len(schedule),
    )

    all_rounds: list[pd.DataFrame] = []

    for rnd in rounds:
        event_name = completed.loc[
            completed["RoundNumber"] == rnd, "EventName"
        ].values[0]
        logger.info("── Round %d / %d  (%s)", rnd, len(rounds), event_name)

        round_df = collect_round(year, rnd)
        if round_df is not None and not round_df.empty:
            all_rounds.append(round_df)

    if not all_rounds:
        logger.error("No data collected for season %d.", year)
        return pd.DataFrame()

    season_df = pd.concat(all_rounds, ignore_index=True)

    logger.info(
        "Season %d complete — %d rows across %d round(s).",
        year, len(season_df), season_df["round"].nunique(),
    )

    if save:
        out_path = RAW_DIR / f"raw_{year}.csv"
        season_df.to_csv(out_path, index=False)
        logger.info("💾 Saved → %s", out_path.resolve())

    return season_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect F1 season data via FastF1.",
    )
    parser.add_argument(
        "--year", type=int, nargs="+", default=[2026],
        help="Season year(s) to collect (e.g. --year 2023 2024 2025 2026).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip writing CSV output to data/raw_{year}.csv.",
    )
    args = parser.parse_args()

    for yr in args.year:
        collect_season(yr, save=not args.no_save)