"""
src/retrain.py  (Final)
~~~~~~~~~~~~~~~~~~~~~~~~
Incremental retraining pipeline for the F1 2026 Race Outcome Predictor.

Run after every completed race weekend:
    python src/retrain.py --year 2026 --race "Spanish Grand Prix"

Pipeline
--------
1. Fetch new race result via data_loader.collect_round()
2. Append to raw_{year}.csv (skip if already present)
3. Re-engineer features via features.build_features()
4. Recompute scale_pos_weight dynamically
5. Race-level TimeSeriesSplit CV
6. Train final model on ALL data
7. Overwrite model.pkl
8. Update model_metadata.json:
       last_updated_race, races_used, scale_pos_weight, all CV metrics
9. Print summary line
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    DATA_DIR,
    META_PATH,
    MODEL_PKL_PATH,
    WEIGHT_2026,
    WEIGHT_PRE_2026,
)
from data_loader import collect_round
from features import build_features
from model import (
    _read_feature_cols,
    _sort_and_validate,
    _time_series_cv,
    _plot_feature_importance,
    _plot_roc_curve,
    _plot_pr_curve,
    _plot_cv_metrics,
    compute_scale_pos_weight,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA APPEND
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_round_number(year: int, race_name: str | int) -> int:
    """
    Resolve race_name (string or int) to a FastF1 round number.

    If race_name is already an int, return it directly.
    If string, look it up in the event schedule.
    """
    if isinstance(race_name, int):
        return race_name
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        match = schedule[
            schedule["EventName"].str.lower() == race_name.lower()
        ]
        if match.empty:
            # Partial match fallback
            match = schedule[
                schedule["EventName"].str.lower().str.contains(
                    race_name.lower(), na=False
                )
            ]
        if match.empty:
            raise ValueError(
                f"Race '{race_name}' not found in {year} schedule. "
                f"Available: {schedule['EventName'].tolist()}"
            )
        return int(match["RoundNumber"].iloc[0])
    except Exception as exc:
        raise RuntimeError(f"Could not resolve round number: {exc}") from exc


def append_new_race(year: int, race_name: str | int) -> tuple[pd.DataFrame, int]:
    """
    Fetch newly completed race via FastF1 and append to raw_{year}.csv.

    Skips fetch if the round already exists in the CSV to prevent duplicates.

    Parameters
    ----------
    year : int
    race_name : str or int
        Grand Prix name or round number.

    Returns
    -------
    updated_df : pd.DataFrame
        Full season DataFrame after appending.
    round_number : int
        Resolved round number.
    """
    round_number = _resolve_round_number(year, race_name)
    csv_path     = DATA_DIR / f"raw_{year}.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if ((existing["year"] == year) & (existing["round"] == round_number)).any():
            logger.info(
                "Round %d / %d already in %s — skipping fetch.",
                round_number, year, csv_path.name,
            )
            return existing, round_number
    else:
        existing = pd.DataFrame()

    logger.info("Fetching %d Round %d (%s) from FastF1...", year, round_number, race_name)
    new_round = collect_round(year, round_number)

    if new_round is None or new_round.empty:
        raise RuntimeError(
            f"FastF1 returned no data for {year} Round {round_number}. "
            "Race may not be completed yet."
        )

    updated = pd.concat([existing, new_round], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    logger.info(
        "💾 Appended Round %d → %s  (%d total rows)",
        round_number, csv_path, len(updated),
    )
    return updated, round_number


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def retrain_after_race(
    year: int,
    race_name: str | int,
    all_years: list[int] | None = None,
    n_cv_splits: int = 5,
    save_model: bool = True,
) -> tuple[XGBClassifier, dict]:
    """
    Full incremental retraining pipeline after a completed race.

    Parameters
    ----------
    year : int
        Season year of the new race.
    race_name : str or int
        Grand Prix name (e.g. "Spanish Grand Prix") or round number.
    all_years : list[int], optional
        All seasons to include. Defaults to [2022, 2023, 2024, 2025, 2026].
    n_cv_splits : int
        Race-level TimeSeriesSplit folds (default 5).
    save_model : bool
        If True, overwrite model.pkl.

    Returns
    -------
    model : XGBClassifier
    metadata : dict

    Side effects
    ------------
    - Appends new race to data/raw_{year}.csv
    - Overwrites model.pkl
    - Updates model_metadata.json with last_updated_race, races_used,
      scale_pos_weight, and all CV metrics
    - Prints summary line to stdout
    """
    if all_years is None:
        all_years = [2022, 2023, 2024, 2025, 2026]

    logger.info("━━━ RETRAIN: %s %d ━━━", race_name, year)

    # ── Step 1: Append new race data ──────────────────────────────────────
    _, round_number = append_new_race(year, race_name)

    # ── Step 2: Load all raw data ─────────────────────────────────────────
    frames = []
    for yr in all_years:
        p = DATA_DIR / f"raw_{yr}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
            logger.info("Loaded %s (%d rows)", p.name, len(frames[-1]))
        else:
            logger.warning("Missing %s — skipping.", p.name)

    if not frames:
        raise FileNotFoundError("No raw data found. Check data_loader output.")

    df_raw = pd.concat(frames, ignore_index=True)

    # ── Step 3: Feature engineering ───────────────────────────────────────
    X, y, weights = build_features(df_raw, write_metadata=True)
    feature_cols  = _read_feature_cols()
    missing       = [c for c in feature_cols if c not in X.columns]
    if missing:
        logger.warning("Dropping missing features: %s", missing)
        feature_cols = [c for c in feature_cols if c in X.columns]

    # ── Step 4: Sort + validate ───────────────────────────────────────────
    X, y, weights, df_sorted = _sort_and_validate(X, y, weights, df_raw)

    # ── Step 5: Dynamic scale_pos_weight ──────────────────────────────────
    scale_pos_weight = compute_scale_pos_weight(y)

    xgb_params = dict(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.03,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = "logloss",
        scale_pos_weight  = scale_pos_weight,
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )

    # ── Step 6: Race-level CV ─────────────────────────────────────────────
    cv_results = _time_series_cv(
        X=X, y=y, weights=weights,
        df_sorted=df_sorted,
        feature_cols=feature_cols,
        xgb_params=xgb_params,
        n_splits=n_cv_splits,
    )
    _plot_cv_metrics(cv_results)

    # ── Step 7: Final model on ALL data ───────────────────────────────────
    logger.info("Training final model on %d rows...", len(X))
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(
        X[feature_cols], y,
        sample_weight=weights,
        verbose=False,
    )

    # ── Step 8: Evaluation charts ─────────────────────────────────────────
    _plot_feature_importance(final_model, feature_cols)
    y_test_last, y_prob_last = cv_results["last_test"]
    _plot_roc_curve(y_test_last, y_prob_last)
    _plot_pr_curve(y_test_last, y_prob_last)

    # ── Step 9: Save model ────────────────────────────────────────────────
    if save_model:
        joblib.dump(final_model, MODEL_PKL_PATH)
        logger.info("💾 Model saved → %s", MODEL_PKL_PATH)

    # ── Step 10: Update model_metadata.json ───────────────────────────────
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    mean     = cv_results["mean"]
    n_pos    = int((y == 1).sum())
    n_neg    = int((y == 0).sum())
    n_races  = int(df_sorted[["year", "round"]].drop_duplicates().shape[0])

    # Resolve event name for metadata
    event_name_str = (
        race_name if isinstance(race_name, str)
        else f"{year} Round {round_number}"
    )

    metadata.update({
        # ── Key fields updated after every retrain ──────────────────────
        "last_updated_race" : {
            "year"       : year,
            "round"      : round_number,
            "event_name" : event_name_str,
            "updated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "races_used"        : n_races,          # ← updated
        "num_samples"       : int(X.shape[0]),
        # ── Model metrics ───────────────────────────────────────────────
        "model_accuracy"    : round(float(mean["avg_precision"]), 4),
        "avg_precision"     : round(float(mean["avg_precision"]), 4),
        "accuracy_score"    : round(float(mean["accuracy"]), 4),
        "precision_score"   : round(float(mean["precision"]), 4),
        "recall_score"      : round(float(mean["recall"]), 4),
        "f1_score"          : round(float(mean["f1"]), 4),
        "roc_auc_score"     : round(float(mean["roc_auc"]), 4),
        # ── Class balance (dynamic) ─────────────────────────────────────
        "class_balance"     : {
            "pos_samples"      : n_pos,
            "neg_samples"      : n_neg,
            "scale_pos_weight" : scale_pos_weight,   # ← recomputed
            "top3_rate"        : round(float(y.mean()), 4),
        },
        "safe_feature_cols" : feature_cols,
        "xgb_params"        : xgb_params,
        "training_date"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_weights"    : {
            "2026"    : WEIGHT_2026,
            "pre_2026": WEIGHT_PRE_2026,
        },
    })

    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("💾 Metadata updated → %s", META_PATH)

    # ── Step 11: Summary print ────────────────────────────────────────────
    ap_score = round(float(mean["avg_precision"]), 4)
    print(
        f"\n✅ Model updated after {event_name_str}. "
        f"Races used: {n_races}. "
        f"New accuracy (AP): {ap_score:.1%}"
    )

    return final_model, metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrain F1 predictor after a completed race."
    )
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--race",  type=str, required=True,
                        help='e.g. "Spanish Grand Prix" or round number')
    parser.add_argument(
        "--all-years", type=int, nargs="+",
        default=[2022, 2023, 2024, 2025, 2026],
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    retrain_after_race(
        year         = args.year,
        race_name    = args.race,
        all_years    = args.all_years,
        n_cv_splits  = args.cv_splits,
        save_model   = not args.no_save,
    )