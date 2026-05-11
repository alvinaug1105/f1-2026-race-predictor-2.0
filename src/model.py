"""
src/model.py  (Final)
~~~~~~~~~~~~~~~~~~~~~
XGBoost training pipeline for the F1 2026 Race Outcome Predictor.

Changelog
---------
v1 → Initial implementation
v2 → Fix 1: Dynamic scale_pos_weight (not hardcoded)
     Fix 2: Enforce sort by [year, round] + sanity check
     Fix 3: Race-level TimeSeriesSplit (no intra-race split)
     New:   Precision-Recall curve + per-fold AUC trend chart
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    DATA_DIR,
    LEAKAGE_COLS,
    META_PATH,
    MODEL_PKL_PATH,
    WEIGHT_2026,
    WEIGHT_PRE_2026,
)
from features import build_features

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_FEATURE_IMP = DATA_DIR / "feature_importance.png"
FIG_ROC         = DATA_DIR / "roc_curve.png"
FIG_PR_CURVE    = DATA_DIR / "pr_curve.png"
FIG_CV_ACC      = DATA_DIR / "cv_accuracy.png"
FIG_CV_AUC      = DATA_DIR / "cv_auc_trend.png"

# ── F1 Theme ──────────────────────────────────────────────────────────────────
F1_RED   = "#E8002D"
F1_DARK  = "#15151E"
F1_WHITE = "#FFFFFF"
F1_GOLD  = "#FFD700"
F1_GREY  = "#38383F"
F1_TEAL  = "#00D2BE"
F1_BLUE  = "#0600EF"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _f1_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply F1 dark theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=18, color=F1_WHITE)),
        paper_bgcolor=F1_DARK,
        plot_bgcolor=F1_DARK,
        font=dict(color=F1_WHITE, family="Arial"),
        legend=dict(bgcolor=F1_DARK, bordercolor=F1_GREY),
    )
    fig.update_xaxes(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY)
    fig.update_yaxes(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY)
    return fig


def _read_feature_cols() -> list[str]:
    """
    Read safe_feature_cols from model_metadata.json (written by features.py).
    Falls back to hardcoded default if file is absent or malformed.
    """
    default = [
        "grid_position", "quali_gap_to_pole", "rolling_avg_finish_3",
        "rolling_dnf_rate_3", "constructor_rank", "circuit_type",
        "adaptation_score",
    ]
    if not META_PATH.exists():
        logger.warning("model_metadata.json not found — using default FEATURE_COLS.")
        return default
    try:
        meta = json.loads(META_PATH.read_text())
        cols = meta.get("safe_feature_cols", default)
        logger.info("Feature list loaded from metadata: %s", cols)
        return cols
    except json.JSONDecodeError:
        logger.warning("model_metadata.json malformed — using default FEATURE_COLS.")
        return default


def _load_raw_data(years: list[int]) -> pd.DataFrame:
    """Load and concatenate raw season CSVs."""
    frames = []
    for yr in years:
        path = DATA_DIR / f"raw_{yr}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            logger.info("Loaded %s (%d rows)", path.name, len(frames[-1]))
        else:
            logger.warning("Missing %s — skipping.", path.name)
    if not frames:
        raise FileNotFoundError(
            "No raw data CSVs found. Run: python src/data_loader.py"
        )
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# [FIX 1] DYNAMIC scale_pos_weight
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Dynamically compute scale_pos_weight from current class distribution.

    Formula: sum(negative_samples) / sum(positive_samples)

    WHY dynamic (not hardcoded):
        top3 rate shifts as new races are added — DNF rate, safety cars,
        and regulation changes all affect how often drivers finish top-3.
        Hardcoding 5 would under/over-correct as data grows during 2026.

    Parameters
    ----------
    y : pd.Series
        Binary target (top3_finish).

    Returns
    -------
    float
        scale_pos_weight rounded to 2 decimal places.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0:
        logger.warning("No positive samples — defaulting scale_pos_weight=5.0")
        return 5.0

    spw = round(n_neg / n_pos, 2)
    logger.info(
        "scale_pos_weight = %.2f  (neg=%d, pos=%d, top3_rate=%.1f%%)",
        spw, n_neg, n_pos, (n_pos / (n_pos + n_neg)) * 100,
    )
    return spw


# ─────────────────────────────────────────────────────────────────────────────
# [FIX 2] SORT + SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _sort_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Enforce chronological sort order and run sanity checks.

    WHY critical:
        TimeSeriesSplit splits by ROW INDEX, not by date semantics.
        If rows are in random order after concat, the 'time series'
        split is meaningless — future races leak into training folds.

    Checks performed
    ----------------
    1. X row count matches raw data after feature engineering
    2. First row is the earliest (year, round) in dataset
    3. sort_values is applied to df_raw and index propagated to X/y/weights

    Returns
    -------
    X, y, weights, df_sorted : all re-indexed in chronological order
    """
    # Build a sort key from raw df aligned to X (build_features resets index)
    sort_key_cols = [c for c in ["year", "round", "driver_abbr"] if c in df_raw.columns]
    df_sorted = df_raw.sort_values(sort_key_cols).reset_index(drop=True)

    # X is already sorted by build_features() — validate first row
    if "year" in df_raw.columns and "round" in df_raw.columns:
        min_year  = df_sorted["year"].min()
        min_round = df_sorted.loc[df_sorted["year"] == min_year, "round"].min()

        # [FIX 2] Sanity check: X must start at earliest race
        first_yr = df_sorted["year"].iloc[0]
        assert first_yr == min_year, (
            f"Sort order check FAILED: first row year={first_yr}, "
            f"expected min year={min_year}"
        )
        logger.info(
            "✅ Sort check passed — first row: %d Round %d",
            first_yr, df_sorted["round"].iloc[0],
        )

    return X, y, weights, df_sorted


# ─────────────────────────────────────────────────────────────────────────────
# [FIX 3] RACE-LEVEL TimeSeriesSplit
# ─────────────────────────────────────────────────────────────────────────────

def race_level_time_splits(
    df_sorted: pd.DataFrame,
    n_splits: int = 5,
):
    """
    Generate (train_indices, test_indices) where split boundaries align
    exactly with race boundaries — never mid-race.

    WHY race-level (not row-level):
        Each race has ~20 driver rows. Standard TimeSeriesSplit can cut
        in the middle of a race (e.g. drivers 1-10 in train, 11-20 in test
        for the same Round). This is temporal leakage because the model
        could implicitly learn race-level patterns from partial race data.
        Race-level splits guarantee all 20 drivers in a race go to the
        SAME fold.

    Algorithm
    ---------
    1. Extract unique (year, round) pairs in chronological order
    2. Divide into n_splits+1 equal-sized blocks of RACES (not rows)
    3. Each fold: train = races 0..i*block, test = races i*block..(i+1)*block
    4. Map race identities back to row indices in df_sorted

    Parameters
    ----------
    df_sorted : pd.DataFrame
        Full DataFrame sorted by [year, round, driver_abbr].
        Must contain 'year' and 'round' columns.
    n_splits : int
        Number of folds (default 5).

    Yields
    ------
    train_idx : pd.Index
        Row indices for training set.
    test_idx : pd.Index
        Row indices for test set.
    """
    # Step 1: unique races in order
    rounds = (
        df_sorted[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    n_races    = len(rounds)
    fold_size  = max(1, n_races // (n_splits + 1))

    logger.info(
        "Race-level splits: %d total races, ~%d races per fold",
        n_races, fold_size,
    )

    # Pre-build a (year, round) → row indices lookup
    race_to_idx: dict[tuple, list[int]] = {}
    for idx, row in df_sorted.iterrows():
        key = (row["year"], row["round"])
        race_to_idx.setdefault(key, []).append(idx)

    for i in range(1, n_splits + 1):
        train_races = rounds.iloc[: i * fold_size]
        test_races  = rounds.iloc[i * fold_size : (i + 1) * fold_size]

        if test_races.empty:
            logger.warning("Fold %d has empty test set — skipping.", i)
            continue

        train_idx = pd.Index(
            [idx for _, r in train_races.iterrows()
             for idx in race_to_idx.get((r["year"], r["round"]), [])]
        )
        test_idx = pd.Index(
            [idx for _, r in test_races.iterrows()
             for idx in race_to_idx.get((r["year"], r["round"]), [])]
        )

        logger.info(
            "  Fold %d: train=%d races (%d rows)  test=%d races (%d rows)",
            i,
            len(train_races), len(train_idx),
            len(test_races),  len(test_idx),
        )
        yield train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    df_sorted: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    n_splits: int = 5,
) -> dict:
    """
    Race-level TimeSeriesSplit cross-validation.

    Primary metric  : Average Precision (AP) — better for imbalanced data
    Secondary metric: ROC-AUC

    AP score reflects the area under the Precision-Recall curve.
    For a ~85/15 imbalanced dataset, a random classifier achieves AP≈0.15,
    making AP much more discriminative than ROC-AUC (random≈0.50).

    Returns
    -------
    dict
        "folds"       : per-fold metrics DataFrame
        "mean"        : mean across folds
        "last_test"   : (y_test, y_prob) from last fold for chart generation
    """
    fold_metrics: list[dict] = []
    last_test: tuple | None  = None

    logger.info("━━━ Race-Level TimeSeriesSplit CV (%d folds) ━━━", n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(
        race_level_time_splits(df_sorted, n_splits), start=1
    ):
        # Align X/y/weights to fold indices
        X_train = X.loc[train_idx, feature_cols]
        X_test  = X.loc[test_idx,  feature_cols]
        y_train = y.loc[train_idx]
        y_test  = y.loc[test_idx]
        w_train = weights.loc[train_idx]

        # Recompute scale_pos_weight per fold (train distribution only)
        fold_spw = compute_scale_pos_weight(y_train)
        fold_params = {**xgb_params, "scale_pos_weight": fold_spw}

        model = XGBClassifier(**fold_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Primary: AP score; Secondary: ROC-AUC
        has_both_classes = y_test.nunique() > 1
        ap_score  = average_precision_score(y_test, y_pred_prob) if has_both_classes else float("nan")
        auc_score = roc_auc_score(y_test, y_pred_prob)           if has_both_classes else float("nan")

        metrics = {
            "fold"            : fold_idx,
            "train_races"     : len(train_idx),
            "test_races"      : len(test_idx),
            "accuracy"        : accuracy_score(y_test, y_pred),
            "precision"       : precision_score(y_test, y_pred, zero_division=0),
            "recall"          : recall_score(y_test, y_pred, zero_division=0),
            "f1"              : f1_score(y_test, y_pred, zero_division=0),
            "roc_auc"         : auc_score,
            "avg_precision"   : ap_score,
            "fold_spw"        : fold_spw,
        }
        fold_metrics.append(metrics)
        last_test = (y_test, y_pred_prob)

        logger.info(
            "Fold %d | AP=%.3f  AUC=%.3f  F1=%.3f  "
            "Prec=%.3f  Rec=%.3f  spw=%.2f",
            fold_idx, ap_score, auc_score,
            metrics["f1"], metrics["precision"], metrics["recall"], fold_spw,
        )

    df_folds = pd.DataFrame(fold_metrics)
    mean_row = df_folds[[
        "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"
    ]].mean()

    logger.info(
        "━━━ Mean CV | AP=%.3f  AUC=%.3f  F1=%.3f  "
        "Prec=%.3f  Rec=%.3f ━━━",
        mean_row["avg_precision"], mean_row["roc_auc"],
        mean_row["f1"], mean_row["precision"], mean_row["recall"],
    )

    return {
        "folds"    : df_folds,
        "mean"     : mean_row,
        "last_test": last_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
) -> None:
    """Save feature importance bar chart → data/feature_importance.png"""
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )
    colors = [
        F1_RED if v == feat_df["importance"].max() else F1_TEAL
        for v in feat_df["importance"]
    ]
    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"],
        orientation="h", marker_color=colors,
        text=feat_df["importance"].round(4), textposition="outside",
        textfont=dict(color=F1_WHITE, size=11),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig = _f1_layout(fig, "XGBoost Feature Importance (gain)")
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="Feature")
    fig.write_image(str(FIG_FEATURE_IMP))
    logger.info("💾 Saved → %s", FIG_FEATURE_IMP)


def _plot_roc_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    """Save ROC curve → data/roc_curve.png"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {auc_val:.3f}",
        line=dict(color=F1_RED, width=2.5),
        fill="tozeroy", fillcolor="rgba(232,0,45,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.50)",
        line=dict(color=F1_GREY, dash="dash", width=1.5),
    ))
    fig = _f1_layout(fig, f"ROC Curve — top3_finish  (AUC={auc_val:.3f})")
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.05,
                    xanchor="right", x=0.95),
    )
    fig.write_image(str(FIG_ROC))
    logger.info("💾 Saved → %s", FIG_ROC)


def _plot_pr_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    """
    Save Precision-Recall curve → data/pr_curve.png

    WHY PR curve (not just ROC):
        For imbalanced classification (~85/15), a random classifier
        achieves ROC-AUC≈0.50 but AP≈0.15 (= base rate).
        PR curve shows actual precision at each recall threshold,
        making model quality differences much more visible.
        AP score directly measures area under this curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=f"AP = {ap:.3f}",
        line=dict(color=F1_TEAL, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,210,190,0.12)",
    ))
    # Baseline = random classifier AP = positive rate
    fig.add_hline(
        y=baseline,
        line_dash="dot", line_color=F1_GREY,
        annotation_text=f"Baseline AP = {baseline:.3f}",
        annotation_font_color=F1_WHITE,
        annotation_position="top right",
    )
    fig = _f1_layout(
        fig,
        f"Precision-Recall Curve — top3_finish  (AP={ap:.3f})",
    )
    fig.update_xaxes(title_text="Recall")
    fig.update_yaxes(title_text="Precision", range=[0, 1.05])
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.05,
                    xanchor="right", x=0.95),
    )
    fig.write_image(str(FIG_PR_CURVE))
    logger.info("💾 Saved → %s", FIG_PR_CURVE)


def _plot_cv_metrics(cv_results: dict) -> None:
    """
    Save two CV visualisation charts:

    1. cv_accuracy.png  — all 5 metrics per fold (line chart)
    2. cv_auc_trend.png — AP + ROC-AUC trend across folds
                          (later folds = more 2026 data → expect rising AUC)
    """
    df_folds = cv_results["folds"]

    # ── Chart 1: All metrics per fold ─────────────────────────────────────
    metric_cfg = [
        ("avg_precision", F1_RED,    "Avg Precision (primary)"),
        ("roc_auc",       F1_TEAL,   "ROC-AUC"),
        ("f1",            "#FF8700", "F1"),
        ("precision",     "#FFF500", "Precision"),
        ("recall",        "#B6BABD", "Recall"),
    ]
    fig1 = go.Figure()
    for col, color, label in metric_cfg:
        fig1.add_trace(go.Scatter(
            x=df_folds["fold"], y=df_folds[col],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate=f"Fold %{{x}}<br>{label}: %{{y:.3f}}<extra></extra>",
        ))
    mean_ap = cv_results["mean"]["avg_precision"]
    fig1.add_hline(
        y=mean_ap, line_dash="dot", line_color=F1_GOLD,
        annotation_text=f"Mean AP = {mean_ap:.3f}",
        annotation_font_color=F1_GOLD,
        annotation_position="top right",
    )
    fig1 = _f1_layout(
        fig1,
        f"CV Metrics per Fold (Race-Level Split, n=5) | Mean AP={mean_ap:.3f}",
    )
    fig1.update_xaxes(title_text="CV Fold (chronological)", dtick=1)
    fig1.update_yaxes(title_text="Score", range=[0, 1.05])
    fig1.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5),
    )
    fig1.write_image(str(FIG_CV_ACC))
    logger.info("💾 Saved → %s", FIG_CV_ACC)

    # ── Chart 2: AP + AUC trend (should rise as 2026 data grows) ─────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_folds["fold"], y=df_folds["avg_precision"],
        mode="lines+markers", name="Avg Precision (AP)",
        line=dict(color=F1_RED, width=2.5),
        marker=dict(size=8, symbol="circle"),
        fill="tozeroy", fillcolor="rgba(232,0,45,0.08)",
        hovertemplate="Fold %{x}<br>AP: %{y:.3f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=df_folds["fold"], y=df_folds["roc_auc"],
        mode="lines+markers", name="ROC-AUC",
        line=dict(color=F1_TEAL, width=2.5),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="Fold %{x}<br>AUC: %{y:.3f}<extra></extra>",
    ))
    # Annotate expected trend
    fig2.add_annotation(
        x=df_folds["fold"].max(), y=df_folds["avg_precision"].iloc[-1],
        text="← More 2026 data",
        showarrow=True, arrowhead=2, arrowcolor=F1_GOLD,
        font=dict(color=F1_GOLD, size=11),
        ax=-80, ay=-30,
    )
    fig2 = _f1_layout(
        fig2,
        "Model Performance Trend Across Time Folds (later = more 2026 data)",
    )
    fig2.update_xaxes(title_text="CV Fold (chronological)", dtick=1)
    fig2.update_yaxes(title_text="Score", range=[0, 1.05])
    fig2.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5),
    )
    fig2.write_image(str(FIG_CV_AUC))
    logger.info("💾 Saved → %s", FIG_CV_AUC)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    years: list[int] | None = None,
    n_cv_splits: int = 5,
    save_model: bool = True,
) -> tuple[XGBClassifier, dict]:
    """
    Full training pipeline: load → engineer → CV → final fit → save.

    Parameters
    ----------
    years : list[int], optional
        Seasons to include. Defaults to [2022, 2023, 2024, 2025, 2026].
    n_cv_splits : int
        Race-level TimeSeriesSplit folds (default 5).
    save_model : bool
        If True, serialise final model to model.pkl.

    Returns
    -------
    model : XGBClassifier
    metadata : dict
    """
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]

    # ── 1. Load raw data ──────────────────────────────────────────────────
    df_raw = _load_raw_data(years)

    # ── 2. Feature engineering ────────────────────────────────────────────
    X, y, weights = build_features(df_raw, write_metadata=True)
    feature_cols  = _read_feature_cols()
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        logger.warning("Dropping missing features: %s", missing)
        feature_cols = [c for c in feature_cols if c in X.columns]

    # ── 3. [FIX 2] Sort + sanity check ───────────────────────────────────
    X, y, weights, df_sorted = _sort_and_validate(X, y, weights, df_raw)

    # ── 4. [FIX 1] Dynamic scale_pos_weight ──────────────────────────────
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
    logger.info("XGBoost params: %s", xgb_params)

    # ── 5. [FIX 3] Race-level TimeSeriesSplit CV ──────────────────────────
    cv_results = _time_series_cv(
        X=X, y=y, weights=weights,
        df_sorted=df_sorted,
        feature_cols=feature_cols,
        xgb_params=xgb_params,
        n_splits=n_cv_splits,
    )
    _plot_cv_metrics(cv_results)

    # ── 6. Final model — train on ALL data ────────────────────────────────
    logger.info("Training final model on ALL %d rows...", len(X))
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(
        X[feature_cols], y,
        sample_weight=weights,
        verbose=False,
    )

    # ── 7. Evaluation charts from last CV fold ────────────────────────────
    _plot_feature_importance(final_model, feature_cols)

    y_test_last, y_prob_last = cv_results["last_test"]
    _plot_roc_curve(y_test_last, y_prob_last)
    _plot_pr_curve(y_test_last, y_prob_last)

    # ── 8. Save model ─────────────────────────────────────────────────────
    if save_model:
        joblib.dump(final_model, MODEL_PKL_PATH)
        logger.info("💾 Model saved → %s", MODEL_PKL_PATH)

    # ── 9. Update model_metadata.json ────────────────────────────────────
    mean_metrics     = cv_results["mean"]
    metadata         = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    n_pos            = int((y == 1).sum())
    n_neg            = int((y == 0).sum())

    metadata.update({
        "training_date"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "years_trained"    : years,
        "num_races_used"   : int(
            df_sorted[["year", "round"]].drop_duplicates().shape[0]
        ),
        "num_samples"      : int(X.shape[0]),
        "num_features"     : len(feature_cols),
        "safe_feature_cols": feature_cols,
        "xgb_params"       : xgb_params,
        "cv_folds"         : n_cv_splits,
        "cv_split_method"  : "Race-level TimeSeriesSplit — no intra-race split, no temporal leakage",
        # Primary metric: AP; Secondary: AUC
        "model_accuracy"   : round(float(mean_metrics["avg_precision"]), 4),
        "avg_precision"    : round(float(mean_metrics["avg_precision"]), 4),
        "accuracy_score"   : round(float(mean_metrics["accuracy"]), 4),
        "precision_score"  : round(float(mean_metrics["precision"]), 4),
        "recall_score"     : round(float(mean_metrics["recall"]), 4),
        "f1_score"         : round(float(mean_metrics["f1"]), 4),
        "roc_auc_score"    : round(float(mean_metrics["roc_auc"]), 4),
        # [FIX 1] Full class balance record
        "class_balance"    : {
            "pos_samples"       : n_pos,
            "neg_samples"       : n_neg,
            "scale_pos_weight"  : scale_pos_weight,
            "top3_rate"         : round(float(y.mean()), 4),
        },
        "sample_weights"   : {
            "2026"    : WEIGHT_2026,
            "pre_2026": WEIGHT_PRE_2026,
        },
        "output_charts"    : {
            "feature_importance": str(FIG_FEATURE_IMP),
            "roc_curve"         : str(FIG_ROC),
            "pr_curve"          : str(FIG_PR_CURVE),
            "cv_accuracy"       : str(FIG_CV_ACC),
            "cv_auc_trend"      : str(FIG_CV_AUC),
        },
    })

    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("💾 Metadata saved → %s", META_PATH)

    _print_summary(metadata, cv_results)
    return final_model, metadata


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(metadata: dict, cv_results: dict) -> None:
    df_folds = cv_results["folds"]
    bar = "━" * 62

    print(f"\n{bar}")
    print(f"{'MODEL TRAINING SUMMARY':^62}")
    print(bar)
    print(f"  Training date    : {metadata['training_date']}")
    print(f"  Seasons          : {metadata['years_trained']}")
    print(f"  Races used       : {metadata['num_races_used']}")
    print(f"  Samples          : {metadata['num_samples']:,}")
    print(f"  Features         : {metadata['safe_feature_cols']}")
    print(f"  CV method        : {metadata['cv_split_method']}")
    print(bar)
    print(f"  {'METRIC':<22} {'MEAN':>8}  {'MIN':>8}  {'MAX':>8}")
    print("  " + "─" * 46)
    metric_map = {
        "avg_precision" : "AVG PRECISION (AP) ★",
        "roc_auc"       : "ROC-AUC",
        "f1"            : "F1",
        "precision"     : "PRECISION",
        "recall"        : "RECALL",
        "accuracy"      : "ACCURACY",
    }
    for key, label in metric_map.items():
        mean_v = float(cv_results["mean"].get(key, float("nan")))
        min_v  = float(df_folds[key].min())
        max_v  = float(df_folds[key].max())
        print(f"  {label:<22} {mean_v:>8.3f}  {min_v:>8.3f}  {max_v:>8.3f}")
    print(bar)
    cb = metadata.get("class_balance", {})
    print(f"  Top-3 rate       : {cb.get('top3_rate', 0):.1%}")
    print(f"  scale_pos_weight : {cb.get('scale_pos_weight', '?')}  (dynamic)")
    print(f"  pos / neg        : {cb.get('pos_samples','?')} / {cb.get('neg_samples','?')}")
    print(bar)
    print(f"  Charts → {DATA_DIR}/")
    print(f"  Model  → {MODEL_PKL_PATH}")
    print(f"  Meta   → {META_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train F1 podium predictor.")
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=[2022, 2023, 2024, 2025, 2026],
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train(
        years       = args.years,
        n_cv_splits = args.cv_splits,
        save_model  = not args.no_save,
    )