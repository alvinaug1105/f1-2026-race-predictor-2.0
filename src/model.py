"""
src/model.py (Optuna + LightGBM Ensemble Upgrade)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Upgrades:
- Optuna bayesian hyperparameter tuning for XGBoost + LightGBM
- Weighted ensemble of calibrated XGB + LGB models
- CalibratedClassifierCV for better probability output
- All existing charts + race-level TimeSeriesSplit preserved
"""
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import CalibratedClassifierCV
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
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import DATA_DIR, LEAKAGE_COLS, META_PATH, MODEL_PKL_PATH, WEIGHT_2026, WEIGHT_PRE_2026
from features import build_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_FEATURE_IMP = DATA_DIR / "feature_importance.png"
FIG_ROC = DATA_DIR / "roc_curve.png"
FIG_PR_CURVE = DATA_DIR / "pr_curve.png"
FIG_CV_ACC = DATA_DIR / "cv_accuracy.png"
FIG_CV_AUC = DATA_DIR / "cv_auc_trend.png"

F1_RED = "#E8002D"
F1_DARK = "#15151E"
F1_WHITE = "#FFFFFF"
F1_GOLD = "#FFD700"
F1_GREY = "#38383F"
F1_TEAL = "#00D2BE"


def _f1_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color=F1_WHITE)),
        paper_bgcolor=F1_DARK, plot_bgcolor=F1_DARK,
        font=dict(color=F1_WHITE, family="Arial"),
        legend=dict(bgcolor=F1_DARK, bordercolor=F1_GREY),
    )
    fig.update_xaxes(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY)
    fig.update_yaxes(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY)
    return fig


def _read_feature_cols() -> list[str]:
    default = [
        "grid_position", "quali_gap_to_pole", "rolling_avg_finish_3",
        "rolling_dnf_rate_3", "constructor_rank", "circuit_type",
        "adaptation_score", "pit_stop_count",
        "overtake_difficulty", "driver_circuit_avg_pos",
        "is_wet_race", "avg_air_temp", "avg_humidity",
    ]
    if not META_PATH.exists():
        return default
    try:
        meta = json.loads(META_PATH.read_text())
        return meta.get("safe_feature_cols", default)
    except json.JSONDecodeError:
        return default


def _load_raw_data(years: list[int]) -> pd.DataFrame:
    frames = []
    for yr in years:
        path = DATA_DIR / f"raw_{yr}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            logger.info("Loaded %s (%d rows)", path.name, len(frames[-1]))
        else:
            logger.warning("Missing %s — skipping.", path.name)
    if not frames:
        raise FileNotFoundError("No raw data CSVs found. Run: python src/data_loader.py")
    return pd.concat(frames, ignore_index=True)


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 5.0
    spw = round(n_neg / n_pos, 2)
    logger.info("scale_pos_weight = %.2f (neg=%d, pos=%d)", spw, n_neg, n_pos)
    return spw


def _sort_and_validate(X, y, weights, df_raw):
    sort_key_cols = [c for c in ["year", "round", "driver_abbr"] if c in df_raw.columns]
    df_sorted = df_raw.sort_values(sort_key_cols).reset_index(drop=True)
    if "year" in df_raw.columns:
        min_year = df_sorted["year"].min()
        assert df_sorted["year"].iloc[0] == min_year
        logger.info("✅ Sort check passed — first row: %d Round %d", df_sorted["year"].iloc[0], df_sorted["round"].iloc[0])
    return X, y, weights, df_sorted


def race_level_time_splits(df_sorted: pd.DataFrame, n_splits: int = 5):
    rounds = (
        df_sorted[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    n_races = len(rounds)
    fold_size = max(1, n_races // (n_splits + 1))
    race_to_idx: dict = {}
    for idx, row in df_sorted.iterrows():
        key = (row["year"], row["round"])
        race_to_idx.setdefault(key, []).append(idx)

    for i in range(1, n_splits + 1):
        train_races = rounds.iloc[: i * fold_size]
        test_races = rounds.iloc[i * fold_size : (i + 1) * fold_size]
        if test_races.empty:
            continue
        train_idx = pd.Index([idx for _, r in train_races.iterrows() for idx in race_to_idx.get((r["year"], r["round"]), [])])
        test_idx = pd.Index([idx for _, r in test_races.iterrows() for idx in race_to_idx.get((r["year"], r["round"]), [])])
        yield train_idx, test_idx


# ── Optuna objective functions ─────────────────────────────────────────────────

def _objective_xgb(trial, X, y, weights, df_sorted, feature_cols, n_splits=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }
    ap_scores = []
    for train_idx, test_idx in race_level_time_splits(df_sorted, n_splits):
        X_tr, X_te = X.loc[train_idx, feature_cols], X.loc[test_idx, feature_cols]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
        w_tr = weights.loc[train_idx]
        spw = compute_scale_pos_weight(y_tr)
        model = XGBClassifier(**params, scale_pos_weight=spw)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        if y_te.nunique() > 1:
            ap_scores.append(average_precision_score(y_te, model.predict_proba(X_te)[:, 1]))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def _objective_lgb(trial, X, y, weights, df_sorted, feature_cols, n_splits=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    ap_scores = []
    for train_idx, test_idx in race_level_time_splits(df_sorted, n_splits):
        X_tr, X_te = X.loc[train_idx, feature_cols], X.loc[test_idx, feature_cols]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
        w_tr = weights.loc[train_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        if y_te.nunique() > 1:
            ap_scores.append(average_precision_score(y_te, model.predict_proba(X_te)[:, 1]))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def _plot_feature_importance(model, feature_cols, suffix="xgb"):
    try:
        importances = model.feature_importances_
    except AttributeError:
        return
    feat_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=True)
    )
    colors = [F1_RED if v == feat_df["importance"].max() else F1_TEAL for v in feat_df["importance"]]
    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"], orientation="h",
        marker_color=colors, text=feat_df["importance"].round(4),
        textposition="outside", textfont=dict(color=F1_WHITE, size=11),
    ))
    fig = _f1_layout(fig, f"Feature Importance ({suffix.upper()})")
    fig.write_image(str(DATA_DIR / f"feature_importance_{suffix}.png"))
    feat_df[["feature", "importance"]].to_csv(DATA_DIR / "feature_importance.csv", index=False)
    logger.info("💾 Feature importance saved")


def _plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_val:.3f}",
                              line=dict(color=F1_RED, width=2.5), fill="tozeroy",
                              fillcolor="rgba(232,0,45,0.15)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(color=F1_GREY, dash="dash")))
    fig = _f1_layout(fig, f"ROC Curve (AUC={auc_val:.3f})")
    fig.write_image(str(FIG_ROC))


def _plot_pr_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"AP={ap:.3f}",
                              line=dict(color=F1_TEAL, width=2.5)))
    fig.add_hline(y=baseline, line_dash="dot", line_color=F1_GREY,
                  annotation_text=f"Baseline AP={baseline:.3f}")
    fig = _f1_layout(fig, f"Precision-Recall Curve (AP={ap:.3f})")
    fig.write_image(str(FIG_PR_CURVE))


def train(
    years: list[int] | None = None,
    n_cv_splits: int = 5,
    n_optuna_trials: int = 50,
    save_model: bool = True,
) -> tuple[dict, dict]:
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]

    df_raw = _load_raw_data(years)
    X, y, weights = build_features(df_raw, write_metadata=True)
    feature_cols = _read_feature_cols()
    feature_cols = [c for c in feature_cols if c in X.columns]

    X, y, weights, df_sorted = _sort_and_validate(X, y, weights, df_raw)
    scale_pos_weight = compute_scale_pos_weight(y)

    # ── Optuna tuning ────────────────────────────────────────────────────────
    logger.info("🔍 Optuna tuning XGBoost (%d trials)...", n_optuna_trials)
    study_xgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(
        lambda t: _objective_xgb(t, X, y, weights, df_sorted, feature_cols, n_cv_splits),
        n_trials=n_optuna_trials, show_progress_bar=True,
    )
    best_xgb_params = {**study_xgb.best_params, "scale_pos_weight": scale_pos_weight,
                       "eval_metric": "logloss", "use_label_encoder": False,
                       "random_state": 42, "n_jobs": -1}
    logger.info("✅ XGBoost best AP: %.4f | Params: %s", study_xgb.best_value, study_xgb.best_params)

    logger.info("🔍 Optuna tuning LightGBM (%d trials)...", n_optuna_trials)
    study_lgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(
        lambda t: _objective_lgb(t, X, y, weights, df_sorted, feature_cols, n_cv_splits),
        n_trials=n_optuna_trials, show_progress_bar=True,
    )
    best_lgb_params = {**study_lgb.best_params, "class_weight": "balanced",
                       "random_state": 42, "verbose": -1, "n_jobs": -1}
    logger.info("✅ LightGBM best AP: %.4f | Params: %s", study_lgb.best_value, study_lgb.best_params)

    # ── Train final models ───────────────────────────────────────────────────
    logger.info("🏁 Training final ensemble on ALL %d rows...", len(X))
    final_xgb = XGBClassifier(**best_xgb_params)
    final_lgb = lgb.LGBMClassifier(**best_lgb_params)
    final_xgb.fit(X[feature_cols], y, sample_weight=weights, verbose=False)
    final_lgb.fit(X[feature_cols], y, sample_weight=weights)

    # ── Calibrate probabilities ─────────────────────────────────────────────
    # 校準概率輸出（cv=None 兼容 sklearn 1.4+）
    cal_xgb = CalibratedClassifierCV(final_xgb, cv=5, method="isotonic")
    cal_lgb = CalibratedClassifierCV(final_lgb, cv=5, method="isotonic")
    cal_xgb.fit(X[feature_cols], y, sample_weight=weights)
    cal_lgb.fit(X[feature_cols], y, sample_weight=weights)

    ensemble = {
        "xgb": cal_xgb,
        "lgb": cal_lgb,
        "xgb_weight": study_xgb.best_value,
        "lgb_weight": study_lgb.best_value,
        "feature_cols": feature_cols,
    }

    # ── Save ensemble ────────────────────────────────────────────────────────
    if save_model:
        joblib.dump(ensemble, MODEL_PKL_PATH)
        logger.info("💾 Ensemble saved → %s", MODEL_PKL_PATH)

    # ── Charts ───────────────────────────────────────────────────────────────
    _plot_feature_importance(final_xgb, feature_cols, suffix="xgb")
    _plot_feature_importance(final_lgb, feature_cols, suffix="lgb")

    # Quick eval on last 20% of data for charts
    split_idx = int(len(X) * 0.8)
    X_eval, y_eval = X.iloc[split_idx:][feature_cols], y.iloc[split_idx:]
    w_xgb = study_xgb.best_value / (study_xgb.best_value + study_lgb.best_value)
    w_lgb = study_lgb.best_value / (study_xgb.best_value + study_lgb.best_value)
    ensemble_prob = (
        w_xgb * cal_xgb.predict_proba(X_eval)[:, 1] +
        w_lgb * cal_lgb.predict_proba(X_eval)[:, 1]
    )
    if y_eval.nunique() > 1:
        _plot_roc_curve(y_eval, ensemble_prob)
        _plot_pr_curve(y_eval, ensemble_prob)

    # ── Save metadata ────────────────────────────────────────────────────────
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    metadata.update({
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "years_trained": years,
        "num_races_used": int(df_sorted[["year", "round"]].drop_duplicates().shape[0]),
        "num_samples": int(X.shape[0]),
        "num_features": len(feature_cols),
        "safe_feature_cols": feature_cols,
        "model_type": "XGBoost+LightGBM Calibrated Ensemble",
        "avg_precision": round(float((study_xgb.best_value + study_lgb.best_value) / 2), 4),
        "xgb_best_ap": round(study_xgb.best_value, 4),
        "lgb_best_ap": round(study_lgb.best_value, 4),
        "xgb_params": best_xgb_params,
        "lgb_params": best_lgb_params,
        "model_performance": {
            "average_precision": round(float((study_xgb.best_value + study_lgb.best_value) / 2), 4),
            "roc_auc": 0.0,
            "cv_fold_ap_scores": [],
            "cv_fold_ap_std": 0.0,
        },
        "class_balance": {
            "pos_samples": n_pos,
            "neg_samples": n_neg,
            "scale_pos_weight": scale_pos_weight,
            "top3_rate": round(float(y.mean()), 4),
        },
        "sample_weights": {"2022": 0.2, "2023": 0.3, "2024": 0.5, "2025": 0.8, "2026": 1.0},
    })
    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("💾 Metadata saved → %s", META_PATH)
    logger.info("\n🏆 Training complete! XGB AP=%.4f | LGB AP=%.4f | Ensemble AP≈%.4f",
                study_xgb.best_value, study_lgb.best_value,
                (study_xgb.best_value + study_lgb.best_value) / 2)
    return ensemble, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per model (default 50, use 100 for max accuracy)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    train(years=args.years, n_cv_splits=args.cv_splits,
          n_optuna_trials=args.trials, save_model=not args.no_save)
