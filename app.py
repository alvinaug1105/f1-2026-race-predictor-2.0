"""
app.py  (v5 — final)
F1 2026 Race Outcome Predictor — Streamlit web app.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="F1 2026 Predictor",
    page_icon="🏎️",
    initial_sidebar_state="expanded",
)

ROOT_DIR  = Path(__file__).resolve().parent
META_PATH = ROOT_DIR / "model_metadata.json"
MODEL_PKL = ROOT_DIR / "model.pkl"
DATA_DIR  = ROOT_DIR / "data"

F1_RED    = "#E8002D"
F1_DARK   = "#15151E"
F1_SIDEBAR = "#1e1e2e"
F1_CARD   = "#2a2a3e"
F1_WHITE  = "#FFFFFF"
F1_GOLD   = "#FFD700"
F1_SILVER = "#C0C0C0"
F1_BRONZE = "#CD7F32"
F1_TEAL   = "#00D2BE"
F1_GREY   = "#38383F"

FEATURE_DESCRIPTIONS = {
    "grid_position":         "Starting grid slot (lower = better)",
    "quali_gap_to_pole":     "Qualifying gap to pole (seconds)",
    "rolling_avg_finish_3":  "Avg finish position, last 3 races",
    "rolling_dnf_rate_3":    "DNF rate, last 3 races",
    "constructor_rank":      "Team championship position at race time",
    "circuit_type":          "Track type: 0=street 1=technical 2=fast",
    "adaptation_score":      "2026 regulation adaptation score (team)",
    "pit_stop_count":        "Avg pit stops, last 3 races",
}

DEFAULT_TEAM_COLORS = {
    "Red Bull":     "#3671C6",
    "Ferrari":      "#E8002D",
    "McLaren":      "#FF8000",
    "Mercedes":     "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine":       "#0093CC",
    "Williams":     "#64C4FF",
    "RB":           "#6692FF",
    "Haas":         "#B6BABD",
    "Kick Sauber":  "#52E252",
}

import sys
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from constants import TEAM_COLORS  # type: ignore
except ImportError:
    TEAM_COLORS = DEFAULT_TEAM_COLORS

try:
    from predict import predict_race  # type: ignore
    PREDICT_AVAILABLE = True
except ImportError:
    PREDICT_AVAILABLE = False


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    css_parts = [
        f".stApp, .main .block-container {{background-color:{F1_DARK};color:{F1_WHITE};}}",
        f"[data-testid='stSidebar'] {{background-color:{F1_SIDEBAR};border-right:1px solid {F1_RED};box-shadow:2px 0 12px rgba(232,0,45,0.15);}}",
        f"[data-testid='stSidebar'] *{{color:{F1_WHITE} !important;}}",
        f"[data-testid='stSidebar'] .stRadio label[aria-checked='true']{{border-left:3px solid {F1_RED} !important;padding-left:10px !important;background-color:{F1_CARD} !important;}}",
        f"[data-testid='stSidebar'] [data-testid='stMetricValue']{{color:{F1_RED} !important;font-size:1.1rem !important;font-weight:700 !important;}}",
        f"[data-testid='stMetric']{{background-color:{F1_CARD};border:1px solid {F1_RED};border-radius:8px;padding:12px 16px;}}",
        f"[data-testid='stMetricValue']{{color:{F1_RED} !important;font-size:1.4rem !important;font-weight:700 !important;}}",
        f"[data-testid='stMetricLabel']{{color:{F1_WHITE} !important;}}",
        f"[data-testid='stDataFrame']{{border:1px solid {F1_GREY};border-radius:6px;}}",
        f"[data-testid='stDataFrame'] thead tr th{{background-color:{F1_RED} !important;color:{F1_WHITE} !important;font-weight:700 !important;}}",
        f"[data-testid='stDataFrame'] tbody tr:nth-child(even) td{{background-color:#2a2a3e;}}",
        f"[data-testid='stDataFrame'] tbody tr:nth-child(odd) td{{background-color:{F1_DARK};}}",
        f"[data-testid='stDataFrame'] tbody tr:hover td{{background-color:{F1_GREY};transition:background-color 0.15s ease;}}",
        f".stButton > button{{background-color:{F1_RED};color:{F1_WHITE};border:none;border-radius:6px;font-weight:700;font-size:1rem;width:100%;padding:0.6rem 1rem;cursor:pointer;transition:background-color 0.2s ease,transform 0.1s ease;}}",
        f".stButton > button:hover{{background-color:#c0001f;color:{F1_WHITE};}}",
        f".stButton > button:active{{background-color:#c0001f;transform:scale(0.98);}}",
        f"h2{{border-left:4px solid {F1_RED};padding-left:12px;letter-spacing:1px;text-shadow:0 0 20px rgba(232,0,45,0.3);}}",
        f"[data-testid='stExpander']{{border:1px solid {F1_GREY};border-radius:8px;transition:border-color 0.2s ease;}}",
        f"[data-testid='stExpander']:hover{{border-color:{F1_RED};}}",
        f".streamlit-expanderHeader{{background-color:{F1_CARD};border-radius:8px;color:{F1_WHITE} !important;}}",
        f".streamlit-expanderHeader svg{{fill:{F1_RED} !important;}}",
        f"[data-testid='stAlert']{{border-radius:8px !important;}}",
        f".element-container div[class*='success']{{border-left:4px solid {F1_TEAL} !important;}}",
        f".element-container div[class*='error']{{border-left:4px solid {F1_RED} !important;}}",
        f".element-container div[class*='info']{{border-left:4px solid #3671C6 !important;}}",
        f".element-container div[class*='warning']{{border-left:4px solid {F1_GOLD} !important;}}",
        f".stSelectbox > div > div,.stFileUploader > div{{background-color:{F1_CARD};border:1px solid {F1_GREY};color:{F1_WHITE};border-radius:6px;}}",
        f".stRadio label{{background-color:{F1_CARD};border:1px solid {F1_GREY};border-radius:6px;padding:0.4rem 0.8rem;cursor:pointer;transition:border-color 0.2s;}}",
        f".stRadio label:hover{{border-color:{F1_RED};}}",
        f"hr{{border-color:{F1_RED};opacity:0.4;}}",
        f"::-webkit-scrollbar{{width:6px;}}",
        f"::-webkit-scrollbar-track{{background:{F1_DARK};}}",
        f"::-webkit-scrollbar-thumb{{background:{F1_RED};border-radius:3px;}}",
    ]
    st.markdown(f"<style>{''.join(css_parts)}</style>", unsafe_allow_html=True)


# ── LOADERS ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_model():
    if not MODEL_PKL.exists():
        return None
    import joblib
    return joblib.load(MODEL_PKL)


@st.cache_resource
def _load_metadata() -> dict:
    defaults = {
        "last_updated_race":   {"event_name": "N/A"},
        "races_used":          0,
        "avg_precision":       0.0,
        "roc_auc_score":       0.0,
        "model_performance":   {
            "average_precision": 0.0,
            "roc_auc":           0.0,
            "cv_fold_ap_scores": [],
            "cv_fold_ap_std":    0.0,
        },
        "class_balance":       {"top3_positive_rate": 0.15},
        "weight_justification": {
            "pearson_r_grid_finish_2022_2025": "N/A",
            "pearson_r_grid_finish_2026":      "N/A",
        },
        "training_data":       {"races_used": 0},
        "safe_feature_cols":   [],
    }
    if not META_PATH.exists():
        return defaults
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        for k, v in defaults.items():
            meta.setdefault(k, v)
        if not meta.get("races_used"):
            meta["races_used"] = meta.get("training_data", {}).get("races_used", 0)
        if not meta.get("avg_precision"):
            meta["avg_precision"] = meta.get("model_performance", {}).get("average_precision", 0.0)
        return meta
    except (json.JSONDecodeError, OSError):
        return defaults


@st.cache_data(ttl=3600)
def _load_race_schedule(year: int) -> pd.DataFrame:
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        schedule = schedule[schedule["EventFormat"] != "testing"]
        return schedule[["RoundNumber", "EventName", "EventDate"]].copy()
    except Exception:
        raw = DATA_DIR / f"raw_{year}.csv"
        if raw.exists():
            try:
                df = pd.read_csv(raw)
                if {"round", "event_name"}.issubset(df.columns):
                    events = (
                        df[["round", "event_name"]]
                        .drop_duplicates()
                        .rename(columns={"round": "RoundNumber", "event_name": "EventName"})
                        .sort_values("RoundNumber")
                    )
                    events["EventDate"] = pd.NaT
                    return events
            except Exception:
                pass
        return pd.DataFrame(columns=["RoundNumber", "EventName", "EventDate"])


@st.cache_data(ttl=300)
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_feature_importance() -> pd.DataFrame:
    return _load_csv(DATA_DIR / "feature_importance.csv")


# ── CHART HELPERS ─────────────────────────────────────────────────────────────

def _f1_layout(title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=17, color=F1_WHITE)),
        paper_bgcolor=F1_DARK,
        plot_bgcolor=F1_DARK,
        font=dict(color=F1_WHITE, family="Arial, sans-serif"),
        legend=dict(bgcolor=F1_DARK, bordercolor=F1_GREY),
        margin=dict(l=20, r=30, t=60, b=55),
        height=height,
        xaxis=dict(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY, color=F1_WHITE),
        yaxis=dict(gridcolor=F1_GREY, linecolor=F1_GREY, zerolinecolor=F1_GREY, color=F1_WHITE),
    )


def _pos_emoji(pos) -> str:
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return "🏎️"
    if p == 1:  return "🥇"
    if p == 2:  return "🥈"
    if p == 3:  return "🥉"
    if p <= 10: return "🏎️"
    return "   "


def _podium_bar_chart(result_df: pd.DataFrame, race_name: str, cv_ap_std: float = 0.0) -> go.Figure:
    required = {"predicted_position", "driver", "team", "podium_probability"}
    if not required.issubset(result_df.columns):
        fig = go.Figure()
        fig.update_layout(
            **_f1_layout("Prediction data missing required columns"),
            annotations=[dict(text="Missing columns in result_df", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=F1_WHITE, size=14))]
        )
        return fig

    df = result_df.sort_values("podium_probability", ascending=True).copy()
    df["display_label"] = df.apply(
        lambda r: f"{_pos_emoji(r['predicted_position'])} {r['driver']}", axis=1
    )
    prob_std   = cv_ap_std * df["podium_probability"]
    bar_colors = [TEAM_COLORS.get(str(t), F1_TEAL) for t in df["team"].values]
    custom     = [[str(team), float(std)] for team, std in zip(df["team"].values, prob_std.values)]

    fig = go.Figure(go.Bar(
        x=df["podium_probability"],
        y=df["display_label"],
        orientation="h",
        marker_color=bar_colors,
        error_x=dict(
            type="data", array=prob_std.tolist(),
            visible=cv_ap_std > 0, color=F1_GOLD, thickness=1.5, width=4,
        ),
        text=(df["podium_probability"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(color=F1_WHITE, size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Podium prob: %{x:.1%}<br>"
            "CI: %{customdata[1]:.1%}<extra></extra>"
        ),
        customdata=custom,
        cliponaxis=False,
    ))
    fig.add_vline(
        x=0.15, line_dash="dot", line_color=F1_GREY, line_width=1.5,
        annotation_text="Random baseline (15%)",
        annotation_font_color=F1_GREY,
        annotation_position="top right",
    )
    layout = _f1_layout(f"{race_name} — Podium Probability", height=max(380, len(df) * 34))
    layout["xaxis"]["tickformat"] = ".0%"
    layout["xaxis"]["title"]      = "Podium Probability"
    layout["yaxis"]["title"]      = "Driver"
    fig.update_layout(**layout)
    fig.add_annotation(
        text="Error bars = +/-1 std across 5 CV folds",
        xref="paper", yref="paper",
        x=0, y=-0.09, showarrow=False,
        font=dict(size=10, color=F1_GREY),
    )
    return fig


def _feature_importance_chart(feat_df: pd.DataFrame) -> go.Figure:
    feat_df = feat_df.sort_values("importance", ascending=True).copy()
    import plotly.colors as pc
    n       = len(feat_df)
    palette = pc.sample_colorscale("Reds", [i / max(n - 1, 1) for i in range(n)])
    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"],
        orientation="h", marker=dict(color=palette),
        text=feat_df["importance"].round(4), textposition="outside",
        textfont=dict(color=F1_WHITE, size=10),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    layout = _f1_layout("What Drives the Model Predictions?", height=max(300, n * 40))
    layout["xaxis"]["title"] = "Importance (gain)"
    layout["yaxis"]["title"] = "Feature"
    fig.update_layout(**layout)
    return fig


def _learning_curve_chart(lc_df, meta: dict):
    if lc_df is not None and not lc_df.empty:
        df = lc_df.copy()
    else:
        n  = int(meta.get("races_used", 0))
        ap = float(meta.get("avg_precision", 0.0))
        if n == 0:
            return None
        df = pd.DataFrame({"race_number": [n], "ap_score": [ap], "ap_std": [0.0]})

    fig = go.Figure()
    if "ap_std" in df.columns and float(df["ap_std"].sum()) > 0:
        upper  = (df["ap_score"] + df["ap_std"]).tolist()
        lower  = (df["ap_score"] - df["ap_std"]).iloc[::-1].tolist()
        x_band = df["race_number"].tolist() + df["race_number"].iloc[::-1].tolist()
        fig.add_trace(go.Scatter(
            x=x_band, y=upper + lower,
            fill="toself", fillcolor="rgba(232,0,45,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=df["race_number"], y=df["ap_score"],
        mode="lines+markers", name="AP Score",
        line=dict(color=F1_RED, width=2.5), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(232,0,45,0.08)",
        hovertemplate="Race %{x}<br>AP: %{y:.3f}<extra></extra>",
    ))
    layout = _f1_layout("Model Learning Curve (AP vs Race Count)")
    layout["xaxis"]["title"] = "Race Number"
    layout["yaxis"]["title"] = "Average Precision (AP)"
    layout["yaxis"]["range"] = [0, 1.05]
    fig.update_layout(**layout)
    return fig


def _race_accuracy_bar(acc_df: pd.DataFrame) -> go.Figure:
    x_vals = acc_df["event_name"].tolist() if "event_name" in acc_df.columns else list(range(len(acc_df)))
    colors = [F1_TEAL if v >= 0.6 else F1_RED for v in acc_df["accuracy"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals, y=acc_df["accuracy"],
        marker_color=colors, name="Race Accuracy",
        text=(acc_df["accuracy"] * 100).round(1).astype(str) + "%",
        textposition="outside", textfont=dict(color=F1_WHITE, size=10),
        hovertemplate="%{x}<br>Accuracy: %{y:.1%}<extra></extra>",
    ))
    if len(acc_df) >= 3:
        trend = acc_df["accuracy"].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=x_vals, y=trend, mode="lines", name="3-race trend",
            line=dict(color=F1_GOLD, width=2, dash="dot"),
        ))
    layout = _f1_layout("Prediction Accuracy per 2026 Race")
    layout["xaxis"]["title"]     = "Race"
    layout["yaxis"]["title"]     = "Accuracy (Top 5)"
    layout["yaxis"]["range"]     = [0, 1.15]
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig


# ── TABLE STYLER ──────────────────────────────────────────────────────────────

def _style_pred_table(df: pd.DataFrame):
    try:
        pct_cols = [c for c in ["Podium Prob%", "Win Share%"] if c in df.columns]
        fmt      = {c: "{:.1f}%" for c in pct_cols}

        def _row_bg(row):
            pos = row.name
            if pos == 0:
                return [f"background-color:{F1_GOLD}33;color:{F1_GOLD};font-weight:700"] * len(row)
            if pos == 1:
                return [f"background-color:{F1_SILVER}22;color:{F1_SILVER};font-weight:700"] * len(row)
            if pos == 2:
                return [f"background-color:{F1_BRONZE}22;color:{F1_BRONZE};font-weight:700"] * len(row)
            bg = F1_CARD if pos % 2 == 0 else F1_DARK
            return [f"background-color:{bg};color:{F1_WHITE}"] * len(row)

        return df.style.apply(_row_bg, axis=1).format(fmt).hide(axis="index")
    except Exception:
        return df.style.hide(axis="index")


# ── CHAMPIONSHIP CARDS ────────────────────────────────────────────────────────

def _render_championship_cards(raw_2026: pd.DataFrame, meta: dict, acc_df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        try:
            if not raw_2026.empty and {"driver_abbr", "points"}.issubset(raw_2026.columns):
                standings  = raw_2026.groupby("driver_abbr")["points"].sum().sort_values(ascending=False)
                leader     = str(standings.index[0])
                leader_pts = int(standings.iloc[0])
                st.metric("Championship Leader", leader, delta=f"{leader_pts} pts")
            else:
                st.metric("Championship Leader", "N/A", delta="No 2026 data yet")
        except Exception:
            st.metric("Championship Leader", "—")

    with c2:
        try:
            sp_path = DATA_DIR / "season_predictions.csv"
            if sp_path.exists():
                sp = _load_csv(sp_path)
                if not sp.empty and {"driver", "win_probability"}.issubset(sp.columns):
                    champ      = sp.groupby("driver")["win_probability"].sum().sort_values(ascending=False)
                    champ_name = str(champ.index[0])
                    champ_prob = float(champ.iloc[0])
                    st.metric("Predicted Champion", champ_name, delta=f"+{champ_prob:.0%} win rate")
                else:
                    raise ValueError("unexpected schema")
            else:
                st.metric("Predicted Champion", "Run full season prediction")
        except Exception:
            st.metric("Predicted Champion", "—")

    with c3:
        try:
            if not acc_df.empty and "accuracy" in acc_df.columns:
                mean_acc = float(acc_df["accuracy"].mean())
                last_acc = float(acc_df["accuracy"].iloc[-1])
                prev_acc = float(acc_df["accuracy"].iloc[-2]) if len(acc_df) > 1 else last_acc
                delta_v  = last_acc - prev_acc
                st.metric(
                    "Season Accuracy", f"{mean_acc:.0%}",
                    delta=f"{abs(delta_v):.0%} vs last race",
                    delta_color="normal" if delta_v >= 0 else "inverse",
                )
            else:
                st.metric("Season Accuracy", "No races completed yet")
        except Exception:
            st.metric("Season Accuracy", "—")

    st.markdown("---")


# ── WINNER EXPLANATION ────────────────────────────────────────────────────────

def _render_winner_explanation(result_df: pd.DataFrame, meta: dict, raw_features_df=None) -> None:
    if result_df.empty or "podium_probability" not in result_df.columns:
        return
    try:
        winner_row = result_df.sort_values("podium_probability", ascending=False).iloc[0]
    except IndexError:
        return

    winner_name = str(winner_row.get("driver", "Unknown"))
    winner_team = str(winner_row.get("team", ""))
    winner_prob = float(winner_row.get("podium_probability", 0.0))

    feat_df = _load_feature_importance()
    if not feat_df.empty and "feature" in feat_df.columns:
        top5_feats = feat_df.sort_values("importance", ascending=False)["feature"].head(5).tolist()
    else:
        top5_feats = list(meta.get("safe_feature_cols", []))[:5]

    with st.expander("Why did the model predict this?", expanded=False):
        st.markdown(f"**Predicted winner: {winner_name} ({winner_team})**")
        st.caption(f"Podium probability: {winner_prob:.1%}")
        st.markdown("---")
        st.markdown("##### Top 5 Model Signals")
        if not top5_feats:
            st.info("Feature list unavailable. Run `python src/model.py`.")
            return

        winner_vals = {f: None for f in top5_feats}
        if raw_features_df is not None and not raw_features_df.empty:
            drv_col = next((c for c in ["driver", "driver_abbr"] if c in raw_features_df.columns), None)
            if drv_col:
                wf = raw_features_df[raw_features_df[drv_col] == winner_name]
                for feat in top5_feats:
                    if feat in wf.columns and not wf.empty:
                        try:
                            winner_vals[feat] = float(wf[feat].iloc[0])
                        except (ValueError, TypeError):
                            pass

        INVERT = {"grid_position", "quali_gap_to_pole", "rolling_avg_finish_3",
                  "rolling_dnf_rate_3", "constructor_rank"}

        for feat in top5_feats:
            col_name, col_bar = st.columns([1, 2])
            raw_val = winner_vals.get(feat)
            with col_name:
                st.markdown(f"`{feat}`")
                st.caption(FEATURE_DESCRIPTIONS.get(feat, ""))
            with col_bar:
                if raw_val is not None:
                    norm = 0.5
                    if raw_features_df is not None and feat in raw_features_df.columns:
                        try:
                            col_min = float(raw_features_df[feat].min())
                            col_max = float(raw_features_df[feat].max())
                            span    = col_max - col_min
                            norm    = (raw_val - col_min) / span if span > 0 else 0.5
                            if feat in INVERT:
                                norm = 1.0 - norm
                            norm = max(0.0, min(1.0, norm))
                        except Exception:
                            norm = 0.5
                    st.progress(norm)
                    st.caption(f"Value: {raw_val:.3f}")
                else:
                    st.caption("Feature values unavailable — run `python src/model.py`")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

def _render_sidebar(meta: dict) -> str:
    with st.sidebar:
        title_style = f"color:{F1_RED};font-size:1.4rem;font-weight:900;letter-spacing:2px;"
        st.markdown(f"<h1 style='{title_style}'>F1 2026 PREDICTOR</h1>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "nav",
            ["Next Race", "Model Performance", "Race Replay"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        last_event = meta.get("training_date", "N/A")[:10]   # 顯示 "2026-05-11"
        races_used = int(meta.get("num_races_used", 0))       # 顯示 96
        ap_score   = float(meta.get("avg_precision", 0.0))    # 已正確
        st.metric("Last Trained",   str(last_event))
        st.metric("Races in Model", str(races_used))
        st.metric("Model AP Score", f"{ap_score:.3f}")
        st.markdown("---")
        disc_style = f"color:{F1_GREY};font-size:0.75rem;text-align:center;"
        st.markdown(
            f"<p style='{disc_style}'>Predictions are probabilistic, not guaranteed.</p>",
            unsafe_allow_html=True,
        )
    return page


# ── PAGE 1: NEXT RACE ─────────────────────────────────────────────────────────

def _page_next_race(meta: dict) -> None:
    heading_style = f"color:{F1_RED};"
    st.markdown(f"<h2 style='{heading_style}'>Next Race Prediction</h2>", unsafe_allow_html=True)

    if _load_model() is None:
        st.error("Model not trained yet. Run: `python src/model.py`")
        return
    if not PREDICT_AVAILABLE:
        st.error("`src/predict.py` not found. Check project structure.")
        return

    raw_2026 = _load_csv(DATA_DIR / "raw_2026.csv")
    acc_df   = _load_csv(DATA_DIR / "race_accuracy_log.csv")
    _render_championship_cards(raw_2026, meta, acc_df)

    schedule = _load_race_schedule(2026)
    try:
        if "EventDate" in schedule.columns and schedule["EventDate"].notna().any():
            schedule["EventDate"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
            future = schedule[schedule["EventDate"].dt.date >= date.today()]
        else:
            future = schedule
    except Exception:
        future = schedule

    race_options = future["EventName"].tolist() if not future.empty else ["No upcoming races found"]

    col_in, col_out = st.columns([1, 1.6], gap="large")

    with col_in:
        st.markdown("#### Race Settings")
        selected_race = st.selectbox("Select Race", race_options)
        uploaded_file = st.file_uploader("Upload Grid CSV (optional)", type=["csv"])
        predict_btn   = st.button("Predict Race", use_container_width=True)
        if st.button("Clear", use_container_width=False):
            st.rerun()

        with st.expander("Grid CSV Format"):
            grid_csv_example = (
                "driver_abbr,team,grid_position,gap_to_pole_s\n"
                "NOR,McLaren,1,0.000\n"
                "VER,Red Bull,2,0.082\n"
                "LEC,Ferrari,3,0.134\n"
                "..."
            )
            st.code(grid_csv_example, language="csv")
            st.caption("20 rows expected — one per driver.")

    with col_out:
        if predict_btn and selected_race != "No upcoming races found":
            grid_path = None
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.read())
                    grid_path = tmp.name
            try:
                with st.spinner("Running prediction..."):
                    result_df = predict_race(
                        year=2026,
                        race_name=selected_race,
                        grid_csv_path=grid_path,
                    )

                if result_df is None or result_df.empty:
                    st.warning("predict_race() returned no results.")
                    return

                cv_ap_std = float(meta.get("model_performance", {}).get("cv_fold_ap_std", 0.0))
                st.plotly_chart(
                    _podium_bar_chart(result_df, selected_race, cv_ap_std=cv_ap_std),
                    use_container_width=True,
                )

                col_map = {
                    "predicted_position": "Predicted Pos",
                    "driver":             "Driver",
                    "team":               "Team",
                    "podium_probability": "Podium Prob%",
                    "win_probability":    "Win Share%",
                    "grid_position":      "Grid Pos",
                }
                available  = {k: v for k, v in col_map.items() if k in result_df.columns}
                display_df = result_df[list(available.keys())].rename(columns=available).copy()
                for pct_col in ["Podium Prob%", "Win Share%"]:
                    if pct_col in display_df.columns:
                        display_df[pct_col] = display_df[pct_col] * 100
                if "Grid Pos" in display_df.columns:
                    display_df["Grid Pos"] = pd.array(display_df["Grid Pos"], dtype="Int64")
                if "Predicted Pos" in display_df.columns:
                    display_df = display_df.sort_values("Predicted Pos").reset_index(drop=True)

                st.dataframe(_style_pred_table(display_df), use_container_width=True, height=400)
                _render_winner_explanation(result_df, meta)

                with st.expander("Model Confidence"):
                    mp          = meta.get("model_performance", {})
                    ap          = float(meta.get("avg_precision", 0.0))
                    n           = int(meta.get("races_used", 0))
                    n26         = meta.get("class_balance", {}).get("pos_samples", "?")
                    fold_scores = mp.get("cv_fold_ap_scores", [])
                    lines_body  = [
                        f"- **Model AP Score:** `{ap:.3f}` (random baseline approx 0.15)",
                        f"- **Races in training:** `{n}` total",
                        f"- **2026 podium events:** `{n26}`",
                    ]
                    if fold_scores:
                        lines_body.append(f"- **CV fold AP scores:** `{[round(x, 3) for x in fold_scores]}`")
                    lines_body.append("")
                    lines_body.append("> Confidence increases as the 2026 season progresses.")
                    st.markdown("\n".join(lines_body))

                st.toast("Prediction complete!", icon="🏎️")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check internet access (FastF1) and confirm model.pkl exists.")
            finally:
                if grid_path is not None:
                    Path(grid_path).unlink(missing_ok=True)

        elif not predict_btn:
            placeholder_style = f"text-align:center;padding:60px 0;color:{F1_GREY};"
            st.markdown(
                f"<div style='{placeholder_style}'>"
                "<p style='font-size:3rem;'>🏎️</p>"
                "<p>Select a race and click <b>Predict Race</b></p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ── PAGE 2: MODEL PERFORMANCE ─────────────────────────────────────────────────

def _page_model_performance(meta: dict) -> None:
    heading_style = f"color:{F1_RED};"
    st.markdown(f"<h2 style='{heading_style}'>Model Performance</h2>", unsafe_allow_html=True)

    mp        = meta.get("model_performance", {})
    ap_score  = float(meta.get("avg_precision", 0.0))
    roc_auc   = float(mp.get("roc_auc") or meta.get("roc_auc_score", 0.0))
    top3_rate = float(meta.get("class_balance", {}).get("top3_positive_rate", 0.15))

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.metric("Average Precision (AP)", f"{ap_score:.3f}", help="Random baseline approx 0.15.")
    with m2:
        st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Random baseline = 0.50.")
    with m3:
        st.metric("Top-3 Base Rate", f"{top3_rate:.1%}", help="Approx 3/20 drivers per race.")

    st.markdown("---")
    st.markdown("#### Model Learning Curve")
    lc_df  = _load_csv(DATA_DIR / "learning_curve.csv")
    lc_fig = _learning_curve_chart(lc_df if not lc_df.empty else None, meta)
    if lc_fig:
        st.plotly_chart(lc_fig, use_container_width=True)
    else:
        st.info("Learning curve unavailable — builds after each retrain.")

    st.markdown("#### Feature Importance")
    feat_df = _load_feature_importance()
    if not feat_df.empty and "feature" in feat_df.columns:
        st.plotly_chart(_feature_importance_chart(feat_df), use_container_width=True)
    else:
        st.info("Run `python src/model.py` to generate feature importance.")
        feat_cols = meta.get("safe_feature_cols", [])
        if feat_cols:
            st.caption(f"Features in model: `{', '.join(feat_cols)}`")

    st.markdown("---")
    st.markdown("#### Regulation Reset Evidence")
    wj = meta.get("weight_justification", {})
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.metric("Grid to Finish r (2022-2025)", str(wj.get("pearson_r_grid_finish_2022_2025", "N/A")))
    with c2:
        st.metric("Grid to Finish r (2026)", str(wj.get("pearson_r_grid_finish_2026", "N/A")))
    st.caption(
        "Lower Pearson r in 2026 confirms regulations disrupted grid-to-finish "
        "relationships — justifying the 0.4x discount on pre-2026 training data."
    )


# ── PAGE 3: RACE REPLAY ───────────────────────────────────────────────────────

def _page_race_replay(meta: dict) -> None:
    heading_style = f"color:{F1_RED};"
    st.markdown(f"<h2 style='{heading_style}'>Race Replay</h2>", unsafe_allow_html=True)

    raw_2026 = _load_csv(DATA_DIR / "raw_2026.csv")
    if raw_2026.empty:
        st.info("No completed 2026 data found in `data/raw_2026.csv`.")
        return

    completed = []
    if "event_name" in raw_2026.columns and "round" in raw_2026.columns:
        completed = (
            raw_2026[["round", "event_name"]]
            .drop_duplicates()
            .sort_values("round")["event_name"]
            .tolist()
        )
    if not completed:
        st.info("No completed races found in raw_2026.csv.")
        return

    selected  = st.selectbox("Select Completed 2026 Race", completed)
    if not selected:
        return

    race_data = raw_2026[raw_2026["event_name"] == selected].copy()
    if race_data.empty:
        st.warning(f"No data found for {selected}.")
        return

    act_cols    = [c for c in ["driver_abbr", "team", "final_position"] if c in race_data.columns]
    actual_top5 = (
        race_data[act_cols]
        .sort_values("final_position").head(5)
        .rename(columns={"driver_abbr": "Driver", "team": "Team", "final_position": "Actual Pos"})
        .reset_index(drop=True)
    )

    pred_top5  = pd.DataFrame()
    cache_name = selected.replace(" ", "_")
    pred_cache = DATA_DIR / f"predictions_{cache_name}.csv"

    col_pred, col_actual = st.columns(2, gap="large")

    with col_pred:
        st.markdown("#### Predicted Top 5")
        if pred_cache.exists():
            raw_pred = _load_csv(pred_cache)
            required = {"driver", "team", "predicted_position", "podium_probability"}
            if required.issubset(raw_pred.columns):
                pred_top5 = (
                    raw_pred[list(required)]
                    .sort_values("predicted_position").head(5)
                    .rename(columns={
                        "driver": "Driver", "team": "Team",
                        "predicted_position": "Predicted Pos",
                        "podium_probability": "Podium Prob",
                    })
                    .reset_index(drop=True)
                )
                pred_top5["Podium Prob"] = (pred_top5["Podium Prob"] * 100).round(1).astype(str) + "%"
                st.dataframe(pred_top5, use_container_width=True, hide_index=True)
            else:
                st.warning(f"Unexpected columns in prediction cache: {list(raw_pred.columns)}")
        else:
            st.info(f"No cached prediction for **{selected}**. Run Next Race before each race.")

    with col_actual:
        st.markdown("#### Actual Result")
        st.dataframe(actual_top5, use_container_width=True, hide_index=True)

    if not pred_top5.empty and "Driver" in pred_top5.columns:
        st.markdown("---")
        st.markdown("#### Prediction vs Reality")
        actual_podium = set(actual_top5["Driver"].head(3))
        pred_podium   = set(pred_top5["Driver"].head(3))
        correct       = actual_podium & pred_podium
        incorrect     = pred_podium - actual_podium
        cc1, cc2, cc3 = st.columns(3, gap="medium")
        with cc1:
            for d in correct:
                st.success(f"{d} — correctly predicted podium")
        with cc2:
            for d in incorrect:
                st.error(f"{d} — predicted podium, finished outside top 3")
        with cc3:
            hits = len(set(actual_top5["Driver"]) & set(pred_top5["Driver"]))
            st.metric("Top-5 Race Accuracy", f"{hits / 5:.0%}")

    st.markdown("---")
    st.markdown("#### Season Accuracy Tracker")
    acc_df = _load_csv(DATA_DIR / "race_accuracy_log.csv")
    if not acc_df.empty and "accuracy" in acc_df.columns:
        st.plotly_chart(_race_accuracy_bar(acc_df), use_container_width=True)
    else:
        st.info("Accuracy log builds at `data/race_accuracy_log.csv` after each race replay.")


# ── WARNINGS ──────────────────────────────────────────────────────────────────

def _render_warnings(meta_ok: bool) -> None:
    if not MODEL_PKL.exists():
        st.warning("model.pkl not found. Run `python src/model.py` to train.")
    if not meta_ok:
        st.warning("model_metadata.json missing or malformed. Sidebar metrics show defaults.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    meta    = _load_metadata()
    meta_ok = META_PATH.exists()
    _render_warnings(meta_ok)
    page = _render_sidebar(meta)

    if page == "Next Race":
        _page_next_race(meta)
    elif page == "Model Performance":
        _page_model_performance(meta)
    elif page == "Race Replay":
        _page_race_replay(meta)


if __name__ == "__main__":
    main()
