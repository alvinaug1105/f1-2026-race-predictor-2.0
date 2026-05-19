# 🏎️ F1 2026 Race Outcome Predictor

> **Live Streamlit App** | XGBoost + LightGBM Calibrated Ensemble | 13 Features | Optuna Tuning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-2026-race-predictor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

A **machine learning pipeline** that predicts the probability of each F1 driver finishing in the **top 3 (podium)** for the 2026 season. The app fetches live qualifying data via FastF1, engineers 13 race features, and serves predictions through an interactive Streamlit dashboard.

**Key highlights:**
- 🧠 **Ensemble model**: Calibrated XGBoost + LightGBM, weighted by Optuna CV score
- 📊 **Primary metric**: Average Precision (AP = 0.715) — 4.8× above random baseline (0.15)
- 🔄 **No temporal leakage**: Race-level TimeSeriesSplit CV (never splits mid-race)
- ⚖️ **2026 regulation reset**: Exponential decay sample weights (2022: 0.2× → 2026: 1.0×)
- 🌐 **Live deployment**: Streamlit Cloud, auto-fetches qualifying via FastF1

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| **Next Race** | Select upcoming race → upload grid CSV → get podium probability chart |
| **Model Performance** | AP score, ROC-AUC, feature importance, learning curve |
| **Race Replay** | Compare predictions vs actual results for completed 2026 races |

---

## 🏗️ Architecture

```
f1-2026-race-predictor-2.0/
├── app.py                    # Streamlit frontend (3 pages)
├── model.pkl                 # Trained ensemble (XGB + LGB calibrated)
├── model_metadata.json       # Training metadata, feature list, metrics
├── requirements.txt
├── src/
│   ├── constants.py          # Shared paths, leakage cols, year weights
│   ├── features.py           # Feature engineering pipeline (13 features)
│   ├── data_loader.py        # FastF1 data fetcher → raw_{year}.csv
│   ├── model.py              # Training pipeline + Optuna tuning
│   └── predict.py            # Inference module (public API)
└── data/
    ├── raw_2022..2026.csv    # Historical race data
    ├── cache/                # FastF1 local cache
    └── *.png                 # Model performance charts
```

---

## ⚙️ Feature Engineering

| # | Feature | Description | Type |
|---|---|---|---|
| 1 | `grid_position` | Qualifying grid slot | Numerical |
| 2 | `quali_gap_to_pole` | Gap to pole position (seconds) | Numerical |
| 3 | `rolling_avg_finish_3` | Avg finish, last 3 races | Rolling |
| 4 | `rolling_dnf_rate_3` | DNF rate, last 3 races | Rolling |
| 5 | `constructor_rank` | Team championship rank at race time | Derived |
| 6 | `circuit_type` | 0=street, 1=technical, 2=high-speed | Categorical |
| 7 | `adaptation_score` | 2026 regulation adaptation score | 2026-specific |
| 8 | `pit_stop_count` | Avg pit stops, last 3 races | Rolling |
| 9 | `overtake_difficulty` | Circuit overtaking difficulty (0–1) | Lookup |
| 10 | `driver_circuit_avg_pos` | Driver's historical avg at this circuit | Historical |
| 11 | `is_wet_race` | Rainfall detected during race | Weather |
| 12 | `avg_air_temp` | Average air temperature (°C) | Weather |
| 13 | `avg_humidity` | Average humidity (%) | Weather |

---

## 📈 Model Performance

| Metric | Score | Random Baseline |
|---|---|---|
| **Average Precision (AP)** ★ | **0.715** | 0.15 |
| ROC-AUC | 0.936 | 0.50 |
| Accuracy | 0.880 | — |
| F1 Score | 0.680 | — |
| Recall | 0.854 | — |

> AP is the primary metric. For ~85/15 imbalanced data, AP is far more discriminative than accuracy.

### Why Race-Level TimeSeriesSplit?
Standard `TimeSeriesSplit` can split mid-race (e.g. drivers 1–10 in train, 11–20 in test for the same round), leaking race-level patterns. This pipeline groups all 20 drivers per race into the same fold.

---

## 🔬 Model Design Decisions

### 1. Regulation Reset Weighting
The 2026 regulation changes disrupted the grid-to-finish correlation from pre-2026 seasons. Pre-2026 data is discounted using exponential decay:

```
2022: 0.20× | 2023: 0.30× | 2024: 0.50× | 2025: 0.80× | 2026: 1.00×
```

### 2. Calibrated Ensemble
Both XGBoost and LightGBM are wrapped with `CalibratedClassifierCV` (isotonic regression) to produce well-calibrated probabilities. The ensemble weight is determined by each model's Optuna CV AP score.

### 3. Dynamic `scale_pos_weight`
Rather than hardcoding class imbalance correction, `scale_pos_weight` is recomputed per fold and for the final model from the actual class distribution.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Fetch Data
```bash
python src/data_loader.py --years 2022 2023 2024 2025 2026
```

### 2. Train Model
```bash
python src/model.py --years 2022 2023 2024 2025 2026
# Add --optuna-trials 100 for more tuning (default: 50)
```

### 3. Run App
```bash
streamlit run app.py
```

### 4. CLI Prediction
```bash
python src/predict.py --year 2026 --race "Canadian Grand Prix" --grid-csv data/canadian_gp_grid.csv
```

---

## 📄 Grid CSV Format

To override FastF1 qualifying data, upload a CSV with these columns:

```csv
driver_abbr,team,grid_position,gap_to_pole_s,rolling_avg_finish_3,rolling_dnf_rate_3,constructor_rank,circuit_type,adaptation_score,pit_stop_count
NOR,McLaren,1,0.000,2.3,0.00,2,1,0.88,2
VER,Red Bull,2,0.142,1.7,0.00,1,1,0.92,2
LEC,Ferrari,3,0.287,3.1,0.33,3,1,0.91,2
```

> 20 rows expected (one per driver). `gap_to_pole` is also accepted as a column alias.

---

## 🔄 Retraining Workflow

Run after each race weekend to keep the model fresh:

```bash
# 1. Fetch latest race data
python src/data_loader.py --years 2026

# 2. Retrain with new data
python src/model.py --years 2022 2023 2024 2025 2026

# 3. Upload model.pkl + model_metadata.json to GitHub
# Streamlit Cloud will auto-redeploy
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `XGBoost` | Primary gradient boosting classifier |
| `LightGBM` | Secondary gradient boosting (ensemble) |
| `Optuna` | Bayesian hyperparameter optimisation |
| `scikit-learn` | CalibratedClassifierCV, metrics |
| `FastF1` | Live F1 qualifying & race data |
| `Streamlit` | Web app frontend |
| `Plotly` | Interactive charts |
| `pandas / numpy` | Data processing |
| `joblib` | Model serialisation |

---

## 📁 Data Sources

- **FastF1** — Official F1 timing & telemetry data (2022–2026)
- **Ergast API** (via FastF1) — Historical race results
- **Custom weather scraping** — Via FastF1 `session.weather_data`

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

1. Fork the repo
2. Create feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'feat: your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**alvinaug1105**
- GitHub: [@alvinaug1105](https://github.com/alvinaug1105)
- Project: [f1-2026-race-predictor-2.0](https://github.com/alvinaug1105/f1-2026-race-predictor-2.0)

---

<div align="center">
<i>Built with ❤️ and a lot of race data</i><br>
<i>Predictions are probabilistic, not guaranteed 🏎️</i>
</div>
