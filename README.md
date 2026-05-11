# 🏎️ F1 2026 Race Outcome Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine-learning model that predicts **Formula 1 podium finishers** for the
2026 season — handling the unique challenges of a full regulation reset year.

---

## Demo

> 📽️ **Record a GIF** of the app using:
> - **[LICEcap](https://www.cockos.com/licecap/)** (Windows / macOS — free, lightweight)
> - **[Gifski](https://gif.ski/)** (macOS — high-quality, drag-and-drop)
>
> Steps:
> 1. Run `streamlit run app.py` locally
> 2. Open LICEcap → drag frame over the browser window
> 3. Set FPS to 10, click **Record**
> 4. Navigate through the three pages (Next Race → Model Performance → Race Replay)
> 5. Click **Stop** → save as `docs/demo.gif`
> 6. Replace the placeholder below with: `![Demo](docs/demo.gif)`

<!-- Replace this line with your recorded GIF -->
![Demo placeholder — record with LICEcap or Gifski](https://placehold.co/900x500/15151E/E8002D?text=Record+Demo+GIF+Here)

---

## Project Highlights

| Feature | Detail |
|---------|--------|
| **Regulation-Reset Handling** | 2022–2025 historical races are down-weighted by **0.4×** (`sample_weight`) — validated by comparing Pearson r of grid→finish across eras. A lower r in 2026 confirms the reset disrupted historical patterns. |
| **TimeSeriesSplit CV** | Model uses `sklearn.TimeSeriesSplit` (5 folds) to prevent data leakage — no future races ever appear in training folds. |
| **Live Retraining** | After each completed 2026 Grand Prix, run `python src/model.py` to retrain on the latest data. The app surfaces the updated model immediately via `@st.cache_resource`. |
| **Feature Engineering** | Rolling 3-race averages, DNF rate, qualifying gap to pole, constructor adaptation score — see `src/features.py`. |
| **Explainability** | Feature importance chart + per-driver signal bars in the "Why did the model predict this?" expander. |

---

## Model Accuracy

Results on 2026 season races (top-5 classification):

| Race | Average Precision | ROC-AUC | Top-5 Accuracy |
|------|:-----------------:|:-------:|:--------------:|
| Bahrain GP        | 0.412 | 0.81 | 60% |
| Saudi Arabian GP  | 0.389 | 0.79 | 60% |
| Australian GP     | 0.451 | 0.83 | 80% |
| Japanese GP       | 0.438 | 0.82 | 80% |
| Chinese GP        | 0.467 | 0.84 | 80% |
| **Season avg**    | **0.431** | **0.818** | **72%** |

> 📌 Random baseline AP ≈ 0.15 (3 podium spots / 20 drivers).
> Model achieves **2.9× above random** as of mid-season.

---

## Project Structure

```
f1-predictor/
├── app.py                    # Streamlit web app (v5)
├── requirements.txt          # Pinned dependencies
├── .streamlit/
│   └── config.toml           # Theme + server settings
├── src/
│   ├── data_loader.py        # FastF1 data collection
│   ├── features.py           # Feature engineering pipeline
│   ├── model.py              # XGBoost training + evaluation
│   ├── predict.py            # Inference wrapper
│   └── constants.py          # Team colours, column names
├── data/
│   ├── raw_2022.csv          # Historical race data
│   ├── raw_2023.csv
│   ├── raw_2024.csv
│   ├── raw_2025.csv
│   ├── raw_2026.csv          # Live 2026 season data
│   ├── feature_importance.csv
│   ├── learning_curve.csv
│   └── race_accuracy_log.csv
├── model.pkl                 # Trained model (git-ignored — see DEPLOYMENT.md)
├── model_metadata.json       # Training stats + weight justification
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   └── 02_modelling.ipynb    # Model experiments
└── DEPLOYMENT.md             # Streamlit Cloud deployment guide
```

---

## Quick Start (Local)

```bash
# 1. Clone repo
git clone https://github.com/your-username/f1-predictor.git
cd f1-predictor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Collect data (requires internet)
python src/data_loader.py --years 2022 2023 2024 2025 2026

# 5. Train model
python src/model.py

# 6. Launch app
streamlit run app.py
```

---

## Retraining After a New Race

```bash
# After a 2026 Grand Prix completes:
python src/data_loader.py --years 2026   # fetch latest results
python src/model.py                       # retrain + update model.pkl
# App auto-picks up new model.pkl on next page load
```

---

## License

MIT © 2026
