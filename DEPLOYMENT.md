# Deployment Guide — Streamlit Cloud

This guide walks you through deploying the F1 2026 Predictor to
**Streamlit Community Cloud** (free tier, 512 MB RAM).

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GitHub account | App must live in a public (or private*) GitHub repo |
| Streamlit Cloud account | Sign up free at [share.streamlit.io](https://share.streamlit.io) |
| Trained model artefact | See **Model Storage** section below |

*Private repos supported on free tier.

---

## Step 1 — Prepare the Repository

Ensure your repo contains:

```
✅ app.py
✅ requirements.txt          (all versions pinned)
✅ .streamlit/config.toml
✅ src/                      (data_loader, features, model, predict, constants)
✅ data/raw_2022.csv … raw_2026.csv
✅ model_metadata.json
❌ model.pkl                 (git-ignored — see Model Storage below)
❌ .venv/                    (git-ignored)
❌ __pycache__/              (git-ignored)
```

Push everything:

```bash
git add .
git commit -m "feat: add deployment config"
git push origin main
```

---

## Step 2 — Connect GitHub to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
2. Click **"New app"**
3. Select **"From existing repo"**
4. Authorise Streamlit to access your GitHub account
5. Choose your repository (e.g. `your-username/f1-predictor`)

---

## Step 3 — Configure the App

Fill in the deployment form:

| Field | Value |
|-------|-------|
| **Repository** | `your-username/f1-predictor` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **Python version** | `3.11` |

> ⚠️ Do **not** select Python 3.12+ — some FastF1 dependencies have
> wheel build issues on 3.12 as of fastf1==3.4.0.

Click **"Advanced settings"** and set:

```
Python version: 3.11
```

---

## Step 4 — Set Secrets (if needed)

If your `src/data_loader.py` uses any API keys or tokens:

1. In the app dashboard → **"⋮" menu** → **"Settings"** → **"Secrets"**
2. Add secrets in TOML format:

```toml
[fastf1]
# No API key needed — FastF1 uses public Ergast/F1 API

[general]
ENVIRONMENT = "production"
```

---

## Step 5 — Deploy

Click **"Deploy!"**

Streamlit Cloud will:
1. Clone your repo
2. Install `requirements.txt`  (~2-3 min on first deploy)
3. Run `app.py`
4. Assign a URL: `https://your-app-name.streamlit.app`

Copy this URL into your `README.md` badge:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
```

---

## Model Storage — Critical

### Why model.pkl must be git-ignored

`model.pkl` (XGBoost + LightGBM calibrated ensemble) can be **80–300 MB**.
GitHub blocks files >100 MB and flags files >50 MB.
Committing it will eventually break `git push`.

### Solution A — Retrain on First Run (Recommended for free tier)

Add this block to the **top of `app.py`**, before any page renders:

```python
# app.py — add after imports
import subprocess

def _ensure_model() -> None:
    if not MODEL_PKL.exists():
        with st.spinner("First run — training model (~15-20 min, Optuna 50 trials)..."):
            result = subprocess.run(
                ["python", "src/model.py"],
                capture_output=True, text=True,
                cwd=str(ROOT_DIR),
            )
            if result.returncode != 0:
                st.error(f"Model training failed:\n{result.stderr}")
                st.stop()
            st.success("Model trained successfully!")
            st.rerun()

_ensure_model()
```

Then in `main()`, remove the `st.error("Model not trained")` early-exit.

**Pros:** Zero external dependencies, works on free tier.  
**Cons:** ~15-20 minute cold start on first deploy (and after dyno restart).

### Solution B — GitHub Releases (Recommended for faster deploys)

1. After training locally, create a GitHub Release:

```bash
gh release create v1.0 model.pkl model_metadata.json \
  --title "Model v1.0 — Post Race 5" \
  --notes "Trained on 2022-2026 data, 5 completed 2026 races"
```

2. Add a loader to `app.py`:

```python
import requests, io, joblib

@st.cache_resource
def _load_model_from_release():
    url = "https://github.com/your-username/f1-predictor/releases/download/v1.0/model.pkl"
    r   = requests.get(url, timeout=60)
    return joblib.load(io.BytesIO(r.content))
```

**Pros:** Instant load after first download (~5 sec).  
**Cons:** Must manually upload new model after each retrain.

### Solution C — Hugging Face Hub ⭐ (Best for large models)

Recommended for the XGBoost + LightGBM ensemble as model size can exceed
GitHub Release limits.

```bash
# 1. Install huggingface_hub
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Upload model after training
python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="your-username/f1-2026-predictor",
    repo_type="model",
)
EOF
```

Then load in `app.py`:

```python
from huggingface_hub import hf_hub_download
import joblib

@st.cache_resource
def _load_model_hf():
    path = hf_hub_download(
        repo_id="your-username/f1-2026-predictor",
        filename="model.pkl",
    )
    return joblib.load(path)
```

**Pros:** Free storage, no file size limits, version controlled.  
**Cons:** Requires HuggingFace account + manual upload after retrain.

---

## Requirements.txt — Full Pinned List

Ensure these are all present in your `requirements.txt`:

```txt
# Core ML
xgboost==2.0.3
lightgbm==4.3.0
optuna==3.6.1
scikit-learn==1.4.2
joblib==1.3.2

# Data
fastf1==3.4.0
pandas==2.2.2
numpy==1.26.4

# App
streamlit==1.35.0
plotly==5.22.0

# Utils
python-dotenv==1.0.1
requests==2.31.0
huggingface_hub==0.23.0   # only if using Solution C
```

---

## Redeployment After Each Race

```bash
# 1. Retrain locally
python src/data_loader.py --years 2026
python src/model.py

# 2. Commit updated artefacts (NOT model.pkl)
git add data/raw_2026.csv data/feature_importance.csv \
        data/learning_curve.csv data/race_accuracy_log.csv \
        model_metadata.json
git commit -m "data: Race N — retrain ensemble XGB+LGB, AP=0.xxx"
git push origin main

# 3. Upload new model.pkl (choose one)
# Solution B:
gh release upload v1.x model.pkl --clobber
# Solution C:
huggingface-cli upload your-username/f1-2026-predictor model.pkl model.pkl

# Streamlit Cloud auto-redeploys on git push ✅
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: fastf1` | requirements.txt missing or wrong version | Check `fastf1==3.4.0` is in requirements.txt |
| `ModuleNotFoundError: lightgbm` | New dependency missing | Add `lightgbm==4.3.0` to requirements.txt |
| `ModuleNotFoundError: optuna` | New dependency missing | Add `optuna==3.6.1` to requirements.txt |
| `CalibratedClassifierCV` error | sklearn version too old | Ensure `scikit-learn==1.4.2` in requirements.txt |
| App crashes on schedule load | FastF1 tries to write disk cache | Ensure no `Cache.enable_cache()` calls in code |
| `model.pkl not found` | File git-ignored, not loaded from release | Implement Solution A, B, or C above |
| Blank white screen | CSS injection error | Check `_inject_css()` f-strings are valid |
| `st.columns() gap= error` | Old Streamlit version | Ensure `streamlit==1.35.0` in requirements.txt |
| Memory limit exceeded (512 MB) | Ensemble model too large | Reduce `n_estimators` or use Solution C (HuggingFace) |
| Training OOM on Streamlit Cloud | Optuna trials too many | Add `--optuna-trials 20` flag in Solution A subprocess call |

---

## Resource Limits (Free Tier)

| Resource | Limit |
|----------|-------|
| RAM | 512 MB |
| CPU | 1 vCPU |
| Storage | 1 GB (ephemeral) |
| Concurrent users | Unlimited |
| Sleep after inactivity | 7 days (wakes on visit) |

> ⚠️ The XGBoost + LightGBM ensemble may approach the 512 MB RAM limit
> during inference. If you hit OOM errors, switch to Solution C (HuggingFace)
> and reduce `n_estimators` to 300 in both models.

---

## Custom Domain (Optional)

1. In app dashboard → **Settings** → **Custom domain**
2. Add a CNAME record in your DNS provider:
   ```
   CNAME  f1.yourdomain.com  cname.streamlit.app
   ```
3. Enter `f1.yourdomain.com` in the custom domain field

---

*Last updated: 2026-05-20*
