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

`model.pkl` (XGBoost trained model) can be **50–200 MB**.
GitHub blocks files >100 MB and flags files >50 MB.
Committing it will eventually break `git push`.

### Solution A — Retrain on First Run (Recommended for free tier)

Add this block to the **top of `app.py`**, before any page renders:

```python
# app.py — add after imports
import subprocess

def _ensure_model() -> None:
    if not MODEL_PKL.exists():
        with st.spinner("First run — training model (~60 sec)..."):
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
**Cons:** ~60 second cold start on first deploy (and after dyno restart).

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

### Solution C — Hugging Face Hub (Best for large models)

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

Free storage, no file size limits.

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
git commit -m "data: add Race N results + retrain metadata"
git push origin main

# 3. Upload new model.pkl to GitHub Releases (Solution B)
gh release upload v1.x model.pkl --clobber

# Streamlit Cloud auto-redeploys on git push ✅
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: fastf1` | requirements.txt missing or wrong version | Check `fastf1==3.4.0` is in requirements.txt |
| App crashes on schedule load | FastF1 tries to write disk cache | Ensure no `Cache.enable_cache()` calls in code |
| `model.pkl not found` | File git-ignored, not loaded from release | Implement Solution A, B, or C above |
| Blank white screen | CSS injection error | Check `_inject_css()` f-strings are valid |
| `st.columns() gap= error` | Old Streamlit version | Ensure `streamlit==1.35.0` in requirements.txt |
| Memory limit exceeded (512 MB) | XGBoost model too large | Reduce `n_estimators` in `src/model.py` |

---

## Resource Limits (Free Tier)

| Resource | Limit |
|----------|-------|
| RAM | 512 MB |
| CPU | 1 vCPU |
| Storage | 1 GB (ephemeral) |
| Concurrent users | Unlimited |
| Sleep after inactivity | 7 days (wakes on visit) |

---

## Custom Domain (Optional)

1. In app dashboard → **Settings** → **Custom domain**
2. Add a CNAME record in your DNS provider:
   ```
   CNAME  f1.yourdomain.com  cname.streamlit.app
   ```
3. Enter `f1.yourdomain.com` in the custom domain field

---

*Last updated: 2026-05-11*
