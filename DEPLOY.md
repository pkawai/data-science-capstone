# Deployment Guide

Step-by-step instructions to deploy the dashboard to Streamlit Cloud.

## Prerequisites

- [ ] GitHub account with this repo pushed
- [ ] `streamlit_app.py` at repo root ✅
- [ ] `requirements.txt` at repo root ✅
- [ ] `.streamlit/config.toml` at repo root ✅

## Deploy Steps

### 1. Go to Streamlit Cloud

Open **https://share.streamlit.io** and sign in with your GitHub account.

### 2. Create a new app

Click **"New app"** → **"From existing repo"**.

### 3. Fill in deployment details

| Field | Value |
|---|---|
| Repository | `pkawai/data-science-capstone` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| App URL (optional) | Choose a slug like `forex-bot-orgil` |

### 4. Click "Deploy"

Wait 2–4 minutes for the build to complete. Streamlit will:
- Pull the latest code
- Install dependencies from `requirements.txt`
- Start the Streamlit server

### 5. Get your URL

Once deployed, you'll have a URL like:
```
https://forex-bot-orgil.streamlit.app
```

### 6. Verify

- [ ] Dashboard loads without errors
- [ ] All 3 tabs work (Monitor / Backtest / About)
- [ ] Sidebar shows project info
- [ ] Backtest tab shows equity curves and feature importance
- [ ] Open the URL on your phone — looks OK

### 7. Update README

Replace this in `README.md`:
```
🔗 **Dashboard:** [https://YOUR-STREAMLIT-URL.streamlit.app](https://YOUR-STREAMLIT-URL.streamlit.app)
```

with your real URL. Commit + push.

### 8. Take screenshots

Capture each tab and save to `screenshots/`:
- `dashboard_monitor.png`
- `dashboard_backtest.png`
- `dashboard_about.png`

Then update the README screenshot links to point to these.

## Troubleshooting

### Build fails: "ModuleNotFoundError"
Add the missing package to `requirements.txt` and push. Streamlit Cloud auto-redeploys on push.

### App shows "File not found: trades.csv"
The Streamlit Cloud deployment doesn't have demo data by default. Run `python bot/generate_demo_data.py` locally and commit the resulting `trades.csv` and `state.json` to `bot/`.

### App is slow to wake up
Free tier apps sleep after inactivity. First load after sleep can take ~30 seconds.
