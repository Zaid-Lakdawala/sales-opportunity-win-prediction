# Sales Win Prediction

Predict whether a sales opportunity will be won or lost using a synthetic CRM-like dataset. This repo shows the full ML lifecycle: data generation, feature engineering, model training/evaluation, and deployment via a FastAPI inference API.

## Table of Contents

- Overview
- Features
- Tech Stack
- Project Structure
- Setup
- Train & Export Artifacts
- Run the API
- Call the Endpoint (Examples)
- Troubleshooting
- License

## Overview

Many teams want reliable win/loss predictions to improve forecasting and prioritisation. Since labeled CRM data is rarely public, this project includes a realistic synthetic dataset generator and an end-to-end pipeline that you can run locally.

- Data: synthetic opportunities with stage progression, activity, and deal size
- Models: RandomForest (tuned), LightGBM (tuned), and a stacked ensemble
- Deployment: persisted artifacts + FastAPI endpoint for real-time scoring

## Features

- Synthetic CRM dataset capturing non-linear relationships (e.g., stage × meetings, sales velocity)
- Shared feature engineering between training and inference (`make_features`)
- Model comparison with calibration and ROC curves
- Artifacts persisted with joblib for production-style inference
- FastAPI `POST /predict/` returns class and probability

## Tech Stack

- Python, NumPy, Pandas, scikit-learn, LightGBM
- FastAPI, Uvicorn
- Matplotlib, Seaborn (EDA/plots)

## Project Structure

```
.
├── salesWinPrediction.ipynb      # End-to-end notebook (data, features, modeling, export)
├── api_app.py                    # Lightweight FastAPI app for inference
├── main.py                       # main file
├── sales_win_model.pkl           # Saved RandomForest model
├── scaler.pkl                    # Saved StandardScaler
├── feature_columns.pkl           # Saved training feature order
└── sales_opportunities_balanced.csv  # (optional) exported synthetic dataset
```

## Setup

Python 3.10+ is recommended.

Using conda (recommended for Windows):

```pwsh
conda create -n saleswin python=3.10 -y
conda activate saleswin
conda install -y numpy pandas scikit-learn matplotlib seaborn joblib
conda install -y -c conda-forge lightgbm
pip install fastapi uvicorn
```

Using pip only:

```pwsh
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn joblib fastapi uvicorn
```

## Train & Export Artifacts

Run all cells in `salesWinPrediction.ipynb`. This will:

- Generate data and engineer features
- Train/tune models and compare performance
- Save artifacts to the project root:
  - `sales_win_model.pkl`
  - `scaler.pkl`
  - `feature_columns.pkl`

Optional: export the notebook code to a single Python file.

```pwsh
python .\export_notebook_code.py .\salesWinPrediction.ipynb
# Creates/updates main.py with the notebook code cells verbatim
```

## Run the API

The lightweight app `api_app.py` loads the saved artifacts and exposes `POST /predict/`.

```pwsh
# From the project folder
python -m uvicorn api_app:app --host 127.0.0.1 --port 8000
# or
uvicorn api_app:app --host 127.0.0.1 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json

## Call the Endpoint (Examples)

Required JSON fields:

- `opportunity_size` (int)
- `meetings_held` (int)
- `actions_completed` (int)
- `days_open` (int)
- `stage` (int)

Windows PowerShell (single line):

```pwsh
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"opportunity_size":55000,"meetings_held":3,"actions_completed":12,"days_open":40,"stage":2}'
```

Bash (macOS/Linux):

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"opportunity_size":55000,"meetings_held":3,"actions_completed":12,"days_open":40,"stage":2}'
```

Example response:

```json
{
  "prediction": "Won",
  "probability": 0.8123
}
```

## Troubleshooting

- 404 Not Found for `/predict/`: ensure the server is running (see Run the API) and you’re calling `/predict/` (trailing slash matters in FastAPI routes).
- Import/model errors on startup: make sure `sales_win_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` exist in the same folder as `api_app.py`.
- Port already in use: change `--port 8000` to a free port (e.g., 9000).
- PowerShell quoting: prefer the one-liner example above; multiline with backticks works too.

## License

MIT License
