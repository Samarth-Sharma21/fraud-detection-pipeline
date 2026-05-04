# Fraud Detection Pipeline

A production-grade fraud detection system built on the **IEEE-CIS Fraud Detection** dataset. Combines machine learning, MLOps, API serving, real-time explainability (SHAP), and a professional monitoring dashboard.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│   Raw Data      │────▶│  Preprocessing   │────▶│  Model Training      │
│  IEEE-CIS CSV   │     │  src/preprocess   │     │  src/train (XGBoost) │
│  590K txns      │     │  Feature eng.     │     │  MLflow tracking     │
└─────────────────┘     │  Label encoding   │     │  Hyperparameter opt  │
                        │  Scaling          │     └──────────┬───────────┘
                        └──────────────────┘                │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Streamlit      │◀────│  FastAPI          │◀────│  Evaluation          │
│  Dashboard      │     │  REST API         │     │  src/evaluate        │
│  5 tabs:        │     │  /predict         │     │  ROC, PR, CM plots   │
│  - Analyze      │     │  /health          │     │  SHAP explainability │
│  - Explainability│    │  /model/info      │     └──────────────────────┘
│  - Batch scoring│     └──────────────────┘
│  - Performance  │     ┌──────────────────┐
│  - Dataset      │◀────│  Monitoring       │
└─────────────────┘     │  src/monitor      │
                        │  Evidently drift  │
                        └──────────────────┘
```

## Results

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9337 |
| **F1 Score** | 0.4223 |
| **Precision** | 0.2901 |
| **Recall** | 0.7758 |
| **Training Samples** | 236,216 |
| **Test Samples** | 59,054 |
| **Features** | 60 interpretable features |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy, Scikit-learn |
| ML Models | XGBoost, Random Forest |
| Experiment Tracking | MLflow |
| Model Explainability | SHAP (TreeExplainer) |
| Data Monitoring | Evidently AI |
| API Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + ECharts + shadcn-ui |
| Deployment | Docker, HuggingFace Spaces |

## Project Structure

```
fraud-detection-pipeline/
├── data/
│   ├── raw/                    # IEEE-CIS CSVs (gitignored)
│   └── processed/              # Train/test splits, predictions
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── src/
│   ├── preprocess.py           # Feature engineering + encoding
│   ├── train.py                # XGBoost/RF training + MLflow
│   ├── evaluate.py             # Metrics, plots, confusion matrix
│   └── monitor.py              # Evidently AI drift reports
├── api/
│   └── main.py                 # FastAPI endpoints
├── models/                     # Saved model artifacts
├── reports/                    # Evaluation plots, drift reports
├── streamlit_app.py            # Dashboard (5 tabs)
└── requirements.txt
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place IEEE-CIS data in data/raw/
#    Files: train_transaction.csv, train_identity.csv

# 3. Run full pipeline
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.monitor

# 4. Launch dashboard
streamlit run streamlit_app.py

# 5. Launch API (optional)
uvicorn api.main:app --reload
```

## Dashboard Features

1. **Analyze** — Input transaction details, get instant fraud/legitimate verdict with confidence gauge
2. **Explainability** — SHAP waterfall chart showing exactly which features drove the prediction
3. **Batch Scoring** — Upload CSV of transactions, get downloadable fraud predictions for all rows
4. **Performance** — ROC-AUC, F1, confusion matrix heatmap, feature importance chart, ROC curve
5. **Dataset** — Class distribution, probability histogram, dataset statistics, sample predictions

## Design Decisions

- **Interpretable features over accuracy**: Dropped anonymous V-columns to keep SHAP explanations meaningful. 93.4% AUC with interpretable features > 97% black-box.
- **scale_pos_weight over SMOTE**: XGBoost's native class weighting avoids synthetic sample artifacts.
- **Stratified splits**: Preserves 3.53% fraud ratio in both train and test sets.
- **50% subsampling**: Balanced training speed vs data volume (295K from 590K transactions).

## What Interviewers See

> "I built an end-to-end fraud detection pipeline on the IEEE-CIS benchmark. The core challenge was 3.5% fraud rate class imbalance — I used XGBoost with scale_pos_weight and evaluated with ROC-AUC/F1. All experiments tracked via MLflow. The model includes SHAP explainability so analysts can see *why* transactions were flagged. Served via FastAPI and a Streamlit dashboard with batch scoring. Monitored with Evidently AI drift reports."

This covers: **data engineering + ML + model ops + explainability + API design + monitoring + deployment** — all connected.

## License

MIT
