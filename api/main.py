"""
main.py — FastAPI Backend for Fraud Detection
===============================================
REST API serving the trained fraud detection model.
Endpoints: /predict, /health, /model/info
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.schemas import (
    TransactionInput, PredictionResponse,
    ModelInfoResponse, HealthResponse
)
from api.model import FraudDetector

# ─── Initialize App ──────────────────────────────────────────────────
app = FastAPI(
    title="🛡️ Fraud Detection Pipeline API",
    description="Real-time fraud detection for e-commerce transactions using XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow Streamlit and any frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
detector = None

@app.on_event("startup")
async def load_model():
    global detector
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        detector = FraudDetector(models_dir=models_dir)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("   API will start but /predict will return errors until model is trained.")


# ─── Endpoints ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (required for HuggingFace Spaces)."""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        version=detector.metadata.get('version', 'unknown') if detector else 'no-model'
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict whether a transaction is fraudulent.

    Accepts transaction features and returns:
    - prediction: FRAUD or LEGITIMATE
    - confidence: probability score (0-1)
    - risk_level: LOW, MEDIUM, HIGH, or CRITICAL
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    try:
        result = detector.predict(transaction.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get current model metadata and performance metrics."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    meta = detector.metadata
    return ModelInfoResponse(
        model_name=meta['model_name'],
        version=meta['version'],
        f1_score=meta['metrics']['f1_score'],
        roc_auc=meta['metrics']['roc_auc'],
        precision=meta['metrics']['precision'],
        recall=meta['metrics']['recall'],
        training_samples=meta['training_samples'],
        test_samples=meta['test_samples'],
        fraud_rate_train=meta['fraud_rate_train'],
        fraud_rate_test=meta['fraud_rate_test'],
        n_features=meta['n_features'],
    )


@app.get("/report/drift", response_class=HTMLResponse)
async def drift_report():
    """Serve the latest Evidently AI drift report."""
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'drift_report.html')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Drift report not generated yet. Run src/monitor.py first.")

    with open(report_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
