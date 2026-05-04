"""
train.py — Model Training with MLflow Experiment Tracking
==========================================================
Trains XGBoost and Random Forest on the preprocessed IEEE-CIS data.
Logs all parameters, metrics, and artifacts to MLflow for experiment tracking.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ───────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MLFLOW_TRACKING_URI = 'file:///' + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns')).replace('\\', '/')


def load_processed_data():
    """Load the preprocessed train/test splits."""
    print("📂 Loading processed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all relevant classification metrics."""
    return {
        'f1_score': round(f1_score(y_true, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
    }


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with class imbalance handling via scale_pos_weight."""
    print("\n🚀 Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    params = {
        'n_estimators': 600,
        'max_depth': 8,
        'learning_rate': 0.02,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"   ✅ XGBoost — F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")
    return model, params, metrics, y_pred, y_prob


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with balanced class weights."""
    print("\n🌲 Training Random Forest...")

    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print(f"   ✅ Random Forest — F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")
    return model, params, metrics, y_pred, y_prob


def log_to_mlflow(model, model_name, params, metrics, feature_names):
    """Log a training run to MLflow with params, metrics, and artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection-ieee")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for key, val in params.items():
            mlflow.log_param(key, val)

        # Log metrics
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        # Log model
        if 'xgb' in model_name.lower():
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            mlflow.log_dict(importance, "feature_importance.json")

        print(f"   📝 Logged {model_name} to MLflow")


def run_training():
    """Full training pipeline: load data → train models → log to MLflow → save best."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_data()
    feature_names = X_train.columns.tolist()

    # Train both models
    xgb_model, xgb_params, xgb_metrics, xgb_pred, xgb_prob = train_xgboost(X_train, X_test, y_train, y_test)
    rf_model, rf_params, rf_metrics, rf_pred, rf_prob = train_random_forest(X_train, X_test, y_train, y_test)

    # Log both to MLflow
    log_to_mlflow(xgb_model, "XGBoost", xgb_params, xgb_metrics, feature_names)
    log_to_mlflow(rf_model, "RandomForest", rf_params, rf_metrics, feature_names)

    # Pick the best model based on ROC-AUC
    print("\n🏆 Model Comparison:")
    print(f"   XGBoost  — F1: {xgb_metrics['f1_score']}, ROC-AUC: {xgb_metrics['roc_auc']}, Precision: {xgb_metrics['precision']}, Recall: {xgb_metrics['recall']}")
    print(f"   RF       — F1: {rf_metrics['f1_score']}, ROC-AUC: {rf_metrics['roc_auc']}, Precision: {rf_metrics['precision']}, Recall: {rf_metrics['recall']}")

    if xgb_metrics['roc_auc'] >= rf_metrics['roc_auc']:
        best_model, best_name, best_metrics = xgb_model, 'XGBoost', xgb_metrics
        best_pred, best_prob = xgb_pred, xgb_prob
    else:
        best_model, best_name, best_metrics = rf_model, 'RandomForest', rf_metrics
        best_pred, best_prob = rf_pred, rf_prob

    print(f"\n   🥇 Best model: {best_name}")

    # Save the best model and metadata
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))

    model_metadata = {
        'model_name': best_name,
        'version': 'v1.0',
        'metrics': best_metrics,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'fraud_rate_train': float(round(y_train.mean() * 100, 2)),
        'fraud_rate_test': float(round(y_test.mean() * 100, 2)),
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)

    # Save confusion matrix data for the dashboard
    cm = confusion_matrix(y_test, best_pred)
    cm_data = {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1]),
    }
    with open(os.path.join(MODELS_DIR, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_data, f, indent=2)

    # Save test predictions for monitoring
    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': best_pred,
        'y_prob': best_prob
    })
    test_results.to_csv(os.path.join(PROCESSED_DIR, 'test_predictions.csv'), index=False)

    print(f"\n💾 Best model saved to {MODELS_DIR}")
    print(f"📊 Classification Report:\n")
    print(classification_report(y_test, best_pred, target_names=['Legitimate', 'Fraud']))

    return best_model, best_metrics


if __name__ == '__main__':
    run_training()
