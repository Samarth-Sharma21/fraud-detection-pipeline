"""
model.py — Model Loader & Prediction Logic
============================================
Loads the trained model, scaler, and encoders from disk.
Handles feature transformation and inference.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib


class FraudDetector:
    """Wrapper class for the trained fraud detection model."""

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metadata = None
        self._load()

    def _load(self):
        """Load all model artifacts from disk."""
        self.model = joblib.load(os.path.join(self.models_dir, 'best_model.joblib'))
        self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
        self.label_encoders = joblib.load(os.path.join(self.models_dir, 'label_encoders.joblib'))
        self.feature_names = joblib.load(os.path.join(self.models_dir, 'feature_names.joblib'))

        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'r') as f:
            self.metadata = json.load(f)

    def predict(self, transaction_data: dict) -> dict:
        """
        Make a fraud prediction for a single transaction.

        Args:
            transaction_data: dict with transaction features

        Returns:
            dict with prediction, confidence, and risk_level
        """
        # Build feature vector with defaults for missing features
        features = {}
        for fname in self.feature_names:
            if fname in transaction_data:
                features[fname] = transaction_data[fname]
            else:
                features[fname] = 0  # Default for missing features

        # Encode categorical features
        categorical_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain',
                                'R_emaildomain', 'DeviceType', 'DeviceInfo_cleaned']

        for col in categorical_features:
            if col in features and col in self.label_encoders:
                le = self.label_encoders[col]
                val = str(features[col])
                if val in le.classes_:
                    features[col] = le.transform([val])[0]
                else:
                    # Unknown category — use the most common class
                    features[col] = 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([features], columns=self.feature_names)

        # Scale numeric features (same ones used during training)
        from src.preprocess import NUMERIC_FEATURES
        numeric_cols = [c for c in NUMERIC_FEATURES + ['TransactionAmt_log', 'Transaction_hour', 'Transaction_dayofweek']
                        if c in df.columns]

        # Add log transform for TransactionAmt
        if 'TransactionAmt' in transaction_data and 'TransactionAmt_log' in df.columns:
            df['TransactionAmt_log'] = np.log1p(transaction_data['TransactionAmt'])

        # Add day of week default
        if 'Transaction_dayofweek' in df.columns:
            df['Transaction_dayofweek'] = transaction_data.get('Transaction_dayofweek', 3)

        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Predict
        prob = self.model.predict_proba(df)[0][1]
        prediction = "FRAUD" if prob >= 0.5 else "LEGITIMATE"

        # Risk level
        if prob >= 0.85:
            risk_level = "CRITICAL"
        elif prob >= 0.6:
            risk_level = "HIGH"
        elif prob >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            'prediction': prediction,
            'confidence': round(float(prob), 4),
            'risk_level': risk_level,
            'model_version': self.metadata.get('version', 'v1.0')
        }
