"""
schemas.py — Pydantic Request/Response Models
==============================================
Strict type validation for the FastAPI prediction endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionInput(BaseModel):
    """Input schema for a single transaction prediction request."""
    TransactionAmt: float = Field(..., description="Transaction amount in USD", ge=0)
    ProductCD: str = Field(default="W", description="Product code: W, H, C, R, or S")
    card4: str = Field(default="visa", description="Card network: visa, mastercard, discover, american express")
    card6: str = Field(default="debit", description="Card type: debit, credit, charge, debit or credit")
    P_emaildomain: str = Field(default="Gmail", description="Purchaser email domain group")
    R_emaildomain: str = Field(default="Unknown", description="Recipient email domain group")
    DeviceType: str = Field(default="desktop", description="Device type: desktop or mobile")
    DeviceInfo_cleaned: str = Field(default="Windows", description="Device OS/brand")
    Transaction_hour: int = Field(default=12, description="Hour of transaction (0-23)", ge=0, le=23)

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 150.0,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "Gmail",
                "R_emaildomain": "Unknown",
                "DeviceType": "desktop",
                "DeviceInfo_cleaned": "Windows",
                "Transaction_hour": 14
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for the prediction endpoint."""
    prediction: str = Field(..., description="FRAUD or LEGITIMATE")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    model_version: str = Field(..., description="Current model version identifier")


class ModelInfoResponse(BaseModel):
    """Output schema for model metadata endpoint."""
    model_name: str
    version: str
    f1_score: float
    roc_auc: float
    precision: float
    recall: float
    training_samples: int
    test_samples: int
    fraud_rate_train: float
    fraud_rate_test: float
    n_features: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
