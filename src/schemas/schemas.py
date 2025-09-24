"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class Transaction(BaseModel):
    """Transaction data model for fraud detection."""
    
    id: Optional[str] = Field(None, description="Transaction ID")
    Time: float = Field(..., description="Time in seconds since first transaction")
    Amount: float = Field(..., description="Transaction amount")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "txn_001",
                "Time": 0.0,
                "Amount": 149.62,
                "V1": -0.073403184690714,
                "V2": 0.27723157600516,
                "V3": -0.11047391018877,
                "V4": 0.0669280749146731,
                "V5": -0.108300452035545,
                "V6": -0.159786378772829,
                "V7": -0.270532677192782,
                "V8": 0.133502274842453,
                "V9": -0.0210530534538215,
                "V10": -0.0203958894663294,
                "V11": 0.0596169545912925,
                "V12": -0.101197302988088,
                "V13": -0.073403184690714,
                "V14": 0.27723157600516,
                "V15": -0.11047391018877,
                "V16": 0.0669280749146731,
                "V17": -0.108300452035545,
                "V18": -0.159786378772829,
                "V19": -0.270532677192782,
                "V20": 0.133502274842453,
                "V21": -0.0210530534538215,
                "V22": -0.0203958894663294,
                "V23": 0.0596169545912925,
                "V24": -0.101197302988088,
                "V25": -0.073403184690714,
                "V26": 0.27723157600516,
                "V27": -0.11047391018877,
                "V28": 0.0669280749146731
            }
        }

class PredictionResponse(BaseModel):
    """Response model for single transaction prediction."""
    
    transaction_id: str = Field(..., description="Transaction ID")
    prediction: int = Field(..., description="Prediction (0=genuine, 1=fraud)")
    is_fraud: bool = Field(..., description="Boolean fraud indicator")
    probability: Optional[float] = Field(None, description="Fraud probability score")
    risk_score: float = Field(..., description="Risk score (0-100)")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "prediction": 0,
                "is_fraud": False,
                "probability": 0.15,
                "risk_score": 15.0,
                "timestamp": "2024-01-15T10:30:00Z",
                "model_version": "1.0.0"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch transaction predictions."""
    
    transactions: List[Transaction] = Field(..., description="List of transactions to predict")

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "id": "txn_001",
                        "Time": 0.0,
                        "Amount": 149.62,
                        "V1": -0.073403184690714,
                        "V2": 0.27723157600516,
                        "V3": -0.11047391018877,
                        "V4": 0.0669280749146731,
                        "V5": -0.108300452035545,
                        "V6": -0.159786378772829,
                        "V7": -0.270532677192782,
                        "V8": 0.133502274842453,
                        "V9": -0.0210530534538215,
                        "V10": -0.0203958894663294,
                        "V11": 0.0596169545912925,
                        "V12": -0.101197302988088,
                        "V13": -0.073403184690714,
                        "V14": 0.27723157600516,
                        "V15": -0.11047391018877,
                        "V16": 0.0669280749146731,
                        "V17": -0.108300452035545,
                        "V18": -0.159786378772829,
                        "V19": -0.270532677192782,
                        "V20": 0.133502274842453,
                        "V21": -0.0210530534538215,
                        "V22": -0.0203958894663294,
                        "V23": 0.0596169545912925,
                        "V24": -0.101197302988088,
                        "V25": -0.073403184690714,
                        "V26": 0.27723157600516,
                        "V27": -0.11047391018877,
                        "V28": 0.0669280749146731
                    }
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Response model for batch transaction predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    fraud_count: int = Field(..., description="Number of transactions predicted as fraud")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Batch processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "transaction_id": "txn_001",
                        "prediction": 0,
                        "is_fraud": False,
                        "probability": 0.15,
                        "risk_score": 15.0,
                        "timestamp": "2024-01-15T10:30:00Z",
                        "model_version": "1.0.0"
                    }
                ],
                "total_transactions": 1,
                "fraud_count": 0,
                "processing_time": 0.05,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }

class ModelInfoResponse(BaseModel):
    """Model information response model."""
    
    model_path: str = Field(..., description="Path to the model file")
    scaler_path: Optional[str] = Field(None, description="Path to the scaler file")
    model_type: str = Field(..., description="Type of the model")
    has_scaler: bool = Field(..., description="Whether a scaler is available")
    version: str = Field(..., description="Model version")
    config: Dict[str, Any] = Field(..., description="Model configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "models/trained/fraud_model.pkl",
                "scaler_path": "models/trained/scaler.pkl",
                "model_type": "RandomForestClassifier",
                "has_scaler": True,
                "version": "1.0.0",
                "config": {"model_version": "1.0.0"}
            }
        }
