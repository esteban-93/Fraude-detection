"""
Fraud Detection API package.

This package contains the FastAPI application for fraud detection service.
"""

__version__ = "1.0.0"

from .main import app
from .schemas import (
    Transaction,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from .dependencies import get_predictor, get_config
from .config import settings, get_settings

__all__ = [
    "app",
    "Transaction",
    "PredictionResponse", 
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "get_predictor",
    "get_config",
    "settings",
    "get_settings"
]
