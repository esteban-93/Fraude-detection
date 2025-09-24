"""
API endpoints for fraud detection service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import logging
from datetime import datetime

from .schemas import (
    Transaction, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, ModelInfoResponse
)
from .dependencies import get_predictor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "predict_batch": "/api/v1/predict_batch",
            "model_info": "/api/v1/model_info"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(predictor=Depends(get_predictor)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None,
        version="1.0.0"
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: Transaction,
    predictor=Depends(get_predictor)
):
    """Predict fraud for a single transaction."""
    try:
        # Convert Pydantic model to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict_single(transaction_dict)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    predictor=Depends(get_predictor)
):
    """Predict fraud for multiple transactions."""
    try:
        start_time = datetime.now()
        
        # Convert Pydantic models to dicts
        transactions = [t.dict() for t in request.transactions]
        
        # Make predictions
        results = predictor.predict_batch(transactions)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        fraud_count = sum(1 for r in results if r['is_fraud'])
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_transactions=len(results),
            fraud_count=fraud_count,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info(predictor=Depends(get_predictor)):
    """Get information about the loaded model."""
    try:
        model_info = predictor.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/stats", response_model=dict)
async def get_stats(predictor=Depends(get_predictor)):
    """Get API statistics and metrics."""
    try:
        # This could be expanded to include actual metrics
        return {
            "total_predictions": 0,  # Would be tracked in production
            "fraud_predictions": 0,
            "uptime": "0:00:00",  # Would be calculated in production
            "model_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
