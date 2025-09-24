"""
FastAPI main application for fraud detection service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from datetime import datetime
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .schemas import (
    Transaction, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, ModelInfoResponse
)
from .dependencies import get_predictor
from .endpoints import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for credit card transactions using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["fraud-detection"])

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     """Global exception handler for unhandled errors."""
#     logger.error(f"Unhandled exception: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error", "error": str(exc)}
#     )

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the application on startup."""
#     logger.info("Starting Fraud Detection API...")
#     try:
#         # Initialize the predictor
#         predictor = get_predictor()
#         logger.info("Model loaded successfully")
#     except Exception as e:
#         logger.error(f"Failed to load model: {e}")
#         raise

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on application shutdown."""
#     logger.info("Shutting down Fraud Detection API...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
