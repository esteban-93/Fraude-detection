"""
FastAPI dependencies for fraud detection service.
"""

from functools import lru_cache
import os
import logging
from typing import Optional

# from ..models.inference.predictor import FraudPredictor
from src.utils.inference.predictor import FraudPredictor
from ..utils.config_loader import load_config
from src.config import settings

logger = logging.getLogger(__name__)

# Global predictor instance
_predictor: Optional[FraudPredictor] = None

# @lru_cache()
# def get_config():
#     """Get application configuration."""
#     return load_config()

def get_predictor() -> FraudPredictor:
    """Get or create the fraud predictor instance."""
    global _predictor
    
    if _predictor is None:
        try:
            config = get_config()
            model_config = config.get('model', {})
            
            model_path = os.getenv("MODEL_PATH", model_config.get('model_path', 'models/trained/fraud_model.pkl'))
            scaler_path = os.getenv("SCALER_PATH", model_config.get('scaler_path', 'models/trained/scaler.pkl'))
            
            # Ensure paths are absolute
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)
            if not os.path.isabs(scaler_path):
                scaler_path = os.path.join(os.getcwd(), scaler_path)
            
            _predictor = FraudPredictor(
                model_path=model_path,
                scaler_path=scaler_path,
                config=model_config
            )
            
            logger.info(f"Predictor initialized with model: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise
    
    return _predictor

def reset_predictor():
    """Reset the predictor instance (useful for testing)."""
    global _predictor
    _predictor = None
    get_config.cache_clear()
    logger.info("Predictor instance reset")
