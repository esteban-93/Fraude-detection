"""
Model inference module for fraud detection.
Handles real-time predictions and model serving.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FraudPredictor:
    """Handles fraud detection predictions."""
    
    def __init__(self, model_path: str, scaler_path: str = None, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            if self.scaler_path:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_single_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess a single transaction for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Feature engineering (same as training)
        if 'Time' in df.columns:
            df['hour'] = (df['Time'] % (24 * 3600)) // 3600
            df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7
        
        if 'Amount' in df.columns:
            df['amount_log'] = np.log1p(df['Amount'])
            df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Select features (exclude Time and Class)
        feature_columns = [col for col in df.columns if col not in ['Class', 'Time']]
        X = df[feature_columns]
        
        return X
    
    def preprocess_batch(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess a batch of transactions."""
        df = pd.DataFrame(transactions)
        
        # Feature engineering
        if 'Time' in df.columns:
            df['hour'] = (df['Time'] % (24 * 3600)) // 3600
            df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7
        
        if 'Amount' in df.columns:
            df['amount_log'] = np.log1p(df['Amount'])
            df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Select features
        feature_columns = [col for col in df.columns if col not in ['Class', 'Time']]
        X = df[feature_columns]
        
        return X
    
    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single transaction."""
        try:
            # Preprocess
            X = self.preprocess_single_transaction(transaction)
            
            # Scale if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            else:
                X_scaled = X
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get probability if available
            try:
                probability = self.model.predict_proba(X_scaled)[0][1]
            except AttributeError:
                probability = None
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(prediction, probability)
            
            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'prediction': int(prediction),
                'is_fraud': bool(prediction),
                'probability': float(probability) if probability is not None else None,
                'risk_score': risk_score,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.config.get('model_version', '1.0')
            }
            
            logger.info(f"Prediction made for transaction {result['transaction_id']}: {result['is_fraud']}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict fraud for a batch of transactions."""
        try:
            # Preprocess
            X = self.preprocess_batch(transactions)
            
            # Scale if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            else:
                X_scaled = X
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba(X_scaled)[:, 1]
            except AttributeError:
                probabilities = [None] * len(predictions)
            
            # Format results
            results = []
            for i, transaction in enumerate(transactions):
                risk_score = self._calculate_risk_score(predictions[i], probabilities[i])
                
                result = {
                    'transaction_id': transaction.get('id', f'transaction_{i}'),
                    'prediction': int(predictions[i]),
                    'is_fraud': bool(predictions[i]),
                    'probability': float(probabilities[i]) if probabilities[i] is not None else None,
                    'risk_score': risk_score,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': self.config.get('model_version', '1.0')
                }
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(transactions)} transactions")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch prediction: {e}")
            raise
    
    def _calculate_risk_score(self, prediction: int, probability: float = None) -> float:
        """Calculate a risk score between 0 and 100."""
        if probability is not None:
            return float(probability * 100)
        else:
            return float(prediction * 100)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'model_type': type(self.model).__name__,
            'has_scaler': self.scaler is not None,
            'config': self.config
        }
