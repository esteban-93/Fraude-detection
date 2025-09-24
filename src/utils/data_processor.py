"""
Data preprocessing module for fraud detection.
Handles data cleaning, feature engineering, and preparation for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data preprocessing for fraud detection."""
    
    # def __init__(self, config: Dict[str, Any]):
    def __init__(self):
        # self.config = config
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset."""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(data.median())
        
        logger.info(f"Data cleaned. Shape: {data.shape}")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance."""
        
        # Time-based features
        data['hour'] = (data['Time'] % (24 * 3600)) // 3600
        data['day_of_week'] = (data['Time'] // (24 * 3600)) % 7

        # other features
        data['log_amount'] = np.log1p(data['Amount'])
        data['V1*V2'] = data['V1'] * data['V2']
        data['V17*V14'] = data['V17'] * data['V14']
        data['V12*V10'] = data['V12'] * data['V10']
        data['V14_inv'] = 1 / (data['V14'] + 1e-5)
        data['V17_inv'] = 1 / (data['V17'] + 1e-5)
        
        # Amount-based features
        data['scaled_amount'] = StandardScaler().fit_transform(data[['Amount']])
        data['log_scaled_amount'] = np.log1p(data['scaled_amount'])
        logger.info("Feature engineering completed")
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Separate features and target
        # feature_columns = [col for col in data.columns if col not in ['Class', 'Time']]
        # X = data[feature_columns]

        feature_columns = ['log_amount', 'V1*V2', 'V17*V14', 'V12*V10', 'V14_inv', 'V17_inv', 'scaled_amount', 'log_scaled_amount']
        X = data[feature_columns]
        y = data['Class']
        
        self.feature_columns = feature_columns
        logger.info(f"Features prepared. Shape: {X.shape}")
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using RobustScaler."""
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
