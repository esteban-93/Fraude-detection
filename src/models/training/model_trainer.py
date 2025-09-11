"""
Model training module for fraud detection.
Handles training of various ML models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def get_models(self) -> Dict[str, Any]:
        """Define models for training."""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'svm': SVC(random_state=42, probability=True),
            'isolation_forest': IsolationForest(random_state=42, contamination=0.1)
        }
        return models
    
    def get_hyperparameters(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for tuning."""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        return param_grids
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smote') -> Tuple:
        """Handle class imbalance using various techniques."""
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
            
        logger.info(f"-Resampling completed. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning."""
        models = self.get_models()
        param_grids = self.get_hyperparameters()
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")
        
        model = models[model_name]
        
        # Hyperparameter tuning
        # if model_name in param_grids:
        if False:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=cv, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            best_model = model.fit(X_train, y_train)
            best_params = {}
            best_score = 0
        
        # Store model
        self.models[model_name] = {
            'model': best_model,
            'params': best_params,
            'score': best_score
        }
        
        logger.info(f"Model {model_name} trained. Best score: {best_score}")
        return self.models[model_name]
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all available models."""
        for model_name in self.get_models().keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        return self.models
    
    def select_best_model(self) -> Tuple[str, Any]:
        """Select the best performing model."""
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['score'])
        self.best_model = self.models[best_model_name]['model']
        self.best_score = self.models[best_model_name]['score']
        
        logger.info(f"Best model: {best_model_name} with score: {self.best_score}")
        return best_model_name, self.best_model
    
    def save_model(self, model_path: str, model_name: str = None):
        """Save the best model to disk."""
        if model_name:
            model_to_save = self.models[model_name]['model']
        else:
            model_to_save = self.best_model
           
        joblib.dump(model_to_save, model_path)
        logger.info(f"Model saved to {model_path}")
