"""
Model training module for fraud detection.
Handles training of various ML models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
import mlflow
from mlflow.tracking import MlflowClient


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("/experiments")

class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    # def __init__(self, config: Dict[str, Any]):
    def __init__(self):
        # self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def get_models(self) -> Dict[str, Any]:
        """Define models for training."""
        models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            'isolation_forest': IsolationForest()
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
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
        }
        }
        return param_grids

    def get_fixed_params(self):
        params = {
            'logistic_regression': {
                'C': 1,
                'penalty': 'l1',
                'solver': 'saga'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 1.0
            }
        }
        return params
    
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
        """Train a single model with hyperparameter tuning & register them in MLflow"""

        hyperparam_optimization = False
        models = self.get_models()
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")
        
        if hyperparam_optimization:
            param_grids = self.get_hyperparameters()

            for model_name, model in models.items():
                with mlflow.start_run(run_name=f'best_{model_name}') as parent_run:
                    run_id = parent_run.info.run_id
                    experiment_id = parent_run.info.experiment_id

                    mlflow.log_param("dataset_length", len(X_train))
                    mlflow.log_param("features", X_train.shape[0])

                    # if model_name in param_grids:
                    if model_name in param_grids:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        grid_search = GridSearchCV(
                                model, param_grids[model_name], 
                                cv=cv, scoring='f1', n_jobs=-1, verbose=1
                            )
                        grid_search.fit(X_train, y_train)

                        best_params = grid_search.best_params_
                        best_score = grid_search.best_score_
                        mlflow.log_params(best_params)
                        mlflow.log_metric("best_score_f1", grid_search.best_score_)
                        
                        best_model = grid_search.best_estimator_

                        # mlflow.sklearn.log_model(sk_model=best_model, 
                        #                         artifact_path="model",
                        #                         registered_model_name=f"{model_name}-best_model"
                        # )
                        # params_model = params[model_name]
                        # model.set_params(**params_model)
                        # mlflow.log_params(params_model)
                        # model.fit(X_train, y_train)

                        y_pred = best_model.predict(X_val)

                        metrics = {
                            'accuracy': accuracy_score(y_val, y_pred),
                            'precision': precision_score(y_val, y_pred),
                            'recall': recall_score(y_val, y_pred),
                            'f1_score': f1_score(y_val, y_pred),
                        }
                        mlflow.log_metrics(metrics)

                        mlflow.sklearn.log_model(
                                sk_model=best_model,
                                name=f"sklear-model-{model_name}",
                                input_example=X_train,
                                registered_model_name=f"{model_name}-best_model"
                        )
            return experiment_id, run_id
        
        else:
            with mlflow.start_run(run_name=f'unique_{model_name}') as parent_run:
                run_id = parent_run.info.run_id
                experiment_id = parent_run.info.experiment_id

                model = models[model_name]
                params = self.get_fixed_params()
                params_model = params[model_name]
                model.set_params(**params_model)

                mlflow.log_params(params_model)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                metrics = {
                            'accuracy': accuracy_score(y_val, y_pred),
                            'precision': precision_score(y_val, y_pred),
                            'recall': recall_score(y_val, y_pred),
                            'f1_score': f1_score(y_val, y_pred),
                        }
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                            sk_model=model,
                            name=f"sklear-model-{model_name}",
                            input_example=X_train,
                            registered_model_name=f"{model_name}-best_model"
                    )



            # break
        
        # Hyperparameter tuning
        # if model_name in param_grids:
        #     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        #     grid_search = GridSearchCV(
        #         model, param_grids[model_name], 
        #         cv=cv, scoring='f1', n_jobs=-1, verbose=1
        #     )
        #     grid_search.fit(X_train, y_train)
        #     best_model = grid_search.best_estimator_
        #     best_params = grid_search.best_params_
        #     best_score = grid_search.best_score_
        # else:
        #     best_model = model.fit(X_train, y_train)
        #     best_params = {}
        #     best_score = 0
        


        # Store model
        # self.models[model_name] = {
        #     'model': best_model,
        #     'params': best_params,
        #     'score': best_score
        # }
        
        # logger.info(f"Model {model_name} trained. Best score: {best_score}")
        # return self.models[model_name]
            return experiment_id, run_id
    
    # def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    #     """Train all available models."""
    #     for model_name in self.get_models().keys():
    #         try:
    #             self.train_model(model_name, X_train, y_train)
    #         except Exception as e:
    #             logger.error(f"Error training {model_name}: {e}")
        
    #     return self.models
    
    def select_best_model(self, 
            experiment_id: str = None, 
            metric: str = 'f1_score') -> Tuple[str, Any]:
        """Select the best performing model."""

        client = MlflowClient()
        experiment_id = '382050914863043977'
        metric = 'f1_score'

        best_run = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                order_by=[f"metrics.{metric} DESC"],  # sort by metric
                max_results=1
            )[0]

        run_id = best_run.info.run_id
        best_model_name = '_'.join(best_run.info.run_name.split('_')[1:])
        best_f1 =  best_run.data.metrics[metric]
        model_id = best_run.outputs.model_outputs[0].model_id

        model_uri = f"mlflow-artifacts:/{experiment_id}/models/{model_id}/artifacts"
        self.best_model = mlflow.pyfunc.load_model(
                model_uri=model_uri
        )

        logger.info(f"Best model: {best_model_name} with {metric}: {best_f1}")
        return best_model_name, self.best_model
    
    # def save_model(self, model_path: str, model_name: str = None):
    #     """Save the best model to disk."""
    #     if model_name:
    #         model_to_save = self.models[model_name]['model']
    #     else:
    #         model_to_save = self.best_model
           
    #     joblib.dump(model_to_save, model_path)
    #     logger.info(f"Model saved to {model_path}")
