"""
Training script for fraud detection model.
"""

import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_processor import DataProcessor
from src.utils.training.model_trainer import ModelTrainer
from src.utils.evaluation.model_evaluator import ModelEvaluator
from mlflow import sklearn
# from src.fraud_detector.utils.config_loader import load_config
# from src.config import settings
import logging

def main():
    """Main training pipeline."""
    # Load configuration
    # config = load_config()
    # settings = get_settings()
    # print(settings)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting fraud detection model training...")

    # Initialize components
    # data_processor = DataProcessor(settings)
    data_processor = DataProcessor()
    # model_trainer = ModelTrainer(settings)
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    # data = data_processor.load_data(config['data']['raw_data_path'])
    data = data_processor.load_data('/Users/juan_esteban/Documents/data_science/Fraude-detection/data/raw/creditcard.csv')
    data = data_processor.clean_data(data)
    data = data_processor.engineer_features(data)

    # Prepare features
    X, y = data_processor.prepare_features(data)

    # Split data
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

    # Scale features
    # X_train_scaled, X_test_scaled = data_processor.scale_features(X_train, X_test)

    # Handle class imbalance
    X_resampled, y_resampled = model_trainer.handle_imbalance(
        X_train, y_train, 
        # method=config['training']['resampling_method']
        method='smote'
    )

    # Train models
    MODEL = "logistic_regression"
    logger.info("Training models...")
    # model_trainer.train_all_models(X_resampled, y_resampled)
    # model_trainer.train_model(MODEL, X_resampled, y_resampled, X_test, y_test)
    experiment_id, run_id = model_trainer.train_model(MODEL, X_resampled, y_resampled, X_test, y_test)

    # import mlflow
    # best_model = mlflow.pyfunc.load_model(
    #     model_uri="mlflow-artifacts:/382050914863043977/models/m-7d581c0921494964877d699cb1f59a66/artifacts"
    # )





    # Select best model
    best_model_name, best_model = model_trainer.select_best_model()
    logger.info(f"Best model: {best_model_name}")

#     # Evaluate model
    # logger.info("Evaluating model...")
    # evaluation_results = model_evaluator.evaluate_model(best_model, MODEL, X_test, y_test)




#     # Save model
#     model_path_root = config['model']['model_path']
#     model_path = f"{model_path_root}/{MODEL}.pkl"
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     model_trainer.save_model(model_path, best_model_name)

#     # Save scaler
#     scaler_path_root = config['model']['scaler_path']
#     scaler_path = f"{scaler_path_root}/{MODEL}.pkl"
#     os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
#     joblib.dump(data_processor.scaler, scaler_path)

#     logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
