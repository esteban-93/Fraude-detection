"""
Prediction script for fraud detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.inference.predictor import FraudPredictor
from src.utils.config_loader import load_config
import pandas as pd
import json

def main():
    """Main prediction pipeline."""
    # Load configuration
    config = load_config()
    
    # Initialize predictor
    predictor = FraudPredictor(
        model_path=config['model']['model_path'],
        scaler_path=config['model']['scaler_path'],
        config=config['model']
    )
    
    # Example transaction
    sample_transaction = {
        'id': 'sample_001',
        'Time': 0.0,
        'Amount': 149.62,
        'V1': -0.073403184690714,
        'V2': 0.27723157600516,
        'V3': -0.11047391018877,
        'V4': 0.0669280749146731,
        'V5': -0.108300452035545,
        'V6': -0.159786378772829,
        'V7': -0.270532677192782,
        'V8': 0.133502274842453,
        'V9': -0.0210530534538215,
        'V10': -0.0203958894663294,
        'V11': 0.0596169545912925,
        'V12': -0.101197302988088,
        'V13': -0.073403184690714,
        'V14': 0.27723157600516,
        'V15': -0.11047391018877,
        'V16': 0.0669280749146731,
        'V17': -0.108300452035545,
        'V18': -0.159786378772829,
        'V19': -0.270532677192782,
        'V20': 0.133502274842453,
        'V21': -0.0210530534538215,
        'V22': -0.0203958894663294,
        'V23': 0.0596169545912925,
        'V24': -0.101197302988088,
        'V25': -0.073403184690714,
        'V26': 0.27723157600516,
        'V27': -0.11047391018877,
        'V28': 0.0669280749146731
    }
    
    # Make prediction
    result = predictor.predict_single(sample_transaction)
    
    # Print results
    print("Fraud Detection Prediction:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
