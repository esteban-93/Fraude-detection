"""
Model evaluation module for fraud detection.
Handles model performance evaluation and metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance metrics."""
    
    # def __init__(self, config: Dict[str, Any]):
    def __init__(self):
        # self.config = config
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = self._calculate_pr_auc(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_pr_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Precision-Recall AUC."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return np.trapz(precision, recall)
    
    def evaluate_model(self, model, model_name, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba = None
            logger.warning("Model does not support probability predictions")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        evaluation_results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Model evaluation completed. F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Model evaluation completed. recall: {metrics['recall']:.4f}")
        self.plot_confusion_matrix(cm, model_name, f"models/artifacts/plots/{model_name}")

        return evaluation_results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save_path: str = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Fraud'],
                   yticklabels=['Genuine', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str, save_path: str = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str, save_path: str = None):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models' performance."""
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Specificity': metrics['specificity'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'PR-AUC': metrics.get('pr_auc', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
