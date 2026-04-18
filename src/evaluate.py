"""Model evaluation module for fraud detection."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score
)
import seaborn as sns


class ModelEvaluator:
    """Class for evaluating model performance."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None):
        """
        Initialize evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.metrics = {}
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        self.metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1": f1_score(self.y_true, self.y_pred),
        }
        
        if self.y_pred_proba is not None:
            self.metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
        
        return self.metrics
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report as string
        """
        return classification_report(self.y_true, self.y_pred, 
                                     target_names=["Normal", "Fraud"])
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot ROC curve.
        
        Args:
            figsize: Figure size
        """
        if self.y_pred_proba is None:
            raise ValueError("Prediction probabilities required for ROC curve")
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1])
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            figsize: Figure size
        """
        if self.y_pred_proba is None:
            raise ValueError("Prediction probabilities required for PR curve")
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba[:, 1])
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.show()
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        metrics = self.calculate_metrics()
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        print("\n" + self.get_classification_report())
        print("="*50 + "\n")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: Optional[np.ndarray] = None) -> ModelEvaluator:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        ModelEvaluator instance
    """
    return ModelEvaluator(y_true, y_pred, y_pred_proba)
