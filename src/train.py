"""Model training module for fraud detection."""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib


class FraudDetectionModel:
    """Wrapper class for fraud detection models."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ("random_forest", "gradient_boosting", "logistic_regression")
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.trained = False
    
    def _create_model(self, model_type: str, **kwargs):
        """Create model instance based on type."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                random_state=kwargs.get("random_state", 42)
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            FraudDetectionModel instance
        """
        model = joblib.load(filepath)
        instance = FraudDetectionModel.__new__(FraudDetectionModel)
        instance.model = model
        instance.trained = True
        return instance


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    **kwargs
) -> FraudDetectionModel:
    """
    Train a fraud detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train
        **kwargs: Additional parameters for the model
        
    Returns:
        Trained FraudDetectionModel instance
    """
    model = FraudDetectionModel(model_type, **kwargs)
    model.train(X_train, y_train)
    return model
