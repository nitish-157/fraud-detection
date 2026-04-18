"""
Model Training Module for Credit Card Fraud Detection

Trains and manages multiple classification models:
- Logistic Regression
- Random Forest
- XGBoost

Features:
- Modular design for easy extension
- Cross-validation support
- Hyperparameter tuning ready
- Model persistence (save/load)
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {type(e).__name__}")

warnings.filterwarnings('ignore')


class FraudDetectionModelTrainer:
    """Train and manage multiple fraud detection models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_histories = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available models with default hyperparameters."""
        print("🔧 Initializing models...")
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            try:
                self.models['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scale_pos_weight=775,  # Approximate class imbalance ratio
                    eval_metric='logloss'
                )
            except Exception as e:
                print(f"⚠️  XGBoost initialization failed: {str(e)}")
        
        print(f"✓ {len(self.models)} models initialized: {list(self.models.keys())}")
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print training info
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        if verbose:
            print(f"\n🚀 Training {model_name}...")
            print(f"   Data shape: {X_train.shape}")
            print(f"   Sample distribution: {np.bincount(y_train.astype(int))}")
        
        model = self.models[model_name]
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Store trained model
            self.trained_models[model_name] = model
            
            # Get training predictions for metrics
            y_pred = model.predict(X_train)
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            
            # Calculate training metrics
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred),
                'precision': precision_score(y_train, y_pred, zero_division=0),
                'recall': recall_score(y_train, y_pred, zero_division=0),
                'f1': f1_score(y_train, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_train, y_pred_proba)
            }
            
            if verbose:
                print(f"✓ {model_name} trained successfully!")
                print(f"  Training Metrics:")
                for metric, value in train_metrics.items():
                    print(f"    {metric.upper()}: {value:.4f}")
            
            return model, train_metrics
        
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            raise
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print training info
            
        Returns:
            Dictionary with training histories for all models
        """
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        histories = {}
        
        for model_name in self.models.keys():
            try:
                _, metrics = self.train_model(model_name, X_train, y_train, verbose)
                histories[model_name] = metrics
                self.model_histories[model_name] = metrics
            except Exception as e:
                print(f"⚠️  Skipped {model_name}: {str(e)}")
                continue
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        return histories
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print evaluation info
            
        Returns:
            Dictionary with evaluation metrics for all models
        """
        print("\n" + "="*70)
        print("EVALUATING MODELS ON TEST SET")
        print("="*70)
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            evaluation_results[model_name] = metrics
            
            if verbose:
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1']:.4f}")
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n" + "="*70)
        
        return evaluation_results
    
    def compare_models(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metric: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Compare models across a specific metric.
        
        Args:
            evaluation_results: Results from evaluate_models()
            metric: Metric to compare (accuracy, precision, recall, f1, roc_auc)
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1': results['f1'],
                'ROC-AUC': results['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Map metric names
        metric_map = {
            'f1': 'F1',
            'roc_auc': 'ROC-AUC',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        
        sort_column = metric_map.get(metric, 'F1')
        df_comparison = df_comparison.sort_values(sort_column, ascending=False)
        
        return df_comparison
    
    def get_best_model(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metric: str = 'f1'
    ) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            evaluation_results: Results from evaluate_models()
            metric: Metric to use for selection
            
        Returns:
            Tuple of (best_model_name, best_model_object)
        """
        best_model_name = max(
            evaluation_results.keys(),
            key=lambda x: evaluation_results[x][metric]
        )
        best_model = self.trained_models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   {metric.upper()}: {evaluation_results[best_model_name][metric]:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"✓ Model '{model_name}' saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            filepath: Path to load the model from
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"✓ Model loaded from {filepath}")
    
    def save_all_models(self, directory: str = "models") -> None:
        """
        Save all trained models to directory.
        
        Args:
            directory: Directory to save models in
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = f"{directory}/{model_name.replace(' ', '_')}.pkl"
            self.save_model(model_name, filepath)
    
    def get_trained_models(self) -> Dict[str, Any]:
        """
        Get dictionary of all trained models.
        
        Returns:
            Dictionary of {model_name: model_object}
        """
        return self.trained_models


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FRAUD DETECTION - MODEL TRAINING DEMO")
    print("="*70)
    
    # Import preprocessing utilities
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    
    # Load and prepare data
    print("\n📂 Loading dataset...")
    df = pd.read_csv("Fraud.csv")
    
    # Sample for demo
    print("📊 Sampling 100,000 records...")
    df, _ = train_test_split(
        df,
        train_size=100000,
        random_state=42,
        stratify=df['isFraud']
    )
    
    # Prepare features
    print("🔧 Preparing features...")
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop = string_cols + ['isFlaggedFraud']
    X = df.drop(columns=['isFraud'] + cols_to_drop)
    y = df['isFraud']
    
    # Split data
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    print("⚖️  Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train.values)
    
    print(f"\nTraining data shape: {X_train_balanced.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer(random_state=42)
    
    # Train all models
    training_histories = trainer.train_all_models(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test_scaled, y_test.values)
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison_df = trainer.compare_models(evaluation_results, metric='roc_auc')
    print("\n" + comparison_df.to_string(index=False))
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model(evaluation_results, metric='roc_auc')
    
    # Save all models
    print("\n💾 Saving models...")
    trainer.save_all_models(directory="models")
    
    # Get all trained models
    all_models = trainer.get_trained_models()
    print(f"\n✅ {len(all_models)} models trained and ready!")
    
    print("\n" + "="*70)
    print("✨ TRAINING COMPLETE")
    print("="*70)
