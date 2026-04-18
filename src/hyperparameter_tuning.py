"""
Hyperparameter Tuning Module for Fraud Detection

Optimizes:
- Random Forest using GridSearchCV
- XGBoost using RandomizedSearchCV
- Optimizes for Recall Score (catching fraud is critical)

Returns best models and parameters.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any, Optional
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import recall_score, make_scorer

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {type(e).__name__}")

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Hyperparameter tuning for fraud detection models."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize tuner.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs for parallel processing (-1 = all)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tuning_results = {}
        self.best_models = {}
        self.best_params = {}
    
    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        verbose: bool = True
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Tune Random Forest using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_model, best_params)
        """
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING: RANDOM FOREST")
        print("="*70)
        
        if verbose:
            print(f"\n🔍 Setting up GridSearchCV...")
            print(f"   Data shape: {X_train.shape}")
            print(f"   Cross-validation: {cv} folds")
            print(f"   Optimization metric: Recall Score")
        
        # Define parameter grid (optimized for faster execution)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        if verbose:
            print(f"\n📋 Parameter Grid:")
            print(f"   n_estimators: {param_grid['n_estimators']}")
            print(f"   max_depth: {param_grid['max_depth']}")
            print(f"   min_samples_split: {param_grid['min_samples_split']}")
            print(f"   min_samples_leaf: {param_grid['min_samples_leaf']}")
            print(f"   max_features: {param_grid['max_features']}")
            print(f"   class_weight: {param_grid['class_weight']}")
            print(f"\n   Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Create base model
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        
        # Create GridSearchCV
        recall_scorer = make_scorer(recall_score, zero_division=0)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv,
            scoring=recall_scorer,
            n_jobs=self.n_jobs,
            verbose=1 if verbose else 0
        )
        
        if verbose:
            print(f"\n🚀 Training GridSearchCV...")
            start_time = time.time()
        
        # Fit
        grid_search.fit(X_train, y_train)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"✓ GridSearchCV completed in {elapsed:.2f} seconds")
        
        # Extract best results
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Store results
        self.best_models['Random Forest'] = best_model
        self.best_params['Random Forest'] = best_params
        self.tuning_results['Random Forest'] = {
            'best_score': best_score,
            'cv_results': grid_search.cv_results_,
            'best_index': grid_search.best_index_
        }
        
        if verbose:
            print(f"\n🏆 Best Recall Score: {best_score:.4f}")
            print(f"\n📊 Best Parameters:")
            for param, value in best_params.items():
                print(f"   {param}: {value}")
        
        return best_model, best_params
    
    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        n_iter: int = 30,
        verbose: bool = True
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Tune XGBoost using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings to sample
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_model, best_params) or (None, None) if XGBoost unavailable
        """
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost not available. Skipping XGBoost tuning.")
            return None, None
        
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING: XGBOOST")
        print("="*70)
        
        if verbose:
            print(f"\n🔍 Setting up RandomizedSearchCV...")
            print(f"   Data shape: {X_train.shape}")
            print(f"   Cross-validation: {cv} folds")
            print(f"   Number of iterations: {n_iter}")
            print(f"   Optimization metric: Recall Score")
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5],
            'gamma': [0, 0.1, 0.5, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0]
        }
        
        if verbose:
            print(f"\n📋 Parameter Distribution:")
            for param, values in param_dist.items():
                print(f"   {param}: {values}")
            print(f"\n   Total possible combinations: ~10,000+")
            print(f"   Will sample: {n_iter} combinations")
        
        # Create base model
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='logloss',
            scale_pos_weight=775  # Approximate class imbalance
        )
        
        # Create RandomizedSearchCV
        recall_scorer = make_scorer(recall_score, zero_division=0)
        random_search = RandomizedSearchCV(
            xgb_model,
            param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=recall_scorer,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1 if verbose else 0
        )
        
        if verbose:
            print(f"\n🚀 Training RandomizedSearchCV...")
            start_time = time.time()
        
        # Fit
        random_search.fit(X_train, y_train)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"✓ RandomizedSearchCV completed in {elapsed:.2f} seconds")
        
        # Extract best results
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        # Store results
        self.best_models['XGBoost'] = best_model
        self.best_params['XGBoost'] = best_params
        self.tuning_results['XGBoost'] = {
            'best_score': best_score,
            'cv_results': random_search.cv_results_,
            'best_index': random_search.best_index_
        }
        
        if verbose:
            print(f"\n🏆 Best Recall Score: {best_score:.4f}")
            print(f"\n📊 Best Parameters:")
            for param, value in best_params.items():
                print(f"   {param}: {value}")
        
        return best_model, best_params
    
    def compare_tuned_models(self, verbose: bool = True) -> pd.DataFrame:
        """
        Compare all tuned models.
        
        Args:
            verbose: Whether to print comparison
            
        Returns:
            DataFrame with tuning results
        """
        if not self.tuning_results:
            print("No tuned models to compare. Run tuning methods first.")
            return None
        
        comparison_data = []
        
        for model_name, results in self.tuning_results.items():
            comparison_data.append({
                'Model': model_name,
                'Best Recall Score': results['best_score'],
                'Parameters Count': len(self.best_params[model_name])
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Best Recall Score', ascending=False)
        
        if verbose:
            print("\n" + "="*70)
            print("TUNING RESULTS COMPARISON")
            print("="*70)
            print("\n" + df.to_string(index=False))
            print("\n")
        
        return df
    
    def get_best_model(self, metric: str = 'recall') -> Tuple[str, Any, Dict[str, Any]]:
        """
        Get the best tuned model.
        
        Args:
            metric: Metric to use for selection (always recall in this case)
            
        Returns:
            Tuple of (model_name, model, best_params)
        """
        if not self.tuning_results:
            raise ValueError("No tuned models available.")
        
        best_model_name = max(
            self.tuning_results.keys(),
            key=lambda x: self.tuning_results[x]['best_score']
        )
        
        best_model = self.best_models[best_model_name]
        best_params = self.best_params[best_model_name]
        
        print(f"\n🏆 Best Tuned Model: {best_model_name}")
        print(f"   Recall Score: {self.tuning_results[best_model_name]['best_score']:.4f}")
        
        return best_model_name, best_model, best_params
    
    def save_best_model(self, model_name: str, filepath: str) -> None:
        """Save a tuned model."""
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not tuned yet.")
        
        joblib.dump(self.best_models[model_name], filepath)
        print(f"✓ Model '{model_name}' saved to {filepath}")
    
    def save_all_tuned_models(self, directory: str = "models") -> None:
        """Save all tuned models."""
        for model_name, model in self.best_models.items():
            filepath = f"{directory}/{model_name.replace(' ', '_')}_tuned.pkl"
            self.save_best_model(model_name, filepath)
    
    def get_parameter_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get parameter importance from tuning results.
        
        Args:
            model_name: Name of the tuned model
            top_n: Number of top parameters to return
            
        Returns:
            DataFrame with parameter importance
        """
        if model_name not in self.tuning_results:
            raise ValueError(f"Model '{model_name}' not tuned yet.")
        
        cv_results = self.tuning_results[model_name]['cv_results']
        
        # Calculate parameter importance by variance in mean_test_score
        params = cv_results['params']
        scores = cv_results['mean_test_score']
        
        # Create DataFrame
        df_results = pd.DataFrame({
            'params': params,
            'score': scores
        })
        
        # Expand params
        for param in self.best_params[model_name].keys():
            df_results[param] = df_results['params'].apply(lambda x: x.get(param, None))
        
        return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - FRAUD DETECTION MODELS")
    print("="*70)
    
    # Load and prepare data
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    
    print("\n📂 Loading and preparing data...")
    df = pd.read_csv('Fraud.csv')
    
    # Sample for demo (smaller sample for faster tuning)
    df, _ = train_test_split(
        df, train_size=50000, random_state=42, stratify=df['isFraud']
    )
    
    # Prepare features
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop = string_cols + ['isFlaggedFraud']
    X = df.drop(columns=['isFraud'] + cols_to_drop)
    y = df['isFraud']
    
    # Split data
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
    
    print(f"✓ Data prepared:")
    print(f"   Training: {X_train_balanced.shape}")
    print(f"   Test: {X_test_scaled.shape}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(random_state=42, n_jobs=-1)
    
    # Tune Random Forest
    print("\n" + "="*70)
    rf_model, rf_params = tuner.tune_random_forest(
        X_train_balanced, y_train_balanced, cv=5, verbose=True
    )
    
    # Tune XGBoost
    print("\n" + "="*70)
    xgb_model, xgb_params = tuner.tune_xgboost(
        X_train_balanced, y_train_balanced, cv=5, n_iter=20, verbose=True
    )
    
    # Compare results
    print("\n" + "="*70)
    comparison = tuner.compare_tuned_models(verbose=True)
    
    # Get best model
    print("\n" + "="*70)
    best_name, best_model, best_params = tuner.get_best_model()
    
    # Evaluate on test set
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE - BEST MODEL")
    print("="*70)
    
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n📊 {best_name} Metrics on Test Set:")
    print(f"  Recall:    {recall_score(y_test.values, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test.values, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_test.values, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test.values, y_pred_proba):.4f}")
    
    # Save models
    print("\n💾 Saving tuned models...")
    tuner.save_all_tuned_models(directory="models")
    
    print("\n" + "="*70)
    print("✨ HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    print("\n📁 Saved tuned models:")
    print("   - models/Random_Forest_tuned.pkl")
    if XGBOOST_AVAILABLE:
        print("   - models/XGBoost_tuned.pkl")
