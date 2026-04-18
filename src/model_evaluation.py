"""
Model Evaluation Module for Fraud Detection

Comprehensive evaluation with:
- Precision, Recall, F1-score, ROC-AUC
- Confusion Matrix
- Model Comparison
- ROC Curve Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional
import joblib
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve
)

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_results = {}
        self.models_data = {}
    
    def evaluate_single_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        # Store for later use
        self.evaluation_results[model_name] = metrics
        self.models_data[model_name] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Print results
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics in formatted way."""
        print(f"\n📊 METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        print(f"  [[TN: {cm[0, 0]:,}  | FP: {cm[0, 1]:,}]")
        print(f"   [FN: {cm[1, 0]:,}  | TP: {cm[1, 1]:,}]]")
        
        print(f"\n📈 CLASSIFICATION REPORT:")
        print(metrics['classification_report'])
    
    def evaluate_multiple_models(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.
        
        Args:
            models_dict: Dictionary of {model_name: model_object}
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation for all models
        """
        print("\n" + "="*70)
        print("EVALUATING MULTIPLE MODELS")
        print("="*70)
        
        for model_name, model in models_dict.items():
            self.evaluate_single_model(model_name, model, X_test, y_test, verbose)
        
        return self.evaluation_results
    
    def compare_models(
        self,
        metric: str = 'roc_auc',
        display: bool = True
    ) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Args:
            metric: Metric to sort by (accuracy, precision, recall, f1, roc_auc)
            display: Whether to print comparison table
            
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet. Run evaluate_multiple_models() first.")
        
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        df = pd.DataFrame(comparison_data)
        df_sorted = df.sort_values('ROC-AUC', ascending=False)
        
        if display:
            print("\n" + "="*70)
            print("MODEL COMPARISON")
            print("="*70)
            print("\n" + df_sorted.to_string(index=False))
            print("\n")
        
        return df_sorted
    
    def plot_confusion_matrices(
        self,
        figsize: Tuple[int, int] = (16, 10)
    ) -> None:
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            figsize: Figure size
        """
        n_models = len(self.evaluation_results)
        
        if n_models == 0:
            print("No models to plot. Run evaluate_multiple_models() first.")
            return
        
        # Calculate grid
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            cm = metrics['confusion_matrix']
            
            # Plot confusion matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                ax=axes[idx],
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud']
            )
            
            axes[idx].set_title(f'{model_name}\n(ROC-AUC: {metrics["roc_auc"]:.4f})',
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(self.evaluation_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Confusion matrices saved as 'confusion_matrices.png'")
    
    def plot_roc_curves(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True
    ) -> None:
        """
        Plot ROC curves for all evaluated models.
        
        Args:
            figsize: Figure size
            save: Whether to save the plot
        """
        if not self.evaluation_results:
            print("No models to plot. Run evaluate_multiple_models() first.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each model
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
            roc_auc = metrics['roc_auc']
            
            # Plot
            color = colors[idx % len(colors)]
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})'
            )
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
        
        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
            print("✓ ROC curves saved as 'roc_curves.png'")
        
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True
    ) -> None:
        """
        Plot Precision-Recall curves for all evaluated models.
        
        Args:
            figsize: Figure size
            save: Whether to save the plot
        """
        if not self.evaluation_results:
            print("No models to plot. Run evaluate_multiple_models() first.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            # Calculate P-R curve
            precision, recall, _ = precision_recall_curve(
                metrics['y_test'],
                metrics['y_pred_proba']
            )
            
            # Calculate AP (average precision)
            ap = np.mean(precision)
            
            # Plot
            color = colors[idx % len(colors)]
            ax.plot(
                recall, precision,
                color=color,
                lw=2.5,
                label=f'{model_name} (AP = {ap:.4f})'
            )
        
        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight')
            print("✓ Precision-Recall curves saved as 'precision_recall_curves.png'")
        
        plt.show()
    
    def plot_metrics_comparison(
        self,
        figsize: Tuple[int, int] = (14, 8),
        save: bool = True
    ) -> None:
        """
        Plot all metrics as bar charts for comparison.
        
        Args:
            figsize: Figure size
            save: Whether to save the plot
        """
        if not self.evaluation_results:
            print("No models to plot. Run evaluate_multiple_models() first.")
            return
        
        # Prepare data
        comparison_df = self.compare_models(display=False)
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(comparison_df)]
        
        for idx, metric in enumerate(metrics_cols):
            ax = axes[idx]
            
            # Plot bar chart
            bars = ax.bar(
                range(len(comparison_df)),
                comparison_df[metric],
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold'
                )
            
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
        
        # Hide last subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
            print("✓ Metrics comparison saved as 'metrics_comparison.png'")
        
        plt.show()
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, float]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to evaluate (roc_auc, f1, accuracy, etc.)
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet.")
        
        metric_map = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        metric_key = metric_map.get(metric, 'roc_auc')
        
        best_model = max(
            self.evaluation_results.items(),
            key=lambda x: x[1][metric_key]
        )
        
        print(f"\n🏆 Best Model ({metric.upper()}): {best_model[0]}")
        print(f"   Score: {best_model[1][metric_key]:.4f}")
        
        return best_model[0], best_model[1][metric_key]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Load trained models
    print("\n📂 Loading trained models...")
    models = {
        'Logistic Regression': joblib.load('models/Logistic_Regression.pkl'),
        'Random Forest': joblib.load('models/Random_Forest.pkl')
    }
    print(f"✓ {len(models)} models loaded")
    
    # Load test data (for demo, we'll load and preprocess Fraud.csv)
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    
    print("\n📂 Loading and preparing test data...")
    df = pd.read_csv('Fraud.csv')
    
    # Sample for demo
    df, _ = train_test_split(
        df, train_size=100000, random_state=42, stratify=df['isFraud']
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Test data prepared: {X_test_scaled.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    print("\n" + "="*70)
    results = evaluator.evaluate_multiple_models(
        models, X_test_scaled, y_test.values
    )
    
    # Compare models
    print("\n")
    comparison = evaluator.compare_models(metric='roc_auc', display=True)
    
    # Get best model
    print("\n")
    best_model_name, best_score = evaluator.get_best_model(metric='roc_auc')
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n📊 Plotting Confusion Matrices...")
    evaluator.plot_confusion_matrices()
    
    print("\n📊 Plotting ROC Curves...")
    evaluator.plot_roc_curves()
    
    print("\n📊 Plotting Precision-Recall Curves...")
    evaluator.plot_precision_recall_curves()
    
    print("\n📊 Plotting Metrics Comparison...")
    evaluator.plot_metrics_comparison()
    
    print("\n" + "="*70)
    print("✨ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - confusion_matrices.png")
    print("   - roc_curves.png")
    print("   - precision_recall_curves.png")
    print("   - metrics_comparison.png")
