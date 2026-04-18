"""
Threshold Optimization for Fraud Detection

Analyzes how precision and recall change with different classification thresholds.
Finds optimal threshold that maximizes fraud detection while minimizing false alarms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')


class ThresholdOptimizer:
    """Optimize classification threshold for fraud detection."""
    
    def __init__(self, model_path: str = "models/Random_Forest_tuned.pkl"):
        """
        Initialize optimizer.
        
        Args:
            model_path: Path to trained model
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.thresholds = np.linspace(0.0, 1.0, 101)  # 0.0 to 1.0 in 0.01 steps
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.cm_data = {}
    
    def analyze_thresholds(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Calculate metrics for different thresholds.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print progress
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if verbose:
            print("\n" + "="*70)
            print("THRESHOLD OPTIMIZATION")
            print("="*70)
            print(f"\n📊 Analyzing {len(self.thresholds)} thresholds...")
        
        # Get predicted probabilities for fraud class
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store for later use
        self.y_test = y_test
        self.y_pred_proba = y_pred_proba
        
        # Calculate metrics for each threshold
        results = []
        
        for threshold in self.thresholds:
            # Make predictions with custom threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate
                'fraud_predicted': tp + fp  # Total fraud predictions
            })
            
            self.precision_scores.append(precision)
            self.recall_scores.append(recall)
            self.f1_scores.append(f1)
        
        self.results_df = pd.DataFrame(results)
        
        if verbose:
            print(f"✓ Analysis complete: {len(results)} thresholds evaluated")
        
        return self.results_df
    
    def find_optimal_threshold(
        self,
        strategy: str = 'balanced',
        recall_weight: float = 0.7,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Find optimal threshold using different strategies.
        
        Args:
            strategy: 'balanced', 'recall_priority', 'precision_priority', 'f1', 'custom'
            recall_weight: For custom strategy (0-1), weight for recall vs precision
            verbose: Print results
            
        Returns:
            Dict with optimal threshold and its metrics
        """
        if self.results_df is None:
            raise ValueError("Run analyze_thresholds() first")
        
        if verbose:
            print(f"\n" + "="*70)
            print(f"FINDING OPTIMAL THRESHOLD ({strategy.upper()})")
            print("="*70)
        
        # Strategy selection
        if strategy == 'f1':
            # Maximize F1-score
            idx = self.results_df['f1'].idxmax()
            threshold = self.results_df.loc[idx, 'threshold']
            reason = "Maximizes F1-score (harmonic mean of precision and recall)"
            
        elif strategy == 'recall_priority':
            # Focus on recall >= 0.7, then maximize precision
            valid = self.results_df[self.results_df['recall'] >= 0.7]
            if len(valid) > 0:
                idx = valid['precision'].idxmax()
                threshold = self.results_df.loc[idx, 'threshold']
                reason = "Recall >= 0.7 with highest precision"
            else:
                idx = self.results_df['recall'].idxmax()
                threshold = self.results_df.loc[idx, 'threshold']
                reason = "Maximizes recall"
        
        elif strategy == 'precision_priority':
            # Focus on precision >= 0.3, then maximize recall
            valid = self.results_df[self.results_df['precision'] >= 0.3]
            if len(valid) > 0:
                idx = valid['recall'].idxmax()
                threshold = self.results_df.loc[idx, 'threshold']
                reason = "Precision >= 0.3 with highest recall"
            else:
                idx = self.results_df['precision'].idxmax()
                threshold = self.results_df.loc[idx, 'threshold']
                reason = "Maximizes precision"
        
        elif strategy == 'balanced':
            # Balance precision and recall (0.6 recall + 0.4 precision)
            self.results_df['balanced_score'] = (
                0.6 * self.results_df['recall'] + 
                0.4 * self.results_df['precision']
            )
            idx = self.results_df['balanced_score'].idxmax()
            threshold = self.results_df.loc[idx, 'threshold']
            reason = "Balanced: 60% recall weight + 40% precision weight"
        
        elif strategy == 'custom':
            # Custom weighted combination
            self.results_df['custom_score'] = (
                recall_weight * self.results_df['recall'] +
                (1 - recall_weight) * self.results_df['precision']
            )
            idx = self.results_df['custom_score'].idxmax()
            threshold = self.results_df.loc[idx, 'threshold']
            reason = f"Custom: {recall_weight:.1%} recall + {(1-recall_weight):.1%} precision"
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Get metrics at optimal threshold
        optimal_metrics = self.results_df.loc[idx]
        
        result = {
            'threshold': threshold,
            'precision': optimal_metrics['precision'],
            'recall': optimal_metrics['recall'],
            'f1': optimal_metrics['f1'],
            'specificity': optimal_metrics['specificity'],
            'tp': int(optimal_metrics['tp']),
            'fp': int(optimal_metrics['fp']),
            'fn': int(optimal_metrics['fn']),
            'tn': int(optimal_metrics['tn']),
            'strategy': strategy,
            'reason': reason
        }
        
        if verbose:
            print(f"\n Optimal Threshold: {threshold:.2f}")
            print(f"\n Rationale:")
            print(f"   {reason}")
            print(f"\n Metrics at Optimal Threshold:")
            print(f"   Recall:      {result['recall']:.4f} (Catch fraud)")
            print(f"   Precision:   {result['precision']:.4f}   (Minimize false alarms)")
            print(f"   F1-Score:    {result['f1']:.4f}")
            print(f"   Specificity: {result['specificity']:.4f} (True Negative Rate)")
            print(f"\n Confusion Matrix at Threshold {threshold:.2f}:")
            print(f"   True Positives:  {result['tp']:,}   (Correctly caught frauds)")
            print(f"   False Positives: {result['fp']:,}   (False alarms)")
            print(f"   False Negatives: {result['fn']:,}   (Missed frauds)")
            print(f"   True Negatives:  {result['tn']:,}   (Correct normal transactions)")
        
        self.optimal_threshold = threshold
        self.optimal_result = result
        
        return result
    
    def plot_precision_recall(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Plot precision vs recall curve."""
        if self.results_df is None:
            raise ValueError("Run analyze_thresholds() first")
        
        plt.figure(figsize=figsize)
        
        # Plot PR curve with color gradient for threshold
        scatter = plt.scatter(
            self.results_df['recall'],
            self.results_df['precision'],
            c=self.results_df['threshold'],
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar to show threshold values
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classification Threshold', fontsize=11, fontweight='bold')
        
        # Mark optimal threshold
        if hasattr(self, 'optimal_threshold'):
            optimal = self.results_df[self.results_df['threshold'] == self.optimal_threshold].iloc[0]
            plt.scatter(optimal['recall'], optimal['precision'], 
                       s=500, marker='*', color='red', edgecolors='darkred', 
                       linewidth=2, label=f"Optimal ({self.optimal_threshold:.2f})", zorder=5)
        
        # Mark default threshold (0.5)
        default = self.results_df[self.results_df['threshold'] == 0.5].iloc[0]
        plt.scatter(default['recall'], default['precision'],
                   s=300, marker='s', color='orange', edgecolors='darkorange',
                   linewidth=2, label=f"Default (0.50)", zorder=4)
        
        plt.xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision vs Recall for Different Classification Thresholds', 
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('precision_recall_threshold.png', dpi=300, bbox_inches='tight')
        print("✓ Precision-Recall curve saved as 'precision_recall_threshold.png'")
        plt.close()
    
    def plot_metrics_vs_threshold(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot all metrics vs threshold."""
        if self.results_df is None:
            raise ValueError("Run analyze_thresholds() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metrics
        ax.plot(self.results_df['threshold'], self.results_df['precision'],
               label='Precision', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax.plot(self.results_df['threshold'], self.results_df['recall'],
               label='Recall', linewidth=2, marker='s', markersize=4, alpha=0.7)
        ax.plot(self.results_df['threshold'], self.results_df['f1'],
               label='F1-Score', linewidth=2, marker='^', markersize=4, alpha=0.7)
        ax.plot(self.results_df['threshold'], self.results_df['specificity'],
               label='Specificity', linewidth=2, marker='d', markersize=4, alpha=0.7)
        
        # Mark default threshold
        ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, 
                  label='Default (0.50)', alpha=0.7)
        
        # Mark optimal threshold
        if hasattr(self, 'optimal_threshold'):
            ax.axvline(x=self.optimal_threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Optimal ({self.optimal_threshold:.2f})', alpha=0.7)
        
        ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Classification Metrics vs Threshold', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('metrics_vs_threshold.png', dpi=300, bbox_inches='tight')
        print("✓ Metrics vs threshold saved as 'metrics_vs_threshold.png'")
        plt.close()
    
    def plot_confusion_matrix_comparison(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        """Compare confusion matrices at different thresholds."""
        if not hasattr(self, 'optimal_threshold'):
            raise ValueError("Run find_optimal_threshold() first")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        thresholds_to_plot = [0.3, 0.5, self.optimal_threshold]
        titles = [
            f'Threshold: 0.30\n(High Recall)',
            f'Threshold: 0.50\n(Default)',
            f'Threshold: {self.optimal_threshold:.2f}\n(Optimal)'
        ]
        
        for ax, threshold, title in zip(axes, thresholds_to_plot, titles):
            # Get predictions at threshold
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, annot_kws={'size': 11, 'weight': 'bold'})
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_xticklabels(['Normal', 'Fraud'])
            ax.set_yticklabels(['Normal', 'Fraud'])
            
            # Add metrics below
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            
            metrics_text = f"Precision: {precision:.3f}\nRecall: {recall:.3f}"
            ax.text(0.5, -0.3, metrics_text, ha='center', transform=ax.transAxes,
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Confusion Matrices at Different Thresholds', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('confusion_matrices_threshold_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix comparison saved as 'confusion_matrices_threshold_comparison.png'")
        plt.close()
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Plot ROC curve with threshold annotations."""
        from sklearn.metrics import roc_curve, auc
        
        if not hasattr(self, 'y_pred_proba'):
            raise ValueError("Run analyze_thresholds() first")
        
        # Calculate ROC curve
        fpr, tpr, thresholds_roc = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})',
                linewidth=2.5, color='blue')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
        
        # Mark some thresholds on the curve
        thresholds_to_mark = [0.1, 0.3, 0.5, 0.7, 0.9]
        for t in thresholds_to_mark:
            idx = np.argmin(np.abs(thresholds_roc - t))
            if idx < len(fpr):
                plt.scatter(fpr[idx], tpr[idx], s=100, zorder=5)
                plt.annotate(f'{t:.1f}', (fpr[idx], tpr[idx]), 
                           textcoords="offset points", xytext=(5,5), fontsize=9)
        
        # Mark optimal threshold
        if hasattr(self, 'optimal_threshold'):
            idx = np.argmin(np.abs(thresholds_roc - self.optimal_threshold))
            if idx < len(fpr):
                plt.scatter(fpr[idx], tpr[idx], s=300, marker='*', 
                          color='red', edgecolors='darkred', linewidth=2,
                          label=f"Optimal ({self.optimal_threshold:.2f})", zorder=6)
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve with Threshold Annotations', fontsize=13, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('roc_curve_threshold_annotations.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved as 'roc_curve_threshold_annotations.png'")
        plt.close()
    
    def get_recommendations(self) -> pd.DataFrame:
        """Get threshold recommendations for different use cases."""
        if self.results_df is None:
            raise ValueError("Run analyze_thresholds() first")
        
        recommendations = []
        
        # 1. Maximum Recall (catch all frauds, tolerate false alarms)
        idx = self.results_df['recall'].idxmax()
        rec1 = self.results_df.loc[idx]
        recommendations.append({
            'Use Case': 'Maximum Fraud Detection',
            'Threshold': rec1['threshold'],
            'Rationale': 'Catch as many frauds as possible',
            'Precision': f"{rec1['precision']:.4f}",
            'Recall': f"{rec1['recall']:.4f}",
            'TP': int(rec1['tp']),
            'FP': int(rec1['fp']),
            'FN': int(rec1['fn'])
        })
        
        # 2. Balanced (good compromise)
        valid = self.results_df[(self.results_df['recall'] >= 0.6) & 
                               (self.results_df['precision'] >= 0.15)]
        if len(valid) > 0:
            idx = valid['f1'].idxmax()
            rec2 = self.results_df.loc[idx]
        else:
            idx = self.results_df['f1'].idxmax()
            rec2 = self.results_df.loc[idx]
        
        recommendations.append({
            'Use Case': 'Balanced Approach',
            'Threshold': rec2['threshold'],
            'Rationale': 'Good balance between catching fraud and false alarms',
            'Precision': f"{rec2['precision']:.4f}",
            'Recall': f"{rec2['recall']:.4f}",
            'TP': int(rec2['tp']),
            'FP': int(rec2['fp']),
            'FN': int(rec2['fn'])
        })
        
        # 3. Minimize False Alarms (high precision)
        valid = self.results_df[self.results_df['precision'] >= 0.5]
        if len(valid) > 0:
            idx = valid['recall'].idxmax()
            rec3 = self.results_df.loc[idx]
        else:
            idx = self.results_df['precision'].idxmax()
            rec3 = self.results_df.loc[idx]
        
        recommendations.append({
            'Use Case': 'Minimize False Alarms',
            'Threshold': rec3['threshold'],
            'Rationale': 'Only flag very likely frauds to reduce manual review',
            'Precision': f"{rec3['precision']:.4f}",
            'Recall': f"{rec3['recall']:.4f}",
            'TP': int(rec3['tp']),
            'FP': int(rec3['fp']),
            'FN': int(rec3['fn'])
        })
        
        return pd.DataFrame(recommendations)
    
    def export_results(self, filepath: str = 'threshold_analysis.csv') -> None:
        """Export complete analysis results."""
        if self.results_df is None:
            raise ValueError("Run analyze_thresholds() first")
        
        self.results_df.to_csv(filepath, index=False)
        print(f"✓ Results exported to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION - FRAUD DETECTION")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('Fraud.csv')
    
    # Quick preprocessing (same as training)
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop = string_cols + ['isFlaggedFraud']
    X = df.drop(columns=['isFraud'] + cols_to_drop)
    y = df['isFraud']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    print(f"✓ Data loaded: {X_test_scaled.shape}")
    
    # Optimize
    print("\n Initializing optimizer...")
    optimizer = ThresholdOptimizer(model_path="models/Random_Forest_tuned.pkl")
    
    print("\n📊 Analyzing thresholds (0.0 to 1.0)...")
    results = optimizer.analyze_thresholds(X_test_scaled, y_test.values, verbose=True)
    
    print("\n" + "="*70)
    print("STRATEGIES FOR FINDING OPTIMAL THRESHOLD")
    print("="*70)
    
    # Try different strategies
    strategies = ['balanced', 'recall_priority', 'precision_priority', 'f1']
    strategy_results = {}
    
    for strategy in strategies:
        print(f"\n{'─'*70}")
        result = optimizer.find_optimal_threshold(strategy=strategy, verbose=True)
        strategy_results[strategy] = result
    
    # Show recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR DIFFERENT USE CASES")
    print("="*70)
    recommendations = optimizer.get_recommendations()
    print("\n" + recommendations.to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n Creating visualizations...")
    optimizer.plot_precision_recall()
    optimizer.plot_metrics_vs_threshold()
    optimizer.plot_roc_curve()
    optimizer.plot_confusion_matrix_comparison()
    
    # Export results
    optimizer.export_results('threshold_analysis_results.csv')
    
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION COMPLETE!")
    print("="*70)
    print("\n Generated files:")
    print("   - precision_recall_threshold.png")
    print("   - metrics_vs_threshold.png")
    print("   - roc_curve_threshold_annotations.png")
    print("   - confusion_matrices_threshold_comparison.png")
    print("   - threshold_analysis_results.csv")
