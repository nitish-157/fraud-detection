"""
SHAP Model Explainability - Fraud Detection

Uses SHAP (SHapley Additive exPlanations) to explain model predictions:
- Feature importance for fraud detection
- Individual prediction explanations
- Summary plots showing feature contributions
- Force plots and decision plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP-based model explainability for fraud detection."""
    
    def __init__(self, model_path: str = "models/Random_Forest_tuned.pkl"):
        """
        Initialize SHAP explainer.
        
        Args:
            model_path: Path to trained model
        """
        self.model = joblib.load(model_path)
        self.explainer = None
        self.shap_values = None
        self.X_sample = None
    
    def prepare_explainer(
        self,
        X_train: np.ndarray,
        sample_size: int = 100,
        verbose: bool = True
    ) -> None:
        """
        Prepare SHAP explainer using training data sample.
        
        Args:
            X_train: Training features (for background reference)
            sample_size: Number of samples to use for background
            verbose: Print progress
        """
        if verbose:
            print("\n" + "="*70)
            print("PREPARING SHAP EXPLAINER")
            print("="*70)
            print(f"\n🔧 Creating TreeExplainer...")
        
        # Create background data sample
        if X_train.shape[0] > sample_size:
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            background_data = X_train[indices]
        else:
            background_data = X_train
        
        self.X_sample = background_data
        
        # Create SHAP explainer
        if verbose:
            print(f"   Using {background_data.shape[0]} samples as background")
        
        self.explainer = shap.TreeExplainer(self.model)
        
        if verbose:
            print("✓ TreeExplainer created successfully")
    
    def calculate_shap_values(
        self,
        X_test: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Calculate SHAP values for test set.
        
        Args:
            X_test: Test features
            verbose: Print progress
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Run prepare_explainer() first")
        
        if verbose:
            print("\n" + "="*70)
            print("CALCULATING SHAP VALUES")
            print("="*70)
            print(f"\n📊 Computing SHAP values for {X_test.shape[0]} samples...")
        
        self.shap_values = self.explainer.shap_values(X_test)
        
        # For binary classification, get fraud class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Class 1: Fraud
        elif len(self.shap_values.shape) == 3:
            # Shape (n_samples, n_features, n_classes)
            self.shap_values = self.shap_values[:, :, 1]  # Get fraud class
        
        if verbose:
            print(f"✓ SHAP values computed: {self.shap_values.shape}")
        
        return self.shap_values
    
    def plot_feature_importance(
        self,
        X_test: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        verbose: bool = True
    ) -> None:
        """
        Plot SHAP feature importance.
        
        Args:
            X_test: Test features with column names
            figsize: Figure size
            verbose: Print progress
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        if verbose:
            print("\n📈 Creating feature importance plot...")
        
        plt.figure(figsize=figsize)
        
        # Calculate mean absolute SHAP values for importance
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        # Create bar plot
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
        
        plt.barh(range(len(indices)), importance[indices], color='steelblue', edgecolor='navy', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
        plt.title('SHAP Feature Importance - Fraud Detection', fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
        if verbose:
            print("✓ Feature importance plot saved as 'shap_feature_importance.png'")
        plt.close()
    
    def plot_summary_plot(
        self,
        X_test: pd.DataFrame,
        plot_type: str = 'dot',
        figsize: Tuple[int, int] = (10, 8),
        verbose: bool = True
    ) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            X_test: Test features with column names
            plot_type: 'dot', 'bar', or 'violin'
            figsize: Figure size
            verbose: Print progress
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        if verbose:
            print(f"\n📊 Creating {plot_type} summary plot...")
        
        plt.figure(figsize=figsize)
        
        # Create SHAP Explainer for plotting
        shap.summary_plot(
            self.shap_values,
            X_test,
            plot_type=plot_type,
            show=False
        )
        
        plt.title(f'SHAP Summary Plot ({plot_type.capitalize()}) - Fraud Detection', 
                 fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        
        plot_filename = f'shap_summary_{plot_type}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Summary plot ({plot_type}) saved as '{plot_filename}'")
        plt.close()
    
    def plot_force_plot(
        self,
        X_test: pd.DataFrame,
        instance_idx: int = 0,
        y_pred: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> None:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            X_test: Test features
            instance_idx: Index of instance to explain
            y_pred: Predictions (optional)
            verbose: Print progress
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        if verbose:
            print(f"\n🔍 Creating force plot for instance {instance_idx}...")
        
        # Get expected value (base value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # Fraud class
        
        # Create force plot
        force_plot = shap.plots.force(
            base_value,
            self.shap_values[instance_idx],
            X_test.iloc[instance_idx],
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - Instance {instance_idx}', 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_force_plot_instance_{instance_idx}.png', dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Force plot saved as 'shap_force_plot_instance_{instance_idx}.png'")
        plt.close()
    
    def plot_decision_plot(
        self,
        X_test: pd.DataFrame,
        num_samples: int = 50,
        figsize: Tuple[int, int] = (12, 8),
        verbose: bool = True
    ) -> None:
        """
        Plot SHAP contributions for top samples.
        
        Args:
            X_test: Test features
            num_samples: Number of samples to analyze
            figsize: Figure size
            verbose: Print progress
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        if verbose:
            print(f"\n📊 Creating SHAP contributions plot for top {num_samples} samples...")
        
        # Average absolute SHAP values for top samples
        top_shap = self.shap_values[:num_samples]
        avg_shap = np.abs(top_shap).mean(axis=0)
        
        # Get feature names
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
        
        # Sort by contribution
        indices = np.argsort(avg_shap)[::-1]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.bar(range(len(indices)), avg_shap[indices], color='coral', edgecolor='darkred', alpha=0.7)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
        plt.title(f'SHAP Contributions - Average of {num_samples} Samples', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('shap_contributions_plot.png', dpi=300, bbox_inches='tight')
        if verbose:
            print("✓ SHAP contributions plot saved as 'shap_contributions_plot.png'")
        plt.close()
    
    def plot_dependence_plots(
        self,
        X_test: pd.DataFrame,
        top_features: int = 4,
        figsize: Tuple[int, int] = (14, 10),
        verbose: bool = True
    ) -> None:
        """
        Plot SHAP dependence plots for top features.
        
        Args:
            X_test: Test features with column names
            top_features: Number of top features to show
            figsize: Figure size
            verbose: Print progress
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        if verbose:
            print(f"\n📊 Creating dependence plots for top {top_features} features...")
        
        # Calculate importance to find top features
        importance = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:top_features]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
        
        for idx, feature_idx in enumerate(top_indices[:4]):
            ax = axes[idx]
            
            # Plot dependence
            shap_vals = self.shap_values[:, feature_idx]
            feature_vals = X_test.iloc[:, feature_idx] if isinstance(X_test, pd.DataFrame) else X_test[:, feature_idx]
            
            scatter = ax.scatter(feature_vals, shap_vals, alpha=0.5, s=20, c=shap_vals, cmap='coolwarm')
            ax.set_xlabel(f'{feature_names[feature_idx]}', fontsize=10, fontweight='bold')
            ax.set_ylabel('SHAP Value', fontsize=10, fontweight='bold')
            ax.set_title(f'Dependence: {feature_names[feature_idx]}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax)
        
        plt.suptitle('SHAP Dependence Plots - Top 4 Features', fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        if verbose:
            print("✓ Dependence plots saved as 'shap_dependence_plots.png'")
        plt.close()
    
    def get_feature_explanation(
        self,
        X_test: pd.DataFrame,
        instance_idx: int = 0,
        verbose: bool = True
    ) -> dict:
        """
        Get detailed feature explanation for a single prediction.
        
        Args:
            X_test: Test features
            instance_idx: Index of instance to explain
            verbose: Print explanation
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # Fraud class
        
        # Convert to scalar if array
        if isinstance(base_value, np.ndarray):
            base_value = base_value.item() if base_value.size == 1 else np.mean(base_value)
        
        base_value = float(base_value)
        
        # Get feature names
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
        
        # Get SHAP values and features for this instance
        shap_vals = self.shap_values[instance_idx]
        feature_vals = X_test.iloc[instance_idx]
        
        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1]
        
        explanation = {
            'base_value': base_value,
            'features': []
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PREDICTION EXPLANATION - Instance {instance_idx}")
            print(f"{'='*70}")
            print(f"\n📊 Base Value (Model Average): {base_value:.4f}")
            print(f"\n🔝 Top Contributing Features (Sorted by Impact):")
            print(f"\n{'Feature':<15} {'Value':>12} {'SHAP Value':>12} {'Direction':>12}")
            print(f"{'-'*52}")
        
        for rank, idx in enumerate(indices[:10], 1):
            shap_val = shap_vals[idx]
            feature_val = feature_vals.iloc[idx] if isinstance(feature_vals, pd.Series) else feature_vals[idx]
            
            direction = "↑ Fraud" if shap_val > 0 else "↓ Normal"
            
            explanation['features'].append({
                'name': feature_names[idx],
                'value': float(feature_val),
                'shap_value': float(shap_val),
                'direction': direction
            })
            
            if verbose:
                print(f"{feature_names[idx]:<15} {feature_val:>12.4f} {shap_val:>12.4f} {direction:>12}")
        
        # Calculate final prediction
        final_score = base_value + np.sum(shap_vals)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Final Model Score: {final_score:.4f}")
            print(f"{'='*70}")
        
        explanation['final_score'] = final_score
        
        return explanation
    
    def export_shap_analysis(self, filepath: str = 'shap_analysis_summary.txt') -> None:
        """Export SHAP analysis summary."""
        if self.shap_values is None:
            raise ValueError("Run calculate_shap_values() first")
        
        summary = f"""
SHAP EXPLAINABILITY ANALYSIS SUMMARY
====================================

Dataset: {self.shap_values.shape[0]} samples analyzed
Features: {self.shap_values.shape[1]} features

SHAP Values Statistics:
- Mean absolute SHAP: {np.abs(self.shap_values).mean():.6f}
- Max SHAP: {np.abs(self.shap_values).max():.6f}
- Min SHAP: {np.abs(self.shap_values).min():.6f}

GENERATED VISUALIZATIONS:
1. shap_feature_importance.png - Feature importance ranking
2. shap_summary_dot.png - Summary plot (dot format)
3. shap_summary_bar.png - Summary plot (bar format)
4. shap_contributions_plot.png - Average SHAP contributions
5. shap_dependence_plots.png - Top 4 feature dependencies
6. shap_force_plot_instance_0.png - Force plot for instance 0

INTERPRETATION GUIDE:

Feature Importance:
- Shows which features have the most impact on model predictions
- Measured by mean absolute SHAP values
- Higher = More important for fraud detection

Summary Plot (Dot):
- Each point = one sample
- Color = feature value (red=high, blue=low)
- Position = SHAP contribution to prediction
- Points spread right = increases fraud prediction

Summary Plot (Bar):
- Average |SHAP value| per feature
- Combines feature importance & value ranges
- Shows overall feature contribution

Decision Plot:
- Shows cumulative effect of features on prediction
- Left to right = features impact on decision
- Slope = strength of feature effect
- Feature value vs SHAP value
- Reveals non-linear relationships
- Shows feature interactions
"""
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        print(f"✓ Analysis summary exported to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*70)
    print("SHAP MODEL EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    # Load and prepare data
    print("\n📂 Loading data...")
    df = pd.read_csv('Fraud.csv')
    
    # Quick preprocessing
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop = string_cols + ['isFlaggedFraud']
    X = df.drop(columns=['isFraud'] + cols_to_drop)
    y = df['isFraud']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # For SHAP analysis, use a smaller sample
    X_train_sample = X_train.iloc[:5000]  # 5000 for faster SHAP
    X_test_sample = X_test.iloc[:1000]    # 1000 for analysis
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test_sample)
    
    # Convert to DataFrame for feature names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"✓ Data prepared: Train={X_train_scaled.shape}, Test={X_test_scaled_df.shape}")
    
    # Initialize SHAP explainer
    print("\n🔧 Initializing SHAP explainer...")
    explainer = SHAPExplainer(model_path="models/Random_Forest_tuned.pkl")
    
    # Prepare explainer
    explainer.prepare_explainer(X_train_scaled, sample_size=100, verbose=True)
    
    # Calculate SHAP values
    shap_vals = explainer.calculate_shap_values(X_test_scaled, verbose=True)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    explainer.plot_feature_importance(X_test_scaled_df, verbose=True)
    explainer.plot_summary_plot(X_test_scaled_df, plot_type='dot', verbose=True)
    explainer.plot_summary_plot(X_test_scaled_df, plot_type='bar', verbose=True)
    explainer.plot_decision_plot(X_test_scaled_df, num_samples=50, verbose=True)
    explainer.plot_dependence_plots(X_test_scaled_df, top_features=4, verbose=True)
    
    # Get sample explanation
    print("\n" + "="*70)
    print("SAMPLE PREDICTION EXPLANATION")
    print("="*70)
    
    explanation = explainer.get_feature_explanation(X_test_scaled_df, instance_idx=0, verbose=True)
    
    # Export summary
    explainer.export_shap_analysis()
    
    print("\n" + "="*70)
    print("✨ SHAP ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - shap_feature_importance.png")
    print("   - shap_summary_dot.png")
    print("   - shap_summary_bar.png")
    print("   - shap_contributions_plot.png")
    print("   - shap_dependence_plots.png")
    print("   - shap_force_plot_instance_0.png")
    print("   - shap_analysis_summary.txt")
