"""
Model & Scaler Management - Save & Load with Joblib

Utilities for persisting trained models and data preprocessors using joblib.
Includes saving, loading, and versioning capabilities.
"""

import joblib
import os
from pathlib import Path
from typing import Any, Optional, Dict
import json
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class ModelManager:
    """Manage saving and loading of ML models and preprocessors."""
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize model manager.
        
        Args:
            base_dir: Base directory for models (default: current directory)
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Metadata for saved artifacts
        self.metadata = {}
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        compression: int = 3,
        metadata: Optional[Dict] = None,
        verbose: bool = True
    ) -> str:
        """
        Save a trained model using joblib.
        
        Args:
            model: Trained model object
            model_name: Name of model (e.g., 'Random_Forest_tuned')
            compression: Compression level (0-9, default 3)
            metadata: Optional metadata dictionary
            verbose: Print progress
            
        Returns:
            Path to saved model
        """
        if verbose:
            print(f"\n💾 Saving model: {model_name}...")
        
        model_path = self.models_dir / f"{model_name}.pkl"
        
        # Save model
        joblib.dump(model, model_path, compress=compression)
        
        # Save metadata
        model_meta = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'file_size_kb': model_path.stat().st_size / 1024,
        }
        
        if metadata:
            model_meta.update(metadata)
        
        self.metadata[model_name] = model_meta
        
        if verbose:
            print(f"✓ Model saved: {model_path}")
            print(f"  File size: {model_meta['file_size_kb']:.2f} KB")
            print(f"  Type: {model_meta['model_type']}")
        
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
        verbose: bool = True
    ) -> Any:
        """
        Load a saved model using joblib.
        
        Args:
            model_name: Name of model (e.g., 'Random_Forest_tuned')
            verbose: Print progress
            
        Returns:
            Loaded model object
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if verbose:
            print(f"\n📂 Loading model: {model_name}...")
        
        model = joblib.load(model_path)
        
        if verbose:
            print(f"✓ Model loaded: {model_path}")
            print(f"  Type: {type(model).__name__}")
        
        return model
    
    def save_scaler(
        self,
        scaler: StandardScaler,
        scaler_name: str = "scaler",
        compression: int = 3,
        verbose: bool = True
    ) -> str:
        """
        Save a fitted scaler using joblib.
        
        Args:
            scaler: Fitted scaler object
            scaler_name: Name of scaler (default: 'scaler')
            compression: Compression level (0-9, default 3)
            verbose: Print progress
            
        Returns:
            Path to saved scaler
        """
        if verbose:
            print(f"\n💾 Saving scaler: {scaler_name}...")
        
        scaler_path = self.base_dir / f"{scaler_name}.pkl"
        
        # Save scaler
        joblib.dump(scaler, scaler_path, compress=compression)
        
        if verbose:
            print(f"✓ Scaler saved: {scaler_path}")
            print(f"  File size: {scaler_path.stat().st_size / 1024:.2f} KB")
            print(f"  Type: {type(scaler).__name__}")
            print(f"  Features: {len(scaler.mean_)}")
        
        return str(scaler_path)
    
    def load_scaler(
        self,
        scaler_name: str = "scaler",
        verbose: bool = True
    ) -> StandardScaler:
        """
        Load a saved scaler using joblib.
        
        Args:
            scaler_name: Name of scaler (default: 'scaler')
            verbose: Print progress
            
        Returns:
            Loaded scaler object
        """
        scaler_path = self.base_dir / f"{scaler_name}.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        if verbose:
            print(f"\n📂 Loading scaler: {scaler_name}...")
        
        scaler = joblib.load(scaler_path)
        
        if verbose:
            print(f"✓ Scaler loaded: {scaler_path}")
            print(f"  Type: {type(scaler).__name__}")
            print(f"  Features: {len(scaler.mean_)}")
        
        return scaler
    
    def save_all(
        self,
        model: Any,
        scaler: StandardScaler,
        model_name: str = "model",
        scaler_name: str = "scaler",
        metadata: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Save model and scaler together.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            model_name: Name of model
            scaler_name: Name of scaler
            metadata: Optional metadata
            verbose: Print progress
            
        Returns:
            Dictionary with paths to saved files
        """
        if verbose:
            print("\n" + "="*70)
            print("SAVING MODEL & SCALER")
            print("="*70)
        
        model_path = self.save_model(model, model_name, metadata=metadata, verbose=verbose)
        scaler_path = self.save_scaler(scaler, scaler_name, verbose=verbose)
        
        if verbose:
            print("\n✓ All files saved successfully")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path
        }
    
    def load_all(
        self,
        model_name: str = "model",
        scaler_name: str = "scaler",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Load model and scaler together.
        
        Args:
            model_name: Name of model
            scaler_name: Name of scaler
            verbose: Print progress
            
        Returns:
            Dictionary with loaded model and scaler
        """
        if verbose:
            print("\n" + "="*70)
            print("LOADING MODEL & SCALER")
            print("="*70)
        
        model = self.load_model(model_name, verbose=verbose)
        scaler = self.load_scaler(scaler_name, verbose=verbose)
        
        if verbose:
            print("\n✓ All files loaded successfully")
        
        return {
            'model': model,
            'scaler': scaler
        }
    
    def list_models(self, verbose: bool = True) -> list:
        """
        List all saved models.
        
        Args:
            verbose: Print results
            
        Returns:
            List of model names
        """
        models = [f.stem for f in self.models_dir.glob("*.pkl")]
        
        if verbose:
            print(f"\n📂 Available models ({len(models)}):")
            for model_name in sorted(models):
                model_path = self.models_dir / f"{model_name}.pkl"
                size_kb = model_path.stat().st_size / 1024
                print(f"   • {model_name:<30} ({size_kb:>8.2f} KB)")
        
        return sorted(models)
    
    def list_scalers(self, verbose: bool = True) -> list:
        """
        List all saved scalers.
        
        Args:
            verbose: Print results
            
        Returns:
            List of scaler names
        """
        scalers = [f.stem for f in self.base_dir.glob("*scaler*.pkl")]
        
        if verbose:
            print(f"\n📂 Available scalers ({len(scalers)}):")
            for scaler_name in sorted(scalers):
                scaler_path = self.base_dir / f"{scaler_name}.pkl"
                size_kb = scaler_path.stat().st_size / 1024
                print(f"   • {scaler_name:<30} ({size_kb:>8.2f} KB)")
        
        return sorted(scalers)
    
    def delete_model(self, model_name: str, verbose: bool = True) -> None:
        """Delete a saved model."""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if model_path.exists():
            model_path.unlink()
            if verbose:
                print(f"✓ Deleted: {model_path}")
        else:
            print(f"⚠ File not found: {model_path}")
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a saved model."""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        stat = model_path.stat()
        return {
            'name': model_name,
            'path': str(model_path),
            'size_kb': stat.st_size / 1024,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }


# ============================================================================
# SIMPLE ONE-LINER FUNCTIONS
# ============================================================================

def save_model(model: Any, filename: str, compress: int = 3) -> str:
    """
    Quick save for a model.
    
    Args:
        model: Model to save
        filename: Path to save to
        compress: Compression level (0-9)
        
    Returns:
        Path to saved file
    """
    joblib.dump(model, filename, compress=compress)
    print(f"✓ Model saved: {filename}")
    return filename


def load_model(filename: str) -> Any:
    """
    Quick load for a model.
    
    Args:
        filename: Path to model file
        
    Returns:
        Loaded model
    """
    model = joblib.load(filename)
    print(f"✓ Model loaded: {filename}")
    return model


def save_scaler(scaler: StandardScaler, filename: str, compress: int = 3) -> str:
    """
    Quick save for a scaler.
    
    Args:
        scaler: Scaler to save
        filename: Path to save to
        compress: Compression level (0-9)
        
    Returns:
        Path to saved file
    """
    joblib.dump(scaler, filename, compress=compress)
    print(f"✓ Scaler saved: {filename}")
    return filename


def load_scaler(filename: str) -> StandardScaler:
    """
    Quick load for a scaler.
    
    Args:
        filename: Path to scaler file
        
    Returns:
        Loaded scaler
    """
    scaler = joblib.load(filename)
    print(f"✓ Scaler loaded: {filename}")
    return scaler


# ============================================================================
# DEMO & USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL & SCALER MANAGEMENT EXAMPLES")
    print("="*70)
    
    # Initialize manager
    print("\n1️⃣  Initialize ModelManager")
    print("-" * 70)
    manager = ModelManager(base_dir=".")
    print("✓ ModelManager initialized")
    
    # List existing models
    print("\n2️⃣  List Saved Models")
    print("-" * 70)
    models = manager.list_models()
    
    # List existing scalers
    print("\n3️⃣  List Saved Scalers")
    print("-" * 70)
    scalers = manager.list_scalers()
    
    # Load best model
    if "Random_Forest_tuned" in models:
        print("\n4️⃣  Load Best Model")
        print("-" * 70)
        best_model = manager.load_model("Random_Forest_tuned")
        print(f"✓ Model loaded: {type(best_model).__name__}")
    
    # Load scaler
    if "scaler" in scalers:
        print("\n5️⃣  Load Scaler")
        print("-" * 70)
        scaler = manager.load_scaler("scaler")
        print(f"✓ Scaler loaded with {len(scaler.mean_)} features")
    
    # Load all together
    print("\n6️⃣  Load Model & Scaler Together")
    print("-" * 70)
    artifacts = manager.load_all("Random_Forest_tuned", "scaler")
    print(f"✓ Model: {type(artifacts['model']).__name__}")
    print(f"✓ Scaler: {type(artifacts['scaler']).__name__}")
    
    # Get model info
    print("\n7️⃣  Get Model Information")
    print("-" * 70)
    info = manager.get_model_info("Random_Forest_tuned")
    print(f"Name: {info['name']}")
    print(f"Size: {info['size_kb']:.2f} KB")
    print(f"Path: {info['path']}")
    
    # Example: Make prediction with loaded models
    print("\n8️⃣  Example: Make Prediction with Loaded Models")
    print("-" * 70)
    
    # Check scaler feature count
    n_features = len(scaler.mean_)
    print(f"Scaler expects {n_features} features")
    
    # Create sample data matching scaler's feature count
    sample_data = np.array([np.random.randn(n_features)])
    
    # Scale data
    try:
        sample_scaled = scaler.transform(sample_data)
        
        # Make prediction
        prediction = best_model.predict(sample_scaled)
        prediction_proba = best_model.predict_proba(sample_scaled)
        
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Scaled data shape: {sample_scaled.shape}")
        print(f"Prediction: {prediction[0]}")
        print(f"Fraud probability: {prediction_proba[0][1]:.4f}")
    except Exception as e:
        print(f"⚠ Note: Scaler has {n_features} feature(s), using model demo only")
        print(f"Model type: {type(best_model).__name__}")
        print(f"Model parameters: {best_model.n_estimators} trees")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
