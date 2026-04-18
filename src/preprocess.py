"""Data preprocessing module for fraud detection."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ("drop" or "mean")
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
    return df


def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to remove outliers from
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features using specified method.
    
    Args:
        X_train: Training features
        X_test: Testing features
        method: Scaling method ("standard" or "robust")
        
    Returns:
        Tuple of scaled training and testing features
    """
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def prepare_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "Class"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete data preparation pipeline.
    
    Args:
        filepath: Path to the data file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        target_column: Name of the target column
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values
