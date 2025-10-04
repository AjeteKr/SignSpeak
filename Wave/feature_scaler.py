"""
Feature Scaling Utilities
Functions for loading and applying feature scalers for ASL recognition
"""
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)

def load_feature_scaler(filepath: str) -> Optional[object]:
    """
    Load a saved feature scaler from pickle file
    
    Args:
        filepath: Path to the saved scaler file
        
    Returns:
        Loaded scaler object or None if loading fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Scaler file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info(f"Feature scaler loaded from {filepath}")
        logger.info(f"Scaler type: {type(scaler).__name__}")
        
        return scaler
        
    except Exception as e:
        logger.error(f"Error loading feature scaler: {e}")
        return None

def save_feature_scaler(scaler: object, filepath: str) -> bool:
    """
    Save a feature scaler to pickle file
    
    Args:
        scaler: The scaler object to save
        filepath: Path where to save the scaler
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Feature scaler saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving feature scaler: {e}")
        return False

def create_default_scaler(scaler_type: str = "standard") -> object:
    """
    Create a default feature scaler
    
    Args:
        scaler_type: Type of scaler ("standard", "minmax", or "robust")
        
    Returns:
        Initialized scaler object
    """
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type.lower() == "robust":
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown scaler type: {scaler_type}, using StandardScaler")
        scaler = StandardScaler()
    
    logger.info(f"Created {type(scaler).__name__}")
    return scaler

def fit_scaler_on_data(data: np.ndarray, scaler_type: str = "standard") -> object:
    """
    Fit a scaler on training data
    
    Args:
        data: Training data to fit the scaler on
        scaler_type: Type of scaler to use
        
    Returns:
        Fitted scaler object
    """
    scaler = create_default_scaler(scaler_type)
    
    try:
        scaler.fit(data)
        logger.info(f"Scaler fitted on data with shape {data.shape}")
        return scaler
        
    except Exception as e:
        logger.error(f"Error fitting scaler: {e}")
        return scaler

def apply_scaler(scaler: object, data: np.ndarray) -> np.ndarray:
    """
    Apply a fitted scaler to data
    
    Args:
        scaler: Fitted scaler object
        data: Data to transform
        
    Returns:
        Transformed data
    """
    if scaler is None:
        logger.warning("No scaler provided, returning original data")
        return data
    
    try:
        # Handle both single samples and batches
        if data.ndim == 1:
            # Single sample - reshape for scaler
            transformed = scaler.transform(data.reshape(1, -1))[0]
        else:
            # Batch of samples
            transformed = scaler.transform(data)
        
        return transformed
        
    except Exception as e:
        logger.error(f"Error applying scaler: {e}")
        return data

def create_mock_scaler() -> object:
    """
    Create a mock scaler for testing when no real scaler is available
    This scaler performs identity transformation (no change to data)
    """
    class MockScaler:
        """Mock scaler that doesn't change the data"""
        
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
            
        def fit(self, X):
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
            return self
            
        def transform(self, X):
            return X.copy()
            
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
            
        def inverse_transform(self, X):
            return X.copy()
    
    logger.info("Created mock scaler (identity transformation)")
    return MockScaler()

# Convenience function for backward compatibility
def load_scaler(filepath: str) -> Optional[object]:
    """Alias for load_feature_scaler"""
    return load_feature_scaler(filepath)