"""
Scaler Handler Module

This module provides classes for handling data scaling and normalization
for different types of models (LSTM, TFT).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scaler_handler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScalerHandlerError(Exception):
    """Exception class for scaler handler errors."""
    pass

class ScalerHandler:
    """
    Base class for handling data scaling and normalization.
    
    Attributes:
        model_type (str): Type of model ('lstm' or 'tft')
        scaler_type (str): Type of scaler to use
        scalers (dict): Dictionary of fitted scalers
        feature_columns (list): List of columns to scale
    """
    
    def __init__(self, model_type: str = 'lstm', scaler_type: str = 'standard'):
        """
        Initialize the scaler handler.
        
        Args:
            model_type (str): Type of model ('lstm' or 'tft')
            scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.model_type = model_type.lower()
        self.scaler_type = scaler_type.lower()
        self.scalers = {}
        self.feature_columns = []
        
        if self.model_type not in ['lstm', 'tft']:
            raise ScalerHandlerError(f"Unsupported model type: {model_type}")
        
        if self.scaler_type not in ['standard', 'minmax', 'robust']:
            raise ScalerHandlerError(f"Unsupported scaler type: {scaler_type}")
    
    def _create_scaler(self) -> Any:
        """Create a new scaler based on the specified type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ScalerHandlerError(f"Unsupported scaler type: {self.scaler_type}")
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None) -> None:
        """
        Fit scalers to the data.
        
        Args:
            data: Input data to fit scalers on
            columns: List of column names to scale (for DataFrame input)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted scalers.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted scalers to disk.
        
        Args:
            path: Directory path to save scalers
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        for name, scaler in self.scalers.items():
            scaler_path = path / f"{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'feature_columns': self.feature_columns
        }
        joblib.dump(metadata, path / "scaler_metadata.joblib")
        
        logger.info(f"Saved scalers to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load fitted scalers from disk.
        
        Args:
            path: Directory path to load scalers from
        """
        path = Path(path)
        
        if not path.exists():
            raise ScalerHandlerError(f"Path does not exist: {path}")
        
        # Load metadata
        metadata_path = path / "scaler_metadata.joblib"
        if not metadata_path.exists():
            raise ScalerHandlerError(f"Metadata file not found: {metadata_path}")
        
        metadata = joblib.load(metadata_path)
        self.model_type = metadata['model_type']
        self.scaler_type = metadata['scaler_type']
        self.feature_columns = metadata['feature_columns']
        
        # Load scalers
        self.scalers = {}
        for column in self.feature_columns:
            scaler_path = path / f"{column}_scaler.joblib"
            if not scaler_path.exists():
                raise ScalerHandlerError(f"Scaler file not found: {scaler_path}")
            
            self.scalers[column] = joblib.load(scaler_path)
        
        logger.info(f"Loaded scalers from {path}")


class LSTMScalerHandler(ScalerHandler):
    """
    Scaler handler for LSTM models.
    
    Handles scaling of time series data for LSTM models.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the LSTM scaler handler.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust')
        """
        super().__init__(model_type='lstm', scaler_type=scaler_type)
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None) -> None:
        """
        Fit scalers to the data.
        
        Args:
            data: Input data to fit scalers on
            columns: List of column names to scale (for DataFrame input)
        """
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns.tolist()
            
            self.feature_columns = columns
            
            for column in columns:
                scaler = self._create_scaler()
                scaler.fit(data[column].values.reshape(-1, 1))
                self.scalers[column] = scaler
        else:
            # For numpy arrays, fit a single scaler
            scaler = self._create_scaler()
            scaler.fit(data.reshape(-1, 1))
            self.scalers['default'] = scaler
            self.feature_columns = ['default']
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted scalers.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            
            for column in self.feature_columns:
                if column in result.columns:
                    scaler = self.scalers.get(column)
                    if scaler is not None:
                        result[column] = scaler.transform(result[column].values.reshape(-1, 1))
            
            return result
        else:
            # For numpy arrays, use the default scaler
            scaler = self.scalers.get('default')
            if scaler is None:
                raise ScalerHandlerError("No default scaler found for numpy array")
            
            return scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            
            for column in self.feature_columns:
                if column in result.columns:
                    scaler = self.scalers.get(column)
                    if scaler is not None:
                        result[column] = scaler.inverse_transform(result[column].values.reshape(-1, 1))
            
            return result
        else:
            # For numpy arrays, use the default scaler
            scaler = self.scalers.get('default')
            if scaler is None:
                raise ScalerHandlerError("No default scaler found for numpy array")
            
            return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)


class TFTScalerHandler(ScalerHandler):
    """
    Scaler handler for TFT models.
    
    Handles scaling of time series data for TFT models.
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        """
        Initialize the TFT scaler handler.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust')
        """
        super().__init__(model_type='tft', scaler_type=scaler_type)
        self.target_scaler = None
        self.target_column = None
    
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None, target_column: str = 'close') -> None:
        """
        Fit scalers to the data.
        
        Args:
            data: Input data to fit scalers on
            columns: List of column names to scale
            target_column: Name of the target column
        """
        if columns is None:
            # Default columns for TFT models
            columns = ['open', 'high', 'low', 'close', 'volume']
            # Filter to only include columns that exist in the data
            columns = [col for col in columns if col in data.columns]
        
        self.feature_columns = columns
        self.target_column = target_column
        
        # Fit scalers for each column
        for column in columns:
            if column in data.columns:
                scaler = self._create_scaler()
                # Handle multi-index DataFrame
                if isinstance(data.index, pd.MultiIndex):
                    # Extract all values for this column across all symbols
                    values = data[column].values.reshape(-1, 1)
                else:
                    values = data[column].values.reshape(-1, 1)
                
                scaler.fit(values)
                self.scalers[column] = scaler
                
                # Save a separate target scaler
                if column == target_column:
                    self.target_scaler = scaler
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        result = data.copy()
        
        for column in self.feature_columns:
            if column in result.columns and column in self.scalers:
                scaler = self.scalers[column]
                # Handle multi-index DataFrame
                if isinstance(result.index, pd.MultiIndex):
                    # Transform each symbol's data separately to maintain the index
                    for symbol in result.index.get_level_values('symbol').unique():
                        idx = result.index.get_level_values('symbol') == symbol
                        values = result.loc[idx, column].values.reshape(-1, 1)
                        result.loc[idx, column] = scaler.transform(values)
                else:
                    result[column] = scaler.transform(result[column].values.reshape(-1, 1))
        
        return result
    
    def inverse_transform_target(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target predictions.
        
        Args:
            predictions: Scaled predictions to inverse transform
            
        Returns:
            Original scale predictions
        """
        if self.target_scaler is None:
            raise ScalerHandlerError("Target scaler not fitted")
        
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform all scaled data.
        
        Args:
            data: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        result = data.copy()
        
        for column in self.feature_columns:
            if column in result.columns and column in self.scalers:
                scaler = self.scalers[column]
                # Handle multi-index DataFrame
                if isinstance(result.index, pd.MultiIndex):
                    # Transform each symbol's data separately to maintain the index
                    for symbol in result.index.get_level_values('symbol').unique():
                        idx = result.index.get_level_values('symbol') == symbol
                        values = result.loc[idx, column].values.reshape(-1, 1)
                        result.loc[idx, column] = scaler.inverse_transform(values)
                else:
                    result[column] = scaler.inverse_transform(result[column].values.reshape(-1, 1))
        
        return result 