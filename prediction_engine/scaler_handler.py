"""Scaler handler module for stock prediction.

This module provides functionality for scaling and normalizing data
for different types of models (TFT).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
from pathlib import Path
import joblib

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ScalerHandlerError(Exception):
    """Custom exception for scaler handler related errors."""
    pass

class ScalerHandler:
    """
    Base scaler handler for time series data.
    
    This class provides functionality for scaling and normalizing time series data
    for different types of models (TFT).
    
    Attributes:
        model_type (str): Type of model ('tft')
        scaler_type (str): Type of scaler to use ('standard', 'robust', or 'minmax')
        scaler: The actual scaler instance
    """
    
    def __init__(self, model_type: str = 'tft', scaler_type: str = 'standard'):
        """
        Initialize the scaler handler.
        
        Args:
            model_type (str): Type of model ('tft')
            scaler_type (str): Type of scaler to use ('standard', 'robust', or 'minmax')
            
        Raises:
            ScalerHandlerError: If model_type or scaler_type is invalid
        """
        if model_type != 'tft':
            raise ScalerHandlerError(f"Unsupported model type: {model_type}. Must be 'tft'.")
        
        if scaler_type not in ['standard', 'robust', 'minmax']:
            raise ScalerHandlerError(
                f"Unsupported scaler type: {scaler_type}. "
                "Must be 'standard', 'robust', or 'minmax'."
            )
        
        self.model_type = model_type
        self.scaler_type = scaler_type
        self.scaler = None
        
        # Initialize appropriate scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:  # minmax
            self.scaler = MinMaxScaler()
        
        logger.info(f"Initialized {scaler_type} scaler for {model_type} model")
    
    def fit_scaler(self, data: pd.DataFrame) -> Optional[object]:
        """
        Fit the scaler to the data.
        
        Args:
            data: DataFrame to fit the scaler to
            
        Returns:
            Fitted scaler instance or None if fitting fails
            
        Raises:
            ScalerHandlerError: If fitting fails
        """
        try:
            if data.empty:
                raise ScalerHandlerError("Cannot fit scaler to empty data")
            
            # Fit scaler
            self.scaler.fit(data)
            logger.info(f"Fitted {self.scaler_type} scaler to data")
            return self.scaler
            
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
            raise ScalerHandlerError(f"Failed to fit scaler: {str(e)}")
    
    def transform_data(
        self,
        data: pd.DataFrame,
        scaler: Optional[object] = None
    ) -> pd.DataFrame:
        """
        Transform data using the scaler.
        
        Args:
            data: DataFrame to transform
            scaler: Optional scaler instance to use (defaults to self.scaler)
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ScalerHandlerError: If transformation fails
        """
        try:
            if data.empty:
                raise ScalerHandlerError("Cannot transform empty data")
            
            # Use provided scaler or self.scaler
            scaler_to_use = scaler or self.scaler
            if scaler_to_use is None:
                raise ScalerHandlerError("No scaler available for transformation")
            
            # Transform data
            transformed_data = pd.DataFrame(
                scaler_to_use.transform(data),
                columns=data.columns,
                index=data.index
            )
            
            logger.info(f"Transformed data using {self.scaler_type} scaler")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise ScalerHandlerError(f"Failed to transform data: {str(e)}")
    
    def inverse_transform_data(
        self,
        data: pd.DataFrame,
        scaler: Optional[object] = None
    ) -> pd.DataFrame:
        """
        Inverse transform data using the scaler.
        
        Args:
            data: DataFrame to inverse transform
            scaler: Optional scaler instance to use (defaults to self.scaler)
            
        Returns:
            Inverse transformed DataFrame
            
        Raises:
            ScalerHandlerError: If inverse transformation fails
        """
        try:
            if data.empty:
                raise ScalerHandlerError("Cannot inverse transform empty data")
            
            # Use provided scaler or self.scaler
            scaler_to_use = scaler or self.scaler
            if scaler_to_use is None:
                raise ScalerHandlerError("No scaler available for inverse transformation")
            
            # Inverse transform data
            original_data = pd.DataFrame(
                scaler_to_use.inverse_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            logger.info(f"Inverse transformed data using {self.scaler_type} scaler")
            return original_data
            
        except Exception as e:
            logger.error(f"Error inverse transforming data: {str(e)}")
            raise ScalerHandlerError(f"Failed to inverse transform data: {str(e)}")
    
    def save_scaler(self, path: Union[str, Path]) -> None:
        """
        Save the scaler to disk.
        
        Args:
            path: Path to save the scaler to
            
        Raises:
            ScalerHandlerError: If saving fails
        """
        try:
            if self.scaler is None:
                raise ScalerHandlerError("No scaler to save")
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.scaler, path)
            logger.info(f"Saved scaler to {path}")
            
        except Exception as e:
            logger.error(f"Error saving scaler: {str(e)}")
            raise ScalerHandlerError(f"Failed to save scaler: {str(e)}")
    
    @classmethod
    def load_scaler(cls, path: Union[str, Path]) -> 'ScalerHandler':
        """
        Load a scaler from disk.
        
        Args:
            path: Path to load the scaler from
            
        Returns:
            ScalerHandler instance with loaded scaler
            
        Raises:
            ScalerHandlerError: If loading fails
        """
        try:
            path = Path(path)
            if not path.exists():
                raise ScalerHandlerError(f"Scaler file not found: {path}")
            
            # Create instance
            instance = cls()
            
            # Load scaler
            instance.scaler = joblib.load(path)
            logger.info(f"Loaded scaler from {path}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise ScalerHandlerError(f"Failed to load scaler: {str(e)}")

class TFTScalerHandler(ScalerHandler):
    """
    Scaler handler for TFT models.
    
    This class extends the base ScalerHandler to provide specific
    functionality for scaling time series data for TFT models.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the TFT scaler handler.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'robust', or 'minmax')
        """
        super().__init__(model_type='tft', scaler_type=scaler_type)
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features for TFT model input.
        
        Args:
            data: DataFrame containing features to scale
            
        Returns:
            Scaled DataFrame
            
        Raises:
            ScalerHandlerError: If scaling fails
        """
        try:
            if data.empty:
                raise ScalerHandlerError("Cannot scale empty data")
            
            # Fit scaler if not already fitted
            if self.scaler is None:
                self.fit_scaler(data)
            
            # Transform data
            scaled_data = self.transform_data(data)
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise ScalerHandlerError(f"Failed to scale features: {str(e)}")
    
    def prepare_tft_input(
        self,
        data: pd.DataFrame,
        target_col: str = 'target'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare input data for TFT model.
        
        Args:
            data: DataFrame containing features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (scaled_features, scaled_target)
            
        Raises:
            ScalerHandlerError: If preparation fails
        """
        try:
            if data.empty:
                raise ScalerHandlerError("Cannot prepare empty data")
            
            if target_col not in data.columns:
                raise ScalerHandlerError(f"Target column '{target_col}' not found in data")
            
            # Split features and target
            features = data.drop(columns=[target_col])
            target = data[[target_col]]
            
            # Scale features
            scaled_features = self.scale_features(features)
            
            # Scale target
            if self.scaler is None:
                self.fit_scaler(target)
            scaled_target = self.transform_data(target)
            
            return scaled_features, scaled_target
            
        except Exception as e:
            logger.error(f"Error preparing TFT input: {str(e)}")
            raise ScalerHandlerError(f"Failed to prepare TFT input: {str(e)}") 