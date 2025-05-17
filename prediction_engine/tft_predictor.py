"""
TFT Predictor Module

This module provides a class for loading and using a trained Temporal Fusion Transformer (TFT)
model for stock price predictions.
"""

import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tft_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TFTPredictorError(Exception):
    """Exception class for TFT predictor errors."""
    pass

class TFTPredictor:
    """
    A predictor class that uses a trained Temporal Fusion Transformer (TFT) model
    to make stock price predictions.
    
    Attributes:
        model_path (Path): Path to the directory containing the trained model
        config_path (Path): Path to the configuration file
        model: Loaded TFT model instance
        context_length (int): Number of time steps used for context
        prediction_horizon (int): Number of time steps to predict
    """
    
    def __init__(self, model_path, config_path):
        """
        Initialize the TFT predictor.
        
        Args:
            model_path (str or Path): Path to the directory containing the trained model
            config_path (str or Path): Path to the configuration file
            
        Raises:
            TFTPredictorError: If model_path or config_path is invalid
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Validate paths
        if not self.model_path.exists():
            raise TFTPredictorError(f"Model path does not exist: {self.model_path}")
        if not self.config_path.exists():
            raise TFTPredictorError(f"Config path does not exist: {self.config_path}")
        
        # Load configuration
        self._load_config()
        
        # Initialize model as None - will be loaded when needed
        self.model = None
        
        logger.info(f"TFT predictor initialized with model from {self.model_path}")
    
    def _load_config(self):
        """
        Load configuration from YAML file and model config.
        
        Raises:
            TFTPredictorError: If configuration cannot be loaded
        """
        try:
            # Load main config
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Load model config
            model_config_path = self.model_path / "model_config.json"
            if not model_config_path.exists():
                raise TFTPredictorError(f"Model config not found: {model_config_path}")
            
            with open(model_config_path, 'r') as f:
                self.model_config = json.load(f)
            
            # Load formatter config if available
            formatter_config_path = self.model_path / "formatter_config.json"
            if formatter_config_path.exists():
                with open(formatter_config_path, 'r') as f:
                    self.formatter_config = json.load(f)
            else:
                self.formatter_config = None
            
            # Extract key parameters
            self.context_length = self.model_config["model_config"]["num_encoder_steps"]
            self.prediction_horizon = self.model_config["model_config"]["num_steps"]
            
        except (yaml.YAMLError, json.JSONDecodeError, KeyError) as e:
            raise TFTPredictorError(f"Error loading configuration: {str(e)}")
    
    def _load_model(self):
        """
        Load the trained TFT model.
        
        Returns:
            Loaded TFT model instance
            
        Raises:
            TFTPredictorError: If model cannot be loaded
        """
        try:
            # Import TensorFlow only when needed
            import tensorflow as tf
            
            # Import TFT model from colab_training
            sys.path.append(str(Path(__file__).parent.parent))
            
            try:
                from colab_training.libs.tft_model import TFTModel
                
                # Create model instance with the saved configuration
                model = TFTModel(self.model_config["model_config"])
                
                # Load weights
                weights_path = self.model_path / "best_model.weights.h5"
                if not weights_path.exists():
                    weights_path = self.model_path / "model.weights.h5"
                
                if not weights_path.exists():
                    raise TFTPredictorError(f"Model weights not found in {self.model_path}")
                
                # Load the model weights
                model.load(self.model_path, use_keras_loadings=True)
                
                return model
                
            except ImportError as e:
                raise TFTPredictorError(f"Could not import TFT model: {str(e)}. Make sure the TFT model implementation is available in colab_training/libs/tft_model.py")
                
        except Exception as e:
            raise TFTPredictorError(f"Error loading TFT model: {str(e)}")
    
    def preprocess_data(self, df):
        """
        Preprocess input data for the TFT model.
        
        Args:
            df (pd.DataFrame): Input DataFrame with MultiIndex (symbol, datetime)
            
        Returns:
            dict: Processed data ready for TFT model
            
        Raises:
            TFTPredictorError: If data cannot be preprocessed
        """
        try:
            # Validate input data
            if df.empty:
                raise TFTPredictorError("Input DataFrame is empty")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise TFTPredictorError(f"Missing required columns: {missing_columns}")
            
            # Check if we have enough data points
            symbols = df.index.get_level_values('symbol').unique()
            for symbol in symbols:
                symbol_data = df.xs(symbol, level='symbol')
                if len(symbol_data) < self.context_length:
                    raise TFTPredictorError(
                        f"Insufficient data for symbol {symbol}. "
                        f"Need at least {self.context_length} data points, but got {len(symbol_data)}"
                    )
            
            # Format data for TFT model
            # This is a simplified version - in production, you would need to implement
            # the full data formatting logic based on your TFT model's requirements
            
            # For now, we'll create a basic structure that matches what the model expects
            processed_data = {
                "inputs": df.reset_index(),
                "time": df.index.get_level_values('datetime').values,
                "identifier": df.index.get_level_values('symbol').values
            }
            
            return processed_data
            
        except Exception as e:
            if not isinstance(e, TFTPredictorError):
                raise TFTPredictorError(f"Error preprocessing data: {str(e)}")
            raise
    
    def predict(self, df):
        """
        Make predictions using the TFT model.
        
        Args:
            df (pd.DataFrame): Input DataFrame with MultiIndex (symbol, datetime)
            
        Returns:
            dict: Dictionary containing predictions and metadata
            
        Raises:
            TFTPredictorError: If prediction fails
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.model = self._load_model()
            
            # Validate input
            if df.empty:
                raise TFTPredictorError("Input DataFrame is empty")
            
            # Preprocess data
            processed_data = self.preprocess_data(df)
            
            # Make predictions
            raw_predictions = self.model.predict(processed_data["inputs"])
            
            # Extract median predictions (p50)
            if 'p50' not in raw_predictions:
                raise TFTPredictorError("Model did not return p50 predictions")
            
            median_predictions = raw_predictions['p50']
            
            # Format predictions
            forecast_dates = median_predictions['forecast_time'].unique()
            
            # Extract the actual prediction values
            prediction_cols = [col for col in median_predictions.columns if col.startswith('t+')]
            predictions_array = median_predictions[prediction_cols].values
            
            # Create a dictionary with the results
            result = {
                'predictions': predictions_array,
                'forecast_dates': forecast_dates,
                'symbols': median_predictions['identifier'].unique(),
                'raw_predictions': raw_predictions
            }
            
            return result
            
        except Exception as e:
            if not isinstance(e, TFTPredictorError):
                raise TFTPredictorError(f"Error making predictions: {str(e)}")
            raise
    
    def predict_next(self, df, symbol, n_steps=1):
        """
        Predict the next n_steps for a specific symbol.
        
        Args:
            df (pd.DataFrame): Input DataFrame with MultiIndex (symbol, datetime)
            symbol (str): Symbol to predict for
            n_steps (int): Number of steps to predict
            
        Returns:
            dict: Dictionary containing predictions and metadata
            
        Raises:
            TFTPredictorError: If prediction fails
        """
        try:
            # Validate input
            if n_steps <= 0:
                raise TFTPredictorError(f"n_steps must be positive, got {n_steps}")
            
            if symbol not in df.index.get_level_values('symbol').unique():
                raise TFTPredictorError(f"Symbol {symbol} not found in input data")
            
            # Filter data for the specific symbol
            symbol_data = df.xs(symbol, level='symbol')
            
            # Get the last date and price
            last_date = symbol_data.index.max()
            last_price = symbol_data['close'].iloc[-1]
            
            # Make predictions
            predictions = self.predict(df)
            
            # Extract predictions for the specific symbol
            symbol_predictions = predictions['raw_predictions']['p50']
            symbol_predictions = symbol_predictions[symbol_predictions['identifier'] == symbol]
            
            # Get the latest forecast
            latest_forecast = symbol_predictions[symbol_predictions['forecast_time'] == symbol_predictions['forecast_time'].max()]
            
            # Extract prediction values
            prediction_cols = [col for col in latest_forecast.columns if col.startswith('t+')][:n_steps]
            prediction_values = latest_forecast[prediction_cols].values.flatten()[:n_steps]
            
            # Generate timestamps for predictions
            timestamps = [last_date + timedelta(days=i+1) for i in range(n_steps)]
            
            # Create result dictionary
            result = {
                'predictions': prediction_values.tolist(),
                'timestamps': timestamps,
                'last_date': last_date,
                'last_price': last_price
            }
            
            return result
            
        except Exception as e:
            if not isinstance(e, TFTPredictorError):
                raise TFTPredictorError(f"Error predicting next steps: {str(e)}")
            raise 