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
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
#     handlers=[
#         logging.FileHandler('logs/tft_predictor.log'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)
# Clear existing handlers to prevent duplicates if they're set by basicConfig
# if len(logger.handlers) > 1:
#     logger.handlers = [logger.handlers[0]]  # Keep only the first handler

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
    
    def get_attention_weights(self, inputs) -> np.ndarray:
        """Get attention weights for interpretability."""
        # Not implemented in this version
        raise NotImplementedError("Attention weights visualization is not implemented")
        
    def __call__(self, sequence_tensor):
        """Handle PyTorch tensor inputs for prediction.
        
        Args:
            sequence_tensor: PyTorch tensor of shape [batch_size, sequence_length, features]
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        logger.info(f"TFTPredictor.__call__ with input shape: {sequence_tensor.shape}")
        
        if hasattr(sequence_tensor, 'cpu'):
            sequence_np = sequence_tensor.cpu().numpy()
        else:
            sequence_np = sequence_tensor
            
        logger.info(f"Converted to numpy with shape: {sequence_np.shape}")

        # Initial reshaping based on input dimensions
        if sequence_np.ndim == 1: # Input is 1D (e.g. [features])
            # Assume batch_size=1, seq_len=1 if only features are provided
            # Or if it's a sequence of features for one sample, seq_len=len, features=1
            # This part is ambiguous without knowing the exact model input structure
            # For now, let's assume it's (seq_len) and needs to become (1, seq_len, 1)
            logger.info(f"Reshaping 1D array {sequence_np.shape} to 3D (1, seq_len, 1)")
            sequence_np = sequence_np.reshape(1, -1, 1) 
        elif sequence_np.ndim == 2: # Input is 2D
            if sequence_np.shape[1] == 1 and sequence_np.shape[0] != 1: # (seq_len, 1)
                logger.info(f"Input is 2D {sequence_np.shape}, potentially (seq_len, 1). Flattening and reshaping to (1, seq_len, 1).")
                sequence_np = sequence_np.flatten().reshape(1, -1, 1)
            elif sequence_np.shape[0] == 1: # (1, features_or_seq_len)
                logger.info(f"Input is 2D {sequence_np.shape}, potentially (1, N). Reshaping to (1, N, 1) or (1, 1, N).")
                # This case is also ambiguous. If N is seq_len, num_features=1. If N is num_features, seq_len=1.
                # Assuming (1, seq_len) and num_features = 1
                sequence_np = sequence_np.reshape(1, -1, 1)
            else: # (seq_len, features)
                logger.info(f"Input is 2D {sequence_np.shape}, assuming (seq_len, features). Adding batch dimension.")
                sequence_np = sequence_np.reshape(1, sequence_np.shape[0], sequence_np.shape[1])
        elif sequence_np.ndim == 3: # Input is 3D (batch, seq_len, features)
            logger.info(f"Input is already 3D: {sequence_np.shape}")
        else:
            raise ValueError(f"Unexpected tensor ndim: {sequence_np.ndim} for shape {sequence_np.shape}")

        logger.info(f"Numpy array shape after initial processing: {sequence_np.shape}")
        
        batch_size, seq_len, n_features = sequence_np.shape
        
        try:
            df_data = []
            for i in range(batch_size):
                # For each item in the batch, construct the features for the DataFrame
                # The original TFT model expects multiple features and known/unknown inputs.
                
                # We take the *last* time step of the sequence to form the input for prediction.
                current_features = sequence_np[i, -1, :] # Shape: (n_features,)

                # Ensure current_features is 1D for DataFrame construction if n_features is 1
                if n_features == 1 and current_features.ndim > 1: 
                    current_features = current_features.flatten()
                
                # These column names MUST match what the TFT model was trained on.
                # The model's hparams should contain the list of these feature names.
                # Typically stored in self.model.hparams.time_varying_unknown_reals
                expected_feature_names = self.model.hparams.get("time_varying_unknown_reals", [])
                
                if not expected_feature_names:
                    # This fallback is risky and assumes a certain number of features if names are not in hparams.
                    # It's better if hparams always contain the feature names.
                    logger.warning(
                        "Feature names ('time_varying_unknown_reals') not found in model hparams. "
                        "Attempting to proceed with assumed feature names based on n_features. "
                        "This may lead to errors if the assumption is incorrect."
                    )
                    # Placeholder: generate generic names if specific ones aren't found.
                    # This part might need adjustment based on how features were named during training.
                    if n_features > 0 :
                        expected_feature_names = [f'feature_{k}' for k in range(n_features)]
                        # A common case is that the first feature is 'close' or the target variable.
                        # If your model was trained with 'close' as the first unknown real, for example:
                        # expected_feature_names[0] = 'close' # Or whatever the actual target/main feature is.
                        logger.warning(f"Using generic feature names: {expected_feature_names}")
                    else: # n_features is 0
                         raise TFTPredictorError("Cannot proceed with 0 features and no feature names in hparams.")


                if len(expected_feature_names) != n_features:
                    # This error can also occur if the data fed to prepare_features resulted in an unexpected number of feature columns.
                    raise TFTPredictorError(
                        f"Mismatch between number of expected (time_varying_unknown_reals) features ({len(expected_feature_names)}) "
                        f"and actual features provided in sequence ({n_features}). Expected names: {expected_feature_names}. "
                        "Check data preprocessing and feature engineering steps."
                    )

                row = {
                    # Standard columns required by PyTorch Forecasting
                    # The actual group ID column name might be different, check training TimeSeriesDataSet.
                    self.model.hparams.get("group_ids", ["symbol"])[0]: f'DUMMY_{i}', 
                    # The actual time index column name might be different.
                    self.model.hparams.get("time_idx", "time_idx"): pd.Timestamp.now().timestamp(), # Needs to be a numerical time index
                    # Add any static categorical or real features if model expects them for DUMMY_{i}
                    # Add any time-varying known features if model expects them
                }
                
                # Populate the time-varying unknown features
                for idx, feature_name in enumerate(expected_feature_names):
                    row[feature_name] = current_features[idx]
                
                # PyTorch Forecasting also needs a 'target' column for prediction, often set to a dummy value like 0
                # The actual target column name(s) are in self.model.hparams.target
                # If it's a multi-target model, all target names must be present.
                if isinstance(self.model.hparams.target, str):
                    targets = [self.model.hparams.target]
                else:
                    targets = self.model.hparams.target
                
                for target_name in targets:
                    if target_name not in row: # Avoid overwriting if it's also a feature
                         row[target_name] = 0.0 # Dummy value for prediction input

                df_data.append(row)

            if not df_data:
                raise ValueError("Could not construct DataFrame, df_data is empty.")

            df = pd.DataFrame(df_data)
            df.set_index(['symbol', 'datetime'], inplace=True)
            
            logger.info(f"Constructed DataFrame for prediction with shape: {df.shape}")
            logger.debug(f"DataFrame head:\n{df.head()}")

            # Call the original predict method which expects a DataFrame
            logger.info("Calling self.predict(df) within __call__")
            result = self.predict(df) # This calls the TFT model's predict
            
            # Extract prediction - assuming it's for the first symbol and first forecast step
            # The structure of 'result['predictions']' depends on your TFT model output
            predictions_val = result['predictions'][0, 0] 
            
            # For now, we'll use a fixed uncertainty for simplicity
            uncertainty_val = 0.01  # 1% uncertainty, as a float
            
            logger.info(f"Prediction: {predictions_val}, Uncertainty: {uncertainty_val}")
            
            import torch
            # Ensure tensors are 2D: [[value]]
            predictions_tensor = torch.tensor([[predictions_val]], dtype=torch.float32)
            uncertainty_tensor = torch.tensor([[uncertainty_val]], dtype=torch.float32)
            
            return predictions_tensor, uncertainty_tensor
            
        except Exception as e:
            logger.error(f"Error in TFTPredictor call: {str(e)}")
            # It's often helpful to log the state that led to the error
            logger.error(f"Input sequence_np shape: {sequence_np.shape if 'sequence_np' in locals() else 'not defined'}")
            logger.error(f"Constructed df_data: {df_data if 'df_data' in locals() else 'not defined'}")
            raise TFTPredictorError(f"Failed in TFTPredictor call: {str(e)}") 