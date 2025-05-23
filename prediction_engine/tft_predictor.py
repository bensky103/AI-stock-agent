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
                
                # Ensure weights_path is absolute
                absolute_weights_path = weights_path.resolve()

                if not absolute_weights_path.exists():
                    # Log which paths were checked if still not found
                    checked_paths = [
                        str(self.model_path / "best_model.weights.h5"), 
                        str(self.model_path / "model.weights.h5"),
                        str(absolute_weights_path) # Log the resolved absolute path attempted
                    ]
                    logger.error(f"Model weights .h5 file not found after checking paths: {checked_paths}")
                    raise TFTPredictorError(f"Model weights .h5 file not found. Checked: {checked_paths}")
                
                logger.info(f"Attempting to load weights from directory: {str(self.model_path.resolve())}")
                # Load the model using its custom load method, passing the directory
                model.load(str(self.model_path.resolve()))
                
                return model
                
            except ImportError as e:
                raise TFTPredictorError(f"Could not import TFT model: {str(e)}. Make sure the TFT model implementation is available in colab_training/libs/tft_model.py")
                
        except Exception as e:
            raise TFTPredictorError(f"Error loading TFT model: {str(e)}")

    def preprocess_data(self, df):
        """
        Preprocess input data for the TFT model.
        This method might be used if TFTPredictor.predict() is called directly
        with a raw DataFrame that needs standard preprocessing before hitting
        a PyTorch Forecasting like model. 
        For the __call__ path with a Keras model, this is likely bypassed if __call__
        formats the DataFrame completely.
        
        Args:
            df (pd.DataFrame): Input DataFrame, indexed by group ID and time ID.
            
        Returns:
            dict: Processed data ready for a PyTorch Forecasting style TFT model's predict method.
            
        Raises:
            TFTPredictorError: If data cannot be preprocessed
        """
        try:
            if self.model is None or not hasattr(self.model, 'hparams'):
                logger.info("Model or hparams not available in preprocess_data, attempting to load model.")
                self._load_model() 
                if self.model is None or not hasattr(self.model, 'hparams'):
                    # If still no hparams (e.g. Keras model), this path might not be usable.
                    logger.warning("Model.hparams not available after load; preprocess_data might not function as expected for PyTorch Forecasting models.")
                    # For a Keras model, self.model.hparams might not exist. 
                    # Fallback to model_config if hparams are PyTorch Forecasting specific.
                    gid_col_name = self.model_config.get("model_config", {}).get("identifier_col_name", "symbol")
                    tid_col_name = self.model_config.get("model_config", {}).get("time_idx_col_name", "time_idx")
                    time_varying_unknown_reals = self.model_config.get("model_config", {}).get("time_varying_unknown_reals", [])
                else:
                    gid_col_name = self.model.hparams.get("group_ids", ["symbol"])[0]
                    tid_col_name = self.model.hparams.get("time_idx", "time_idx")
                    time_varying_unknown_reals = self.model.hparams.get("time_varying_unknown_reals", [])
            else: # hparams exist
                gid_col_name = self.model.hparams.get("group_ids", ["symbol"])[0]
                tid_col_name = self.model.hparams.get("time_idx", "time_idx")
                time_varying_unknown_reals = self.model.hparams.get("time_varying_unknown_reals", [])

            if df.empty:
                raise TFTPredictorError("Input DataFrame is empty for preprocess_data")

            missing_reals = [col for col in time_varying_unknown_reals if col not in df.columns]
            if missing_reals:
                raise TFTPredictorError(
                    f"Missing required time_varying_unknown_reals in df for preprocess_data: {missing_reals}. "
                    f"Available: {df.columns.tolist()}. Expected: {time_varying_unknown_reals}."
                )

            df_flat = df.reset_index()

            if tid_col_name in df_flat:
                if not pd.api.types.is_numeric_dtype(df_flat[tid_col_name]):
                    logger.warning(f"Time column '{tid_col_name}' is not numeric ({df_flat[tid_col_name].dtype}).")
                time_values = df_flat[tid_col_name].values
            else:
                raise TFTPredictorError(f"Time index column '{tid_col_name}' not found in flattened DataFrame for preprocess_data.")

            if gid_col_name not in df_flat:
                raise TFTPredictorError(f"Group ID column '{gid_col_name}' not found in flattened DataFrame for preprocess_data.")

            processed_data = {
                "inputs": df_flat, 
                "time": time_values, 
                "identifier": df_flat[gid_col_name].values
            }
            
            return processed_data
            
        except Exception as e:
            if not isinstance(e, TFTPredictorError):
                raise TFTPredictorError(f"Error preprocessing data: {str(e)}")
            raise
    
    def predict(self, df):
        """
        Make predictions using the TFT model.
        The __call__ method is now the primary path for sequence prediction and prepares the DataFrame.
        This method passes the already formatted DataFrame to the model.
        
        Args:
            df (pd.DataFrame): Input DataFrame, expected to be formatted correctly 
                               for the self.model.predict() call (e.g. by __call__).
            
        Returns:
            dict: Dictionary containing predictions and metadata (structure adapted for Keras model output).
            
        Raises:
            TFTPredictorError: If prediction fails
        """
        try:
            if self.model is None:
                self.model = self._load_model()
            
            if df.empty:
                raise TFTPredictorError("Input DataFrame for predict() is empty")
            
            logger.info(f"[{self.__class__.__name__}] TFTPredictor.predict called with df of shape: {df.shape}. Dtypes:\n{df.dtypes}. Assuming it's preformatted by __call__ or similar.")
            logger.debug(f"[{self.__class__.__name__}] TFTPredictor.predict input df head:\n{df.head()}")

            raw_predictions_output = self.model.predict(df) 
            logger.info(f"[{self.__class__.__name__}] Output type from self.model.predict: {type(raw_predictions_output)}")
            if isinstance(raw_predictions_output, np.ndarray):
                logger.info(f"[{self.__class__.__name__}] Output shape from self.model.predict: {raw_predictions_output.shape}")
            elif isinstance(raw_predictions_output, dict):
                logger.info(f"[{self.__class__.__name__}] Output keys from self.model.predict: {list(raw_predictions_output.keys())}")

            if isinstance(raw_predictions_output, np.ndarray):
                predictions_array = raw_predictions_output 
                
                m_config = self.model_config.get("model_config", {})
                gid_col_name = m_config.get("identifier_col_name", "symbol")
                tid_col_name = m_config.get("time_idx_col_name", "time_idx")

                num_sequences_in_batch = df[gid_col_name].nunique() if gid_col_name in df else 1
                pred_horizon = predictions_array.shape[1] if predictions_array.ndim > 1 else (predictions_array.shape[0] if predictions_array.ndim == 1 and num_sequences_in_batch == 1 else 1)

                # Create placeholder forecast dates. For real dates, more context from `df` or `sequence_tensor` would be needed.
                # This assumes predictions start from the day after the last known time step.
                # If df contains actual timestamps, those could be used as a basis.
                last_input_time = df[tid_col_name].max() if tid_col_name in df else pd.Timestamp.now().timestamp()
                forecast_dates_placeholder = [pd.to_datetime(last_input_time, unit='s') + timedelta(days=i+1) for i in range(pred_horizon)]
                
                symbols_placeholder = df[gid_col_name].unique() if gid_col_name in df else [f"DUMMY_GROUP_{k}" for k in range(num_sequences_in_batch)]

                # Adapt raw_predictions_output to fit the old dict structure for 'raw_predictions':{'p50': ...}
                # This is a simplification. If the Keras model outputs quantiles, they should be mapped here.
                # For now, assuming the output `predictions_array` is the equivalent of 'p50'.
                raw_pred_dict_p50_equivalent = predictions_array 

                result = {
                    'predictions': predictions_array, 
                    'forecast_dates': forecast_dates_placeholder, 
                    'symbols': symbols_placeholder, 
                    'raw_predictions': {'p50': raw_pred_dict_p50_equivalent}
                }
                logger.info(f"[{self.__class__.__name__}] Formatted numpy predictions. Array shape: {predictions_array.shape}")

            elif isinstance(raw_predictions_output, dict) and 'p50' in raw_predictions_output : # Previous structure
                # This path is less likely if self.model.predict() returns a numpy array (typical for Keras)
                # but kept for some compatibility if the custom model still outputs this dict.
                median_predictions_df = raw_predictions_output['p50']
                # Check if median_predictions_df is a DataFrame (expected by old code)
                if isinstance(median_predictions_df, pd.DataFrame):
                    prediction_cols = [col for col in median_predictions_df.columns if col.startswith('t+')]
                    predictions_array = median_predictions_df[prediction_cols].values
                    forecast_dates = median_predictions_df['forecast_time'].unique()
                    symbols = median_predictions_df['identifier'].unique()
                # If median_predictions_df is actually the numpy array of predictions directly:
                elif isinstance(median_predictions_df, np.ndarray):
                    predictions_array = median_predictions_df
                    # Placeholders for dates/symbols if not available in this dict structure
                    pred_horizon = predictions_array.shape[1] if predictions_array.ndim > 1 else predictions_array.shape[0]
                    forecast_dates = [pd.Timestamp.now() + timedelta(days=i) for i in range(pred_horizon)] # Placeholder
                    symbols = ["DUMMY_SYMBOL"] # Placeholder
                else:
                    raise TFTPredictorError(f"'p50' in raw_predictions_output is not DataFrame or ndarray: {type(median_predictions_df)}")

                result = {
                    'predictions': predictions_array,
                    'forecast_dates': forecast_dates,
                    'symbols': symbols,
                    'raw_predictions': raw_predictions_output
                }
                logger.info("Processed dict predictions (PyTorch Forecasting style).")
            else:
                raise TFTPredictorError(f"The custom model.predict() returned an unexpected type or structure: {type(raw_predictions_output)}. Content: {str(raw_predictions_output)[:200]}")

            return result
            
        except Exception as e:
            if not isinstance(e, TFTPredictorError):
                # Log the df shape and head that was passed to self.model.predict()
                logger.error(f"Error making predictions with df shape {df.shape if 'df' in locals() else 'df not defined'}. Head:\n{df.head() if 'df' in locals() and isinstance(df, pd.DataFrame) else 'N/A'}")
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

        # Ensure model is loaded
        if self.model is None:
            self.model = self._load_model()

        # Convert to numpy if it's a PyTorch tensor
        if hasattr(sequence_tensor, 'cpu') and hasattr(sequence_tensor, 'numpy'):
            model_input_np = sequence_tensor.cpu().numpy()
            logger.info(f"Converted to numpy with shape: {model_input_np.shape}")
        elif isinstance(sequence_tensor, np.ndarray):
            model_input_np = sequence_tensor
            logger.info(f"Input is already numpy with shape: {model_input_np.shape}")
        else:
            raise TFTPredictorError(f"Unsupported input type for sequence_tensor: {type(sequence_tensor)}")

        # Reshape if necessary (e.g. if it's 2D [seq_len, features] and model expects 3D [batch, seq_len, features])
        if model_input_np.ndim == 2:
            model_input_np = np.expand_dims(model_input_np, axis=0)
            logger.info(f"Expanded to 3D with shape: {model_input_np.shape}")
        elif model_input_np.ndim != 3:
            raise TFTPredictorError(f"Input numpy array must be 2D or 3D, got {model_input_np.ndim}D with shape {model_input_np.shape}")
        logger.info(f"Numpy array shape after initial processing: {model_input_np.shape}")

        # The model.predict method expects a DataFrame for its internal _prepare_inputs_from_normalized logic
        # We need to construct a DataFrame that TFTModel.predict can handle.
        # This requires knowing the feature names the model was trained on or expects.
        # This is a challenging part if feature names are not stored or passed correctly.

        # Assuming model_input_np is (batch, seq_len, num_features)
        # For now, we pass the numpy array directly, assuming TFTModel.predict can handle it
        # or that it will be wrapped appropriately by TFTModel.predict or its helpers.
        # The TFTModel.predict now uses _prepare_inputs_from_normalized which takes a DataFrame.
        # We need to create this DataFrame.

        num_features = model_input_np.shape[2]
        # Try to get feature names from model_config or generate generic ones
        feature_names = self.model_config.get("model_config", {}).get("input_feature_names", None)
        if not feature_names or len(feature_names) != num_features:
            logger.warning(f"Could not find explicit feature names in model_config.json. Generating generic feature names: feature_0 to feature_{num_features-1}. This is a fallback and might lead to errors if names don't match training.")
            feature_names = [f"feature_{i}" for i in range(num_features)]

        # We need to create a DataFrame for each item in the batch (currently batch_size=1 is assumed for prediction)
        # The DataFrame should have `self.context_length` rows and `num_features` columns.
        # The TFTModel.predict method and its helpers expect the DataFrame to have a specific structure,
        # including a time index and a group identifier if the model was trained with them.
        
        # For simplicity in this __call__ method, which receives a raw sequence tensor,
        # we'll make a DataFrame from the first (and only) item in the batch.
        # This assumes self.context_length matches model_input_np.shape[1]
        if model_input_np.shape[1] != self.context_length:
            # This should not happen if data prep upstream uses correct sequence length from config
            logger.error(f"CRITICAL: Input sequence length {model_input_np.shape[1]} does not match model context length {self.context_length}!")
            # Fallback or raise error - for now, will likely cause issues in TFTModel.predict
        
        # Create a DataFrame from the first sequence in the batch
        # This step matches how TFTModel.predict expects its input DataFrame based on its internal logic.
        # This needs self.context_length (num_encoder_steps) and feature_names.
        # The target column is also typically expected by the formatter inside TFTModel.predict.
        # For prediction, the target column in the input df to TFTModel.predict is usually a placeholder.
        
        df_for_model = pd.DataFrame(model_input_np[0], columns=feature_names)
        df_for_model['time_idx'] = np.arange(self.context_length)
        df_for_model['symbol'] = 'DUMMY_GROUP_0' # Placeholder group ID
        df_for_model['target'] = 0.0 # Placeholder target
        
        logger.info(f"[TFTPredictor] Constructed DataFrame for model prediction with shape: {df_for_model.shape}")
        df_dtypes_str = df_for_model.dtypes.to_string()
        logger.info(f"[TFTPredictor] Dtypes of constructed DataFrame:\n{df_dtypes_str}")

        raw_predictions = self.model.predict(df_for_model) # TFTModel.predict method
        # raw_predictions shape is (batch_size, self.context_length + self.prediction_horizon, 1)
        # e.g. (1, 30 + 5, 1) = (1, 35, 1)
        
        # Extract the first future prediction step
        # The future predictions start after self.context_length (num_encoder_steps)
        if raw_predictions.shape[0] > 0 and raw_predictions.shape[1] > self.context_length:
            prediction = raw_predictions[0, self.context_length, 0]
        else:
            logger.error(f"Raw predictions shape {raw_predictions.shape} is not as expected for extracting future prediction at index {self.context_length}.")
            prediction = np.nan # Fallback to nan if shape is unexpected
        
        logger.info(f"Extracted prediction value: {prediction} from result shape {raw_predictions.shape}")

        # For now, uncertainty is a placeholder
        # TODO: Implement uncertainty extraction if model provides it
        uncertainty = 0.01  # Placeholder
        logger.info(f"Prediction: {prediction}, Uncertainty: {uncertainty}")
        
        return prediction, uncertainty 