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
                model.load(self.model_path)
                
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
            # Ensure model config is loaded to get necessary column names etc.
            if not hasattr(self, 'model_config') or not self.model_config:
                self._load_config() # Loads self.model_config
            if not hasattr(self, 'model') or self.model is None: # Ensure model is loaded to access hparams if needed by _load_config or here
                self._load_model()

            # Attempt to get column name configurations from self.model_config["model_config"]
            m_config = self.model_config.get("model_config", {})
            logger.info(f"[{self.__class__.__name__}] Loaded m_config from model_config.json. Keys: {list(m_config.keys()) if isinstance(m_config, dict) else 'm_config is not a dict'}")
            if isinstance(m_config, dict):
                logger.debug(f"[{self.__class__.__name__}] m_config content: {m_config}")
            
            expected_feature_names = m_config.get("time_varying_unknown_reals", 
                                       m_config.get("input_obs_loc", 
                                       m_config.get("input_feature_names", [])))
            
            if not expected_feature_names and n_features > 0:
                logger.warning(
                    f"Could not find explicit feature names in model_config.json. "
                    f"Generating generic feature names: feature_0 to feature_{n_features-1}. "
                    "This is a fallback and might lead to errors if names don't match training."
                )
                expected_feature_names = [f'feature_{k}' for k in range(n_features)]
            
            if len(expected_feature_names) != n_features:
                raise TFTPredictorError(
                    f"Mismatch between number of features in input sequence ({n_features}) "
                    f"and expected feature names in model_config.json ({len(expected_feature_names)}: {expected_feature_names})."
                )

            gid_col_name = m_config.get("identifier_col_name", "symbol")
            tid_col_name = m_config.get("time_idx_col_name", "time_idx")
            
            target_col_name_config = m_config.get("target_col_name", "target")
            if isinstance(target_col_name_config, str):
                target_names = [target_col_name_config]
            elif isinstance(target_col_name_config, list):
                target_names = target_col_name_config
            else:
                target_names = ["target"]

            all_dfs_data_for_predict_call = []
            for i in range(batch_size):
                df_dict = {}
                df_dict[tid_col_name] = np.arange(seq_len)
                df_dict[gid_col_name] = [f'DUMMY_GROUP_{i}'] * seq_len
                
                for feature_idx, feature_name in enumerate(expected_feature_names):
                    df_dict[feature_name] = sequence_np[i, :, feature_idx]
                    
                for target_name in target_names:
                    if target_name not in df_dict:
                        df_dict[target_name] = np.zeros(seq_len)

                current_sequence_df = pd.DataFrame(df_dict)
                all_dfs_data_for_predict_call.append(current_sequence_df)
            
            final_df_for_model = pd.concat(all_dfs_data_for_predict_call).reset_index(drop=True)

            logger.info(f"[{self.__class__.__name__}] Constructed DataFrame for model prediction with shape: {final_df_for_model.shape}")
            logger.info(f"[{self.__class__.__name__}] Dtypes of constructed DataFrame:\n{final_df_for_model.dtypes}")
            logger.debug(f"[{self.__class__.__name__}] DataFrame head for model:\n{final_df_for_model.head()}")

            # Load model if not already loaded (moved from self.predict to here)
            if self.model is None:
                self.model = self._load_model()

            # Pass the fully constructed DataFrame representing the sequence(s)
            # directly to the custom model's predict method.
            result = self.model.predict(final_df_for_model)
            
            if isinstance(result, np.ndarray):
                if result.ndim == 1:
                    predictions_val = result[0]
                elif result.ndim == 2:
                    if result.shape[0] == batch_size:
                         predictions_val = result[0, 0]
                    else:
                         predictions_val = result[0,0]
                elif result.ndim == 3:
                    predictions_val = result[0, 0, 0]
                else:
                    raise TFTPredictorError(f"Unexpected prediction result ndim: {result.ndim}, shape: {result.shape}")
                logger.info(f"Extracted prediction value: {predictions_val} from result shape {result.shape}")
            elif isinstance(result, dict) and 'predictions' in result :
                predictions_val = result['predictions'][0,0]
                logger.info(f"Extracted prediction value from dict result: {predictions_val}")
            else:
                raise TFTPredictorError(f"Prediction result from custom model is of unexpected type: {type(result)}. Expected numpy array or dict with 'predictions'.")

            uncertainty_val = 0.01 # Placeholder

            logger.info(f"Prediction: {predictions_val}, Uncertainty: {uncertainty_val}")
            
            import torch
            predictions_tensor = torch.tensor([[predictions_val]], dtype=torch.float32)
            uncertainty_tensor = torch.tensor([[uncertainty_val]], dtype=torch.float32)
            
            return predictions_tensor, uncertainty_tensor
            
        except Exception as e:
            logger.error(f"Error in TFTPredictor call: {str(e)}")
            logger.error(f"Input sequence_np shape: {sequence_np.shape if 'sequence_np' in locals() else 'not defined'}")
            logger.error(f"Constructed final_df_for_model (sample):\n{final_df_for_model.head() if 'final_df_for_model' in locals() else 'not defined'}")
            raise TFTPredictorError(f"Failed in TFTPredictor call: {str(e)}") 