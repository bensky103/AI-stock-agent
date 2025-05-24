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
from typing import Optional

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
        # Placeholder for getting attention weights if implemented
        return np.array([])

    def __call__(self, features_df: pd.DataFrame, symbol: Optional[str] = None):
        """
        Make predictions using the TFT model from a feature DataFrame.

        Args:
            features_df (pd.DataFrame): DataFrame containing all necessary features, 
                                        scaled and processed by FeatureEngineer. 
                                        The DataFrame should have enough rows to extract 
                                        a sequence of `context_length`.
            symbol (Optional[str]): Symbol for context, mainly for logging.

        Returns:
            np.ndarray: Prediction array, typically (prediction_horizon,)
            
        Raises:
            TFTPredictorError: If prediction fails
        """
        if self.model is None:
            logger.info("Keras model not loaded. Loading now...")
            self.model = self._load_model() # Ensure Keras model is loaded

        if features_df.empty:
            raise TFTPredictorError(f"Input features_df is empty for symbol '{symbol}'.")

        if len(features_df) < self.context_length:
            raise TFTPredictorError(
                f"Insufficient rows in features_df for symbol '{symbol}' to form a sequence. "
                f"Got {len(features_df)}, need {self.context_length}."
            )

        # Use StockFormatter to get the canonical list of features expected by Keras TFTModel
        # The Keras TFTModel derives its num_historical_features etc. from this.
        from colab_training.data_formatters.stock_formatter import StockFormatter, InputTypes
        formatter = StockFormatter() # Uses its internal column_definition

        # 1. Prepare Historical Inputs (the (1, sequence_length, num_historical_features) array)
        # These are features of type TARGET, OBSERVED, KNOWN in StockFormatter.column_definition
        # Their order is derived from StockFormatter.column_definition
        historical_feature_names = [
            col for col, type_ in formatter.column_definition 
            if type_ in [InputTypes.TARGET, InputTypes.OBSERVED, InputTypes.KNOWN]
        ]
        
        num_expected_hist_features = self.model.input_shape[1][-1] # From Keras model: (batch, seq_len, num_hist_features)
        
        if len(historical_feature_names) != num_expected_hist_features:
            logger.warning(
                f"Number of historical features from StockFormatter ({len(historical_feature_names)}) "
                f"does not match Keras model expected historical features ({num_expected_hist_features}). "
                f"StockFormatter historical features: {historical_feature_names[:5]}..."
            )
            # This is a critical mismatch if it occurs, means Keras model init vs. this list are out of sync.
            # For now, we will trust num_expected_hist_features and try to select if possible, 
            # but ideally these should always match.

        # Select the last `self.context_length` rows for the input sequence
        sequence_df = features_df.iloc[-self.context_length:]

        # Ensure all historical_feature_names are in sequence_df
        missing_hist_cols = [col for col in historical_feature_names if col not in sequence_df.columns]
        if missing_hist_cols:
            raise TFTPredictorError(
                f"Missing historical features in input DataFrame for symbol '{symbol}': {missing_hist_cols}. "
                f"Available columns: {sequence_df.columns.tolist()}"
            )
        
        # Select and order historical features
        # If len(historical_feature_names) > num_expected_hist_features, we must only pick the ones Keras wants.
        # However, the Keras model itself determines num_historical_features from the *full* list from StockFormatter.
        # So, historical_feature_names should be the definitive list.
        historical_inputs_df = sequence_df[historical_feature_names]
        historical_inputs_np = historical_inputs_df.to_numpy().astype(np.float32)
        historical_inputs_np = historical_inputs_np.reshape(1, self.context_length, len(historical_feature_names))

        # 2. Prepare Static Inputs
        static_feature_names = [
            col for col, type_ in formatter.column_definition if type_ == InputTypes.STATIC
        ]
        num_expected_static_features = self.model.input_shape[0][-1]

        static_inputs_list = []
        if static_feature_names:
            # For static features, take the values from the *last* row of the input df (or any, should be constant for the sequence)
            for sf_name in static_feature_names:
                if sf_name in features_df.columns:
                    static_inputs_list.append(features_df[sf_name].iloc[-1]) 
                else:
                    logger.warning(f"Static feature '{sf_name}' not found in features_df for '{symbol}'. Using 0.")
                    static_inputs_list.append(0) # Default to 0 if missing
        
        if len(static_inputs_list) != num_expected_static_features:
            logger.warning(
                f"Number of static features prepared ({len(static_inputs_list)}) from StockFormatter "
                f"does not match Keras model expected static features ({num_expected_static_features}). Using zeros if needed."
            )
            # Pad with zeros or truncate if mismatch, or rely on Keras error
            # Forcing to the Keras expected shape with zeros for missing ones.
            final_static_list = [0.0] * num_expected_static_features
            for i in range(min(len(static_inputs_list), num_expected_static_features)):
                final_static_list[i] = static_inputs_list[i]
            static_inputs_np = np.array(final_static_list).astype(np.float32).reshape(1, num_expected_static_features)
        else:
            static_inputs_np = np.array(static_inputs_list).astype(np.float32).reshape(1, num_expected_static_features)

        # 3. Prepare Future Inputs (Known Future Inputs)
        # These are features of type KNOWN in StockFormatter.column_definition
        # For this Keras model, num_future_features is likely 0 from logs `Future=(1, 5, 0)`
        future_feature_names = [
            col for col, type_ in formatter.column_definition if type_ == InputTypes.KNOWN
        ]
        # Filter out those already in historical_feature_names to avoid duplication if a KNOWN is also OBSERVED/TARGET
        # However, the Keras TFTModel defines num_future_features independently based on KNOWN only.
        # The `historical_inputs` layer uses TARGET, OBSERVED, KNOWN.
        # The `future_inputs` layer uses only KNOWN.
        # This implies that if a feature is KNOWN, it can appear in *both* historical (as past known values) and future inputs (as future known values).
        # For prediction, we only have historical data up to 'now'. We don't typically have *actual* future knowns unless they are like 'day_of_week' for future dates.
        # If FeatureEngineer doesn't produce true future values for these, this array will be based on past values or zeros.

        num_expected_future_features = self.model.input_shape[2][-1] # (batch, horizon, num_future_features)
        
        # For prediction, we might not have actual future values for these knowns. 
        # The Keras model might be trained to handle this (e.g. if future knowns are only for training targets).
        # If num_expected_future_features is 0, this array is just a placeholder.
        if num_expected_future_features > 0:
            logger.warning(
                f"Keras model expects {num_expected_future_features} future_features, but constructing them for prediction is complex "
                f"unless they are trivial (e.g., time flags). For now, using zeros or last known values. "
                f"Future feature names from StockFormatter (type KNOWN): {future_feature_names}"
            )
            # Create placeholder future inputs: (1, prediction_horizon, num_future_features)
            # If these features are genuinely needed and are not just time flags, this needs more sophisticated handling.
            future_inputs_np = np.zeros((1, self.prediction_horizon, num_expected_future_features), dtype=np.float32)
            for i, f_name in enumerate(future_feature_names):
                if i < num_expected_future_features: # Ensure we don't exceed Keras expected dimension
                    if f_name in features_df.columns:
                        # Repeat last known value for all future steps - very basic imputation
                        last_known_val = features_df[f_name].iloc[-1]
                        future_inputs_np[0, :, i] = last_known_val
                    else:
                        logger.warning(f"Future feature '{f_name}' not in features_df for '{symbol}'. Using 0 for future inputs.")
                        future_inputs_np[0, :, i] = 0 # Default to 0
        else: # num_expected_future_features is 0
            future_inputs_np = np.zeros((1, self.prediction_horizon, 0), dtype=np.float32)


        logger.info(f"Prepared inputs for Keras model '{symbol}': "
                    f"Static shape: {static_inputs_np.shape}, "
                    f"Historical shape: {historical_inputs_np.shape}, "
                    f"Future shape: {future_inputs_np.shape}")

        try:
            # Ensure the Keras model's predict or __call__ can handle this input structure.
            # The Keras TFTModel has a `predict(self, df)` which calls `_prepare_inputs`.
            # We are bypassing that internal _prepare_inputs and constructing the arrays directly.
            # The Keras Model itself (tf.keras.Model) is callable.
            raw_predictions = self.model.predict([static_inputs_np, historical_inputs_np, future_inputs_np])
            
            # Assuming raw_predictions is (batch_size, prediction_horizon) or (batch_size, prediction_horizon, 1)
            # For a single prediction, batch_size is 1. We want (prediction_horizon,)
            if raw_predictions.ndim == 3 and raw_predictions.shape[0] == 1 and raw_predictions.shape[2] == 1:
                processed_predictions = raw_predictions.reshape(self.prediction_horizon)
            elif raw_predictions.ndim == 2 and raw_predictions.shape[0] == 1:
                processed_predictions = raw_predictions.reshape(self.prediction_horizon)
            else:
                logger.error(f"Unexpected prediction output shape from Keras model for '{symbol}': {raw_predictions.shape}")
                raise TFTPredictorError("Unexpected Keras prediction output shape")
            
            logger.info(f"Raw prediction for '{symbol}' (first 5): {processed_predictions[:5]}")
            return processed_predictions
            
        except Exception as e:
            logger.error(f"Error during Keras model prediction for '{symbol}': {e}")
            logger.error(f"Historical input sample (first 5 features, first 3 steps):\n{historical_inputs_np[0, :3, :5] if historical_inputs_np.size > 0 else 'N/A'}")
            raise TFTPredictorError(f"Keras model prediction failed for '{symbol}': {e}")

    def predict_actual(self, input_sequence: np.ndarray, symbol: Optional[str] = None) -> np.ndarray:
        """
        DEPRECATED: This method is kept for now if other parts of the system call it directly with a NumPy array.
        The main prediction path should now use __call__ with a DataFrame.
        """
        logger.warning(
            f"[{self.__class__.__name__}] predict_actual is DEPRECATED. Use __call__ with a DataFrame. "
            f"Symbol: {symbol}, Input shape: {input_sequence.shape if input_sequence is not None else 'None'}"
        )
        # Return an empty array or an array of NaNs with the expected prediction_horizon length
        # to avoid breaking downstream code that expects a certain shape.
        # self.prediction_horizon should be available from _load_config()
        return np.full(self.prediction_horizon if hasattr(self, 'prediction_horizon') else 5, np.nan)

    # Placeholder for potential future methods or end of class

    # Placeholder for getting attention weights if implemented
    def get_attention_weights(self, inputs) -> np.ndarray:
        return np.array([]) 