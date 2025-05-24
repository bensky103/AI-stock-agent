"""Prediction engine module for stock price prediction.

This module provides:
1. Enhanced stock predictor with TFT model
2. Feature engineering and sequence preparation
3. Model training and prediction pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import pytz
import json
import torch
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.trial import Trial
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yaml
import sys

from .feature_engineering import FeatureEngineer
from .sequence_preprocessor import SequencePreprocessor
from .scaler_handler import ScalerHandler, ScalerHandlerError
from .tft_predictor import TFTPredictor, TFTPredictorError
from data_input.market_feed import get_market_data
from data_input.sentiment_manager import get_sentiment_data

class StockPredictorError(Exception):
    """Exception class for stock predictor errors."""
    pass

# Configure logging
logger = logging.getLogger(__name__)
# Don't configure handlers here, just get the logger
logger.setLevel(logging.INFO)  # Change from WARNING to INFO for our debug logs

class EnhancedStockPredictor:
    """
    Enhanced stock predictor using TFT model.
    
    This class provides a complete pipeline for stock price prediction using
    the Temporal Fusion Transformer (TFT) model, including:
    1. Feature engineering and sequence preparation
    2. Data scaling and normalization
    3. Model training and evaluation
    4. Prediction generation with uncertainty estimates
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        prediction_horizon: int = 5,
        device: str = 'cpu',
        model_type: str = 'tft',
        use_feature_selection: bool = True,
        use_pca: bool = False,
        detect_regime: bool = True
    ):
        """
        Initialize the enhanced stock predictor.
        
        Args:
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict
            device: Device to use for model ('cpu' or 'cuda')
            model_type: Type of model to use (only 'tft' supported)
            use_feature_selection: Whether to use feature selection
            use_pca: Whether to use PCA for dimensionality reduction
            detect_regime: Whether to detect market regimes
        """
        if model_type != 'tft':
            raise ValueError("Only 'tft' model type is supported")
        
        # Set up logger for this class instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set to INFO to see debug messages
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.model_type = model_type
        self.model = None  # Will be initialized when loading a model
        self.models = {}  # Dictionary to store models by symbol
        self.default_model_dir = Path("colab_training/tft_model")
        self.saved_models_dir = Path("saved_models")
        self.saved_models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            use_feature_selection=use_feature_selection,
            use_pca=use_pca,
            detect_regime=detect_regime
        )
        self.sequence_preprocessor = SequencePreprocessor(sequence_length)
        self.scaler_handler = ScalerHandler(model_type='tft')
        
        # Load global training config
        config_path = self.default_model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.training_config = yaml.safe_load(f)
            logger.info("Loaded global training configuration")
        else:
            logger.warning("No global training configuration found")
            self.training_config = {}
        
        logger.info(
            f"Initialized enhanced stock predictor with {model_type} model, "
            f"sequence length {sequence_length}, prediction horizon {prediction_horizon}"
        )
    
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a trained model from disk."""
        try:
            if model_path is None:
                model_path = self.default_model_dir
            config_path = model_path / "config.yaml"
            
            # Check paths first and raise StockPredictorError directly
            if not model_path.exists():
                raise StockPredictorError(f"Model path does not exist: {model_path}")
            if not config_path.exists():
                raise StockPredictorError(f"Config path does not exist: {config_path}")
            
            try:
                self.model = TFTPredictor(
                    model_path=str(model_path),
                    config_path=str(config_path)
                )
                self.logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                self.logger.error(f"Error initializing model: {str(e)}")
                raise StockPredictorError(f"Failed to initialize model: {str(e)}")
            
        except StockPredictorError:
            # Re-raise StockPredictorError without wrapping
            raise
        except Exception as e:
            # Wrap other unexpected errors
            self.logger.error(f"Unexpected error loading model: {str(e)}")
            raise StockPredictorError(f"Failed to load model: {str(e)}")
    
    def predict(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Make predictions for a list of symbols.
        
        Args:
            symbols: List of stock symbols to predict
            start_date: Optional start date for fetching historical data
            end_date: Optional end date for fetching historical data
            
        Returns:
            Dictionary with predictions for each symbol
        """
        predictions: Dict[str, Dict[str, Union[float, str]]] = {}
        
        for symbol in symbols:
            self.logger.info(f"[{self.__class__.__name__}] ===== Starting Prediction for: {symbol} =====")
            try:
                # Ensure data is up-to-date for prediction
                # Using a recent period that should cover self.sequence_length
                # The actual length of data needed will be sequence_length for feature prep.
                fetch_end_date = datetime.now(pytz.utc)
                # Max lookback from FeatureEngineer for SMAs/EMAs is 200. Sequence length is 30.
                # Need at least 200 (max_lookback) + 30 (sequence_length) = 230 days.
                # Add buffer for non-trading days and to be safe.
                required_days_for_features_and_sequence = 200 + self.feature_engineer.sequence_length + 50 # 200 (max_lookback) + seq_len + buffer
                fetch_start_date_predict = fetch_end_date - timedelta(days=required_days_for_features_and_sequence) 
                
                self.logger.info(f"[{self.__class__.__name__}] Fetching latest raw market data for {symbol} to build the prediction input sequence...")
                latest_data_df = get_market_data(
                    [symbol],
                    start_date=fetch_start_date_predict.strftime('%Y-%m-%d'),
                    end_date=fetch_end_date.strftime('%Y-%m-%d')
                )
                
                if latest_data_df.empty:
                    self.logger.warning(f"[{self.__class__.__name__}] No raw market data was fetched for {symbol}. Cannot make a prediction.")
                    predictions[symbol] = {"error": "No data available"}
                    continue
                
                last_available_date_str = 'N/A'
                if not latest_data_df.empty:
                    last_index_item = latest_data_df.index[-1]
                    # market_feed._process_data for a single symbol in a list returns an index of (Symbol, Date)
                    # So last_index_item would be like ('TSLA', TimestampObject)
                    date_obj_to_format = None
                    if isinstance(last_index_item, tuple) and len(last_index_item) == 2:
                        date_obj_to_format = last_index_item[1] # Get the TimestampObject
                    elif hasattr(last_index_item, 'strftime'): # If it's already a Timestamp (e.g., DatetimeIndex)
                        date_obj_to_format = last_index_item
                    
                    if hasattr(date_obj_to_format, 'strftime'):
                        last_available_date_str = date_obj_to_format.strftime('%Y-%m-%d')
                    elif date_obj_to_format is not None:
                        self.logger.warning(f"[{self.__class__.__name__}] Date object for formatting (last available date) is not a Timestamp: {type(date_obj_to_format)}. Logging as string: {str(date_obj_to_format)}")
                        last_available_date_str = str(date_obj_to_format)
                    else:
                        self.logger.warning(f"[{self.__class__.__name__}] Could not determine a valid date object from last_index_item: {last_index_item}")

                self.logger.info(f"[{self.__class__.__name__}] Successfully fetched {latest_data_df.shape[0]} data points for {symbol}. Last available date from raw fetch: {last_available_date_str}.")
                
                # The fetched data might have a MultiIndex [('symbol', 'date')]
                # Select data for the current symbol if it's a MultiIndex
                if isinstance(latest_data_df.index, pd.MultiIndex):
                    latest_data_df = latest_data_df.xs(symbol, level=0) # Assuming symbol is the first level
                
                self.logger.info(f"[{self.__class__.__name__}] Shape of latest_data_df for {symbol} after potential .xs(): {latest_data_df.shape}")
                if latest_data_df.shape[0] < self.sequence_length:
                    self.logger.warning(f"[{self.__class__.__name__}] Insufficient data for {symbol} to form a sequence. Has {latest_data_df.shape[0]}, needs {self.sequence_length}. Skipping prediction.")
                    predictions[symbol] = {"error": "Insufficient data for sequence"}
                    continue

                # Check if scaler is fitted. If not, fit it using the historical data available for this symbol.
                if not self.feature_engineer.is_scaler_fitted(symbol=symbol):
                    self.logger.info(f"[{self.__class__.__name__}] Data scaler for {symbol} (or global) not fitted. Attempting to fit it now using a larger historical dataset.")
                    
                    # Fetch a larger dataset for scaler fitting
                    fetch_start_date_scaler_fit = fetch_end_date - timedelta(days=365 * 2) # Approx 2 years for robust scaler fitting
                    self.logger.info(f"[{self.__class__.__name__}] Fetching extended historical data for {symbol} for scaler fitting: {fetch_start_date_scaler_fit.strftime('%Y-%m-%d')} to {fetch_end_date.strftime('%Y-%m-%d')}")
                    historical_data_for_scaler = get_market_data(
                        [symbol],
                        start_date=fetch_start_date_scaler_fit.strftime('%Y-%m-%d'),
                        end_date=fetch_end_date.strftime('%Y-%m-%d')
                    )

                    if isinstance(historical_data_for_scaler.index, pd.MultiIndex):
                        historical_data_for_scaler = historical_data_for_scaler.xs(symbol, level=0)

                    if historical_data_for_scaler.empty or historical_data_for_scaler.shape[0] < self.feature_engineer.sequence_length: # Basic check
                        self.logger.error(f"[{self.__class__.__name__}] Insufficient historical data fetched for scaler fitting for {symbol} (shape: {historical_data_for_scaler.shape}). Cannot fit scaler.")
                        predictions[symbol] = {"error": "Insufficient data for scaler fitting."}
                        continue

                    self.logger.info(f"[{self.__class__.__name__}] Shape of historical_data_for_scaler for {symbol} (for fitting): {historical_data_for_scaler.shape}")

                    # 1. Calculate TAs on data intended for scaler fitting.
                    df_with_ta_for_fitting, _ = self.feature_engineer.calculate_technical_indicators(
                        historical_data_for_scaler.copy(), 
                        symbol=symbol, 
                        fit_scalers=True 
                    )
                    
                    # 2. Fit the scaler if TA calculation was successful and produced data.
                    if df_with_ta_for_fitting is not None and not df_with_ta_for_fitting.empty:
                        self.logger.info(f"[{self.__class__.__name__}] Fitting scaler for {symbol} with TA-data (shape {df_with_ta_for_fitting.shape}).")
                        try:
                            self.feature_engineer.fit_scaler(df_with_ta_for_fitting, symbol=symbol)
                            self.logger.info(f"[{self.__class__.__name__}] Scaler fitted and stored for {symbol}.")
                        except ScalerHandlerError as e_sh: # Catch potential errors from fit_scaler
                            self.logger.error(f"[{self.__class__.__name__}] Failed to fit scaler for {symbol}: {e_sh}")
                            predictions[symbol] = {"error": f"Scaler fitting failed: {e_sh}"}
                            continue
                    else:
                        self.logger.warning(f"[{self.__class__.__name__}] Data for scaler fitting for {symbol} is None or empty after TA calculation. Scaler not fitted.")
                        predictions[symbol] = {"error": "Data for scaler fitting empty/None post-TA."}
                        continue
                else:
                    self.logger.info(f"[{self.__class__.__name__}] Scaler for {symbol} (or global) is already fitted. Proceeding with feature preparation.")

                # Prepare features for the latest data
                self.logger.info(f"[{self.__class__.__name__}] Preparing input features for {symbol} from the latest data (current shape: {latest_data_df.shape}). This involves calculating indicators and then scaling.")
                
                prediction_input_sequence_np, last_close_for_pred = self.feature_engineer.prepare_features_for_prediction(
                    raw_data=latest_data_df.copy(), 
                    symbol=symbol
                )
                
                self.logger.info(f"[{self.__class__.__name__}] Features for {symbol} prepared by prepare_features_for_prediction. Resulting NumPy array shape: {prediction_input_sequence_np.shape if isinstance(prediction_input_sequence_np, np.ndarray) else type(prediction_input_sequence_np)}.")

                if not isinstance(prediction_input_sequence_np, np.ndarray) or prediction_input_sequence_np.size == 0:
                    self.logger.warning(f"[{self.__class__.__name__}] Feature preparation for {symbol} (prepare_features_for_prediction) returned empty or invalid result. Shape: {prediction_input_sequence_np.shape if isinstance(prediction_input_sequence_np, np.ndarray) else 'N/A'}. Skipping prediction.")
                    predictions[symbol] = {"error": "Feature preparation (predict path) failed"}
                    continue

                last_sequence_np = None
                if isinstance(prediction_input_sequence_np, np.ndarray):
                    if prediction_input_sequence_np.ndim == 2: # Expected: (sequence_length, num_features)
                        last_sequence_np = prediction_input_sequence_np
                        self.logger.info(f"[{self.__class__.__name__}] Using the 2D sequence from prepare_features_for_prediction directly for {symbol}. Shape: {last_sequence_np.shape}.")
                    else:
                        self.logger.error(f"[{self.__class__.__name__}] prepare_features_for_prediction for {symbol} returned unexpected ndim: {prediction_input_sequence_np.ndim}, shape: {prediction_input_sequence_np.shape}. Expected 2D. Skipping.")
                        predictions[symbol] = {"error": "Features from prep_for_pred have unexpected dimensions"}
                        continue
                else:
                    self.logger.error(f"[{self.__class__.__name__}] prediction_input_sequence_np is not an ndarray after prepare_features_for_prediction for {symbol}.")
                    predictions[symbol] = {"error": "Feature preparation (predict path) failed - not an array"}
                    continue

                if last_sequence_np is None: # Should be caught by earlier checks, but as a safeguard
                     self.logger.error(f"[{self.__class__.__name__}] last_sequence_np is None for {symbol} after processing. Skipping.")
                     predictions[symbol] = {"error": "Could not derive last_sequence_np for model input."}
                     continue

                # Ensure data has the right dimensionality for the model input tensor
                # Standard TFT models expect input of shape [batch_size, sequence_length, num_features]
                # last_sequence_np is currently (sequence_length, num_features)
                if last_sequence_np.ndim == 2:
                    model_input_np = np.expand_dims(last_sequence_np, axis=0) # Add batch dimension -> (1, sequence_length, num_features)
                    self.logger.info(f"[{self.__class__.__name__}] Added batch dimension to the sequence for {symbol}. Model input NumPy array shape: {model_input_np.shape} (batch_size, sequence_length, num_features).")
                else:
                    # This case should not be reached if above logic is correct
                    self.logger.error(f"[{self.__class__.__name__}] last_sequence_np for {symbol} is not 2D after selection. Shape: {last_sequence_np.shape}. Skipping.")
                    predictions[symbol] = {"error": "Processed sequence is not 2D"}
                    continue
                
                # Convert to tensor and move to device
                # This assumes PyTorch. If TensorFlow, use tf.convert_to_tensor.
                # For TF, device placement is usually handled by TF itself or via tf.device context.
                if "torch" in sys.modules:
                    import torch
                    model_input_tensor = torch.FloatTensor(model_input_np).to(self.device)
                    self.logger.info(f"[{self.__class__.__name__}] Converted NumPy input sequence to PyTorch tensor for {symbol}. Shape: {model_input_tensor.shape}, Device: {model_input_tensor.device}. This is ready for the model.")
                elif "tensorflow" in sys.modules:
                    import tensorflow as tf
                    # For TensorFlow, ensure data type matches model expectation (e.g., tf.float32)
                    model_input_tensor = tf.convert_to_tensor(model_input_np, dtype=tf.float32)
                    self.logger.info(f"[{self.__class__.__name__}] Converted NumPy input sequence to TensorFlow tensor for {symbol}. Shape: {model_input_tensor.shape}. This is ready for the model.")
                else:
                    self.logger.warning("[{self.__class__.__name__}] Neither PyTorch nor TensorFlow found. Passing NumPy array directly to the model for {symbol}.")
                    model_input_tensor = model_input_np # Pass as numpy if framework unclear

                # Make prediction
                try:
                    model_to_use = self._get_or_load_model(symbol)
                    self.logger.info(f"[{self.__class__.__name__}] Passing the prepared input tensor to the underlying model ({type(model_to_use).__name__ if model_to_use is not None else 'N/A'}) for {symbol} to get forecasts...")
                    
                    if model_to_use is None:
                        self.logger.error(f"[{self.__class__.__name__}] No model found for {symbol} or default via _get_or_load_model. Skipping.")
                        predictions[symbol] = {"error": f"Model not found for {symbol}"}
                        continue
                        
                    self.logger.info(f"[{self.__class__.__name__}] Using model type: {type(model_to_use).__name__} for {symbol}")
                    # TFTPredictor.__call__ now returns an array of scaled predictions and an array of uncertainties
                    scaled_predictions_array, uncertainties_array = model_to_use(model_input_tensor)
                    
                    self.logger.info(f"[{self.__class__.__name__}] Model ({type(model_to_use).__name__}) for {symbol} has returned scaled predictions. Number of steps predicted: {len(scaled_predictions_array) if isinstance(scaled_predictions_array, (np.ndarray, list)) else 'N/A'}.")
                    
                    denormalized_predictions_list = []
                    if isinstance(scaled_predictions_array, np.ndarray) and self.feature_engineer.target_scaler_params is not None:
                        self.logger.info(f"[{self.__class__.__name__}] Denormalizing the {len(scaled_predictions_array)} scaled prediction steps for {symbol}...")
                        for i, scaled_pred in enumerate(scaled_predictions_array):
                            if scaled_pred is not np.nan:
                                denormalized_pred = self.feature_engineer.inverse_transform_target(scaled_pred, symbol=symbol)
                                denormalized_predictions_list.append(denormalized_pred)
                                self.logger.info(f"  [{self.__class__.__name__}] Step T+{i+1} for {symbol}: Scaled = {scaled_pred:.4f} -> Denormalized = {denormalized_pred:.2f}")
                            else:
                                denormalized_predictions_list.append(np.nan)
                                self.logger.warning(f"  [{self.__class__.__name__}] Step T+{i+1} for {symbol}: Scaled prediction is NaN. Cannot denormalize.")
                    elif not isinstance(scaled_predictions_array, np.ndarray):
                        self.logger.error(f"[{self.__class__.__name__}] Scaled predictions for {symbol} is not a numpy array: {type(scaled_predictions_array)}. Cannot process.")
                        denormalized_predictions_list = [np.nan] * self.prediction_horizon # Fallback
                    else: # target_scaler_params is None or other issue
                        self.logger.warning(f"[{self.__class__.__name__}] Target scaler parameters are not available for {symbol}, or scaled predictions are not in the expected array format. Using raw scaled predictions: {scaled_predictions_array}")
                        # Convert to list if it's an ndarray, otherwise, it might be a single nan or similar
                        if isinstance(scaled_predictions_array, np.ndarray):
                            denormalized_predictions_list = scaled_predictions_array.tolist()
                        else: # If it was a single NaN from TFTPredictor fallback
                             denormalized_predictions_list = [scaled_predictions_array] * self.prediction_horizon

                    self.logger.info(f"[{self.__class__.__name__}] Final denormalized predictions for {symbol} ({self.prediction_horizon} steps): {denormalized_predictions_list}")
                    predictions[symbol] = {
                        "predictions": denormalized_predictions_list, # List of floats
                        "uncertainties": uncertainties_array, # List of floats
                        "predicted_at": datetime.now(pytz.utc).isoformat(),
                        "last_actual_date": last_available_date_str
                    }
                    
                except ValueError as ve:
                    # This is where the "Data must be 1-dimensional" error is caught
                    self.logger.error(f"[{self.__class__.__name__}] ValueError during model call for {symbol}: {str(ve)}")
                    self.logger.error(f"[{self.__class__.__name__}] Model input tensor shape at error: {model_input_tensor.shape if 'model_input_tensor' in locals() and hasattr(model_input_tensor, 'shape') else 'N/A'}")
                    self.logger.error(f"[{self.__class__.__name__}] Type of model_input_tensor: {type(model_input_tensor) if 'model_input_tensor' in locals() else 'N/A'}")
                    # Log details of the exception for more context
                    self.logger.exception(f"[{self.__class__.__name__}] Full exception details for {symbol}:")
                    predictions[symbol] = {"error": f"ValueError in model: {str(ve)}"}
                except Exception as e:
                    self.logger.error(f"[{self.__class__.__name__}] Unexpected error during model prediction for {symbol}: {str(e)}")
                    self.logger.exception(f"[{self.__class__.__name__}] Full exception details for {symbol}:")
                    predictions[symbol] = {"error": f"General error in model: {str(e)}"}

            except Exception as e:
                self.logger.error(f"[{self.__class__.__name__}] Error processing symbol {symbol}: {str(e)}")
                self.logger.exception(f"[{self.__class__.__name__}] Full exception details for {symbol} processing:")
                predictions[symbol] = {"error": str(e)}
                
        return predictions

    def train(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        epochs: int = 50,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Train models for the given symbols.
        
        Args:
            symbols: List of stock symbols to train models for
            start_date: Start date for training data
            end_date: End date for training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Optional path to save trained models
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history for each symbol
        """
        history = {}
        
        # Get market data for all symbols
        market_data = get_market_data(symbols, start_date, end_date)
        if market_data.empty:
            raise ValueError("No market data available for training")
        
        # Get sentiment data if available
        try:
            sentiment_data = get_sentiment_data(symbols, start_date, end_date)
        except Exception as e:
            logger.warning(f"Could not fetch sentiment data: {str(e)}")
            sentiment_data = None
        
        # Train model for each symbol
        for symbol in symbols:
            try:
                logger.info(f"Training model for {symbol}")
                
                # Get symbol-specific data
                symbol_data = market_data[market_data['Symbol'] == symbol].copy()
                if symbol_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Prepare features
                features, sentiment_features = self.feature_engineer.prepare_prediction_data(
                    symbol_data,
                    sentiment_data[symbol] if sentiment_data is not None else None
                )
                
                if features is None or len(features) < self.sequence_length:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Create model
                # input_size = features.shape[-1]
                # model = TFTPredictor(
                #     sequence_length=self.sequence_length,
                #     prediction_horizon=self.prediction_horizon,
                #     hidden_size=64,  # Default values for testing
                #     num_heads=8,
                #     num_layers=2,
                #     dropout=0.2,
                #     device=self.device
                # )
                # model.to(self.device)
                raise NotImplementedError("Training TFT models is not supported in this method. Please use the dedicated TFT training pipeline.")
                
                # Prepare data loaders
                train_size = int(0.8 * len(features))
                val_size = int(0.1 * len(features))
                
                train_data = features[:train_size]
                val_data = features[train_size:train_size + val_size]
                test_data = features[train_size + val_size:]
                
                train_loader = torch.utils.data.DataLoader(
                    torch.FloatTensor(train_data),
                    batch_size=batch_size,
                    shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    torch.FloatTensor(val_data),
                    batch_size=batch_size
                )
                
                # Initialize optimizer and loss function
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                # Training loop
                model.train()
                symbol_history = {'loss': [], 'val_loss': []}
                
                for epoch in range(epochs):
                    # Training
                    model.train()
                    train_loss = 0
                    for batch in train_loader:
                        batch = batch.to(self.device)
                        optimizer.zero_grad()
                        output, _ = model(batch)
                        loss = criterion(output, batch[:, -1, 0:1])  # Predict next close price
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(self.device)
                            output, _ = model(batch)
                            loss = criterion(output, batch[:, -1, 0:1])
                            val_loss += loss.item()
                    
                    # Record history
                    symbol_history['loss'].append(train_loss / len(train_loader))
                    symbol_history['val_loss'].append(val_loss / len(val_loader))
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}/{epochs} - "
                                  f"Loss: {train_loss/len(train_loader):.4f} - "
                                  f"Val Loss: {val_loss/len(val_loader):.4f}")
                
                # Save model if path provided
                if save_path:
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save model state
                    model_path = save_dir / f"{symbol}_model.pth"
                    torch.save(model.state_dict(), model_path)
                    
                    # Save model configuration
                    config_path = save_dir / f"{symbol}_model.json"
                    config = {
                        'sequence_length': self.sequence_length,
                        'prediction_horizon': self.prediction_horizon,
                        'hidden_size': 64,
                        'num_heads': 8,
                        'num_layers': 2,
                        'dropout': 0.2
                    }
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                
                # Store model
                self.models[symbol] = model
                history[symbol] = symbol_history
                
                logger.info(f"Completed training for {symbol}")
            
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
                raise
        
        return history

    def _get_or_load_model(self, symbol: str) -> Any:
        """Get a model from cache or load it from disk."""
        if symbol in self.models:
            self.logger.info(f"[{self.__class__.__name__}] Found existing model for {symbol} in cache.")
            return self.models[symbol]

        model_loaded = False
        # Try to load symbol-specific model
        symbol_model_path = self.saved_models_dir / symbol
        symbol_config_path = symbol_model_path / "config.yaml" # Assuming TFTPredictor uses config.yaml

        if symbol_model_path.exists() and symbol_config_path.exists():
            try:
                self.logger.info(f"[{self.__class__.__name__}] Attempting to load model for {symbol} from {symbol_model_path}")
                model = TFTPredictor(
                    model_path=str(symbol_model_path), # TFTPredictor expects model_path to be a dir
                    config_path=str(symbol_config_path)
                )
                self.models[symbol] = model
                self.logger.info(f"[{self.__class__.__name__}] Successfully loaded model for {symbol} from {symbol_model_path}")
                model_loaded = True
                return model
            except TFTPredictorError as e:
                self.logger.warning(f"[{self.__class__.__name__}] Failed to load symbol-specific model for {symbol} from {symbol_model_path}: {str(e)}. Falling back.")
            except Exception as e:
                self.logger.error(f"[{self.__class__.__name__}] Unexpected error loading symbol-specific model for {symbol} from {symbol_model_path}: {str(e)}")
                # Optionally re-raise or handle as critical failure for the symbol

        # Fallback: Try to load/use a "default" model if symbol-specific one failed or doesn't exist
        if "default" in self.models:
            self.logger.info(f"[{self.__class__.__name__}] Using cached default model for {symbol}.")
            return self.models["default"]
        
        # Try to load the default model if not already loaded (e.g., from self.default_model_dir)
        # This part assumes self.load_model() would populate self.model, and we might cache it as "default"
        # Or directly load the default model here.
        # The current self.load_model() loads into self.model, not self.models['default']
        # Let's adjust to use self.default_model_dir if no specific or cached default is found
        
        default_config_path = self.default_model_dir / "config.yaml"
        if self.default_model_dir.exists() and default_config_path.exists():
            try:
                self.logger.info(f"[{self.__class__.__name__}] Attempting to load default model from {self.default_model_dir} for {symbol}")
                default_model = TFTPredictor(
                    model_path=str(self.default_model_dir),
                    config_path=str(default_config_path)
                )
                self.models["default"] = default_model # Cache it as default
                self.logger.info(f"[{self.__class__.__name__}] Successfully loaded and cached default model from {self.default_model_dir}")
                return default_model
            except TFTPredictorError as e:
                self.logger.warning(f"[{self.__class__.__name__}] Failed to load default model from {self.default_model_dir}: {str(e)}")
            except Exception as e:
                self.logger.error(f"[{self.__class__.__name__}] Unexpected error loading default model from {self.default_model_dir}: {str(e)}")

        self.logger.warning(f"[{self.__class__.__name__}] No model could be loaded for {symbol} (specific or default).")
        return None

if __name__ == "__main__":
    # Example usage with memory-efficient settings
    predictor = EnhancedStockPredictor(
        sequence_length=10,
        ensemble_size=1,  # Single model for memory efficiency
        memory_efficient=True,  # Enable memory-efficient mode
        model_type='tft',
        use_feature_selection=True,
        use_pca=False,
        detect_regime=True
    )
    
    # Train model with reduced parameters
    history = predictor.train(
        symbols=['AAPL'],  # Start with single symbol
        start_date='2023-01-01',
        end_date='2023-12-31',  # Reduced date range
        epochs=20,  # Reduced epochs
        batch_size=8,  # Small batch size
        optimize_hyperparams=True,
        n_trials=20,  # Reduced trials
        save_path='models/enhanced_predictor.pt'
    )
    
    # Make predictions
    predictions = predictor.predict(
        symbols=['AAPL'],
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    print("\nPredictions:")
    print(json.dumps(predictions, indent=4)) 