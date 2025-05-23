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
            self.logger.info(f"[{self.__class__.__name__}] Processing symbol: {symbol}")
            try:
                # Fetch latest market data for the symbol
                # Ensure data is up-to-date for prediction
                # Using a recent period that should cover self.sequence_length
                # The actual length of data needed will be sequence_length for feature prep.
                fetch_end_date = datetime.now(pytz.utc)
                fetch_start_date = fetch_end_date - timedelta(days=60) # Fetch more to be safe, e.g. 60 days
                
                self.logger.info(f"[{self.__class__.__name__}] Fetching latest data for {symbol} from {fetch_start_date} to {fetch_end_date}")
                latest_data_df = get_market_data(
                    [symbol],
                    start_date=fetch_start_date.strftime('%Y-%m-%d'),
                    end_date=fetch_end_date.strftime('%Y-%m-%d')
                )
                
                if latest_data_df.empty:
                    self.logger.warning(f"[{self.__class__.__name__}] No data fetched for {symbol}, cannot make prediction.")
                    predictions[symbol] = {"error": "No data available"}
                    continue
                
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
                if not self.feature_engineer.is_scaler_fitted():
                    self.logger.info(f"[{self.__class__.__name__}] Scaler not fitted for symbol {symbol} or globally. Fitting now.")
                    # Use the full available `latest_data_df` to fit the scaler appropriately for this symbol context
                    # This assumes `latest_data_df` is the historical data up to the point of prediction for the symbol.
                    # 1. Calculate technical indicators on this data
                    data_for_scaler_fitting_with_indicators = self.feature_engineer.calculate_technical_indicators(latest_data_df.copy()) # Use a copy
                    # 2. Fit the scaler
                    if not data_for_scaler_fitting_with_indicators.empty:
                        self.logger.info(f"[{self.__class__.__name__}] Fitting scaler with data of shape {data_for_scaler_fitting_with_indicators.shape} for {symbol}.")
                        _ = self.feature_engineer.normalize_features(data_for_scaler_fitting_with_indicators, fit=True)
                        self.logger.info(f"[{self.__class__.__name__}] Scaler fitted for {symbol}.")
                    else:
                        self.logger.warning(f"[{self.__class__.__name__}] Data for scaler fitting is empty for {symbol} after adding indicators. Scaler not fitted.")
                        predictions[symbol] = {"error": "Could not fit scaler due to empty data"}
                        continue # Or handle as a more critical error
                else:
                    self.logger.info(f"[{self.__class__.__name__}] Scaler already fitted. Proceeding with feature preparation for {symbol}.")

                # Prepare features for the latest data
                # The feature engineer expects a DataFrame with a simple DatetimeIndex for a single symbol here.
                self.logger.info(f"[{self.__class__.__name__}] Preparing features for {symbol} using latest_data_df with shape: {latest_data_df.shape}")
                features_np = self.feature_engineer.prepare_features(latest_data_df) # Fit should be False for prediction
                self.logger.info(f"[{self.__class__.__name__}] Shape of features_np for {symbol} from feature_engineer: {features_np.shape if isinstance(features_np, np.ndarray) else type(features_np)}")

                if not isinstance(features_np, np.ndarray) or features_np.size == 0:
                    self.logger.warning(f"[{self.__class__.__name__}] Feature preparation for {symbol} returned empty or invalid result. Shape: {features_np.shape if isinstance(features_np, np.ndarray) else 'N/A'}. Skipping prediction.")
                    predictions[symbol] = {"error": "Feature preparation failed"}
                    continue

                # Select the last sequence for prediction
                # features_np could be (num_sequences, sequence_length, num_features) or (sequence_length, num_features) if only one sequence
                if features_np.ndim == 3: # (num_sequences, seq_len, num_features)
                    last_sequence_np = features_np[-1] # Take the most recent sequence
                    self.logger.info(f"[{self.__class__.__name__}] Selected last_sequence_np (from 3D features_np) for {symbol}, shape: {last_sequence_np.shape}")
                elif features_np.ndim == 2: # (seq_len, num_features)
                    last_sequence_np = features_np
                    self.logger.info(f"[{self.__class__.__name__}] Using features_np directly as last_sequence_np (2D) for {symbol}, shape: {last_sequence_np.shape}")
                elif features_np.ndim == 1: # (features_at_last_step_if_flattened_unexpectedly_by_FE)
                    # This case should ideally not happen if prepare_features returns sequences.
                    # If it does, it implies prepare_features might have returned a 1D array of the last timestep's features from a single sequence.
                    # The model expects (batch_size, sequence_length, num_features).
                    # We need to reshape it to (1, sequence_length (if known, or 1), num_features (or len of array if seq_len=1) )
                    self.logger.warning(f"[{self.__class__.__name__}] features_np for {symbol} is 1D (shape: {features_np.shape}). This is unexpected for sequence input. Attempting reshape.")
                    # Assuming this 1D array represents features for a single time step in a sequence of length 1 for prediction context
                    # Or it could be (seq_len,) if only 1 feature. This is ambiguous.
                    # For TFT, typically expects sequence_length > 1. For now, assuming it needs to be (1, len, 1) or (1, 1, len)
                    # This path is problematic and indicates an issue in feature_engineer output for sequence models.
                    # Let's assume it means (num_features,) for a single step of a sequence.
                    # TFTPredictor expects (batch, seq_len, features).
                    # If sequence_length for model is self.sequence_length, but we only have one step of features:
                    # This is likely an error state. The feature engineer should provide a full sequence.
                    # For now, to avoid immediate crash, we could try to pad or error out.
                    self.logger.error(f"[{self.__class__.__name__}] Cannot form a sequence from 1D features_np. Feature engineer should provide at least 2D. Skipping {symbol}.")
                    predictions[symbol] = {"error": "Feature engineer returned 1D features for sequence model"}
                    continue
                else:
                    self.logger.error(f"[{self.__class__.__name__}] features_np for {symbol} has unexpected ndim: {features_np.ndim}, shape: {features_np.shape}. Skipping.")
                    predictions[symbol] = {"error": "Features have unexpected dimensions"}
                    continue

                # Ensure data has the right dimensionality for the model input tensor
                # Standard TFT models expect input of shape [batch_size, sequence_length, num_features]
                # last_sequence_np is currently (sequence_length, num_features)
                if last_sequence_np.ndim == 2:
                    model_input_np = np.expand_dims(last_sequence_np, axis=0) # Add batch dimension -> (1, sequence_length, num_features)
                    self.logger.info(f"[{self.__class__.__name__}] Expanded last_sequence_np for {symbol} to model_input_np shape: {model_input_np.shape}")
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
                    self.logger.info(f"[{self.__class__.__name__}] Converted model_input_np to PyTorch tensor for {symbol}, shape: {model_input_tensor.shape}, device: {model_input_tensor.device}")
                elif "tensorflow" in sys.modules:
                    import tensorflow as tf
                    # For TensorFlow, ensure data type matches model expectation (e.g., tf.float32)
                    model_input_tensor = tf.convert_to_tensor(model_input_np, dtype=tf.float32)
                    self.logger.info(f"[{self.__class__.__name__}] Converted model_input_np to TensorFlow tensor for {symbol}, shape: {model_input_tensor.shape}")
                else:
                    self.logger.warning("[{self.__class__.__name__}] Neither PyTorch nor TensorFlow found in sys.modules. Passing numpy array to model.")
                    model_input_tensor = model_input_np # Pass as numpy if framework unclear

                # Make prediction
                try:
                    self.logger.info(f"[{self.__class__.__name__}] Calling model for {symbol} with tensor of shape {model_input_tensor.shape if hasattr(model_input_tensor, 'shape') else type(model_input_tensor)}")
                    model_to_use = self._get_or_load_model(symbol)
                    
                    if model_to_use is None:
                        self.logger.error(f"[{self.__class__.__name__}] No model found for {symbol} or default via _get_or_load_model. Skipping.")
                        predictions[symbol] = {"error": f"Model not found for {symbol}"}
                        continue
                        
                    self.logger.info(f"[{self.__class__.__name__}] Using model type: {type(model_to_use).__name__} for {symbol}")
                    pred_output, uncertainty_output = model_to_use(model_input_tensor) # TFTPredictor.__call__ expects a tensor
                    
                    # Process predictions (example, adjust as per your model output)
                    # Assuming pred_output and uncertainty_output are tensors that need to be converted to numbers
                    if hasattr(pred_output, 'cpu') and hasattr(pred_output, 'numpy'): # PyTorch tensor
                        pred_value = pred_output.cpu().numpy().flatten()[0]
                        uncertainty_value = uncertainty_output.cpu().numpy().flatten()[0] if uncertainty_output is not None else None
                    elif hasattr(pred_output, 'numpy'): # TensorFlow tensor or other numpy-compatible
                        pred_value = pred_output.numpy().flatten()[0]
                        uncertainty_value = uncertainty_output.numpy().flatten()[0] if uncertainty_output is not None and hasattr(uncertainty_output, 'numpy') else None
                    else: # Fallback if not a recognized tensor type
                        pred_value = pred_output[0][0] if isinstance(pred_output, list) or isinstance(pred_output, tuple) else pred_output
                        uncertainty_value = uncertainty_output[0][0] if isinstance(uncertainty_output, list) or isinstance(uncertainty_output, tuple) else uncertainty_output

                    self.logger.info(f"[{self.__class__.__name__}] Prediction for {symbol}: {pred_value}, Uncertainty: {uncertainty_value}")    
                    predictions[symbol] = {
                        "prediction": pred_value,
                        "uncertainty": uncertainty_value,
                        "predicted_at": datetime.now(pytz.utc).isoformat()
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