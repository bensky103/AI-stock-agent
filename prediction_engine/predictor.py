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
logger.setLevel(logging.WARNING)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

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
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.model_type = model_type
        self.model = None  # Will be initialized when loading a model
        self.default_model_dir = Path("colab_training/tft_model")
        self.saved_models_dir = Path("saved_models")
        self.saved_models_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
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
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Config path does not exist: {config_path}")
            
            self.model = TFTPredictor(
                model_path=str(model_path),
                config_path=str(config_path)
            )
            self.logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise StockPredictorError(f"Failed to load model: {str(e)}")
    
    def predict(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Make predictions for the given symbols.
        
        Args:
            symbols: List of stock symbols to predict
            start_date: Optional start date for prediction (defaults to last sequence_length days)
            end_date: Optional end date for prediction (defaults to today)
        
        Returns:
            Dictionary of predictions for each symbol
        """
        predictions = {}
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                         timedelta(days=self.sequence_length * 2)).strftime('%Y-%m-%d')
        
        # Get market data for all symbols
        market_data = get_market_data(symbols, start_date, end_date)
        if market_data.empty:
            raise ValueError("No market data available for prediction")
        
        # Process each symbol
        for symbol in symbols:
            try:
                # Load model if not already loaded
                if symbol not in self.models:
                    self.load_model(symbol)
                
                # Get symbol-specific data
                symbol_data = market_data[market_data['Symbol'] == symbol].copy()
                if symbol_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Prepare features
                features = self.feature_engineer.prepare_features(symbol_data)
                if features is None or len(features) < self.sequence_length:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Get the last sequence for prediction
                last_sequence = features[-self.sequence_length:]
                last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    model = self.models[symbol]
                    pred, uncertainty = model(last_sequence)
                    pred = pred.cpu().numpy()[0][0]
                    uncertainty = uncertainty.cpu().numpy()[0][0] if uncertainty is not None else None
                
                # Store prediction
                predictions[symbol] = {
                    'prediction': float(pred),
                    'uncertainty': float(uncertainty) if uncertainty is not None else None,
                    'last_date': symbol_data.index[-1].strftime('%Y-%m-%d'),
                    'last_price': float(symbol_data['Close'].iloc[-1]),
                    'prediction_date': end_date
                }
                
                logger.info(f"Made prediction for {symbol}: {pred:.2f} ± {uncertainty:.2f if uncertainty else 'N/A'}")
            
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
                predictions[symbol] = {
                    'error': str(e),
                    'last_date': None,
                    'last_price': None,
                    'prediction_date': end_date
                }
        
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