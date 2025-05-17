"""Enhanced prediction engine for stock price prediction."""

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

from .model import EnhancedLSTM, create_model
from .feature_engineering import FeatureEngineer
from data_input.market_feed import get_market_data
from data_input.sentiment_manager import get_sentiment_data

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

class EnhancedStockPredictor:
    """
    Enhanced prediction engine for stock price prediction.
    
    This class combines:
    1. Market data from market_data_manager
    2. Sentiment data from sentiment_manager
    3. Enhanced LSTM model with attention and uncertainty
    4. Advanced feature engineering
    5. Model ensemble support
    """
    
    def __init__(
        self,
        sequence_length: int = 20,  # Updated to match saved models
        prediction_horizon: int = 5,  # Updated to match saved models
        device: str = 'cpu'  # Force CPU usage for VM
    ):
        """Initialize the enhanced stock predictor."""
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.models = {}  # Dictionary to store models for each symbol
        self.feature_engineer = None
        self.saved_models_dir = Path("saved_models")
        
        # Initialize feature engineer with settings from saved models
        self.feature_engineer = FeatureEngineer(
            sequence_length=sequence_length,
            n_technical_indicators=20,
            use_feature_selection=True,
            use_pca=True,
            n_features=20,
            n_pca_components=10,
            use_regime_detection=True
        )
        
        # Load global training config
        config_path = self.saved_models_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
            logger.info("Loaded global training configuration")
        else:
            logger.warning("No global training configuration found")
            self.training_config = {}
        
        logger.info(f"Initialized enhanced stock predictor with sequence length {sequence_length}, "
                   f"prediction horizon {prediction_horizon}")
    
    def load_model(self, symbol: str) -> None:
        """Load model for a specific symbol from saved_models directory."""
        try:
            # Construct paths for this symbol
            model_path = self.saved_models_dir / f"{symbol}_model.pth"
            config_path = self.saved_models_dir / f"{symbol}_model.json"
            
            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"No saved model found for symbol {symbol}")
            
            # Load model configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create model with saved configuration
            model = create_model(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                use_sentiment=config.get('use_sentiment', True),
                bidirectional=config.get('bidirectional', True),
                use_attention=config.get('use_attention', True),
                use_residual=config.get('use_residual', True),
                estimate_uncertainty=config.get('estimate_uncertainty', True)
            )
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Store model for this symbol
            self.models[symbol] = model
            logger.info(f"Loaded model for {symbol} from {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            raise
    
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
                input_size = features.shape[-1]
                model = create_model(
                    input_size=input_size,
                    hidden_size=64,  # Default values for testing
                    num_layers=2,
                    dropout=0.2,
                    use_sentiment=sentiment_features is not None,
                    bidirectional=True,
                    use_attention=True,
                    use_residual=True,
                    estimate_uncertainty=True
                )
                model.to(self.device)
                
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
                        'input_size': input_size,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.2,
                        'use_sentiment': sentiment_features is not None,
                        'bidirectional': True,
                        'use_attention': True,
                        'use_residual': True,
                        'estimate_uncertainty': True
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
        memory_efficient=True  # Enable memory-efficient mode
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