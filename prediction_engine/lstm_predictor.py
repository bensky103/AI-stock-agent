"""LSTM-based stock price predictor module.

This module provides functionality for loading and using the pre-trained LSTM model
for stock price prediction. It handles model loading, data preprocessing, and
prediction generation with proper error handling and logging.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import yaml
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSTMPredictorError(Exception):
    """Custom exception for LSTM predictor related errors."""
    pass

class LSTMPredictor:
    """LSTM-based stock price predictor."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Union[str, Path],
        sequence_length: int = 100,
        device: Optional[str] = None
    ):
        """Initialize the LSTM predictor.
        
        Args:
            model_path: Path to the model file
            config_path: Path to the configuration file
            sequence_length: Length of input sequences
            device: Device to use for predictions ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.sequence_length = sequence_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Load model
        self.model = self._load_model()
        
        # Load model metadata if available
        self.metadata = self._load_metadata()
        
        logger.info(f"LSTM predictor initialized on {self.device}")
    
    def _load_config(self) -> dict:
        """Load and validate model configuration.
        
        Returns:
            dict: Model configuration
            
        Raises:
            LSTMPredictorError: If config is invalid or missing required fields
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('model', {}).get('lstm', {})
            if not model_config:
                raise LSTMPredictorError("Missing LSTM model configuration")
            
            required_fields = ['sequence_length', 'hidden_size', 'num_layers']
            missing_fields = [field for field in required_fields if field not in model_config]
            if missing_fields:
                raise LSTMPredictorError(f"Missing required fields: {missing_fields}")
            
            return model_config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            raise LSTMPredictorError(f"Failed to load config: {e}")
    
    def _load_model(self) -> keras.Model:
        """Load the pre-trained LSTM model.
        
        Returns:
            keras.Model: Loaded model
            
        Raises:
            LSTMPredictorError: If model loading fails
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            model = keras.models.load_model(self.model_path)
            
            # Verify model architecture
            if not isinstance(model, keras.Model):
                raise LSTMPredictorError("Invalid model type")
            
            # Set model to evaluation mode
            model.trainable = False
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise LSTMPredictorError(f"Failed to load model: {e}")
    
    def _load_metadata(self) -> Optional[dict]:
        """Load model metadata if available.
        
        Returns:
            Optional[dict]: Model metadata or None if not available
        """
        metadata_path = self.model_path.with_suffix('.json')
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model metadata: {e}")
        return None
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model input.
        
        Args:
            df: DataFrame with price data
            price_col: Column to use for prediction
            
        Returns:
            Tuple of (X, y) arrays for model input
            
        Raises:
            LSTMPredictorError: If preprocessing fails
        """
        try:
            # Extract price data
            data = df[price_col].values.reshape(-1, 1)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise LSTMPredictorError(f"Failed to preprocess data: {e}")
    
    def predict(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions for the given data.
        
        Args:
            df: DataFrame with price data
            price_col: Column to use for prediction
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Array of predictions and optionally confidence scores
            
        Raises:
            LSTMPredictorError: If prediction fails
        """
        try:
            # Preprocess data
            X, _ = self.preprocess_data(df, price_col)
            
            if len(X) == 0:
                raise LSTMPredictorError("Insufficient data for prediction")
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions)
            
            if return_confidence:
                # Calculate prediction confidence (using model's last layer variance)
                confidence = np.ones_like(predictions)  # Placeholder for actual confidence
                return predictions, confidence
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise LSTMPredictorError(f"Failed to generate predictions: {e}")
    
    def predict_next(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        n_steps: int = 1
    ) -> Dict[str, np.ndarray]:
        """Predict next n steps.
        
        Args:
            df: DataFrame with price data
            price_col: Column to use for prediction
            n_steps: Number of steps to predict
            
        Returns:
            Dictionary with predictions and timestamps
            
        Raises:
            LSTMPredictorError: If prediction fails
        """
        try:
            # Get the last sequence
            last_sequence = df[price_col].values[-self.sequence_length:].reshape(-1, 1)
            scaled_sequence = self.scaler.transform(last_sequence)
            
            predictions = []
            current_sequence = scaled_sequence.copy()
            
            # Generate predictions for n steps
            for _ in range(n_steps):
                # Reshape for model input
                X = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
                
                # Generate prediction
                pred = self.model.predict(X, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence, pred)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            # Generate timestamps for predictions
            last_date = df.index[-1]
            if isinstance(last_date, pd.Timestamp):
                timestamps = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=n_steps,
                    freq='B'  # Business days
                )
            else:
                timestamps = None
            
            return {
                'predictions': predictions.flatten(),
                'timestamps': timestamps,
                'last_price': df[price_col].iloc[-1],
                'last_date': last_date
            }
            
        except Exception as e:
            logger.error(f"Error predicting next steps: {e}")
            raise LSTMPredictorError(f"Failed to predict next steps: {e}")
    
    def save_state(self, path: Union[str, Path]):
        """Save model state and metadata.
        
        Args:
            path: Path to save state to
            
        Raises:
            LSTMPredictorError: If saving fails
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = path / 'model.h5'
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = path / 'scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'config': self.config,
                'metadata': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            metadata_path = path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model state saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            raise LSTMPredictorError(f"Failed to save model state: {e}")
    
    @classmethod
    def load_state(cls, path: Union[str, Path], **kwargs) -> 'LSTMPredictor':
        """Load model state from saved files.
        
        Args:
            path: Path to load state from
            **kwargs: Additional arguments for initialization
            
        Returns:
            LSTMPredictor: Loaded predictor instance
            
        Raises:
            LSTMPredictorError: If loading fails
        """
        try:
            path = Path(path)
            
            # Load metadata
            metadata_path = path / 'metadata.json'
            if not metadata_path.exists():
                raise LSTMPredictorError("Metadata file not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create instance
            instance = cls(
                model_path=path / 'model.h5',
                config_path=metadata['config'],
                sequence_length=metadata['sequence_length'],
                **kwargs
            )
            
            # Load scaler
            scaler_path = path / 'scaler.joblib'
            if scaler_path.exists():
                instance.scaler = joblib.load(scaler_path)
            
            instance.metadata = metadata.get('metadata')
            
            logger.info(f"Model state loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            raise LSTMPredictorError(f"Failed to load model state: {e}")

def main():
    """Example usage of the LSTMPredictor class."""
    try:
        # Initialize predictor
        predictor = LSTMPredictor(
            model_path="downloaded_model/stock_dl_model.h5",
            config_path="config/strategy_config.yaml"
        )
        
        # Example: Load some data
        import yfinance as yf
        df = yf.download('AAPL', start='2024-01-01', end='2024-02-01')
        
        # Generate predictions
        predictions = predictor.predict(df)
        print("\nPredictions:")
        print(predictions)
        
        # Predict next 5 days
        next_pred = predictor.predict_next(df, n_steps=5)
        print("\nNext 5 days predictions:")
        for date, pred in zip(next_pred['timestamps'], next_pred['predictions']):
            print(f"{date.date()}: {pred:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 