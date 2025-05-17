"""Prediction engine package for stock price prediction."""

# Import non-TensorFlow modules directly
from .scaler_handler import (
    ScalerHandler, 
    ScalerHandlerError
)

# Define lazy imports for TensorFlow-dependent modules
def __getattr__(name):
    if name == 'TFTPredictor':
        from .tft_predictor import TFTPredictor
        return TFTPredictor
    elif name == 'TFTPredictorError':
        from .tft_predictor import TFTPredictorError
        return TFTPredictorError
    elif name == 'TFTScalerHandler':
        from .scaler_handler import TFTScalerHandler
        return TFTScalerHandler
    elif name == 'LSTMScalerHandler':
        from .scaler_handler import LSTMScalerHandler
        return LSTMScalerHandler
    elif name == 'EnhancedLSTM':
        from .model import EnhancedLSTM
        return EnhancedLSTM
    elif name == 'create_model':
        from .model import create_model
        return create_model
    elif name == 'EnhancedStockPredictor':
        from .predictor import EnhancedStockPredictor
        return EnhancedStockPredictor
    elif name == 'FeatureEngineer':
        from .feature_engineering import FeatureEngineer
        return FeatureEngineer
    elif name == 'SequencePreprocessor':
        from .sequence_preprocessor import SequencePreprocessor
        return SequencePreprocessor
    raise AttributeError(f"module 'prediction_engine' has no attribute '{name}'")

__all__ = [
    'ScalerHandler',
    'ScalerHandlerError',
    'TFTPredictor',
    'TFTPredictorError',
    'TFTScalerHandler',
    'LSTMScalerHandler',
    'EnhancedLSTM',
    'create_model',
    'EnhancedStockPredictor',
    'FeatureEngineer',
    'SequencePreprocessor'
]
