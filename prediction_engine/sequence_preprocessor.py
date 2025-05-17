# File: prediction_engine/sequence_preprocessor.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

class SequencePreprocessor:
    """
    Transforms a multi‐symbol OHLCV DataFrame into
    formatted data for model training or inference.
    
    Supports both LSTM and TFT models.

    Parameters
    ----------
    sequence_length : int
        Number of time steps per input sequence.
    model_type : str
        Type of model to preprocess for ('lstm' or 'tft')
    """
    def __init__(self, sequence_length: int = 10, model_type: str = 'lstm'):
        self.sequence_length = sequence_length
        self.model_type = model_type.lower()
        
        if self.model_type not in ['lstm', 'tft']:
            raise ValueError(f"Unsupported model type: {model_type}. Must be 'lstm' or 'tft'.")

    def transform_lstm(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform data for LSTM model.
        
        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex[ (symbol, datetime) ] with columns
            ['open','high','low','close','volume'].

        Returns
        -------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        # example for single-symbol only; extend for multi
        symbol = df.index.get_level_values("symbol")[0]
        data = df.xs(symbol).sort_index()
        arr = data[["close"]].values  # or your chosen features

        X, y = [], []
        for i in range(len(arr) - self.sequence_length):
            X.append(arr[i : i + self.sequence_length])
            y.append(arr[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def transform_tft(self, df: pd.DataFrame) -> Dict:
        """
        Transform data for TFT model.
        
        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex[ (symbol, datetime) ] with columns
            ['open','high','low','close','volume'].

        Returns
        -------
        Dict containing formatted data for TFT model
        """
        # Ensure data is sorted
        df = df.sort_index()
        
        # Reset index to get symbol and datetime as columns
        formatted_data = df.reset_index()
        
        # Rename columns to match TFT expectations if needed
        if 'datetime' in formatted_data.columns:
            formatted_data.rename(columns={'datetime': 'date'}, inplace=True)
            
        # Create time and identifier arrays for TFT
        time_array = formatted_data['date'].values
        id_array = formatted_data['symbol'].values
        
        # Return dictionary with formatted data
        return {
            "inputs": formatted_data,
            "time": time_array,
            "identifier": id_array
        }
        
    def transform(self, df: pd.DataFrame) -> Union[Tuple[np.ndarray, np.ndarray], Dict]:
        """
        Transform data based on the model type.
        
        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex[ (symbol, datetime) ] with columns
            ['open','high','low','close','volume'].

        Returns
        -------
        For LSTM: Tuple of (X, y) arrays
        For TFT: Dictionary with formatted data
        """
        if self.model_type == 'lstm':
            return self.transform_lstm(df)
        elif self.model_type == 'tft':
            return self.transform_tft(df)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

