# File: prediction_engine/sequence_preprocessor.py

import pandas as pd
import numpy as np

class SequencePreprocessor:
    """
    Transforms a multi‐symbol OHLCV DataFrame into
    (X, y) numpy arrays for LSTM training or inference.

    Parameters
    ----------
    sequence_length : int
        Number of time steps per input sequence.
    """
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex[ (symbol, datetime) ] with columns
            ['open','high','low','close','adj_close','volume'].

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

