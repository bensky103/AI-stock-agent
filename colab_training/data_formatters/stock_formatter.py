"""Stock data formatter for TFT model.

This module implements the data formatter for stock price prediction using the TFT model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta

# Import base classes using absolute imports
from colab_training.data_formatters.base import GenericDataFormatter, InputTypes

class StockFormatter(GenericDataFormatter):
    """Formats stock data for the TFT model."""
    
    def __init__(self, config=None):
        """Initialize the stock formatter."""
        super().__init__()
        self.config = config
        
        # Define column definitions
        # Ensure only one static input for consistency with TFTModel num_static_features = 1
        # If 'sector' or similar was intended as static, it should be the only one.
        # Adding a placeholder 'static_id_placeholder' if no other clear static feature is present.
        self.column_definition = [
            ('symbol', InputTypes.ID),
            ('date', InputTypes.TIME),
            ('close', InputTypes.TARGET),
            ('open', InputTypes.OBSERVED),
            ('high', InputTypes.OBSERVED),
            ('low', InputTypes.OBSERVED),
            ('volume', InputTypes.OBSERVED),
            ('returns', InputTypes.OBSERVED),
            ('volatility', InputTypes.OBSERVED),
            # ('sector', InputTypes.STATIC), # Example: if sector was the static feature
            ('static_id_placeholder', InputTypes.STATIC), # Placeholder static feature
            ('market_cap', InputTypes.OBSERVED),
            ('pe_ratio', InputTypes.OBSERVED),
            ('dividend_yield', InputTypes.OBSERVED),
            ('rsi', InputTypes.OBSERVED),
            ('macd', InputTypes.OBSERVED),
            ('macd_signal', InputTypes.OBSERVED),
            ('macd_hist', InputTypes.OBSERVED),
            ('bb_upper', InputTypes.OBSERVED),
            ('bb_middle', InputTypes.OBSERVED),
            ('bb_lower', InputTypes.OBSERVED),
            ('atr', InputTypes.OBSERVED),
            ('volume_sma', InputTypes.OBSERVED),
            ('volume_ratio', InputTypes.OBSERVED),
            ('price_sma_5', InputTypes.OBSERVED),
            ('price_sma_10', InputTypes.OBSERVED),
            ('price_sma_20', InputTypes.OBSERVED),
            ('price_sma_50', InputTypes.OBSERVED),
            ('price_sma_200', InputTypes.OBSERVED),
            ('price_ema_5', InputTypes.OBSERVED),
            ('price_ema_10', InputTypes.OBSERVED),
            ('price_ema_20', InputTypes.OBSERVED),
            ('price_ema_50', InputTypes.OBSERVED),
            ('price_ema_200', InputTypes.OBSERVED),
            ('market_regime', InputTypes.CATEGORICAL), # This could also be InputTypes.KNOWN if regimes are forecasted
            ('trading_signal', InputTypes.CATEGORICAL) # Same as above
        ]
        
        # Initialize scalers
        self.real_scalers: Dict[str, StandardScaler] = {}
        self.categorical_scalers: Dict[str, LabelEncoder] = {}
        
    def is_future_known(self, column_name: str) -> bool:
        """Checks if a column is a future-known input based on its definition."""
        # For this formatter, assume KNOWN inputs are future known if they are not the target.
        # A more sophisticated setup might have a separate flag or type.
        for col, type_ in self.column_definition:
            if col == column_name and type_ == InputTypes.KNOWN:
                # Add any further logic here if some KNOWN are historical vs future
                return True # Defaulting all KNOWN to be future for now if any were defined
        return False

    def _get_input_columns(self) -> List[str]:
        """Returns names of all input columns."""
        return [col for col, type_ in self.column_definition if type_ != InputTypes.TARGET]
    
    def _get_timestamp_column(self) -> str:
        """Returns the name of the timestamp column."""
        return 'date'
    
    def _get_single_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets a single prediction for each entity.
        
        Args:
            df: Dataframe to get predictions for.
            
        Returns:
            Dataframe with single prediction per entity.
        """
        # Get the last date for each symbol
        last_dates = df.groupby('symbol')['date'].max()
        
        # Filter to get the last row for each symbol
        predictions = df.merge(last_dates, on=['symbol', 'date'])
        
        return predictions
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates target values as np arrays.
        
        Args:
            df: Dataframe to create targets for.
            
        Returns:
            Dataframe with target values.
        """
        # Calculate future returns
        df['future_returns'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
        
        # Calculate future volatility
        df['future_volatility'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std().shift(-1))
        
        return df
    
    def _get_real_scalers(self, df: pd.DataFrame) -> Dict[str, StandardScaler]:
        """Gets scalers for real-valued columns.
        
        Args:
            df: Dataframe to get scalers for.
            
        Returns:
            Dictionary of scalers for real-valued columns.
        """
        real_cols = [col for col, type_ in self.column_definition if type_ == InputTypes.REAL_VALUED]
        
        scalers = {}
        for col in real_cols:
            if col in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[col].values.reshape(-1, 1))
                scalers[col] = scaler
                
        return scalers
    
    def _get_categorical_scalers(self, df: pd.DataFrame) -> Dict[str, LabelEncoder]:
        """Gets scalers for categorical columns.
        
        Args:
            df: Dataframe to get scalers for.
            
        Returns:
            Dictionary of scalers for categorical columns.
        """
        categorical_cols = [col for col, type_ in self.column_definition if type_ == InputTypes.CATEGORICAL]
        
        scalers = {}
        for col in categorical_cols:
            if col in df.columns:
                scaler = LabelEncoder()
                scaler.fit(df[col].values)
                scalers[col] = scaler
                
        return scalers
    
    def transform_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs feature transformations.
        
        Args:
            df: Dataframe to transform.
            
        Returns:
            Transformed dataframe.
        """
        # Get scalers
        self.real_scalers = self._get_real_scalers(df)
        self.categorical_scalers = self._get_categorical_scalers(df)
        
        # Transform real-valued columns
        for col, scaler in self.real_scalers.items():
            df[col] = scaler.transform(df[col].values.reshape(-1, 1))
            
        # Transform categorical columns
        for col, scaler in self.categorical_scalers.items():
            df[col] = scaler.transform(df[col].values)
            
        return df
    
    def format_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverts any normalisation to give predictions in original scale.
        
        Args:
            df: Dataframe of model predictions.
            
        Returns:
            Dataframe of unnormalised predictions.
        """
        # Revert scaling for target column
        if 'close' in self.real_scalers:
            df['close'] = self.real_scalers['close'].inverse_transform(
                df['close'].values.reshape(-1, 1))
            
        return df
    
    def get_default_model_params(self) -> Dict:
        """Returns default optimised model parameters.
        
        Returns:
            Dictionary of default model parameters.
        """
        return {
            'hidden_layer_size': 64,
            'attention_head_size': 4,
            'dropout_rate': 0.1,
            'max_gradient_norm': 0.01,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'num_encoder_steps': 100,
            'num_steps': 20,
            'context_lengths': [1, 7, 14, 30]
        }
    
    def get_num_samples_for_calibration(self) -> int:
        """Gets the default number of training and validation samples.
        
        Returns:
            Number of samples for calibration.
        """
        return 450000
    
    def get_experiment_params(self) -> Dict:
        """Returns fixed model parameters for experiments.
        
        Returns:
            Dictionary of fixed model parameters.
        """
        return {
            'total_time_steps': 252,  # One trading year
            'num_encoder_steps': 100,
            'num_steps': 20,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'minibatch_size': 64
        }
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data frame into training-validation-test data frames.
        
        Args:
            df: Source dataframe to split.
            
        Returns:
            Tuple of (training data, validation data, test data)
        """
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate split points
        total_rows = len(df)
        train_end = int(total_rows * 0.7)
        valid_end = int(total_rows * 0.85)
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:].copy()
        
        return train_df, valid_df, test_df
    
    def _get_static_input_size(self) -> int:
        """Gets the size of static inputs."""
        return len([col for col, type_ in self.column_definition if type_ == InputTypes.STATIC])
    
    def _get_historical_input_size(self) -> int:
        """Gets the size of historical inputs."""
        return len([col for col, type_ in self.column_definition if type_ in {InputTypes.KNOWN, InputTypes.OBSERVED}])
    
    def _get_future_input_size(self) -> int:
        """Gets the size of future inputs."""
        return len([col for col, type_ in self.column_definition if type_ == InputTypes.KNOWN]) 