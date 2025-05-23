"""Sequence preprocessor module for stock prediction.

This module provides functionality for preparing time series sequences
for TFT models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)
# Don't configure handlers here, just get the logger
logger.setLevel(logging.INFO)

class SequencePreprocessor:
    """
    Preprocessor for time series sequences.
    
    This class handles the preparation of time series data for TFT models,
    including sequence creation, validation, and transformation.
    
    Attributes:
        sequence_length (int): Length of input sequences
        data_frequency (str): Frequency of the data ('1d' for daily, '1W' for weekly)
    """
    
    def __init__(self, sequence_length: int = 8, data_frequency: str = '1W'):
        """
        Initialize the sequence preprocessor.
        
        Args:
            sequence_length: Length of input sequences (defaults to 8 weeks)
            data_frequency: Frequency of the data ('1d' for daily, '1W' for weekly)
        """
        self.sequence_length = sequence_length
        self.data_frequency = data_frequency
        
        # Adjust sequence length based on frequency
        if data_frequency == '1d':
            self.sequence_length = max(20, sequence_length)  # At least 20 days for daily data
        elif data_frequency == '1W':
            self.sequence_length = max(8, sequence_length)  # At least 8 weeks for weekly data
        
        logger.info(
            f"Initialized sequence preprocessor with sequence length {self.sequence_length} "
            f"for {data_frequency} data"
        )

    def prepare_sequence(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_cols: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare sequences for TFT model input.
        
        Args:
            df: DataFrame containing time series data
            target_col: Name of the target column
            feature_cols: List of feature columns to use (defaults to all columns except target)
            
        Returns:
            Dictionary containing prepared sequences and their timestamps
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        try:
            if df.empty:
                raise ValueError("Cannot prepare sequences from empty data")
            
            # Check if we have enough data points
            min_required = self.sequence_length + 1  # +1 for target
            if len(df) < min_required:
                raise ValueError(
                    f"Insufficient data: need at least {min_required} "
                    f"{self.data_frequency} time steps, got {len(df)}"
                )
            
            # Determine feature columns
            if feature_cols is None:
                feature_cols = [col for col in df.columns if col != target_col]
            
            if not feature_cols:
                raise ValueError("No feature columns specified")
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Create sequences
            sequences = []
            targets = []
            sequence_timestamps = []  # Track timestamps for each sequence
            
            # Create sequences with targets (excluding the last sequence)
            # Stop two steps before the end to leave room for the last sequence
            for i in range(len(df) - self.sequence_length - 2):
                # Extract sequence
                sequence = df[feature_cols].iloc[i:i + self.sequence_length].values
                
                # Ensure sequence is 2D with shape (sequence_length, n_features)
                if len(sequence.shape) == 1:
                    sequence = sequence.reshape(-1, 1)
                    
                sequences.append(sequence)
                
                # Store the last timestamp of this sequence
                sequence_timestamps.append(df.index[i + self.sequence_length - 1])
                
                # Extract target (next value after sequence)
                target = df[target_col].iloc[i + self.sequence_length]
                
                # Ensure target is 1D (converting scalar to array if needed)
                if np.isscalar(target):
                    target = np.array([target])
                elif len(target.shape) > 1:
                    # If target is a 2D array, flatten it to 1D
                    target = target.ravel()
                    
                targets.append(target)
            
            # Add the last sequence without a target (for prediction)
            # Take the sequence ending one step before the end date
            last_sequence = df[feature_cols].iloc[-(self.sequence_length + 1):-1].values
            
            # Ensure last_sequence is 2D
            if len(last_sequence.shape) == 1:
                last_sequence = last_sequence.reshape(-1, 1)
                
            sequences.append(last_sequence)
            
            # Store the last timestamp of the final sequence
            sequence_timestamps.append(df.index[-2])  # -2 because we excluded the last date
            
            # Convert to numpy arrays
            X = np.array(sequences)
            
            # If we have targets
            if targets:
                y = np.array(targets)
                
                # Ensure y is properly shaped
                if len(y.shape) == 3 and y.shape[1] == 1:
                    # If y has shape (n_samples, 1, n_features), squeeze out the middle dimension
                    y = y.squeeze(axis=1)
                elif len(y.shape) == 1:
                    # If y is 1D, reshape to (n_samples, 1)
                    y = y.reshape(-1, 1)
            else:
                # Create an empty array with correct shape if no targets
                y = np.array([])
            
            logger.info(
                f"Prepared {len(X)} sequences of length {self.sequence_length} "
                f"with {len(feature_cols)} features for {self.data_frequency} data"
            )
            
            return {
                'features': X,
                'targets': y,
                'feature_names': feature_cols,
                'target_name': target_col,
                'data_frequency': self.data_frequency,
                'sequence_timestamps': sequence_timestamps  # Include timestamps in return value
            }
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise
    
    def validate_sequence(self, sequence: Dict[str, np.ndarray]) -> bool:
        """
        Validate prepared sequences.
        
        Args:
            sequence: Dictionary containing prepared sequences
            
        Returns:
            True if sequences are valid, False otherwise
            
        Raises:
            ValueError: If sequence dictionary is invalid
        """
        try:
            # Check required keys
            required_keys = ['features', 'targets']
            if not all(key in sequence for key in required_keys):
                raise ValueError(
                    f"Missing required keys in sequence dictionary. "
                    f"Expected {required_keys}, got {list(sequence.keys())}"
                )
        
            # Check shapes
            features = sequence['features']
            targets = sequence['targets']
            
            if not isinstance(features, np.ndarray) or not isinstance(targets, np.ndarray):
                raise ValueError("Features and targets must be numpy arrays")
            
            if len(features) != len(targets):
                raise ValueError(
                    f"Number of sequences ({len(features)}) does not match "
                    f"number of targets ({len(targets)})"
                )
            
            if features.shape[1] != self.sequence_length:
                raise ValueError(
                    f"Sequence length mismatch: expected {self.sequence_length}, "
                    f"got {features.shape[1]}"
                )
            
            # Check for NaN values
            if np.isnan(features).any() or np.isnan(targets).any():
                raise ValueError("Sequences contain NaN values")
            
            logger.info("Sequence validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Sequence validation failed: {str(e)}")
            raise
    
    def create_sequence(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_cols: Optional[list] = None,
        target_length: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Create sequences with multiple target steps.
        
        Args:
            df: DataFrame containing time series data
            target_col: Name of the target column
            feature_cols: List of feature columns to use
            target_length: Number of target steps to predict
            
        Returns:
            Dictionary containing prepared sequences with multiple target steps
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        try:
            if df.empty:
                raise ValueError("Cannot create sequences from empty data")
            
            if len(df) < self.sequence_length + target_length:
                raise ValueError(
                    f"Insufficient data: need at least {self.sequence_length + target_length} "
                    f"time steps, got {len(df)}"
                )
            
            # Prepare base sequences
            sequences = self.prepare_sequence(df, target_col, feature_cols)
            
            # Create multi-step targets
            multi_step_targets = []
            for i in range(len(sequences['targets']) - target_length + 1):
                target_sequence = sequences['targets'][i:i + target_length]
                multi_step_targets.append(target_sequence)
            
            # Update sequences dictionary
            sequences['targets'] = np.array(multi_step_targets)
            sequences['target_length'] = target_length
            
            logger.info(
                f"Created {len(sequences['features'])} sequences with "
                f"{target_length} target steps"
            )
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

