"""Base class for data formatters.

This module defines the base class for data formatters and input types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple

class InputTypes(Enum):
    """Defines input types of the model."""
    TARGET = 'target'
    OBSERVED = 'observed'
    KNOWN = 'known'
    STATIC = 'static'
    ID = 'id'
    TIME = 'time'
    
    # Data types
    REAL_VALUED = 'real_valued'
    CATEGORICAL = 'categorical'

class GenericDataFormatter(ABC):
    """Abstract base class for all data formatters.
    
    This class defines the interface that all data formatters must implement.
    """

    @abstractmethod
    def split_data(self, df):
        """Splits data frame into training-validation-test data frames.
        
        Args:
            df: Source dataframe to split.
            
        Returns:
            Tuple of (training data, validation data, test data)
        """
        pass

    @abstractmethod
    def format_predictions(self, df):
        """Reverts any normalisation to give predictions in original scale.
        
        Args:
            df: Dataframe of model predictions.
            
        Returns:
            Data frame of unnormalised predictions.
        """
        pass

    @abstractmethod
    def get_default_model_params(self):
        """Returns default optimised model parameters.
        
        Returns:
            Dictionary of default model parameters.
        """
        pass

    @abstractmethod
    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        
        Returns:
            Number of samples for calibration.
        """
        pass

    @abstractmethod
    def get_experiment_params(self):
        """Returns fixed model parameters for experiments.
        
        Returns:
            Dictionary of fixed model parameters.
        """
        pass

    @abstractmethod
    def _get_input_columns(self):
        """Returns names of all input columns."""
        pass

    @abstractmethod
    def _get_timestamp_column(self):
        """Returns the name of the timestamp column."""
        pass

    @abstractmethod
    def _get_single_prediction(self, df):
        """Gets a single prediction for each entity.
        
        Args:
            df: Dataframe to get predictions for.
            
        Returns:
            Dataframe with single prediction per entity.
        """
        pass

    @abstractmethod
    def _create_targets(self, df):
        """Creates target values as np arrays.
        
        Args:
            df: Dataframe to create targets for.
            
        Returns:
            Dataframe with target values.
        """
        pass

    @abstractmethod
    def _get_real_scalers(self, df):
        """Gets scalers for real-valued columns.
        
        Args:
            df: Dataframe to get scalers for.
            
        Returns:
            Dictionary of scalers for real-valued columns.
        """
        pass

    @abstractmethod
    def _get_categorical_scalers(self, df):
        """Gets scalers for categorical columns.
        
        Args:
            df: Dataframe to get scalers for.
            
        Returns:
            Dictionary of scalers for categorical columns.
        """
        pass

    @abstractmethod
    def transform_inputs(self, df):
        """Performs feature transformations.
        
        Args:
            df: Dataframe to transform.
            
        Returns:
            Transformed dataframe.
        """
        pass 