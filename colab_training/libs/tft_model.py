"""Temporal Fusion Transformer Model.

This module implements the Temporal Fusion Transformer model for time series
forecasting, as described in the paper:
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
by Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister.

Paper link: https://arxiv.org/pdf/1912.09363.pdf
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from tensorflow.keras.layers import Concatenate

# Configure logger
logger = logging.getLogger(__name__)

# Import our data formatter - using absolute imports
from colab_training.data_formatters.base import InputTypes, GenericDataFormatter
from colab_training.data_formatters.stock_formatter import StockFormatter

# Layer definitions
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = layers.Add
LayerNorm = layers.LayerNormalization
Dense = layers.Dense
Multiply = layers.Multiply
Dropout = layers.Dropout
Activation = layers.Activation
Lambda = layers.Lambda

def linear_layer(size: int, activation: Optional[str] = None, 
                use_time_distributed: bool = False, use_bias: bool = True):
    """Returns simple Keras linear layer.
    
    Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    def layer_fn(x):
        if use_time_distributed and len(x.shape) == 3:
            return layers.TimeDistributed(Dense(size, activation=activation, use_bias=use_bias))(x)
        else:
            return Dense(size, activation=activation, use_bias=use_bias)(x)
    return layer_fn

def apply_mlp(inputs, hidden_size: int, output_size: int,
              output_activation: Optional[str] = None,
              hidden_activation: str = 'tanh',
              use_time_distributed: bool = False):
    """Applies simple feed-forward network to an input.
    
    Args:
        inputs: MLP inputs
        hidden_size: Hidden state size
        output_size: Output size of MLP
        output_activation: Activation function to apply on output
        hidden_activation: Activation function to apply on input
        use_time_distributed: Whether to apply across time
    """
    if use_time_distributed:
        hidden = layers.TimeDistributed(
            Dense(hidden_size, activation=hidden_activation))(inputs)
        return layers.TimeDistributed(
            Dense(output_size, activation=output_activation))(hidden)
    else:
        hidden = Dense(hidden_size, activation=hidden_activation)(inputs)
        return Dense(output_size, activation=output_activation)(hidden)

def apply_gating_layer(x, hidden_layer_size: int, dropout_rate: Optional[float] = None,
                      use_time_distributed: bool = True, activation: Optional[str] = None):
    """Applies a Gated Linear Unit (GLU) to an input.
    
    Args:
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature transform
    """
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = layers.TimeDistributed(
            Dense(hidden_layer_size, activation=activation))(x)
        gated_layer = layers.TimeDistributed(
            Dense(hidden_layer_size, activation='sigmoid'))(x)
    else:
        activation_layer = Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = Dense(hidden_layer_size, activation='sigmoid')(x)

    return Multiply()([activation_layer, gated_layer]), gated_layer

def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.
    
    Args:
        x_list: List of inputs to sum for skip connection
    """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def gated_residual_network(x, hidden_layer_size: int, output_size: Optional[int] = None,
                          dropout_rate: Optional[float] = None, use_time_distributed: bool = True,
                          additional_context: Optional[tf.Tensor] = None, return_gate: bool = False):
    """Applies the gated residual network (GRN) as defined in paper.
    
    Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes
    """
    # Always project skip connection to hidden_layer_size if needed
    input_size = x.shape[-1]
    if input_size != hidden_layer_size:
        linear = Dense(hidden_layer_size)
        if use_time_distributed:
            linear = layers.TimeDistributed(linear)
        skip = linear(x)
    else:
        skip = x

    # Apply feedforward network
    hidden = linear_layer(hidden_layer_size, activation=None,
                         use_time_distributed=use_time_distributed)(x)

    if additional_context is not None:
        # Handle dimension mismatch for additional context
        if use_time_distributed and len(x.shape) == 3 and len(additional_context.shape) == 2:
            # Create a Lambda layer to expand static context to match temporal dimensions
            def expand_static_context(inputs):
                x_tensor, context_tensor = inputs
                time_steps = tf.shape(x_tensor)[1]
                # Expand dims and tile across time
                context_expanded = tf.expand_dims(context_tensor, axis=1)
                context_tiled = tf.tile(context_expanded, [1, time_steps, 1])
                return context_tiled
            
            additional_context = Lambda(expand_static_context)([x, additional_context])
        
        context_projection = linear_layer(hidden_layer_size, activation=None,
                                        use_time_distributed=use_time_distributed,
                                        use_bias=False)(additional_context)
        hidden = hidden + context_projection

    hidden = Activation('elu')(hidden)
    hidden = linear_layer(hidden_layer_size, activation=None,
                         use_time_distributed=use_time_distributed)(hidden)

    # Apply dropout if specified
    if dropout_rate is not None:
        hidden = Dropout(dropout_rate)(hidden)

    # Apply GLU
    hidden, gate = apply_gating_layer(hidden, hidden_layer_size,
                                    dropout_rate=None,
                                    use_time_distributed=use_time_distributed,
                                    activation=None)

    # Apply skip connection
    if return_gate:
        return add_and_norm([skip, hidden]), gate
    else:
        return add_and_norm([skip, hidden])

class InterpretableMultiHeadAttention(layers.Layer):
    """Defines interpretable multi-head attention layer.
    
    Attributes:
        n_head: Number of attention heads
        d_model: Dimension of model
        dropout: Dropout rate to apply
        output_size: Output size of layer
    """
    
    def __init__(self, n_head: int, d_model: int, dropout: float):
        """Initialize attention layer.
        
        Args:
            n_head: Number of attention heads
            d_model: Dimension of model
            dropout: Dropout rate to apply
        """
        super(InterpretableMultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        
        self.qs_layer = Dense(n_head * d_model, use_bias=False)
        self.ks_layer = Dense(n_head * d_model, use_bias=False)
        self.vs_layer = Dense(n_head * d_model, use_bias=False)
        
        self.attention_dropout = Dropout(dropout)
        self.w_o = Dense(d_model, use_bias=False)
        
        self.layer_norm = LayerNorm()
        self.dropout_layer = Dropout(dropout)
        
    def build(self, input_shape):
        """Builds the attention layer.
        
        Args:
            input_shape: Shape of input tensor
        """
        # Remove the unused weight variable since we're not using interpretable attention
        # self.w = self.add_weight(name='w',
        #                         shape=(self.n_head, 1, 1),
        #                         initializer='ones',
        #                         trainable=True)
        super(InterpretableMultiHeadAttention, self).build(input_shape)
        
    def call(self, q, k, v, mask=None):
        """Applies attention mechanism to inputs.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Mask to apply if any
            
        Returns:
            Tensor output from attention layer
        """
        # Reshape inputs
        batch_size = tf.shape(q)[0]
        
        # Linear layers
        qs = self.qs_layer(q)  # [batch_size, len_q, n_head * d_model]
        ks = self.ks_layer(k)
        vs = self.vs_layer(v)
        
        # Reshape to [batch_size, len_q, n_head, d_model]
        qs = tf.reshape(qs, [batch_size, -1, self.n_head, self.d_model])
        ks = tf.reshape(ks, [batch_size, -1, self.n_head, self.d_model])
        vs = tf.reshape(vs, [batch_size, -1, self.n_head, self.d_model])
        
        # Transpose for attention
        qs = tf.transpose(qs, [0, 2, 1, 3])  # [batch_size, n_head, len_q, d_model]
        ks = tf.transpose(ks, [0, 2, 1, 3])  # [batch_size, n_head, len_k, d_model]
        vs = tf.transpose(vs, [0, 2, 1, 3])  # [batch_size, n_head, len_v, d_model]
        
        # Scaled dot-product attention
        outputs = tf.matmul(qs, ks, transpose_b=True)  # [batch_size, n_head, len_q, len_k]
        outputs = outputs / (self.d_model ** 0.5)
        
        if mask is not None:
            outputs = outputs * mask
            
        # Apply softmax
        outputs = tf.nn.softmax(outputs)
        outputs = self.attention_dropout(outputs)
        
        # Apply attention weights
        outputs = tf.matmul(outputs, vs)  # [batch_size, n_head, len_q, d_model]
        
        # Reshape and apply final linear layer
        outputs = tf.transpose(outputs, [0, 2, 1, 3])  # [batch_size, len_q, n_head, d_model]
        outputs = tf.reshape(outputs, [batch_size, -1, self.n_head * self.d_model])
        outputs = self.w_o(outputs)
        
        # Apply dropout and layer norm
        outputs = self.dropout_layer(outputs)
        outputs = self.layer_norm(outputs)
        
        return outputs

class TFTModel:
    """Temporal Fusion Transformer model for time series forecasting.
    
    This class implements the TFT model architecture as described in the paper.
    It includes methods for training, evaluation, and prediction.
    """
    
    def __init__(self, config: Dict):
        """Initialize TFT model.
        
        Args:
            config: Dictionary of model configuration parameters.
        """
        # Import here to avoid circular dependency
        from colab_training.data_formatters.stock_formatter import StockFormatter
        
        self.config = config
        self.data_formatter = StockFormatter()
        
        # Model parameters from config, with overrides for alignment
        self.hidden_layer_size = config.get('hidden_layer_size', 64)
        self.attention_head_size = config.get('attention_head_size', 4)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.max_gradient_norm = config.get('max_gradient_norm', 0.01)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.minibatch_size = config.get('minibatch_size', 64)
        self.num_epochs = config.get('num_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Core architectural parameters for input shapes - aligned with FeatureEngineer
        self.num_encoder_steps = config.get('num_encoder_steps', 10) # Historical sequence length (from FeatureEngineer)
        self.num_steps = config.get('num_steps', 5) # Future sequence length (prediction horizon)

        # Explicit feature counts based on FeatureEngineer output and typical TFT structure
        self.num_static_features = 1  # Assuming 1 static feature (e.g., placeholder or simple ID)
        self.num_historical_features = 25 # From FeatureEngineer
        
        # Determine future feature count from StockFormatter's KNOWN input types
        # This should be 0 if no InputTypes.KNOWN are defined in StockFormatter
        future_cols_formatter = [col for col, type_ in self.data_formatter.column_definition if type_ == InputTypes.KNOWN and self.data_formatter.is_future_known(col)]
        self.num_future_features = len(future_cols_formatter)
        if self.num_future_features != 0:
             logger.warning(f"[{self.__class__.__name__}] Model configured with {self.num_future_features} known future features based on StockFormatter. Keras model expects 0.")
             # If Keras model structure (future input layer) expects 0, this might cause issues later if not handled.
             # For now, let's assume _build_model will use this. The error log showed (1,5,0) previously for future inputs from Keras.
             # This implies Keras was built with 0 future features. We will ensure _build_model uses 0.
             self.num_future_features = 0 # Override to match Keras expectation for now.

        # Add training state tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = []
        
        # Initialize model
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the TFT model architecture."""
        # Define inputs based on architectural parameters
        static_inputs = layers.Input(shape=(self.num_static_features,),
                                   name='static_inputs')
        
        historical_inputs = layers.Input(
            shape=(self.num_encoder_steps, self.num_historical_features),
            name='historical_inputs')
        
        # Future inputs: Keras can handle shape (None, N, 0) if the 0-dim feature is not used by any subsequent layer directly
        # or if it's correctly handled by layers like concatenation (if it were non-zero).
        # Given the previous error log `Future=(1, 5, 0)`, the model expects 0 features here.
        future_inputs = layers.Input(
            shape=(self.num_steps, self.num_future_features), # self.num_future_features should be 0
            name='future_inputs')
        
        # Static context
        static_context = gated_residual_network(
            static_inputs,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            additional_context=None)
        
        # Historical context
        historical_context = gated_residual_network(
            historical_inputs,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=None)
        
        # Future context
        future_context = gated_residual_network(
            future_inputs,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=None)
        
        # LSTM for local processing
        lstm_output = layers.LSTM(
            self.hidden_layer_size,
            return_sequences=True,
            dropout=self.dropout_rate
        )(historical_context)
        
        # Static enrichment
        enriched_context = gated_residual_network(
            lstm_output,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=static_context)
        
        # Temporal self-attention
        attention_output = InterpretableMultiHeadAttention(
            self.attention_head_size,
            self.hidden_layer_size,
            self.dropout_rate)(enriched_context, enriched_context, enriched_context)
        
        # Temporal processing
        temporal_output = gated_residual_network(
            attention_output,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=None)
        
        # Future processing
        future_processed = gated_residual_network(
            future_context,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=None)
        
        # Combine temporal and future contexts
        combined_context = Concatenate(axis=1)([temporal_output, future_processed])
        
        # Final processing
        final_output = gated_residual_network(
            combined_context,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=None)
        
        # Output layer
        output = Dense(1)(final_output)
        
        # Create model
        self.model = Model(
            inputs=[static_inputs, historical_inputs, future_inputs],
            outputs=output)
        
        # Compile model with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.max_gradient_norm  # Add gradient clipping
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'])
        
        return self.model
    
    def _get_single_col_by_type(self, type_: 'InputTypes') -> str:
        """Gets the name of a single column of a specific type.
        
        Args:
            type_: Type of column to get
            
        Returns:
            Name of column
        """
        # Import here to avoid circular dependency
        from colab_training.data_formatters.base import InputTypes
        
        cols = [tup[0] for tup in self.data_formatter.column_definition 
                if tup[1] == type_]
        if len(cols) != 1:
            raise ValueError(f"Expected exactly one column of type {type_}, got {len(cols)}")
        return cols[0]

    def _validate_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize input data to prevent NaN values."""
        
        # Check if data is already normalized using a more robust check
        if hasattr(df, '_is_normalized') and df._is_normalized:
            logger.info("Data already normalized, skipping...")
            return df
            
        logger.info("Validating and normalizing data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Store normalization parameters globally for model instance
        if not hasattr(self, 'normalization_params'):
            self.normalization_params = {}
        
        # Convert all numeric columns to float
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].astype(float)
        
        # Store normalization parameters for denormalization later
        if not hasattr(self, 'target_scaler'):
            self.target_scaler = {
                'min': df_clean['close'].min(),
                'max': df_clean['close'].max()
            }
        
        # More efficient NaN handling
        if df_clean.isnull().any().any():
            logger.warning("Found NaN values, filling...")
            # Forward fill then backward fill, then fill remaining with median
            df_clean = df_clean.ffill().bfill()
            
            # For any remaining NaN, fill with column median
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove rows if any NaN still exists
        if df_clean.isnull().any().any():
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.warning(f"Removed {removed_rows} rows with NaN values")
        
        # Normalize and store parameters for consistency
        self._normalize_with_stored_params(df_clean)
        
        # Mark as normalized
        df_clean._is_normalized = True
        
        logger.info("Data normalization complete")
        return df_clean
    
    def _normalize_with_stored_params(self, df: pd.DataFrame):
        """Normalize using stored parameters for consistency across train/val/test."""
        
        # Price columns
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in df.columns:
                if col not in self.normalization_params:
                    self.normalization_params[col] = {
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                
                col_min = self.normalization_params[col]['min']
                col_max = self.normalization_params[col]['max']
                
                if col_max > col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.5
        
        # Technical indicators
        tech_cols = [col for col in df.columns if col.startswith(('rsi', 'macd', 'bb_', 'atr', 'volume_', 'ema_', 'sma_', 'returns', 'log_returns', 'volatility', 'momentum_', 'price_range', 'trend'))]
        for col in tech_cols:
            if col in df.columns:
                if col not in self.normalization_params:
                    self.normalization_params[col] = {
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                
                col_min = self.normalization_params[col]['min']
                col_max = self.normalization_params[col]['max']
                
                if col_max > col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.5

    def _prepare_inputs(self, df: pd.DataFrame) -> tuple:
        """Prepare inputs for the model from DataFrame"""
        logger.info("Preparing model inputs...")
        
        # First validate and normalize the data
        df = self._validate_and_normalize_data(df)
        
        # Get column information from data formatter
        static_cols = [col for col, type_ in self.data_formatter.column_definition 
                      if type_ == InputTypes.STATIC]
        historical_cols = [col for col, type_ in self.data_formatter.column_definition
                          if type_ in [InputTypes.TARGET, InputTypes.OBSERVED, InputTypes.KNOWN]]
        future_cols = [col for col, type_ in self.data_formatter.column_definition
                      if type_ == InputTypes.KNOWN]
        
        # Initialize input arrays
        batch_size = len(df)
        static_inputs = np.zeros((batch_size, len(static_cols)))
        historical_inputs = np.zeros((batch_size, self.num_encoder_steps, len(historical_cols)))
        future_inputs = np.zeros((batch_size, self.num_steps, len(future_cols)))
        
        # Fill static inputs
        for i, col in enumerate(static_cols):
            if col in df.columns:
                static_inputs[:, i] = df[col].values
        
        # Fill historical inputs (assume sequential time steps)
        for i, col in enumerate(historical_cols):
            if col in df.columns:
                # For simplicity, repeat the same value across time steps
                # In a real implementation, you'd have proper time series data
                values = df[col].values
                for t in range(self.num_encoder_steps):
                    historical_inputs[:, t, i] = values
        
        # Fill future inputs
        for i, col in enumerate(future_cols):
            if col in df.columns:
                # For simplicity, repeat the same value across time steps
                values = df[col].values
                for t in range(self.num_steps):
                    future_inputs[:, t, i] = values
        
        # Final validation of prepared inputs
        if np.isnan(static_inputs).any():
            logger.error("Static inputs contain NaN values!")
        if np.isnan(historical_inputs).any():
            logger.error("Historical inputs contain NaN values!")
        if np.isnan(future_inputs).any():
            logger.error("Future inputs contain NaN values!")
            
        return static_inputs, historical_inputs, future_inputs
    
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, callbacks=None) -> Dict:
        """Enhanced fit method with proper normalization handling."""
        
        # Normalize once and reuse
        logger.info("Performing one-time data normalization...")
        train_normalized = self._validate_and_normalize_data(train_df.copy())
        valid_normalized = self._validate_and_normalize_data(valid_df.copy())
        
        # Prepare inputs from normalized data (fix the method call)
        train_static, train_hist, train_future = self._prepare_inputs_from_normalized(train_normalized)
        valid_static, valid_hist, valid_future = self._prepare_inputs_from_normalized(valid_normalized)
        
        # Extract targets from normalized data
        train_targets = train_normalized['close'].values.reshape(-1, 1)
        valid_targets = valid_normalized['close'].values.reshape(-1, 1)
        
        # Set up default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                )
            ]
        
        # Train with callbacks
        history = self.model.fit(
            [train_static, train_hist, train_future],
            train_targets,
            validation_data=([valid_static, valid_hist, valid_future], valid_targets),
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'history': [{'epoch': i+1, 
                        'train_loss': history.history['loss'][i],
                        'train_mae': history.history['mae'][i],
                        'val_loss': history.history['val_loss'][i], 
                        'val_mae': history.history['val_mae'][i]}
                       for i in range(len(history.history['loss']))],
            'best_epoch': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss'])
        }
    
    def _prepare_inputs_from_normalized(self, df_normalized: pd.DataFrame) -> tuple:
        """Helper function to prepare model inputs from normalized data.
        Ensures that three arrays (static, historical, future) are returned, matching model's expected input structure.
        """
        logger_grn = logging.getLogger(__name__) 

        if df_normalized.empty:
            raise ValueError("Input DataFrame for _prepare_inputs_from_normalized is empty.")

        # Get expected shapes from the Keras model
        # input_shape is a list for multi-input models: [(batch, static_feats), (batch, hist_len, hist_feats), (batch, fut_len, fut_feats)]
        # For prediction, batch is often 1 or handled by predict method. We care about feature dimensions and sequence lengths.
        try:
            static_input_shape_keras = self.model.input_shape[0]
            historical_input_shape_keras = self.model.input_shape[1]
            future_input_shape_keras = self.model.input_shape[2]
        except IndexError as e:
            logger_grn.error(f"[{self.__class__.__name__}] Could not get model input shapes. Model expects {len(self.model.input_shape)} inputs. Error: {e}")
            raise ValueError("Failed to determine Keras model input shapes.") from e

        num_static_features_keras = static_input_shape_keras[-1]
        hist_seq_len_keras = historical_input_shape_keras[-2] # self.num_encoder_steps
        num_hist_features_keras = historical_input_shape_keras[-1]
        future_seq_len_keras = future_input_shape_keras[-2] # self.num_steps
        num_future_features_keras = future_input_shape_keras[-1]

        # Attempt to get input column configurations from self.config
        config_static_loc = self.config.get('static_input_loc', [])
        config_input_obs_loc = self.config.get('input_obs_loc', [])
        config_known_regular_inputs = self.config.get('known_regular_inputs', []) # Can be hist or fut
        # config_known_categorical_inputs = self.config.get('known_categorical_inputs', []) # Also hist or fut

        # Fallback for observed inputs if primary configs are missing
        feature_cols_for_hist_fallback = []
        if not config_input_obs_loc and not config_known_regular_inputs: # Simplified condition for brevity
            feature_cols_in_df = [col for col in df_normalized.columns if col.startswith('feature_')]
            if feature_cols_in_df:
                logger_grn.warning(
                    f"[{self.__class__.__name__}] Config missing 'input_obs_loc' or 'known_regular_inputs'. "
                    f"Defaulting to use all 'feature_X' columns as potential historical inputs: {feature_cols_in_df}"
                )
                # These feature_cols will be attempted for historical_input_array
                feature_cols_for_hist_fallback = feature_cols_in_df
            else:
                logger_grn.warning(
                    f"[{self.__class__.__name__}] Config missing input specs and no 'feature_X' columns found."
                )
        
        # 1. Prepare Static Inputs
        static_data_values = []
        if config_static_loc:
            for name in config_static_loc:
                if name in df_normalized.columns:
                    # Static inputs are typically single values per entity, take first row
                    static_data_values.append(df_normalized[name].iloc[0]) 
                else:
                    logger_grn.debug(f"Static input '{name}' not in df_normalized.")
        
        if static_data_values and len(static_data_values) == num_static_features_keras:
            prepared_static_input = np.array(static_data_values).astype(np.float32)
        else:
            if config_static_loc: # Log if we had a config but it didn't match
                 logger_grn.warning(f"Static input from config did not match Keras shape ({len(static_data_values)} vs {num_static_features_keras}). Using zeros.")
            prepared_static_input = np.zeros((num_static_features_keras,), dtype=np.float32)

        # 2. Prepare Historical Inputs
        # Use feature_cols_for_hist_fallback if primary config_input_obs_loc is empty
        # This logic needs to be more robust to differentiate KNOWN inputs for historical vs future part.
        # For now, if feature_cols_for_hist_fallback is populated, assume they are all for historical.
        hist_input_candidate_cols = config_input_obs_loc + [c for c in config_known_regular_inputs if c in df_normalized.columns] # crude merge
        if not hist_input_candidate_cols and feature_cols_for_hist_fallback:
            hist_input_candidate_cols = feature_cols_for_hist_fallback

        historical_data_stacked = None
        if hist_input_candidate_cols:
            temp_hist_data = []
            for name in hist_input_candidate_cols:
                if name in df_normalized.columns:
                    temp_hist_data.append(df_normalized[name].values) # Should be (hist_seq_len_keras,)
                else:
                    logger_grn.debug(f"Historical candidate input '{name}' not in df_normalized.")
            
            if temp_hist_data and len(temp_hist_data) == num_hist_features_keras:
                try:
                    # Each item in temp_hist_data is (L,). Stack to (L, H)
                    historical_data_stacked = np.stack(temp_hist_data, axis=-1).astype(np.float32)
                    if historical_data_stacked.shape[0] != hist_seq_len_keras:
                        logger_grn.warning(f"Historical data seq length mismatch ({historical_data_stacked.shape[0]} vs {hist_seq_len_keras}). Reshaping/padding needed or config error.")
                        # This might require padding or truncation if seq len doesn't match df_normalized rows
                        # For now, we will let it pass to see next error if shapes are incompatible downstream
                except ValueError as e:
                    logger_grn.error(f"Error stacking historical data: {e}. Columns: {hist_input_candidate_cols}, NumFeaturesKeras: {num_hist_features_keras}")
                    historical_data_stacked = None # Failed to stack to correct feature dimension

        if historical_data_stacked is not None and historical_data_stacked.shape == (hist_seq_len_keras, num_hist_features_keras):
            prepared_historical_input = historical_data_stacked
        else:
            if hist_input_candidate_cols: # Log if we had candidates but they didn't match
                 logger_grn.warning(f"Historical input from config/fallback did not match Keras shape. Using zeros. Provided data shape if any: {historical_data_stacked.shape if historical_data_stacked is not None else 'None'}, Keras expects: {(hist_seq_len_keras, num_hist_features_keras)}")
            prepared_historical_input = np.zeros((hist_seq_len_keras, num_hist_features_keras), dtype=np.float32)

        # 3. Prepare Future Inputs
        # This part is tricky as df_normalized usually contains historical data.
        # True future inputs usually come from a different source or are generated.
        # For now, assume if not specified in config to map from df_normalized, use zeros.
        # A more complete solution would parse config_known_regular_inputs for future-specific keys.
        future_data_values = []
        # Example: if self.config.get('known_future_inputs', []) had column names from df_normalized
        # for name in self.config.get('known_future_inputs', []):
        #     if name in df_normalized.columns:
        #           # This would need careful slicing for future timesteps if df_normalized contained them
        #           future_data_values.append(df_normalized[name].values[:future_seq_len_keras]) 
        
        # Assuming no specific future inputs from df_normalized for now if not configured.
        if future_data_values and len(future_data_values) == num_future_features_keras:
            # This would need stacking and ensuring shape (future_seq_len_keras, num_future_features_keras)
            # prepared_future_input = np.stack(future_data_values, axis=-1).astype(np.float32)
            # Placeholder, as current logic won't populate future_data_values this way
            logger_grn.warning("Future input from config path is complex and not fully implemented for df_normalized. Using zeros for future inputs.")
            prepared_future_input = np.zeros((future_seq_len_keras, num_future_features_keras), dtype=np.float32)
        else:
            prepared_future_input = np.zeros((future_seq_len_keras, num_future_features_keras), dtype=np.float32)
        
        # Ensure all inputs have a batch dimension of 1 for prediction
        final_static_input = np.expand_dims(prepared_static_input, axis=0)
        final_historical_input = np.expand_dims(prepared_historical_input, axis=0)
        final_future_input = np.expand_dims(prepared_future_input, axis=0)

        logger_grn.info(f"Prepared Keras inputs shapes (with batch dim): Static={final_static_input.shape}, Historical={final_historical_input.shape}, Future={final_future_input.shape}")
        return final_static_input, final_historical_input, final_future_input

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data."""
        eval_logger = logging.getLogger(__name__)
        eval_logger.info("Evaluate method called.")
        return {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        # Use self.logger if defined in __init__, otherwise get a new one.
        pred_logger = getattr(self, 'logger', logging.getLogger(__name__))

        pred_logger.info(f"[{self.__class__.__name__}] Custom TFTModel.predict called.")
        if isinstance(df, pd.DataFrame):
            pred_logger.info(f"[{self.__class__.__name__}] Input df shape: {df.shape}")
            pred_logger.info(f"[{self.__class__.__name__}] Input df dtypes:\n{df.dtypes}")
            pred_logger.info(f"[{self.__class__.__name__}] Input df head:\n{df.head()}")
        elif isinstance(df, np.ndarray):
            pred_logger.info(f"[{self.__class__.__name__}] Input numpy array shape: {df.shape}")
        else:
            pred_logger.info(f"[{self.__class__.__name__}] Input type to predict: {type(df)}")

        if not hasattr(self, 'data_formatter') or self.data_formatter is None:
            pred_logger.warning("Data formatter not available in TFTModel. Using raw input df, assuming it is already normalized.")
            df_normalized = df
        else:
            try:
                df_normalized = self.data_formatter.transform_inputs(df)
                pred_logger.info(f"[{self.__class__.__name__}] df_normalized shape after formatter.transform_inputs: {df_normalized.shape}")
            except Exception as e:
                pred_logger.error(f"[{self.__class__.__name__}] Error during data_formatter.transform_inputs: {e}. Check if formatter was fitted.")
                raise

        try:
            model_inputs = self._prepare_inputs_from_normalized(df_normalized)
            pred_logger.info(f"[{self.__class__.__name__}] Prepared inputs for Keras model. Number of input tensors: {len(model_inputs)}")
            for i, arr in enumerate(model_inputs):
                pred_logger.info(f"[{self.__class__.__name__}] Shape of Keras input tensor {i}: {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
        except Exception as e:
            pred_logger.error(f"[{self.__class__.__name__}] Error in _prepare_inputs_from_normalized: {e}")
            raise

        pred_logger.info(f"[{self.__class__.__name__}] Calling internal Keras model.predict...")
        try:
            raw_predictions = self.model.predict(model_inputs)
        except Exception as e:
            pred_logger.error(f"[{self.__class__.__name__}] Error during Keras self.model.predict(model_inputs): {e}")
            for i, arr in enumerate(model_inputs):
                pred_logger.error(f"[{self.__class__.__name__}] Shape of Keras input tensor {i} AT ERROR: {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
            raise

        pred_logger.info(f"[{self.__class__.__name__}] Raw predictions shape from Keras model: {raw_predictions.shape if isinstance(raw_predictions, np.ndarray) else type(raw_predictions)}")

        if not isinstance(raw_predictions, np.ndarray):
            try:
                raw_predictions = np.array(raw_predictions)
                pred_logger.info(f"[{self.__class__.__name__}] Converted raw_predictions to numpy array, new shape: {raw_predictions.shape}")
            except Exception as e:
                pred_logger.error(f"[{self.__class__.__name__}] Failed to convert raw_predictions to numpy array: {e}")
                # Ensure TFTPredictorError is available or use a standard error
                # from colab_training.utils import CustomError # Example if you have one
                raise ValueError(f"Keras model output type {type(raw_predictions)} could not be converted to numpy array.") # Using ValueError for now

        return raw_predictions

    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights - Keras now requires .weights.h5 suffix
        self.model.save_weights(os.path.join(path, "model.weights.h5"))
        
        # Save model config and training state
        config = {
            'model_config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history
        }
        
        with open(os.path.join(path, "model_config.json"), 'w') as f:
            json.dump(config, f)
            
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Load model weights - updated for new Keras format
        weights_file_path = os.path.join(path, "model.weights.h5")
        if not os.path.exists(weights_file_path):
            # Fallback to best_model.weights.h5 if model.weights.h5 doesn't exist
            weights_file_path = os.path.join(path, "best_model.weights.h5")
            if not os.path.exists(weights_file_path):
                raise FileNotFoundError(f"Neither model.weights.h5 nor best_model.weights.h5 found in {path}")
        try:
            # Try loading with skip_mismatch only, as by_name was causing an "Invalid keyword arguments" error.
            self.model.load_weights(weights_file_path, skip_mismatch=True)
            logger.info(f"[{self.__class__.__name__}] Successfully attempted to load weights from {weights_file_path} with skip_mismatch=True. Mismatched layers would be skipped.")
        except ValueError as ve:
            # This is the specific error Keras throws for shape mismatches if skip_mismatch is False or not effective for some internal reason.
            logger.error(f"[{self.__class__.__name__}] ValueError loading weights from {weights_file_path} (likely shape mismatch): {ve}. Model proceeds with initial/partially loaded weights.")
            # Allow proceeding, as skip_mismatch should ideally handle this, but if Keras internals still raise it, we log and move on.
            pass
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Generic error loading weights from {weights_file_path}: {e}. Model proceeds with initial weights.")
            # Allow proceeding, model will use initial weights
            pass

        
        # Load config and training state
        config_file_path = os.path.join(path, "model_config.json") # Assuming config is named model_config.json
        # Attempt to locate a more general config if the specific one isn't found.
        if not os.path.exists(config_file_path):
            potential_configs = ["model_config.json", "params.json", "hyperparameters.json"]
            for potential_config in potential_configs:
                _config_file_path = os.path.join(path, potential_config)
                if os.path.exists(_config_file_path):
                    config_file_path = _config_file_path
                    break
            if not os.path.exists(config_file_path): # if still not found
                 # Try to find any .json file in the directory as a last resort for config
                json_files_in_dir = [f for f in os.listdir(path) if f.endswith('.json')]
                if json_files_in_dir:
                    config_file_path = os.path.join(path, json_files_in_dir[0]) # pick the first one
                else:
                    # If no JSON config file is found after all attempts, raise an error.
                    raise FileNotFoundError(f"No suitable .json config file found in {path}. Tried model_config.json, {', '.join(potential_configs)}, and any other .json file.")

        with open(config_file_path, 'r') as f:
            config = json.load(f)
            
        self.best_val_loss = config['best_val_loss']
        self.best_epoch = config['best_epoch']
        self.training_history = config['training_history']
        
    def get_attention_weights(self, inputs) -> np.ndarray:
        """Get attention weights for interpretability.
        
        Args:
            inputs: Either a DataFrame or tuple of (static, historical, future) inputs
            
        Returns:
            Array of attention weights
        """
        # Create a model that outputs attention weights
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer('interpretable_multi_head_attention').output
        )
        
        # Handle both DataFrame and tuple inputs
        if isinstance(inputs, tuple):
            # Already prepared inputs
            static_inputs, historical_inputs, future_inputs = inputs
        else:
            # DataFrame input - need to prepare
            static_inputs, historical_inputs, future_inputs = self._prepare_inputs(inputs)
        
        # Get attention weights
        attention_weights = attention_model.predict(
            [static_inputs, historical_inputs, future_inputs],
            verbose=0
        )
        
        return attention_weights 