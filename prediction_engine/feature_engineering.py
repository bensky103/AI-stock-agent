"""Feature engineering module for stock prediction.

This module provides functionality for generating and transforming features
for TFT-based stock prediction models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator
from .sequence_preprocessor import SequencePreprocessor

# Configure logging
logger = logging.getLogger(__name__)
# Don't configure handlers here, just get the logger
logger.setLevel(logging.INFO)

class MarketRegimeDetector:
    """Detect market regimes using technical indicators."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime using multiple indicators.
        
        Returns
        -------
        pd.Series
            Market regime labels: 1 (bullish), 0 (neutral), -1 (bearish)
        """
        # Calculate trend indicators
        sma_20 = SMAIndicator(close=df['close'], window=20).sma_indicator()
        sma_50 = SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        # ADX calculation with data length check
        adx_window = 14  # Default window for ADXIndicator
        min_periods_adx = (2 * adx_window) - 1
        adx = pd.Series(np.nan, index=df.index, name='adx') # Initialize with NaNs

        if len(df) >= min_periods_adx:
            try:
                adx_indicator = ADXIndicator(
                    high=df['high'], 
                    low=df['low'], 
                    close=df['close'], 
                    window=adx_window, 
                    fillna=True
                )
                adx = adx_indicator.adx()
            except IndexError as ie:
                logger.warning(
                    f"[MarketRegimeDetector] IndexError calculating ADX({adx_window}) with {len(df)} rows (min {min_periods_adx}): {ie}. "
                    f"ADX will contain NaNs."
                )
                # adx is already initialized with NaNs
            except Exception as e:
                logger.error(
                    f"[MarketRegimeDetector] Unexpected error calculating ADX({adx_window}) with {len(df)} rows: {e}. "
                    f"ADX will contain NaNs."
                )
                # adx is already initialized with NaNs
        else:
            logger.warning(
                f"[MarketRegimeDetector] Insufficient data ({len(df)} rows) for ADX({adx_window}). Need {min_periods_adx}. "
                f"ADX will contain NaNs."
            )
            # adx is already initialized with NaNs
        
        # Calculate momentum indicators
        rsi = RSIIndicator(close=df['close']).rsi()
        macd = MACD(close=df['close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        
        # Calculate volatility
        bb = BollingerBands(close=df['close'])
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Combine indicators for regime detection
        regime = pd.Series(0, index=df.index)  # Initialize as neutral
        
        # Trend-based rules
        regime[df['close'] > sma_20] += 1
        regime[df['close'] > sma_50] += 1
        regime[adx > 25] += 1  # Strong trend
        
        # Momentum-based rules
        regime[rsi > 60] += 1
        regime[rsi < 40] -= 1
        regime[macd_line > signal_line] += 1
        regime[macd_line < signal_line] -= 1
        
        # Volatility-based rules
        regime[bb_width > bb_width.rolling(window=self.window).mean()] -= 1  # High volatility
        
        # Normalize to [-1, 1]
        regime = regime / regime.abs().max()
        
        # Apply smoothing
        regime = regime.rolling(window=self.window).mean()
        
        return regime

class FeatureEngineer:
    """
    Enhanced feature engineering for stock prediction.
    
    This class handles:
    1. Technical indicators calculation
    2. Market regime detection
    3. Feature selection and dimensionality reduction
    4. Advanced data normalization
    5. Sequence preparation for TFT models
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        technical_indicators: Optional[List[str]] = None,
        normalize: bool = True,
        use_feature_selection: bool = True,
        n_features: int = 20,
        use_pca: bool = False,
        n_components: int = 10,
        detect_regime: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.use_pca = use_pca
        self.n_components = n_components
        self.detect_regime = detect_regime
        
        # Default technical indicators
        self.technical_indicators = technical_indicators or [
            'sma_20', 'sma_50', 'ema_20', 'ema_50',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr_14', 'vwap', 'mfi_14', 'stoch_k', 'stoch_d',
            'adx_14', 'cci_14', 'williams_r'
        ]
        
        # Initialize components
        self.sequence_preprocessor = SequencePreprocessor(sequence_length)
        self.market_regime_detector = MarketRegimeDetector() if detect_regime else None
        self.feature_selector = SelectKBest(f_regression, k=n_features) if use_feature_selection else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.scaler = RobustScaler() if normalize else None  # More robust to outliers
        
        # Store transformation parameters
        self.feature_means = None
        self.feature_stds = None
        self.selected_features = None
        self.pca_components = None
        self.target_scaler_params = None # To store params for 'close' column
        
        logger.info(
            f"Initialized feature engineer with {len(self.technical_indicators)} "
            f"technical indicators"
        )
    
    def _find_actual_column_name(self, columns_list: Union[pd.Index, List], generic_name: str) -> Optional[Union[str, tuple]]:
        """Helper to find actual column name (str or tuple) matching generic_name (case-insensitive)."""
        ln_generic_name = generic_name.lower()
        for col_name in columns_list:
            if isinstance(col_name, tuple) and col_name and str(col_name[0]).lower() == ln_generic_name:
                return col_name
            elif isinstance(col_name, str) and col_name.lower() == ln_generic_name:
                return col_name
        return None

    def is_scaler_fitted(self) -> bool:
        """Check if the scaler has been fitted."""
        if not self.normalize or self.scaler is None:
            # If normalization is off or no scaler, it's effectively "fitted" or not needed.
            return True
        # RobustScaler sets 'center_' and 'scale_' attributes upon fitting.
        return hasattr(self.scaler, 'center_') and self.scaler.center_ is not None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Convert column names to lowercase for consistency
        df = df.copy()
        
        # Safely convert column names to lowercase
        if isinstance(df.columns, pd.MultiIndex):
            # If we have a MultiIndex, we can't use str.lower() directly
            # Instead, just use the dataframe as is
            logger.warning("MultiIndex detected, skipping column name conversion")
        else:
            # For regular Index, convert column names to lowercase
            df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        result = df.copy()
        
        # Ensure required columns exist (handle both cases)
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        upper_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check available columns
        has_lower = all(col in df.columns for col in required_fields)
        has_upper = all(col in df.columns for col in upper_fields)
        
        # Choose appropriate column names
        if has_lower:
            field_map = {f: f for f in required_fields}
        elif has_upper:
            field_map = dict(zip(required_fields, upper_fields))
        else:
            raise ValueError(f"Missing required columns. Need {required_fields} or {upper_fields}")
        
        # Extract field names based on availability
        open_col = field_map['open']
        high_col = field_map['high']
        low_col = field_map['low']
        close_col = field_map['close']
        volume_col = field_map['volume']
        
        # Helper function to conditionally squeeze if DataFrame
        def _squeeze_if_df(data):
            if isinstance(data, pd.DataFrame):
                return data.squeeze('columns')
            return data

        # Trend indicators
        result['sma_20'] = SMAIndicator(close=_squeeze_if_df(df[close_col]), window=20).sma_indicator()
        result['sma_50'] = SMAIndicator(close=_squeeze_if_df(df[close_col]), window=50).sma_indicator()
        result['ema_20'] = EMAIndicator(close=_squeeze_if_df(df[close_col]), window=20).ema_indicator()
        result['ema_50'] = EMAIndicator(close=_squeeze_if_df(df[close_col]), window=50).ema_indicator()
        
        # Momentum indicators
        result['rsi_14'] = RSIIndicator(close=_squeeze_if_df(df[close_col])).rsi()
        macd = MACD(close=_squeeze_if_df(df[close_col]))
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_hist'] = macd.macd_diff()
        
        # Volatility indicators
        bb = BollingerBands(close=_squeeze_if_df(df[close_col]))
        result['bb_upper'] = bb.bollinger_hband()
        result['bb_middle'] = bb.bollinger_mavg()
        result['bb_lower'] = bb.bollinger_lband()
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['atr_14'] = AverageTrueRange(high=_squeeze_if_df(df[high_col]), low=_squeeze_if_df(df[low_col]), close=_squeeze_if_df(df[close_col])).average_true_range()
        
        # Volume indicators
        result['vwap'] = VolumeWeightedAveragePrice(
            high=_squeeze_if_df(df[high_col]), low=_squeeze_if_df(df[low_col]), close=_squeeze_if_df(df[close_col]), volume=_squeeze_if_df(df[volume_col])
        ).volume_weighted_average_price()
        result['mfi_14'] = MFIIndicator(
            high=_squeeze_if_df(df[high_col]), low=_squeeze_if_df(df[low_col]), close=_squeeze_if_df(df[close_col]), volume=_squeeze_if_df(df[volume_col])
        ).money_flow_index()
        
        # Additional momentum indicators
        stoch = StochasticOscillator(high=_squeeze_if_df(df[high_col]), low=_squeeze_if_df(df[low_col]), close=_squeeze_if_df(df[close_col]))
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()
        
        # Trend strength indicators
        adx_window = 14  # Default window for ADXIndicator, matching 'adx_14'
        # Minimum data points needed for the first ADX value is typically 2*W-1
        min_periods_adx = (2 * adx_window) - 1
        
        if len(df) >= min_periods_adx:
            try:
                adx_indicator = ADXIndicator(
                    high=_squeeze_if_df(df[high_col]),
                    low=_squeeze_if_df(df[low_col]),
                    close=_squeeze_if_df(df[close_col]),
                    window=adx_window,
                    fillna=True # Use fillna=True for graceful handling of initial NaNs
                )
                result['adx_14'] = adx_indicator.adx()
            except IndexError as ie: 
                logger.warning(
                    f"IndexError calculating ADX({adx_window}) with {len(df)} rows (expected min {min_periods_adx}): {ie}. "
                    f"Filling 'adx_14' with NaNs."
                )
                result['adx_14'] = pd.Series(np.nan, index=df.index, name='adx_14')
            except Exception as e: 
                logger.error(
                    f"Unexpected error calculating ADX({adx_window}) with {len(df)} rows: {e}. "
                    f"Filling 'adx_14' with NaNs."
                )
                result['adx_14'] = pd.Series(np.nan, index=df.index, name='adx_14')
        else:
            logger.warning(
                f"Insufficient data ({len(df)} rows) for ADX({adx_window}). "
                f"Need at least {min_periods_adx}. Filling 'adx_14' with NaNs."
            )
            result['adx_14'] = pd.Series(np.nan, index=df.index, name='adx_14')
        
        # Market regime
        if self.detect_regime:
            # Create a temporary DataFrame with standardized column names
            regime_df = pd.DataFrame({
                'close': _squeeze_if_df(df[close_col]),
                'high': _squeeze_if_df(df[high_col]),
                'low': _squeeze_if_df(df[low_col])
            })
            result['market_regime'] = self.market_regime_detector.detect_regime(regime_df)
        
        return result
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """Normalize features using RobustScaler.
        
        If fit is True, scaler is fitted and target scaling parameters are stored.
        """
        # Ensure 'close' column exists if we intend to store its scaling params
        actual_close_col_for_df = self._find_actual_column_name(df.columns, 'close')
        
        if not actual_close_col_for_df:
            logger.warning(f"'close' column (or equivalent like ('Close', ...)) not found in DataFrame columns ({list(df.columns)}) for normalize_features. Cannot store target scaling params.")
            # Proceed with normalization if possible, but target denormalization will fail.
        
        # Identify feature columns (all numeric columns except known non-feature like 'date', 'symbol')
        # This logic should be robust to what df contains at this stage.
        # Typically, df here would be post-indicator calculation.
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Exclude known non-feature columns if they are somehow numeric and present
        # This is a safeguard; ideally, df passed here is purely features + target.
        potential_non_features = ['time_idx', 'group_id', 'identifier', 'date', 'symbol'] # Add any other known ID/time cols
        
        # Handle cases where column names might be tuples (e.g., from MultiIndex)
        filtered_feature_cols = []
        for col in feature_cols:
            # Convert column to string for comparison; join if tuple, else convert to string
            col_str = '_'.join(map(str, col)) if isinstance(col, tuple) else str(col)
            if col_str.lower() not in potential_non_features:
                filtered_feature_cols.append(col) # Keep original column identifier
        feature_cols = filtered_feature_cols

        if not feature_cols:
            logger.warning("No feature columns identified for normalization.")
            return df, None, None # Or self.feature_means, self.feature_stds if they were meant to be set

        # Initialize means and stds if fitting
        if fit:
            self.feature_means = df[feature_cols].mean().values
            self.feature_stds = df[feature_cols].std().values
        
        # Apply RobustScaler for normalization
        if self.normalize and self.scaler is not None:
            if fit:
                logger.info(f"Fitting scaler on feature columns: {feature_cols}")
                self.scaler.fit(df[feature_cols])
                
                # Store scaling parameters for the 'close' column if present
                actual_close_col_in_features = self._find_actual_column_name(feature_cols, 'close')
                
                if actual_close_col_in_features is not None:
                    try:
                        # feature_cols is a list here, get the index of the actual column name
                        close_idx = feature_cols.index(actual_close_col_in_features)
                        self.target_scaler_params = {
                            'center': self.scaler.center_[close_idx],
                            'scale': self.scaler.scale_[close_idx]
                        }
                        logger.info(f"Stored target scaler params for '{actual_close_col_in_features}': {self.target_scaler_params}")
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.error(f"Could not get/store target scaler params for '{actual_close_col_in_features}': {e}. Scaler arrays might not align, or column not in list used for fit.")
                        self.target_scaler_params = None 
                else:
                    logger.warning(f"'close' column (or equivalent) not found among feature_cols used for scaler fitting: {feature_cols}. Cannot store its specific scaling params.")
                    self.target_scaler_params = None


            # Check if scaler is fitted before transforming
            if not self.is_scaler_fitted():
                logger.error("Scaler is not fitted, cannot transform. Call with fit=True first or ensure scaler is pre-fitted.")
                # Potentially return df unmodified or raise error
                return df, self.feature_means, self.feature_stds


            # Perform transformation
            # Ensure columns exist before attempting to transform
            cols_to_transform = [col for col in feature_cols if col in df.columns]
            if not cols_to_transform:
                logger.warning("No columns found in DataFrame to apply scaler transform.")
            else:
                df[cols_to_transform] = self.scaler.transform(df[cols_to_transform])
        
        return df, self.feature_means, self.feature_stds

    def inverse_transform_target(self, scaled_target_value: float) -> float:
        """Inverse transform a scaled target value (e.g., 'close' price)."""
        if self.target_scaler_params is None:
            logger.warning("Target scaler parameters not available. Cannot inverse transform target. Returning scaled value.")
            return scaled_target_value
        
        center = self.target_scaler_params['center']
        scale = self.target_scaler_params['scale']
        
        original_value = (scaled_target_value * scale) + center
        logger.info(f"Inverse transformed target: scaled={scaled_target_value}, original={original_value}")
        return original_value

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Select most relevant features."""
        if not self.use_feature_selection:
            return X, None
        
        if fit:
            # Fit feature selector
            self.feature_selector.fit(X, y)
            self.selected_features = self.feature_selector.get_support()
        
        # Transform data
        if self.selected_features is not None:
            X_selected = X[:, self.selected_features]
        else:
            X_selected = X
        
        return X_selected, self.selected_features
    
    def apply_pca(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply PCA for dimensionality reduction."""
        if not self.use_pca:
            return X, None
        
        if fit:
            # Fit PCA
            self.pca.fit(X)
            self.pca_components = self.pca.components_
        
        # Transform data
        if self.pca_components is not None:
            X_pca = self.pca.transform(X)
        else:
            X_pca = X
        
        return X_pca, self.pca_components
    
    def prepare_sequences(
        self,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for TFT model input.
        
        Args:
            market_data: DataFrame containing market data
            sentiment_data: Optional DataFrame containing sentiment data
            fit: Whether to fit transformers on the data
            
        Returns:
            Tuple of (market_features, targets, sentiment_features)
            market_features: numpy array of shape (n_samples, sequence_length, n_features)
            targets: numpy array of shape (n_samples, prediction_horizon)
            sentiment_features: Optional numpy array of shape (n_samples, n_sentiment_features)
        """
        # Calculate technical indicators
        market_with_indicators = self.calculate_technical_indicators(market_data)
        
        # Normalize market data
        market_normalized, means, stds = self.normalize_features(
            market_with_indicators,
            fit=fit
        )
        
        actual_target_col_name = self._find_actual_column_name(market_normalized.columns, 'close')

        if actual_target_col_name is None:
            logger.error(f"'close' column (or equivalent) not found in market_normalized columns ({list(market_normalized.columns)}) for sequence preparation. This is critical and will likely lead to errors.")
            # Optionally, raise an error to stop mis-processing:
            raise ValueError("Target 'close' column could not be identified in market_normalized data for sequence preparation.")

        # Prepare sequences using the actual identified target column name
        feature_column_names_for_sequence = [col for col in market_normalized.columns if col != actual_target_col_name]
        
        sequences = self.sequence_preprocessor.prepare_sequence(
            market_normalized,
            target_col=actual_target_col_name,
            feature_cols=feature_column_names_for_sequence
        )
        
        X = sequences['features']
        y = sequences['targets']
        
        # Apply feature selection if enabled
        if self.use_feature_selection:
            X, selected_features = self.select_features(X, y, fit=fit)
        
        # Apply PCA if enabled
        if self.use_pca:
            X, pca_components = self.apply_pca(X, fit=fit)
        
        # Prepare sentiment features if available
        sentiment_features = None
        if sentiment_data is not None and not sentiment_data.empty:
            # Align sentiment data with market data
            sentiment_aligned = sentiment_data.set_index(['symbol', 'datetime'])
            sentiment_aligned = sentiment_aligned.reindex(market_data.index)
            
            # Normalize sentiment features
            sentiment_normalized, _, _ = self.normalize_features(
                sentiment_aligned,
                fit=fit
            )
            
            # Get sentiment features for each sequence
            sentiment_features = []
            for i in range(len(X)):
                sentiment_features.append(
                    sentiment_normalized.iloc[i + self.sequence_length].values
                )
            sentiment_features = np.array(sentiment_features)
        
        return X, y, sentiment_features
    
    def prepare_prediction_data(
        self,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for making predictions.
        
        Args:
            market_data: DataFrame containing market data
            sentiment_data: Optional DataFrame containing sentiment data
            
        Returns:
            Tuple of (market_features, sentiment_features)
            market_features: numpy array of shape (n_samples, sequence_length, n_features)
            sentiment_features: Optional numpy array of shape (n_samples, n_sentiment_features)
        """
        # Ensure we have enough data
        if len(market_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} time steps of market data")
        
        # Calculate technical indicators
        market_with_indicators = self.calculate_technical_indicators(market_data)
        
        # Normalize market data
        market_normalized, _, _ = self.normalize_features(market_with_indicators)
        
        # Prepare sequences
        sequences = []
        for i in range(len(market_normalized) - self.sequence_length + 1):
            sequence = market_normalized.iloc[i:i + self.sequence_length].values
            sequences.append(sequence)
        
        market_features = np.array(sequences)
        
        # Prepare sentiment features if available
        sentiment_features = None
        if sentiment_data is not None and not sentiment_data.empty:
            # Get sentiment features for each sequence
            sentiment_sequences = []
            for i in range(len(sentiment_data) - self.sequence_length + 1):
                sentiment_sequence = sentiment_data.iloc[i:i + self.sequence_length].values
                sentiment_sequences.append(sentiment_sequence)
            
            sentiment_features = np.array(sentiment_sequences)
        
        return market_features, sentiment_features

    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction.
        
        This is a wrapper around prepare_prediction_data that returns only the 
        market features.
        
        Args:
            market_data: DataFrame containing market data
            
        Returns:
            numpy array of processed features
        """
        # Ensure we have enough data
        if len(market_data) < self.sequence_length:
            logger.error(f"Not enough data: {len(market_data)} rows, need {self.sequence_length}")
            raise ValueError(f"Need at least {self.sequence_length} time steps of market data")
        
        logger.info(f"[{self.__class__.__name__}] Preparing features from input market_data with shape: {market_data.shape}")
        
        # Calculate technical indicators
        market_with_indicators = self.calculate_technical_indicators(market_data)
        logger.info(f"[{self.__class__.__name__}] Market data with indicators shape: {market_with_indicators.shape}")
        
        # Normalize market data
        market_normalized, _, _ = self.normalize_features(market_with_indicators)
        logger.info(f"[{self.__class__.__name__}] Normalized market_normalized data shape: {market_normalized.shape}")
        if market_normalized.empty:
            logger.error(f"[{self.__class__.__name__}] market_normalized is empty after normalization.")
            # Potentially raise an error or return an empty array, depending on desired handling
            raise ValueError("market_normalized is empty after normalization, cannot prepare features.")

        # Prepare sequences
        sequences = []
        # Ensure there's enough data to form at least one sequence
        if len(market_normalized) >= self.sequence_length:
            for i in range(len(market_normalized) - self.sequence_length + 1):
                # Get data for this sequence window
                window_data = market_normalized.iloc[i:i + self.sequence_length]
                logger.debug(f"[{self.__class__.__name__}] Loop {i}: window_data shape: {window_data.shape}")
                
                # Convert to NumPy array, ensuring we have a 2D shape (sequence_length, features)
                if isinstance(window_data, pd.DataFrame):
                    sequence = window_data.values
                    logger.debug(f"[{self.__class__.__name__}] Loop {i}: sequence from DataFrame shape: {sequence.shape}")
                elif isinstance(window_data, pd.Series):
                    sequence = window_data.values
                    logger.debug(f"[{self.__class__.__name__}] Loop {i}: sequence from Series initial shape: {sequence.shape}")
                    if sequence.ndim == 1:
                        # This implies that market_normalized might have only one column.
                        sequence = sequence.reshape(-1, 1)  # Make it 2D: (sequence_length, 1)
                        logger.debug(f"[{self.__class__.__name__}] Loop {i}: Reshaped 1D sequence to: {sequence.shape}")
                else:
                    logger.warning(f"[{self.__class__.__name__}] Loop {i}: window_data is of unexpected type: {type(window_data)}. Converting to values directly.")
                    sequence = window_data.values # Fallback
                    if sequence.ndim == 1: # Ensure 2D if it became 1D
                        sequence = sequence.reshape(-1,1)
                    logger.debug(f"[{self.__class__.__name__}] Loop {i}: sequence from unexpected type, shape: {sequence.shape}")

                sequences.append(sequence)
        else:
            logger.warning(f"[{self.__class__.__name__}] Not enough data in market_normalized (len: {len(market_normalized)}) to form a sequence of length {self.sequence_length}. Returning empty array.")
            return np.array([]) # Return empty if no sequences can be formed

        if not sequences:
            logger.warning(f"[{self.__class__.__name__}] No sequences were created. market_normalized len: {len(market_normalized)}, sequence_length: {self.sequence_length}. Returning empty array.")
            return np.array([])

        features = np.array(sequences)
        logger.info(f"[{self.__class__.__name__}] Stacked sequences into 'features' with shape: {features.shape}")
        
        # If we have a shape issue with dimensions, resolve it here
        # If features is a 3D array with shape (1, n, m), remove the batch dimension (common if only one sequence was created)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
            logger.info(f"[{self.__class__.__name__}] Squeezed features shape (dim 0 was 1): {features.shape}")
        
        # Handle specific case where we have a 2D array with shape (n, 1) - this should NOT be the final output for most models
        # This might indicate that `market_normalized` only had one feature column, and only one sequence was generated and squeezed.
        if features.ndim == 2 and features.shape[1] == 1:
            logger.warning(f"[{self.__class__.__name__}] 'features' has shape {features.shape} (n, 1). This is unusual for sequence features. Flattening to 1D.")
            features = features.flatten()
            logger.info(f"[{self.__class__.__name__}] Flattened features with shape (n, 1) to 1D: {features.shape}")
        
        # Final check - ensure we don't have a 2D array with shape (n, 1) which causes the error "Data must be 1-dimensional"
        # This specific check might be too aggressive if a (N,1) shape is legitimately needed by some specific downstream Keras layer.
        # However, the error message implies it's not expected.
        if features.ndim == 2 and features.shape[1] == 1:
            logger.warning(f"[{self.__class__.__name__}] Still detected problematic (n, 1) shape post-processing - forcing flatten again. Shape: {features.shape}")
            features = features.flatten()
            logger.info(f"[{self.__class__.__name__}] Final flattened shape: {features.shape}")
        
        logger.info(f"[{self.__class__.__name__}] Final features shape to be returned: {features.shape}")
        return features

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from market data.
        
        Args:
            df: DataFrame containing market data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                or ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with generated features
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Safely convert column names - skip if MultiIndex
        if not isinstance(df.columns, pd.MultiIndex):
            # Convert column names to lowercase using a safe method
            df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Calculate technical indicators
        features = self.calculate_technical_indicators(df)
        
        # Normalize features with fit=True to ensure scaler is fitted
        features, _, _ = self.normalize_features(features, fit=True)
        
        return features

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        sequence_length=10,
        use_feature_selection=True,
        use_pca=True,
        detect_regime=True
    )
    
    # Prepare sequences
    X, y, _ = engineer.prepare_sequences(data, fit=True)
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Selected features: {engineer.selected_features.sum() if engineer.selected_features is not None else 'None'}")
    print(f"PCA components: {engineer.pca_components.shape if engineer.pca_components is not None else 'None'}") 