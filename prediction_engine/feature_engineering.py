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
from sklearn.utils.validation import check_is_fitted # For checking scaler status
from sklearn.exceptions import NotFittedError # For catching scaler errors
import ta
import traceback # For logging exception tracebacks

# Explicitly import from ta submodules to match original usage patterns
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator # Assuming StochasticOscillator might be used
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator # Assuming these might be used
from .sequence_preprocessor import SequencePreprocessor
from .scaler_handler import ScalerHandlerError # Import the custom error

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
        adx_series = pd.Series(np.nan, index=df.index, name='adx') # Initialize with NaNs

        if len(df) >= min_periods_adx:
            try:
                adx_indicator_obj = ADXIndicator(
                    high=df['high'], 
                    low=df['low'], 
                    close=df['close'], 
                    window=adx_window, 
                    fillna=True
                )
                adx_series = adx_indicator_obj.adx()
            except IndexError as ie:
                logger.warning(
                    f"[MarketRegimeDetector] IndexError calculating ADX({adx_window}) with {len(df)} rows (min {min_periods_adx}): {ie}. "
                    f"ADX will contain NaNs."
                )
            except Exception as e:
                logger.error(
                    f"[MarketRegimeDetector] Unexpected error calculating ADX({adx_window}) with {len(df)} rows: {e}. "
                    f"ADX will contain NaNs."
                )
        else:
            logger.warning(
                f"[MarketRegimeDetector] Insufficient data ({len(df)} rows) for ADX({adx_window}). Need {min_periods_adx}. "
                f"ADX will contain NaNs."
            )
        
        # Calculate momentum indicators
        rsi = RSIIndicator(close=df['close']).rsi()
        macd_obj = MACD(close=df['close'])
        macd_line = macd_obj.macd()
        signal_line = macd_obj.macd_signal()
        
        # Calculate volatility
        bb = BollingerBands(close=df['close'])
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Combine indicators for regime detection
        regime = pd.Series(0, index=df.index)  # Initialize as neutral
        
        # Trend-based rules
        regime[df['close'] > sma_20] += 1
        regime[df['close'] > sma_50] += 1
        if not adx_series.empty: # Check if adx_series was populated
             regime[adx_series > 25] += 1  # Strong trend
        
        # Momentum-based rules
        regime[rsi > 60] += 1
        regime[rsi < 40] -= 1
        if not macd_line.empty and not signal_line.empty: # Check if MACD lines were populated
            regime[macd_line > signal_line] += 1
            regime[macd_line < signal_line] -= 1
        
        # Volatility-based rules
        if not bb_width.empty: # Check if bb_width was populated
            regime[bb_width > bb_width.rolling(window=self.window).mean()] -= 1  # High volatility
        
        # Normalize to [-1, 1]
        if not regime.empty and regime.abs().max() != 0:
            regime = regime / regime.abs().max()
        else:
            regime = pd.Series(0, index=df.index) # Default to neutral if max is 0 or empty
        
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
        self.scalers: Dict[str, RobustScaler] = {} # Symbol -> Scaler
        self.global_scaler: Optional[RobustScaler] = None
        self.target_scaler_params: Dict[str, Dict[str, float]] = {} # Initialize as empty dict
        self.global_target_scaler_params: Optional[Dict[str, float]] = None
        self.pca_models: Dict[str, PCA] = {}
        self.global_pca_model: Optional[PCA] = None
        self.feature_columns: Optional[List[str]] = None # Populated after TA calculation
        self.selected_features: Optional[List[str]] = None # Populated after feature selection
        self.target_column_name = 'close' # Default target column name

        logger.info(f"===== FeatureEngineer initialized. Sequence length: {self.sequence_length}, Horizon: {self.prediction_horizon}, Target: '{self.target_column_name}'. Feature selection: {self.use_feature_selection}, PCA: {self.use_pca}, Regime Detection: {self.detect_regime} =====")
        
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

    def is_scaler_fitted(self, symbol: Optional[str] = None) -> bool:
        """Checks if a scaler is fitted for the symbol or globally."""
        scaler_to_check = None
        if symbol and symbol in self.scalers:
            scaler_to_check = self.scalers[symbol]
        elif self.global_scaler is not None:
            scaler_to_check = self.global_scaler
        
        if scaler_to_check is not None:
            try:
                # Check if the scaler has been fitted by looking for attributes like 'center_'
                check_is_fitted(scaler_to_check)
                # Also check if we have target scaling parameters, which are crucial.
                target_params = self._get_target_scaler_params(symbol)
                is_target_params_valid = target_params and \
                                         isinstance(target_params.get('center'), (int, float)) and \
                                         isinstance(target_params.get('scale'), (int, float)) and \
                                         target_params.get('scale') != 0
                if is_target_params_valid:
                    logger.debug(f"===== FeatureEngineer: Scaler for '{symbol if symbol else 'global'}' is fitted and target params are valid. =====")
                    return True
                else:
                    logger.warning(f"===== FeatureEngineer: Scaler for '{symbol if symbol else 'global'}' might be fitted, but target scaler params are missing or invalid: {target_params}. Consider it not fully fitted. =====")
                    return False
            except NotFittedError:
                logger.debug(f"===== FeatureEngineer: Scaler for '{symbol if symbol else 'global'}' is NOT fitted (NotFittedError). =====")
                return False
        logger.debug(f"===== FeatureEngineer: No scaler found for '{symbol if symbol else 'global'}' to check if fitted. =====")
        return False

    def _get_target_scaler_params(self, symbol: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Retrieves the target scaler parameters for the symbol or globally."""
        if symbol and symbol in self.target_scaler_params:
            logger.debug(f"===== FeatureEngineer: Retrieving target scaler params for symbol '{symbol}'. =====")
            return self.target_scaler_params[symbol]
        elif self.global_target_scaler_params is not None:
            logger.debug("===== FeatureEngineer: Retrieving global target scaler params. =====")
            return self.global_target_scaler_params
        
        logger.warning(f"===== FeatureEngineer: No target scaler params available (neither for symbol '{symbol}' nor global). =====")
        return None

    def _get_scaler(self, symbol: Optional[str] = None) -> Optional[RobustScaler]:
        """Retrieves the appropriate scaler (symbol-specific or global)."""
        if symbol and symbol in self.scalers:
            logger.debug(f"===== FeatureEngineer: Retrieving scaler for symbol '{symbol}'. =====")
            return self.scalers[symbol]
        elif self.global_scaler is not None:
            logger.debug("===== FeatureEngineer: Retrieving global scaler. =====")
            return self.global_scaler
        
        logger.warning(f"===== FeatureEngineer: No scaler available (neither for symbol '{symbol}' nor global). Requested scaler but none found. =====")
        return None

    def _convert_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts column names to lowercase and replaces spaces with underscores."""
        if isinstance(df.columns, pd.MultiIndex):
            logger.warning("===== FeatureEngineer: MultiIndex detected, skipping column name conversion =====")
            return df
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        return df

    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: Optional[str] = None, fit_scalers: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """Calculates technical indicators and optionally market regime features."""
        logger.info(f"===== FeatureEngineer: Calculating technical indicators for data with shape {data.shape}. Fit_scalers: {fit_scalers}, Symbol: {symbol}. This adds new columns based on existing price/volume data. =====")
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                logger.info("===== FeatureEngineer: Index converted to DatetimeIndex. =====")
            except Exception as e:
                logger.error(f"===== FeatureEngineer: Failed to convert index to DatetimeIndex: {e}. TA calculation might fail. Index type: {type(df.index)}. First few values: {df.index[:3]} =====")
                # Return None or raise error if DatetimeIndex is critical
                return None, None 

        # Handle MultiIndex columns by taking the first level (e.g., 'Open' from ('Open', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            logger.info(f"===== FeatureEngineer: MultiIndex columns detected: {df.columns.tolist()}. Converting to single-level columns. =====")
            df.columns = df.columns.get_level_values(0)
            logger.info(f"===== FeatureEngineer: Columns after MultiIndex conversion: {df.columns.tolist()} =====")

        df = self._convert_column_names(df)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"===== FeatureEngineer: Missing required columns for TA calculation: {missing_cols}. Available columns: {df.columns.tolist()} =====")
            return None, None

        for col in required_cols:
            if df[col].isnull().any():
                nan_count = df[col].isnull().sum()
                logger.warning(f"===== FeatureEngineer: Column '{col}' has {nan_count} NaN values before TA calculation. Attempting ffill. Data shape: {df.shape} =====")
                df[col] = df[col].ffill()
                if df[col].isnull().any():
                    logger.error(f"===== FeatureEngineer: Column '{col}' still has NaNs after ffill. Cannot proceed with TA calculation. =====")
                    return None, None

        try:
            # Add custom indicators directly using their classes from ta library
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            bbands_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bbands_upper'] = bbands_indicator.bollinger_hband()
            df['bbands_middle'] = bbands_indicator.bollinger_mavg()
            df['bbands_lower'] = bbands_indicator.bollinger_lband()
            
            macd_indicator = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd_line'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()
            
            df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

            logger.info(f"===== FeatureEngineer: Additional custom TAs (SMA, EMA, RSI, BBands, MACD, ATR) applied. Shape after: {df.shape} =====")

        except Exception as e:
            logger.error(f"===== FeatureEngineer: Error during technical indicator calculation: {e} =====")
            logger.error(traceback.format_exc())
            return None, None

        if self.detect_regime:
            logger.info("===== FeatureEngineer: Detecting market regime... =====")
            try:
                # Example: Using rolling volatility to define regimes
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                df['volatility_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(252) # Annualized
                
                # Define regimes (example: low, medium, high volatility)
                low_vol_threshold = df['volatility_20d'].quantile(0.33)
                high_vol_threshold = df['volatility_20d'].quantile(0.67)
                
                df['market_regime'] = 0 # Default (medium)
                df.loc[df['volatility_20d'] < low_vol_threshold, 'market_regime'] = -1 # Low
                df.loc[df['volatility_20d'] > high_vol_threshold, 'market_regime'] = 1  # High
                
                # One-hot encode market_regime
                regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime', dummy_na=False)
                df = pd.concat([df, regime_dummies], axis=1)
                # Drop original market_regime and intermediate columns if not needed
                df.drop(columns=['log_return', 'volatility_20d', 'market_regime'], inplace=True, errors='ignore')
                logger.info("===== FeatureEngineer: Market regime detection complete. Added one-hot encoded regime features. Shape after: {df.shape} =====")
            except Exception as e: 
                logger.error(f"===== FeatureEngineer: Error during market regime detection: {e}. Proceeding without regime features. =====")
                logger.error(traceback.format_exc())

        # Store column names *after* all TAs are calculated, before NaN drop or selection
        # This should ideally be only numeric columns meant for scaling/PCA/selection
        # Exclude original OHLCV if they are not intended to be direct features for the final model
        # For now, assume all numeric columns generated are potential features.
        self.feature_columns = df.select_dtypes(include=np.number).columns.tolist()
        logger.info(f"===== FeatureEngineer: Identified {len(self.feature_columns)} numeric feature columns after TA and regime: {self.feature_columns[:5]}... =====")

        # Handle NaNs from TA calculations (e.g., rolling windows at the start)
        # A common strategy is to drop rows with NaNs, but this reduces sequence length.
        # Another is to fill, but that might introduce noise.
        # For sequence models, preserving sequence integrity is important.
        # If `fit_scalers` is True, we are likely fitting on a larger historical dataset where dropping initial NaNs is fine.
        # If `fit_scalers` is False (i.e., for prediction), we need `sequence_length` valid rows at the end.
        
        nan_rows_before_drop = df.isnull().any(axis=1).sum()
        if nan_rows_before_drop > 0:
            logger.warning(f"===== FeatureEngineer: Data has {nan_rows_before_drop} rows with NaNs after TA calculation (total rows: {len(df)}). This is common due to lookback periods of indicators. =====")
            if fit_scalers: # If fitting scalers, we can afford to drop initial NaNs
                df.dropna(inplace=True)
                logger.info(f"===== FeatureEngineer: Dropped {nan_rows_before_drop} rows with NaNs because fit_scalers=True. Shape after drop: {df.shape}. =====")
                if df.empty:
                    logger.error("===== FeatureEngineer: DataFrame became empty after dropping NaNs during scaler fitting. Cannot proceed. =====")
                    return None, None
            # else: For prediction, NaNs at the start of the required sequence are problematic.
                # The sequence extraction later should handle taking the last `sequence_length` valid rows.
                # We might still have NaNs if the lookback for indicators is > (total_data - sequence_length).
                # For now, let prepare_features_for_prediction handle this by taking the tail.
            pass # Don't dropna if preparing for prediction, sequence logic handles it.

        last_close_price = df[self.target_column_name].iloc[-1] if not df.empty and self.target_column_name in df else None
        if last_close_price is None:
            logger.warning(f"===== FeatureEngineer: Could not determine last close price. Target column '{self.target_column_name}' might be missing or data is empty. =====")

        logger.info(f"===== FeatureEngineer: Technical indicators and market regime (if enabled) calculation complete. DataFrame shape is now {df.shape}. Last close: {last_close_price} =====")
        return df, last_close_price

    def fit_scaler(self, data: pd.DataFrame, symbol: Optional[str] = None):
        """Fits the scaler (RobustScaler) on the provided data."""
        if data.empty:
            logger.error("===== FeatureEngineer: Cannot fit scaler with empty data. =====")
            raise ScalerHandlerError("Cannot fit scaler with empty data.")

        # Ensure only numeric columns are used for fitting the scaler
        numeric_cols = data.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            logger.error("===== FeatureEngineer: No numeric columns found in data to fit scaler. =====")
            raise ScalerHandlerError("No numeric columns to fit scaler.")
        
        # Use self.feature_columns if available and populated, otherwise fall back to all numeric.
        # self.feature_columns should have been set by calculate_technical_indicators.
        cols_to_scale = self.feature_columns if self.feature_columns else numeric_cols.tolist()
        # Filter cols_to_scale to only those present in the current `data` DataFrame
        cols_to_scale = [col for col in cols_to_scale if col in data.columns]

        if not cols_to_scale:
            logger.error(f"===== FeatureEngineer: No valid feature columns (from self.feature_columns or numeric_cols) found in the provided data for symbol '{symbol}'. Data columns: {data.columns.tolist()}. =====")
            raise ScalerHandlerError(f"No columns to scale for {symbol}")

        logger.info(f"===== FeatureEngineer: Fitting the data scaler (RobustScaler) using {len(cols_to_scale)} identified feature columns for symbol '{symbol if symbol else 'global'}'. Columns: {cols_to_scale[:5]}... This learns the centering and scaling parameters. =====")
        
        scaler = RobustScaler()
        try:
            scaler.fit(data[cols_to_scale])
            logger.info(f"===== FeatureEngineer: Scaler fitting complete. Center values (medians): {scaler.center_[:3]}... Scale values (IQRs): {scaler.scale_[:3]}... =====")

            # Store target-specific scaling parameters before it's scaled with other features
            if self.target_column_name in cols_to_scale:
                target_idx = cols_to_scale.index(self.target_column_name)
                center_val = scaler.center_[target_idx]
                scale_val = scaler.scale_[target_idx]
                if scale_val == 0:
                    logger.warning(f"===== FeatureEngineer: Target column '{self.target_column_name}' has a scale (IQR) of 0 for symbol '{symbol}'. This will cause issues with scaling/unscaling. Investigate data. Using scale=1 as fallback. =====")
                    scale_val = 1.0 # Avoid division by zero, though this indicates data issues
                
                target_params = {'center': center_val, 'scale': scale_val}
                if symbol:
                    self.target_scaler_params[symbol] = target_params
                    logger.info(f"===== FeatureEngineer: Stored specific scaling parameters for target column '{self.target_column_name}' for symbol '{symbol}': Center={center_val:.2f}, Scale={scale_val:.2f} =====")
                else:
                    self.global_target_scaler_params = target_params
                    logger.info(f"===== FeatureEngineer: Stored global scaling parameters for target column '{self.target_column_name}': Center={center_val:.2f}, Scale={scale_val:.2f} =====")
            else:
                logger.warning(f"===== FeatureEngineer: Target column '{self.target_column_name}' not found in columns to scale: {cols_to_scale}. Cannot store target-specific scaler params. =====")

            # Store the fitted scaler
            if symbol:
                self.scalers[symbol] = scaler
                logger.info(f"===== FeatureEngineer: Scaler for symbol '{symbol}' has been fitted and stored. =====")
            else:
                self.global_scaler = scaler
                logger.info("===== FeatureEngineer: Global scaler has been fitted and stored. =====")

        except ValueError as e:
            logger.error(f"===== FeatureEngineer: ValueError during scaler fitting for '{symbol if symbol else 'global'}': {e}. This might be due to all-NaN columns or incompatible data. Data shape: {data[cols_to_scale].shape}. NaN counts per column:\n{data[cols_to_scale].isnull().sum()} =====")
            raise ScalerHandlerError(f"ValueError during scaler fitting: {e}") from e
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Unexpected error during scaler fitting for '{symbol if symbol else 'global'}': {e} =====")
            raise ScalerHandlerError(f"Unexpected error during scaler fitting: {e}") from e

    def normalize_features(self, data: pd.DataFrame, symbol: Optional[str] = None, fit: bool = False) -> Optional[pd.DataFrame]:
        """Normalizes features using a fitted RobustScaler."""
        if data.empty:
            logger.warning("===== FeatureEngineer: Data for normalization is empty. Symbol: {symbol}. Returning None. =====")
            return None

        if fit:
            # This path is more for a combined fit_transform, primary fitting is in fit_scaler
            logger.info(f"===== FeatureEngineer: 'fit' is True in normalize_features for symbol '{symbol}'. Calling fit_scaler first. =====")
            self.fit_scaler(data.copy(), symbol=symbol) # Use a copy for fitting
        
        scaler = self._get_scaler(symbol)
        if not self.is_scaler_fitted(symbol): # Check if it (or global) is actually fitted
            logger.error(f"===== FeatureEngineer: Scaler for symbol '{symbol if symbol else 'global'}' is not fitted. Cannot normalize features. Call fit_scaler() first. =====")
            # Fallback: attempt to fit now if critical, though this implies an issue in the calling logic.
            # This might happen if fit_scaler failed or was skipped.
            logger.warning(f"===== FeatureEngineer: Attempting to fit scaler now for '{symbol if symbol else 'global'}' as a fallback within normalize_features. =====")
            try:
                self.fit_scaler(data.copy(), symbol=symbol)
                scaler = self._get_scaler(symbol) # Re-fetch the (now hopefully fitted) scaler
                if not self.is_scaler_fitted(symbol):
                    raise ScalerHandlerError("Fallback scaler fitting failed.")
            except ScalerHandlerError as e_fit:
                 logger.error(f"===== FeatureEngineer: Fallback scaler fitting within normalize_features failed for symbol '{symbol}': {e_fit}. Cannot normalize. =====")
                 return None # Or raise error

        # Determine columns to scale - should match what was used in fit_scaler
        # self.feature_columns should be reliable if fit_scaler was called after calculate_technical_indicators.
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        cols_to_scale = self.feature_columns if self.feature_columns else numeric_cols
        cols_to_scale = [col for col in cols_to_scale if col in data.columns and col in numeric_cols]

        if not cols_to_scale:
            logger.warning(f"===== FeatureEngineer: No numeric columns found or matching self.feature_columns in data for normalization. Symbol: {symbol}. Data columns: {data.columns.tolist()} =====")
            return data # Return original data if no columns to scale

        logger.info(f"===== FeatureEngineer: Normalizing {len(cols_to_scale)} features for symbol '{symbol if symbol else 'global'}' using RobustScaler. Columns: {cols_to_scale[:5]}... Data shape before scaling: {data.shape} =====")
        
        data_to_scale = data[cols_to_scale]
        
        # Check for all-NaN columns before scaling to prevent errors
        if data_to_scale.isnull().all().any():
            all_nan_cols = data_to_scale.columns[data_to_scale.isnull().all()].tolist()
            logger.warning(f"===== FeatureEngineer: One or more columns are all NaN before scaling for symbol '{symbol}': {all_nan_cols}. These will remain NaN. =====")
            # Scaler might handle this or raise error. RobustScaler should be okay if NaNs are consistent.

        try:
            scaled_values = scaler.transform(data_to_scale)
            data[cols_to_scale] = scaled_values
            logger.info(f"===== FeatureEngineer: Normalization applied. Data shape remains {data.shape}. =====")
            # logger.debug(f"Sample of scaled data for symbol '{symbol}':\n{data[cols_to_scale].head().to_string()}")
            return data
        except NotFittedError:
            logger.error(f"===== FeatureEngineer: CRITICAL - Scaler was expected to be fitted for symbol '{symbol if symbol else 'global'}' but NotFittedError occurred during transform. This indicates a flaw in logic or scaler state. =====")
            # This should not happen if is_scaler_fitted() checks are correct and robust.
            raise ScalerHandlerError(f"Scaler not fitted for transform for symbol '{symbol}'. Logic error.")
        except ValueError as e:
            logger.error(f"===== FeatureEngineer: ValueError during feature normalization for '{symbol if symbol else 'global'}': {e}. This can happen if data contains NaNs not handled by scaler or if feature set changed since fit. =====")
            logger.error(f"Data columns: {data.columns.tolist()}")
            logger.error(f"Cols to scale: {cols_to_scale}")
            logger.error(f"Data to scale head:\n{data_to_scale.head()}")
            logger.error(f"Data to scale NaN sum:\n{data_to_scale.isnull().sum()}")
            # It's possible that `cols_to_scale` has columns not present in `scaler.feature_names_in_`
            # or `scaler.n_features_in_` does not match `data_to_scale.shape[1]`
            if hasattr(scaler, 'n_features_in_') and hasattr(data_to_scale, 'shape'):
                 logger.error(f"Scaler expected {scaler.n_features_in_} features, data has {data_to_scale.shape[1]} features to scale.")
            if hasattr(scaler, 'feature_names_in_'):
                 logger.error(f"Scaler was fitted on features: {list(scaler.feature_names_in_)}")
            raise ScalerHandlerError(f"ValueError during normalization: {e}") from e
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Unexpected error during feature normalization for '{symbol if symbol else 'global'}': {e} =====")
            raise ScalerHandlerError(f"Unexpected error during normalization: {e}") from e

    def inverse_transform_target(self, scaled_value: float, symbol: Optional[str] = None) -> float:
        """Inverse transforms a single scaled target value back to its original scale."""
        target_params = self._get_target_scaler_params(symbol)
        
        if target_params is None:
            logger.warning(f"===== FeatureEngineer: No target scaler parameters found for symbol '{symbol if symbol else 'global'}'. Cannot inverse transform. Returning scaled value. =====")
            return scaled_value

        center = target_params['center']
        scale = target_params['scale']
        
        if scale == 0:
            logger.warning(f"===== FeatureEngineer: Target scaler 'scale' is 0 for symbol '{symbol if symbol else 'global'}'. Cannot inverse transform properly. Returning center value. =====")
            return center # Or raise error, as this indicates an issue with data/scaling

        original_value = (scaled_value * scale) + center
        logger.debug(f"===== FeatureEngineer: Inverse transformed target for symbol '{symbol if symbol else 'global'}': scaled={scaled_value:.4f}, original={original_value:.4f} (center={center:.2f}, scale={scale:.2f}) =====")
        return original_value

    def apply_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies feature selection if enabled."""
        if not self.use_feature_selection:
            logger.info("===== FeatureEngineer: Feature selection is disabled. Using all available features. =====")
            self.selected_features = self.feature_columns # Use all if not selecting
            return data

        if data.empty or self.feature_columns is None:
            logger.warning("===== FeatureEngineer: Data is empty or base feature_columns not set. Skipping feature selection. =====")
            return data
        
        # Example: Select top K features based on correlation with a shifted target
        # This is a very basic example. More sophisticated methods (e.g., RFE, SelectFromModel) can be used.
        # Ensure target column is present for correlation calculation
        if self.target_column_name not in data.columns:
            logger.warning(f"===== FeatureEngineer: Target column '{self.target_column_name}' not in data for feature selection. Skipping. =====")
            self.selected_features = self.feature_columns
            return data

        # Shift target for correlation (predict next step)
        # Ensure we have enough data points after shifting to calculate correlation robustly.
        # If data is short, correlation might be noisy or fail.
        if len(data) < self.prediction_horizon + 5: # Need at least a few points post-shift
            logger.warning(f"===== FeatureEngineer: Not enough data (len: {len(data)}) to reliably perform correlation-based feature selection with horizon {self.prediction_horizon}. Skipping selection. =====")
            self.selected_features = self.feature_columns
            return data
            
        shifted_target = data[self.target_column_name].shift(-self.prediction_horizon)
        
        # Consider only feature columns that are numeric and not the target itself for correlation
        features_for_corr = [col for col in self.feature_columns if col in data.columns and col != self.target_column_name and pd.api.types.is_numeric_dtype(data[col])]
        if not features_for_corr:
            logger.warning("===== FeatureEngineer: No valid numeric features found for correlation-based selection. Skipping. =====")
            self.selected_features = self.feature_columns
            return data

        correlations = data[features_for_corr].corrwith(shifted_target).abs().sort_values(ascending=False)
        
        # Select top N features (e.g., N=10 or based on a threshold)
        num_top_features = min(15, len(features_for_corr)) # Example: cap at 15 features
        self.selected_features = correlations.head(num_top_features).index.tolist()
        
        if not self.selected_features:
            logger.warning("===== FeatureEngineer: No features selected by correlation. Using all original features. =====")
            self.selected_features = self.feature_columns
            return data

        # Always include the original target column if it was part of feature_columns, 
        # even if not selected by correlation, as it might be needed for scaling or sequence creation later.
        # However, for model input, target is usually handled separately.
        # For now, selected_features are just the chosen input features.

        logger.info(f"===== FeatureEngineer: Feature selection applied. Selected {len(self.selected_features)} features: {self.selected_features[:5]}... based on correlation with target. =====")
        
        # Return data with only selected features + target (if it was there initially)
        # The actual subsetting for model input happens later in sequence creation.
        # This method primarily determines `self.selected_features`.
        # For consistency, if called, it should return a DataFrame with *only* selected features.
        # However, typically, the full df with all TAs is passed around, and `self.selected_features` list is used later.
        # Let's clarify: this method should determine `self.selected_features` and then `prepare_prediction_input` should use it.
        # It doesn't need to return a subset of `data` here if downstream uses `self.selected_features`.
        # For now, let it just set `self.selected_features`.
        return data # Return the full data, selection is via the list `self.selected_features`

    def apply_pca(self, data: pd.DataFrame, symbol: Optional[str] = None, fit: bool = False) -> pd.DataFrame:
        """Applies PCA if enabled."""
        if not self.use_pca:
            logger.info("===== FeatureEngineer: PCA is disabled. =====")
            return data
        
        features_to_use = self.selected_features if self.selected_features else self.feature_columns
        if not features_to_use:
            logger.warning("===== FeatureEngineer: No features available for PCA. Skipping. =====")
            return data
        
        # Ensure features_to_use are present in the data
        features_present = [f for f in features_to_use if f in data.columns]
        if not features_present:
            logger.warning(f"===== FeatureEngineer: None of the designated PCA features ({features_to_use}) are in the data. Skipping PCA. =====")
            return data
        
        data_for_pca = data[features_present].copy()
        # PCA requires no NaNs
        if data_for_pca.isnull().values.any():
            logger.warning(f"===== FeatureEngineer: Data for PCA for symbol '{symbol}' contains NaNs. Attempting ffill then bfill. Shape: {data_for_pca.shape}, NaN sum:\n{data_for_pca.isnull().sum()[data_for_pca.isnull().sum() > 0]} =====")
            data_for_pca.ffill(inplace=True)
            data_for_pca.bfill(inplace=True)
            if data_for_pca.isnull().values.any():
                all_nan_cols_pca = data_for_pca.columns[data_for_pca.isnull().all()].tolist()
                logger.error(f"===== FeatureEngineer: Data for PCA for symbol '{symbol}' still contains NaNs after fill in columns: {all_nan_cols_pca}. Cannot apply PCA. Original NaN sum:\n{data[features_present].isnull().sum()[data[features_present].isnull().sum() > 0]} =====")
                # Drop all-NaN columns if they are the cause
                if all_nan_cols_pca:
                    data_for_pca.drop(columns=all_nan_cols_pca, inplace=True)
                    features_present = [f for f in features_present if f not in all_nan_cols_pca]
                    if not features_present:
                        logger.error(f"===== FeatureEngineer: No features left for PCA for symbol '{symbol}' after dropping all-NaN columns. =====")
                        return data # Return original data
                if data_for_pca.isnull().values.any(): # Check again
                    logger.error(f"===== FeatureEngineer: PCA cannot proceed with NaNs for symbol '{symbol}'. Returning data without PCA. =====")
                    return data # Return original data
        
        if data_for_pca.empty:
            logger.warning(f"===== FeatureEngineer: Data for PCA for symbol '{symbol}' is empty. Skipping PCA. =====")
            return data

        n_components = min(0.95, len(features_present)) # Keep 95% variance or max possible components
        if isinstance(n_components, float) and n_components < 1.0 and int(n_components * len(features_present)) == 0:
            # If 95% variance leads to 0 components (e.g. very few features), set to 1 component.
             n_components = 1
        elif isinstance(n_components, int) and n_components == 0:
            n_components = 1

        if fit:
            logger.info(f"===== FeatureEngineer: Fitting PCA for symbol '{symbol if symbol else 'global'}' on {len(features_present)} features with n_components={n_components}. Data shape: {data_for_pca.shape} =====")
            pca = PCA(n_components=n_components)
            pca.fit(data_for_pca)
            if symbol:
                self.pca_models[symbol] = pca
            else:
                self.global_pca_model = pca
            logger.info(f"===== FeatureEngineer: PCA fitted. Explained variance ratio: {pca.explained_variance_ratio_}. Number of components: {pca.n_components_} =====")
        else:
            pca = self.pca_models.get(symbol, self.global_pca_model)
            if pca is None:
                logger.warning(f"===== FeatureEngineer: PCA model for '{symbol if symbol else 'global'}' not fitted. Skipping PCA transformation. Call with fit=True first. =====")
                return data
        
        try:
            transformed_data = pca.transform(data_for_pca)
            pca_feature_names = [f'pca_comp_{i}' for i in range(transformed_data.shape[1])]
            pca_df = pd.DataFrame(transformed_data, columns=pca_feature_names, index=data_for_pca.index)
            
            # Replace original features with PCA components
            data_after_pca = data.drop(columns=features_present, errors='ignore')
            data_after_pca = pd.concat([data_after_pca, pca_df], axis=1)
            
            # Update self.selected_features to reflect PCA components if PCA is used
            # The target column should not be in here. Assume it's handled separately.
            self.selected_features = pca_feature_names 
            logger.info(f"===== FeatureEngineer: PCA applied for '{symbol if symbol else 'global'}'. Transformed {len(features_present)} features into {pca.n_components_} components. Updated selected_features. Shape after PCA: {data_after_pca.shape} =====")
            return data_after_pca
        except NotFittedError:
            logger.error(f"===== FeatureEngineer: PCA model for '{symbol if symbol else 'global'}' was not fitted before transform. This is a logic error. =====")
            return data # Return original data
        except ValueError as ve:
            logger.error(f"===== FeatureEngineer: ValueError during PCA transform for '{symbol if symbol else 'global'}': {ve}. Check feature consistency. Data shape: {data_for_pca.shape}, PCA n_features_in: {pca.n_features_in_ if hasattr(pca, 'n_features_in_') else 'N/A'} =====")
            return data
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Unexpected error during PCA transform for '{symbol if symbol else 'global'}': {e} =====")
            return data

    def create_sequences(self, data: pd.DataFrame, symbol: Optional[str] = None) -> Optional[np.ndarray]:
        """Creates sequences from the processed data for model input."""
        if data.empty:
            logger.warning(f"===== FeatureEngineer: Data for sequence creation is empty. Symbol: {symbol}. Returning None. =====")
            return None

        # Determine which features to use for sequencing
        # Priority: self.selected_features (after PCA or selection) > self.feature_columns (all TAs) > all numeric
        features_for_sequencing = []
        if self.selected_features: # This would include PCA components if PCA was run
            features_for_sequencing = [f for f in self.selected_features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
            logger.info(f"===== FeatureEngineer: Using self.selected_features ({len(features_for_sequencing)} numeric available in data) for creating sequences. Symbol: {symbol}. Selected features example: {features_for_sequencing[:5]}... =====")
        elif self.feature_columns:
            features_for_sequencing = [f for f in self.feature_columns if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
            logger.info(f"===== FeatureEngineer: Using self.feature_columns ({len(features_for_sequencing)} numeric available in data) for creating sequences. Symbol: {symbol}. Feature columns example: {features_for_sequencing[:5]}... =====")
        else:
            # Fallback: use all numeric columns if no specific feature set defined
            features_for_sequencing = data.select_dtypes(include=np.number).columns.tolist()
            logger.warning(f"===== FeatureEngineer: No specific feature set (selected_features or feature_columns) defined. Falling back to all {len(features_for_sequencing)} numeric columns in data for sequencing. Symbol: {symbol}. =====")

        if not features_for_sequencing:
            logger.error(f"===== FeatureEngineer: No numeric features available in data to create sequences. Symbol: {symbol}. Data columns: {data.columns.tolist()}. Returning None. =====")
            return None
        
        # Ensure target is not accidentally in features_for_sequencing if it's handled separately by model
        # Most models expect features (X) and target (y) separately.
        # For TFT, target is usually part of the input dataframe but marked.
        # For now, assume features_for_sequencing are all inputs X.
        # If self.target_column_name is in features_for_sequencing, it will be part of the sequence.
        # This is often desired for auto-regressive models or if TFT uses it as `time_varying_unknown_reals`.

        # Check if data has enough rows for at least one sequence
        if len(data) < self.sequence_length:
            logger.warning(f"===== FeatureEngineer: Data length ({len(data)}) is less than sequence length ({self.sequence_length}) for symbol '{symbol}'. Cannot create sequences. Returning None. =====")
            return None

        logger.info(f"===== FeatureEngineer: Creating sequences using {len(features_for_sequencing)} features for symbol '{symbol}'. Input data shape: {data.shape}, Sequence length: {self.sequence_length}. Features: {features_for_sequencing[:5]}... =====")
        
        # Convert selected DataFrame columns to NumPy array for sequence creation
        # Make sure to handle cases where some features might be all NaN after processing, though ideally handled earlier.
        data_subset_for_seq = data[features_for_sequencing]
        if data_subset_for_seq.isnull().values.any():
            nan_cols_in_seq_data = data_subset_for_seq.columns[data_subset_for_seq.isnull().any()].tolist()
            logger.warning(f"===== FeatureEngineer: Data for sequencing for symbol '{symbol}' contains NaNs in columns: {nan_cols_in_seq_data}. Attempting ffill then bfill on this subset. =====")
            data_subset_for_seq.ffill(inplace=True)
            data_subset_for_seq.bfill(inplace=True)
            if data_subset_for_seq.isnull().values.any():
                # Check for columns that are still entirely NaN
                all_nan_cols_in_subset = data_subset_for_seq.columns[data_subset_for_seq.isnull().all()].tolist()
                if all_nan_cols_in_subset:
                    logger.warning(f"===== FeatureEngineer: Columns {all_nan_cols_in_subset} are entirely NaN in data for sequencing for symbol '{symbol}' AFTER fill. Dropping them. =====")
                    data_subset_for_seq = data_subset_for_seq.drop(columns=all_nan_cols_in_subset)
                    # Update features_for_sequencing list to reflect dropped columns
                    features_for_sequencing = [f for f in features_for_sequencing if f not in all_nan_cols_in_subset]
                    if data_subset_for_seq.empty or not features_for_sequencing:
                         logger.error(f"===== FeatureEngineer: Data for sequencing became empty or no features left for symbol '{symbol}' after dropping all-NaN columns. Cannot create sequences. =====")
                         return None
                
                # Re-check for any remaining NaNs (e.g., partial NaNs that couldn't be filled)
                if data_subset_for_seq.isnull().values.any():
                    remaining_nan_cols = data_subset_for_seq.columns[data_subset_for_seq.isnull().any()].tolist()
                    logger.error(f"===== FeatureEngineer: Data for sequencing for symbol '{symbol}' still has NaNs after fill and dropping all-NaN columns (cols: {remaining_nan_cols}). Cannot create sequences reliably. =====")
                    return None

        sequences_np = self.sequence_preprocessor.create_sequences(data_subset_for_seq.values) # Pass NumPy array
        
        if sequences_np is None or sequences_np.size == 0:
            logger.error(f"===== FeatureEngineer: Sequence creation by SequencePreprocessor returned None or empty array for symbol '{symbol}'. Input data for seq was shape {data_subset_for_seq.shape}. =====")
            return None

        logger.info(f"===== FeatureEngineer: Sequences created successfully for symbol '{symbol}'. Shape: {sequences_np.shape} (num_sequences, sequence_length, num_features). =====")
        return sequences_np

    def prepare_features_for_prediction(self, raw_data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Prepares the final input features (as a NumPy array) for making a single prediction step.
        This involves: TA calculation, normalization, feature selection (if any), PCA (if any),
        and taking the *last* sequence of appropriate length.
        Returns a tuple: (feature_sequence_for_prediction_np, last_close_price)
        The feature_sequence_for_prediction_np is (sequence_length, num_features) for ONE prediction.
        """
        logger.info(f"===== FeatureEngineer: Starting feature preparation pipeline for PREDICTION using input data of shape {raw_data.shape} for symbol '{symbol}'. Goal is to produce a single NumPy array sequence for the model. =====")
        
        if raw_data.empty:
            logger.error(f"===== FeatureEngineer (predict): Raw data for symbol '{symbol}' is empty. Cannot prepare features. =====")
            return None, None
        
        # 1. Calculate Technical Indicators
        logger.info(f"===== FeatureEngineer (predict): Calculating technical indicators for symbol '{symbol}'. Initial raw data shape: {raw_data.shape} =====")
        df_with_ta, last_close_price = self.calculate_technical_indicators(raw_data, symbol=symbol, fit_scalers=False) # fit_scalers=False for prediction path
        if df_with_ta is None or df_with_ta.empty:
            logger.error(f"===== FeatureEngineer (predict): TA calculation failed or returned empty DataFrame for symbol '{symbol}'. Shape: {df_with_ta.shape if df_with_ta is not None else 'None'}. =====")
            return None, last_close_price # last_close_price might be None already
        logger.info(f"===== FeatureEngineer (predict): Technical indicators added. Shape after TA for symbol '{symbol}': {df_with_ta.shape} =====")

        # Handle NaNs: For prediction, we need the tail end of the data to be clean.
        # If TA introduces NaNs at the beginning, that's usually fine as long as the last `sequence_length` rows are valid.
        # Let's check the state of the last `sequence_length` rows before normalization.
        if len(df_with_ta) >= self.sequence_length:
            tail_for_check = df_with_ta.tail(self.sequence_length)
            if tail_for_check.isnull().values.any():
                nan_cols_in_tail = tail_for_check.columns[tail_for_check.isnull().any()].tolist()
                logger.warning(f"===== FeatureEngineer (predict): NaNs found in the last {self.sequence_length} rows for symbol '{symbol}' (cols: {nan_cols_in_tail}) BEFORE normalization. Attempting ffill+bfill on df_with_ta. =====")
                # Fill NaNs in the entire df_with_ta before normalization if tail has issues.
                # This is aggressive but might save sequences if NaNs are sporadic.
                df_with_ta.ffill(inplace=True)
                df_with_ta.bfill(inplace=True)
                # Re-check tail
                tail_after_fill = df_with_ta.tail(self.sequence_length)
                if tail_after_fill.isnull().values.any():
                    still_nan_cols = tail_after_fill.columns[tail_after_fill.isnull().any()].tolist()
                    logger.error(f"===== FeatureEngineer (predict): NaNs STILL PRESENT in the last {self.sequence_length} rows for symbol '{symbol}' (cols: {still_nan_cols}) after fill. Prediction will likely fail or be inaccurate. =====")
                    # Proceeding, but this is a warning sign.
        else:
            logger.warning(f"===== FeatureEngineer (predict): Data length ({len(df_with_ta)}) after TA is less than sequence length ({self.sequence_length}) for symbol '{symbol}'. Cannot form a full sequence. =====")
            # Return None if not enough data for even one sequence for prediction
            return None, last_close_price

        # 2. Normalize Features
        # Normalization should use the scaler fitted on historical training data.
        logger.info(f"===== FeatureEngineer (predict): Normalizing features for symbol '{symbol}'. Data shape before norm: {df_with_ta.shape} =====")
        df_normalized = self.normalize_features(df_with_ta, symbol=symbol, fit=False) # fit=False for prediction
        if df_normalized is None or df_normalized.empty:
            logger.error(f"===== FeatureEngineer (predict): Feature normalization failed or returned empty for symbol '{symbol}'. Shape: {df_normalized.shape if df_normalized is not None else 'None'}. =====")
            return None, last_close_price
        logger.info(f"===== FeatureEngineer (predict): Features normalized. Shape after normalization for symbol '{symbol}': {df_normalized.shape} =====")

        # 3. Apply Feature Selection (determines `self.selected_features` list if enabled)
        # This step doesn't usually change the DataFrame for prediction, but sets the list of features to use.
        # It should ideally be done based on training set insights if not dynamic.
        # For simplicity, if `use_feature_selection` is on, it might re-evaluate here, or rely on a pre-fitted selection.
        # Assuming `apply_feature_selection` has been called appropriately during a training/setup phase
        # and `self.selected_features` is set. If not, it might use all features.
        if self.use_feature_selection:
            # Typically, for prediction, we don't re-run selection itself unless it's an adaptive method.
            # We use the `self.selected_features` list that was determined during training/setup.
            # If `self.selected_features` is None, it implies selection hasn't been done or all features are used.
            if self.selected_features is None:
                logger.warning(f"===== FeatureEngineer (predict): Feature selection is enabled for symbol '{symbol}', but no pre-selected features found (`self.selected_features` is None). Attempting to run selection now (might be inconsistent if not based on training data distribution). =====")
                # Running it here might be okay if it's a static method or for testing, but generally, selection criteria are from training.
                self.apply_feature_selection(df_normalized.copy()) # Use copy to avoid modifying df_normalized if selection changes it
            
            if self.selected_features:
                 logger.info(f"===== FeatureEngineer (predict): Using pre-defined selected features ({len(self.selected_features)}) for symbol '{symbol}': {self.selected_features[:5]}... =====")
            else:
                 logger.warning(f"===== FeatureEngineer (predict): Feature selection enabled for symbol '{symbol}', but `self.selected_features` is still None after attempt. Will use all numeric features. =====")
        else:
            logger.info(f"===== FeatureEngineer (predict): Feature selection is disabled for symbol '{symbol}'. All available numeric features will be considered for sequence. =====")
            # Ensure self.selected_features is None or reflects all features if selection is off
            # self.selected_features = None # Or self.feature_columns if that's the policy

        # 4. Apply PCA (transforms data based on `self.selected_features` or `self.feature_columns`)
        # PCA also relies on a fitted model from training data.
        df_processed = df_normalized
        if self.use_pca:
            logger.info(f"===== FeatureEngineer (predict): Applying PCA for symbol '{symbol}'. Data shape before PCA: {df_processed.shape} =====")
            df_processed = self.apply_pca(df_processed.copy(), symbol=symbol, fit=False) # fit=False for prediction
            if df_processed is None or df_processed.empty:
                 logger.error(f"===== FeatureEngineer (predict): PCA application failed or returned empty for symbol '{symbol}'. Shape: {df_processed.shape if df_processed is not None else 'None'}. =====")
                 return None, last_close_price
            logger.info(f"===== FeatureEngineer (predict): PCA applied. Shape after PCA for symbol '{symbol}': {df_processed.shape}. Selected features now PCA components. =====")
            # self.selected_features should now be the PCA component names if PCA was successful.

        # 5. Create the final sequence for prediction from df_processed
        # This should use the final set of features (e.g., PCA components if PCA was run, or selected original features)
        # The `create_sequences` method uses `self.selected_features` or `self.feature_columns` internally.
        
        # Which columns to use for sequencing at this point:
        # If PCA was used, self.selected_features = PCA components.
        # If No PCA but selection, self.selected_features = selected original features.
        # If No PCA and No selection, self.selected_features might be None or all original TAs (self.feature_columns).
        
        # `create_sequences` will pick the right set based on `self.selected_features` then `self.feature_columns`.
        logger.info(f"===== FeatureEngineer (predict): Creating final prediction sequence for symbol '{symbol}'. Data shape for sequence creation: {df_processed.shape}. =====")
        
        # Create sequences from the *entire* processed history for this symbol.
        # This might generate multiple sequences if df_processed is long enough.
        all_sequences_np = self.create_sequences(df_processed, symbol=symbol)

        if all_sequences_np is None or all_sequences_np.size == 0:
            logger.error(f"===== FeatureEngineer (predict): Failed to create any sequences from processed data for symbol '{symbol}'. Shape of data fed to create_sequences: {df_processed.shape}. =====")
            return None, last_close_price
        
        # For prediction, we need only the *last* available sequence.
        # all_sequences_np is (num_sequences, sequence_length, num_features_in_sequence)
        last_sequence_for_prediction_np = all_sequences_np[-1, :, :]
        logger.info(f"===== FeatureEngineer (predict): Extracted the last sequence of length {self.sequence_length}. Final prediction input sequence shape for symbol '{symbol}': {last_sequence_for_prediction_np.shape}. Num features in seq: {last_sequence_for_prediction_np.shape[1]} =====")

        # The number of features in `last_sequence_for_prediction_np` should match model's expectation.
        # This is determined by `features_for_sequencing` logic within `create_sequences`.
        
        # Return the single sequence (sequence_length, num_features) and the last known close price
        return last_sequence_for_prediction_np, last_close_price

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