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

# Import InputTypes from the base formatter if available, or define a placeholder
try:
    from colab_training.data_formatters.base import InputTypes
except ImportError:
    logger.warning("Could not import InputTypes from colab_training.data_formatters.base. Defining a placeholder.")
    class InputTypes: # Placeholder
        ID = 0
        TIME = 1
        TARGET = 2
        OBSERVED = 3
        KNOWN = 4
        STATIC = 5
        CATEGORICAL = 6
        REAL_VALUED = 7 # Assuming this might be used by scalers

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
    1. Technical indicators calculation aligned with StockFormatter
    2. Market regime detection
    3. Feature selection and dimensionality reduction (can be disabled)
    4. Data normalization using StandardScaler (to match StockFormatter)
    5. Sequence preparation for TFT models
    """
    
    # Features defined by StockFormatter (excluding ID, TIME, TARGET)
    STOCK_FORMATTER_FEATURES = [
        ('open', InputTypes.OBSERVED),
        ('high', InputTypes.OBSERVED),
        ('low', InputTypes.OBSERVED),
        ('volume', InputTypes.OBSERVED),
        ('returns', InputTypes.OBSERVED),
        ('volatility', InputTypes.OBSERVED), # Needs window definition
        ('static_id_placeholder', InputTypes.STATIC), 
        ('market_cap', InputTypes.OBSERVED), # Fundamental, might be NaN
        ('pe_ratio', InputTypes.OBSERVED), # Fundamental, might be NaN
        ('dividend_yield', InputTypes.OBSERVED), # Fundamental, might be NaN
        ('rsi', InputTypes.OBSERVED), # Assuming rsi_14 from original
        ('macd', InputTypes.OBSERVED),
        ('macd_signal', InputTypes.OBSERVED),
        ('macd_hist', InputTypes.OBSERVED),
        ('bb_upper', InputTypes.OBSERVED),
        ('bb_middle', InputTypes.OBSERVED),
        ('bb_lower', InputTypes.OBSERVED),
        ('atr', InputTypes.OBSERVED), # Assuming atr_14 from original
        ('volume_sma', InputTypes.OBSERVED), # Needs window definition
        ('volume_ratio', InputTypes.OBSERVED), # Needs base for ratio
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
        ('market_regime', InputTypes.CATEGORICAL), 
        ('trading_signal', InputTypes.CATEGORICAL) # Needs definition
    ]

    def __init__(
        self,
        sequence_length: int = 30, # Default from formatter_config
        prediction_horizon: int = 5, # Default from formatter_config
        normalize: bool = True,
        use_feature_selection: bool = False, # Disable by default, use all StockFormatter features initially
        n_features: int = 25, # Target number of features if selection is enabled
        use_pca: bool = False,
        n_components: int = 10,
        detect_regime: bool = True # Corresponds to 'market_regime'
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.use_pca = use_pca
        self.n_components = n_components
        self.detect_regime = detect_regime
        self.scalers: Dict[str, StandardScaler] = {} # Symbol -> Scaler, changed to StandardScaler
        self.global_scaler: Optional[StandardScaler] = None # Changed to StandardScaler
        self.target_scaler_params: Dict[str, Dict[str, float]] = {} 
        self.global_target_scaler_params: Optional[Dict[str, float]] = None
        self.pca_models: Dict[str, PCA] = {}
        self.global_pca_model: Optional[PCA] = None
        
        # Store the names of features that will be generated
        self.feature_columns = [name for name, type_ in self.STOCK_FORMATTER_FEATURES 
                                if type_ in [InputTypes.OBSERVED, InputTypes.KNOWN, InputTypes.STATIC, InputTypes.CATEGORICAL]]
        
        self.selected_features: Optional[List[str]] = None 
        self.target_column_name = 'close' # Default target column name, matches StockFormatter

        logger.info(f"===== FeatureEngineer initialized. Sequence length: {self.sequence_length}, Horizon: {self.prediction_horizon}, Target: '{self.target_column_name}'. Feature selection: {self.use_feature_selection}, PCA: {self.use_pca}, Regime Detection: {self.detect_regime} =====")
        
        # Removed self.technical_indicators list, will use STOCK_FORMATTER_FEATURES logic
        
        self.sequence_preprocessor = SequencePreprocessor(sequence_length)
        self.market_regime_detector = MarketRegimeDetector() if detect_regime else None
        # Feature selector and PCA remain, but their usage might change based on how many features STOCK_FORMATTER_FEATURES generate
        self.feature_selector = SelectKBest(f_regression, k=n_features) if use_feature_selection else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        # Use StandardScaler to align with StockFormatter
        self.scaler = StandardScaler() if normalize else None 
        
        self.feature_means = None # Deprecate if using StandardScaler's mean_
        self.feature_stds = None # Deprecate if using StandardScaler's scale_
        # self.selected_features is kept
        self.pca_components = None
        
        logger.info(
            f"Initialized feature engineer to generate features based on StockFormatter definition. "
            f"Number of potential features: {len(self.feature_columns)}. "
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
        scaler_to_check: Optional[StandardScaler] = None # Ensure type hint matches
        if symbol and symbol in self.scalers:
            scaler_to_check = self.scalers[symbol]
        elif self.global_scaler is not None:
            scaler_to_check = self.global_scaler
        
        if scaler_to_check is not None:
            try:
                # Check if the scaler has been fitted by looking for attributes like 'mean_' for StandardScaler
                check_is_fitted(scaler_to_check) 
                target_params = self._get_target_scaler_params(symbol)
                is_target_params_valid = target_params and \
                                         isinstance(target_params.get('center'), (int, float)) and \
                                         isinstance(target_params.get('scale'), (int, float)) and \
                                         target_params.get('scale') != 0 # scale for RobustScaler, std for StandardScaler
                # For StandardScaler, scale is 'scale_', center is 'mean_'
                # For RobustScaler, it's 'center_' and 'scale_'
                # We need to ensure this check is compatible with StandardScaler
                # Let's assume 'center' maps to mean and 'scale' maps to std/scale depending on scaler type.
                # The self.target_scaler_params stores them as 'center' and 'scale'
                
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

    def _get_scaler(self, symbol: Optional[str] = None) -> Optional[StandardScaler]: # Changed return type
        """Retrieves the appropriate scaler (symbol-specific or global)."""
        if symbol and symbol in self.scalers:
            logger.debug(f"===== FeatureEngineer: Retrieving scaler for symbol '{symbol}'. =====")
            return self.scalers[symbol]
        elif self.global_scaler is not None:
            logger.debug("===== FeatureEngineer: Retrieving global scaler. =====")
            return self.global_scaler
        
        logger.debug(f"===== FeatureEngineer: No scaler available (neither for symbol '{symbol}' nor global). Will create a new one if fit=True. =====")
        return None # No scaler available or not fitted

    def _convert_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts column names to lowercase and replaces spaces with underscores."""
        if isinstance(df.columns, pd.MultiIndex):
            logger.warning("===== FeatureEngineer: MultiIndex detected, skipping column name conversion =====")
            return df
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        return df

    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: Optional[str] = None, fit_scalers: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """
        Calculate technical indicators and other features based on StockFormatter definition.
        The input 'data' DataFrame is expected to have at least 'open', 'high', 'low', 'close', 'volume'.
        It might also contain 'adj_close'. Column names can be mixed case or MultiIndex.
        
        Args:
            data: Input DataFrame with OHLCV data.
            symbol: Stock symbol for context and logging.
            fit_scalers: If True, implies this data is for fitting scalers (longer history).
                         If False, it's for prediction (shorter, most recent history).

        Returns:
            Tuple: (DataFrame with all engineered features, last_close_price)
                   Returns (None, None) if essential columns are missing or errors occur.
        """
        logger.info(f"===== FeatureEngineer: Calculating technical indicators for data with shape {data.shape}. Fit_scalers: {fit_scalers}, Symbol: {symbol}. This adds new columns based on existing price/volume data. =====")
        
        if data.empty:
            logger.error(f"===== FeatureEngineer: Input data is empty for symbol '{symbol}'. Cannot calculate features. =====")
            return None, None

        # Standardize column names first (e.g., to lowercase, handle MultiIndex)
        # The input 'data' from MarketFeed is already standardized somewhat, but let's ensure.
        # If data.columns is a MultiIndex like [('Open', 'TSLA'), ('Close', 'TSLA'), ...],
        # we need to convert it to a flat index for 'ta' library and general processing.
        original_cols = data.columns.tolist() # For debugging
        if isinstance(data.columns, pd.MultiIndex):
            logger.info(f"===== FeatureEngineer: MultiIndex columns detected: {data.columns.tolist()}. Converting to single-level columns. =====")
            # Assuming the first level of MultiIndex is the feature type (e.g., 'Open', 'Close')
            # And the second level is the symbol, which we don't need here as data is per-symbol.
            # Or, if it's like ('Adj Close', 'TSLA') from predictor, we simplify.
            
            # If the multi-index is like ('feature_name', 'symbol_name')
            if len(data.columns.names) == 2 and data.columns.names[0] is not None and data.columns.names[1] is not None:
                 # Example: columns=[('High', 'TSLA'), ('Low', 'TSLA')]
                data.columns = data.columns.get_level_values(0) # Take the first level, e.g., 'High', 'Low'
            else: # Try to infer, could be just yfinance's default ('Open', 'High', ...) if not processed by MarketFeed for single symbol
                # This case handles when yfinance download for single symbol results in flat columns already.
                # Or if it's like [('Adj Close', 'TSLA'), ...] from predictor calling market_feed for single symbol.
                # We want to extract the primary feature name.
                new_cols = []
                for col_tuple in data.columns:
                    if isinstance(col_tuple, tuple) and len(col_tuple) > 0:
                        new_cols.append(str(col_tuple[0])) # Take the first element, e.g., 'Adj Close' from ('Adj Close', 'TSLA')
                    else:
                        new_cols.append(str(col_tuple)) # If not a tuple, use as is
                data.columns = new_cols
            
            data.columns = [str(c).lower().replace(' ', '_') for c in data.columns]
            logger.info(f"===== FeatureEngineer: Columns after MultiIndex conversion: {data.columns.tolist()} =====")
        else:
            data.columns = [str(c).lower().replace(' ', '_') for c in data.columns] # Ensure lowercase and underscore

        # Make a copy to avoid SettingWithCopyWarning
        df = data.copy()

        # Ensure required base columns are present
        required_base_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"===== FeatureEngineer: Missing required base columns for TA: {missing_cols}. Available: {df.columns.tolist()}. Original was: {original_cols} =====")
            return None, None
        
        # Handle 'adj_close' if present, otherwise use 'close'
        # StockFormatter does not explicitly list 'adj_close' as a primary feature it generates TAs from,
        # but it's good practice to prefer it if available for price-based TAs.
        # However, for consistency with StockFormatter, let's assume 'close' is the primary for TA calculations.
        # If 'adj_close' is needed, StockFormatter should reflect that.
        # The logs show 'adj_close' being part of initial features, then 'close' for TAs.
        # For now, stick to 'close' as the primary for TA calculations as per typical 'ta' library usage.

        last_close_price = df['close'].iloc[-1] if not df['close'].empty else np.nan

        # --- Feature Calculation based on StockFormatter ---
        # Observed features (direct or simple calculations)
        if 'returns' in self.feature_columns:
            df['returns'] = df['close'].pct_change() 
        
        if 'volatility' in self.feature_columns: # Example: 20-day rolling std dev of returns
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std() 

        # Fundamental data (market_cap, pe_ratio, dividend_yield)
        # These are typically not in yfinance history output directly for daily data.
        # They might be available via ticker.info or a different API endpoint.
        # For now, create NaN columns if they are expected.
        # The model training phase with StockFormatter must have handled how these are sourced or imputed.
        for fundamental_col in ['market_cap', 'pe_ratio', 'dividend_yield']:
            if fundamental_col in self.feature_columns and fundamental_col not in df.columns:
                df[fundamental_col] = np.nan 

        # Technical Indicators using 'ta' library
        # Ensure parameters (windows, etc.) match what StockFormatter implies or common defaults if unspecified.
        
        # RSI (rsi_14 was in old list, 'rsi' in StockFormatter)
        if 'rsi' in self.feature_columns:
            df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=False).rsi()

        # MACD
        if 'macd' in self.feature_columns or 'macd_signal' in self.feature_columns or 'macd_hist' in self.feature_columns:
            macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
            if 'macd' in self.feature_columns:
                df['macd'] = macd_indicator.macd()
            if 'macd_signal' in self.feature_columns:
                df['macd_signal'] = macd_indicator.macd_signal()
            if 'macd_hist' in self.feature_columns:
                df['macd_hist'] = macd_indicator.macd_diff() # macd_hist is the difference

        # Bollinger Bands
        if 'bb_upper' in self.feature_columns or 'bb_middle' in self.feature_columns or 'bb_lower' in self.feature_columns:
            bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=False)
            if 'bb_upper' in self.feature_columns:
                df['bb_upper'] = bb_indicator.bollinger_hband()
            if 'bb_middle' in self.feature_columns: # This is typically the SMA20
                df['bb_middle'] = bb_indicator.bollinger_mavg()
            if 'bb_lower' in self.feature_columns:
                df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # ATR (atr_14 was in old list, 'atr' in StockFormatter)
        if 'atr' in self.feature_columns:
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=False).average_true_range()

        # Volume-based indicators
        if 'volume_sma' in self.feature_columns: # e.g., volume_sma_20
            df['volume_sma'] = SMAIndicator(close=df['volume'], window=20, fillna=False).sma_indicator()
        
        if 'volume_ratio' in self.feature_columns: # Ratio of current volume to its SMA
            if 'volume_sma' in df.columns and not df['volume_sma'].empty:
                 # Avoid division by zero or NaN if volume_sma is zero or NaN
                df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            else: # Calculate volume_sma if not already present for this specific feature
                temp_volume_sma = SMAIndicator(close=df['volume'], window=20, fillna=True).sma_indicator() # fillna=True temporarily
                df['volume_ratio'] = df['volume'] / temp_volume_sma.replace(0, np.nan)


        # Price SMAs
        price_sma_windows = [5, 10, 20, 50, 200]
        for window in price_sma_windows:
            col_name = f'price_sma_{window}'
            if col_name in self.feature_columns:
                df[col_name] = SMAIndicator(close=df['close'], window=window, fillna=False).sma_indicator()

        # Price EMAs
        price_ema_windows = [5, 10, 20, 50, 200]
        for window in price_ema_windows:
            col_name = f'price_ema_{window}'
            if col_name in self.feature_columns:
                df[col_name] = EMAIndicator(close=df['close'], window=window, fillna=False).ema_indicator()
        
        # Market Regime (Categorical)
        if 'market_regime' in self.feature_columns and self.detect_regime and self.market_regime_detector:
            try:
                # MarketRegimeDetector.detect_regime returns a Series of numerical values (e.g., -1, 0, 1)
                # StockFormatter treats 'market_regime' as CATEGORICAL, implies it might need label encoding later.
                # The current MarketRegimeDetector already calculates a complex regime.
                # Let's assume its output is suitable for now, or can be mapped.
                # The output of detect_regime might be float; ensure it can be handled by LabelEncoder
                regime_output = self.market_regime_detector.detect_regime(df).fillna(0) # fillna with neutral
                # Discretize or map to specific categories if necessary before label encoding by StockFormatter
                # For now, use its direct output. StockFormatter would handle label encoding.
                df['market_regime'] = regime_output 
                logger.info(f"===== FeatureEngineer: Market regime detection complete. Added one-hot encoded regime features. Shape after: {df.shape} =====") # Log needs update, not one-hot yet
            except Exception as e:
                logger.error(f"===== FeatureEngineer: Error during market regime detection: {e}. Stack trace: {traceback.format_exc()} Skipping market_regime. =====")
                df['market_regime'] = 0 # Default or NaN

        # Trading Signal (Categorical) - Placeholder, as logic is undefined from StockFormatter
        if 'trading_signal' in self.feature_columns:
            logger.warning("===== FeatureEngineer: 'trading_signal' feature is defined in StockFormatter but generation logic is not implemented here. Adding as NaN/0. =====")
            df['trading_signal'] = 0 # Placeholder, assumes 0 is a valid category or will be handled

        # Static ID Placeholder (Static)
        if 'static_id_placeholder' in self.feature_columns:
            # This should be a constant for the given symbol.
            # StockFormatter would use LabelEncoder. The actual value might not matter as much as its consistency.
            # If symbol is 'TSLA', placeholder could be hash('TSLA') % N_CATEGORIES or a fixed int.
            # For now, a simple approach. This needs to align with how StockFormatter creates it.
            # If no symbol provided, this feature might be problematic.
            df['static_id_placeholder'] = hash(symbol if symbol else 'UNKNOWN_SYMBOL') % 1000 # Example placeholder
        
        # Store all generated numeric feature column names before any selection/PCA
        # This should include all OBSERVED and numerical representation of CATEGORICAL/STATIC after processing
        # For now, select columns that are likely numeric or will be made numeric.
        # StockFormatter handles label encoding for categoricals.
        # We are producing the raw features here.
        
        # Select only the columns that are part of self.feature_columns definition to ensure correct feature set
        # and order (though order will be finalized later).
        current_feature_cols = [col for col in self.feature_columns if col in df.columns]
        df = df[current_feature_cols].copy() # Ensure only defined features are kept

        # Re-evaluate self.feature_columns based on what was actually calculable and defined.
        self.feature_columns = df.columns.tolist() # Update to actual generated columns
        
        # Log identified numeric features from the dataframe
        numeric_cols_df = df.select_dtypes(include=np.number).columns.tolist()
        logger.info(f"===== FeatureEngineer: Identified {len(numeric_cols_df)} numeric feature columns after TA and regime: {numeric_cols_df[:5]}... =====")

        # Handle NaNs - This is critical
        # StockFormatter might have its own way (e.g. ffill during sequence creation).
        # For TA calculation, NaNs at the beginning are normal due to lookback.
        # The warning should be about NaNs in the *final sequence* if they persist.
        nan_rows_initial = df.isnull().any(axis=1).sum()
        if nan_rows_initial > 0:
            logger.warning(f"===== FeatureEngineer: Data has {nan_rows_initial} rows with NaNs after TA calculation (total rows: {len(df)}). This is common due to lookback periods of indicators. =====")

        # If fitting scalers, we might drop initial NaNs.
        # If for prediction, we need to be careful as the last row is most important.
        if fit_scalers:
            # Columns that are intentionally all NaN if data isn't found for them.
            # These should not cause all rows to be dropped if other features are valid.
            placeholder_fundamental_cols = ['market_cap', 'pe_ratio', 'dividend_yield']
            
            # Columns to check for NaNs before dropping rows for scaler fitting.
            # Exclude the placeholder fundamental columns from this check.
            cols_for_dropna_check = [col for col in df.columns if col not in placeholder_fundamental_cols]

            if not df.empty: # Proceed only if df is not already empty
                if not cols_for_dropna_check:
                    # This case means df might only contain placeholder_fundamental_cols or is structured unexpectedly.
                    # A full dropna here would behave like the original problematic code if placeholder_fundamental_cols are all NaN.
                    # If df only had placeholder columns, all would be dropped.
                    # If it had other columns that were all NaN, they'd also be dropped.
                    # This is an edge case; ideally, cols_for_dropna_check should not be empty if df has TA features.
                    logger.warning(
                        f"===== FeatureEngineer: No columns identified for subset dropna (excluding {placeholder_fundamental_cols}) "
                        f"during scaler fitting for symbol '{symbol}'. DataFrame columns: {df.columns.tolist()}. "
                        f"Falling back to dropping rows with any NaN. This might drop all rows if placeholders are all NaN."
                    )
                    # Record current number of rows before full dropna
                    rows_before_fallback_dropna = len(df)
                    df.dropna(inplace=True)
                    rows_after_fallback_dropna = len(df)
                    if rows_before_fallback_dropna > 0 and rows_after_fallback_dropna == 0:
                        logger.error(f"===== FeatureEngineer: Fallback dropna removed all rows for symbol '{symbol}'. This was likely due to all-NaN placeholder columns not being excluded from a targeted dropna because cols_for_dropna_check was empty. =====")

                else:
                    # Drop rows only if NaNs exist in the TA-related columns (cols_for_dropna_check)
                    # This preserves rows even if placeholder_fundamental_cols have NaNs, as long as TA features are fine.
                    df.dropna(subset=cols_for_dropna_check, inplace=True)
                
                logger.info(f"===== FeatureEngineer: Shape after dropna for scaler fitting for symbol '{symbol}': {df.shape}. =====")
            else: # df was already empty before dropna
                 logger.info(f"===== FeatureEngineer: DataFrame for symbol '{symbol}' was already empty before attempting dropna for scaler fitting. =====")

            if df.empty: # Check if df became empty after dropna or was already empty
                 logger.error(f"===== FeatureEngineer: DataFrame is empty after processing for scaler fitting for symbol '{symbol}'. Cannot proceed to fit scaler. Check input data length, TA lookbacks, and content of non-fundamental features. =====")
                 return None, last_close_price


        # Final check on feature columns
        self.feature_columns = df.columns.tolist() # Update based on current df state

        logger.info(f"===== FeatureEngineer: Technical indicators and other StockFormatter-based features calculation complete. DataFrame shape is now {df.shape}. Last close: {last_close_price} =====")
        return df, last_close_price

    def fit_scaler(self, data: pd.DataFrame, symbol: Optional[str] = None):
        """Fits the scaler (StandardScaler) on the provided data."""
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

        logger.info(f"===== FeatureEngineer: Fitting the data scaler (StandardScaler) using {len(cols_to_scale)} identified feature columns for symbol '{symbol if symbol else 'global'}'. Columns: {cols_to_scale[:5]}... This learns the centering and scaling parameters. =====")
        
        scaler = StandardScaler()
        try:
            scaler.fit(data[cols_to_scale])
            logger.info(f"===== FeatureEngineer: Scaler fitting complete. Mean sample: {scaler.mean_[:3]}... Scale (std) sample: {scaler.scale_[:3]}... =====")

            # Store scaling parameters for the target column specifically for inverse transformation
            if self.target_column_name in cols_to_scale:
                target_col_idx = cols_to_scale.index(self.target_column_name)
                target_mean = scaler.mean_[target_col_idx]
                target_scale = scaler.scale_[target_col_idx]
                
                if target_scale == 0:
                    logger.warning(f"===== FeatureEngineer: Target column '{self.target_column_name}' has zero standard deviation for symbol '{symbol if symbol else 'global'}'. Inverse transform might be problematic. Mean: {target_mean} =====")
                    # Avoid division by zero, store scale as 1 if it's 0.
                    # Or handle this case more gracefully (e.g. no scaling for this feature).
                    # For now, using 1.0 to prevent NaN/inf in inverse transform if it occurs.
                    target_scale = 1.0


                target_params = {'center': target_mean, 'scale': target_scale} # 'center' is mean, 'scale' is std for StandardScaler
                
                if symbol:
                    self.target_scaler_params[symbol] = target_params
                    logger.info(f"===== FeatureEngineer: Stored specific scaling parameters for target column '{self.target_column_name}' for symbol '{symbol}': Center(mean)={target_mean:.2f}, Scale(std)={target_scale:.2f} =====")
                else:
                    self.global_target_scaler_params = target_params
                    logger.info(f"===== FeatureEngineer: Stored global specific scaling parameters for target column '{self.target_column_name}': Center(mean)={target_mean:.2f}, Scale(std)={target_scale:.2f} =====")
            else:
                logger.warning(f"===== FeatureEngineer: Target column '{self.target_column_name}' not found in columns to scale: {cols_to_scale}. Cannot store its specific scaling parameters. Available numeric features: {data[cols_to_scale].columns.tolist()} =====")

        except ValueError as e:
            logger.error(f"===== FeatureEngineer: ValueError during scaler fitting for symbol '{symbol if symbol else 'global'}': {e}. This might happen if data contains all NaNs or is not numeric. Input shape: {data[cols_to_scale].shape} =====")
            # Potentially re-raise or handle as a critical error
            # raise ScalerHandlerError(f"Scaler fitting failed for {symbol}: {e}") from e # If ScalerHandlerError is appropriate
            return # Exit if fitting fails
            
        logger.info(f"===== FeatureEngineer: Scaler fitting complete for symbol '{symbol if symbol else 'global'}'. =====")


    def normalize_features(self, data: pd.DataFrame, symbol: Optional[str] = None, fit: bool = False) -> Optional[pd.DataFrame]:
        """Normalizes features using a fitted StandardScaler."""
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

        logger.info(f"===== FeatureEngineer: Normalizing {len(cols_to_scale)} features for symbol '{symbol if symbol else 'global'}' using StandardScaler. Columns: {cols_to_scale[:5]}... Data shape before scaling: {data.shape} =====")
        
        data_to_scale = data[cols_to_scale]
        
        # Check for all-NaN columns before scaling to prevent errors
        if data_to_scale.isnull().all().any():
            all_nan_cols = data_to_scale.columns[data_to_scale.isnull().all()].tolist()
            logger.warning(f"===== FeatureEngineer: One or more columns are all NaN before scaling for symbol '{symbol}': {all_nan_cols}. These will remain NaN. =====")
            # Scaler might handle this or raise error. StandardScaler should be okay if NaNs are consistent.

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

        # For StandardScaler: original = (scaled * scale) + center
        # 'center' is mean, 'scale' is std
        center = target_params.get('center') 
        scale = target_params.get('scale')

        if center is None or scale is None:
            logger.warning(f"===== FeatureEngineer: Target scaler parameters 'center' or 'scale' are missing for '{symbol if symbol else 'global'}'. Params: {target_params}. Returning scaled value. =====")
            return scaled_value
            
        if scale == 0:
            logger.warning(f"===== FeatureEngineer: Target scaler 'scale' (std) is zero for '{symbol if symbol else 'global'}'. Inverse transform will only return the 'center' (mean). Returning {center}. =====")
            # If scale is 0, all original values were the same (the mean).
            # So, inverse transform should ideally return the mean.
            if isinstance(scaled_value, np.ndarray):
                return np.full_like(scaled_value, center)
            return center

        original_value = (scaled_value * scale) + center
        logger.debug(f"===== FeatureEngineer: Inverse transformed target for '{symbol if symbol else 'global'}'. Scaled: {scaled_value}, Center(Mean): {center}, Scale(Std): {scale}, Original: {original_value} =====")
        return original_value

    def apply_feature_selection(self, data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Applies feature selection to the data.
        Assumes 'data' contains only numeric features and 'target' is the series to predict.
        Updates self.selected_features.
        """
        if not self.use_feature_selection or self.feature_selector is None:
            logger.info("===== FeatureEngineer: Feature selection is disabled or selector not initialized. Returning all features. =====")
            # If not selecting, all current numeric columns in 'data' are considered "selected"
            self.selected_features = data.columns.tolist() 
            return data

        if data.empty:
            logger.warning("===== FeatureEngineer: Input data for feature selection is empty. Cannot select features. =====")
            self.selected_features = []
            return data
            
        if target.empty:
            logger.warning("===== FeatureEngineer: Target data for feature selection is empty. Cannot select features. =====")
            self.selected_features = data.columns.tolist() # Return all features from data
            return data

        if len(data) != len(target):
            logger.error(f"===== FeatureEngineer: Data ({len(data)}) and target ({len(target)}) lengths mismatch for feature selection. Cannot select. =====")
            self.selected_features = data.columns.tolist()
            return data
            
        # Ensure data and target are aligned and have no NaNs for f_regression
        combined = pd.concat([data, target.rename('__TARGET__')], axis=1).dropna()
        if combined.empty:
            logger.error("===== FeatureEngineer: Data after dropna (for feature selection) is empty. Cannot select features. =====")
            self.selected_features = data.columns.tolist()
            return data

        data_clean = combined.drop(columns=['__TARGET__'])
        target_clean = combined['__TARGET__']

        if data_clean.empty: # All rows might have had NaNs in target or data
            logger.error("===== FeatureEngineer: Data (features part) after dropna (for feature selection) is empty. Cannot select features. =====")
            self.selected_features = data.columns.tolist() # Fallback to original data columns if data_clean is empty.
            return data


        try:
            # Adjust k for SelectKBest if n_features > number of available features
            num_available_features = data_clean.shape[1]
            current_k = self.feature_selector.k
            if current_k == 'all': # if k was set to 'all'
                 effective_k = num_available_features
            elif isinstance(current_k, int) and current_k > num_available_features:
                logger.warning(f"===== FeatureEngineer: Requested k={current_k} for SelectKBest is more than available features ({num_available_features}). Setting k to {num_available_features}. =====")
                self.feature_selector.k = num_available_features # Temporarily adjust k
                effective_k = num_available_features
            else:
                 effective_k = current_k if isinstance(current_k, int) else num_available_features


            if effective_k == 0 and num_available_features > 0: # If k became 0 but features exist
                logger.warning(f"===== FeatureEngineer: Effective k for SelectKBest is 0, but {num_available_features} features are available. Defaulting to select all available. =====")
                self.feature_selector.k = num_available_features
            elif num_available_features == 0 : # No features to select from
                 logger.error("===== FeatureEngineer: No features available to perform feature selection. Returning empty DataFrame. =====")
                 self.selected_features = []
                 return pd.DataFrame(index=data.index)


            self.feature_selector.fit(data_clean, target_clean)
            self.selected_features = data_clean.columns[self.feature_selector.get_support()].tolist()
            
            # Restore original k if it was changed
            if isinstance(current_k, int) and current_k != self.feature_selector.k :
                 self.feature_selector.k = current_k


            if not self.selected_features: # If somehow list is empty but shouldn't be
                logger.warning("===== FeatureEngineer: Feature selection resulted in an empty list of features, but expected some. Defaulting to all original features. =====")
                self.selected_features = data.columns.tolist()
                return data


            logger.info(f"===== FeatureEngineer: Feature selection applied. Selected {len(self.selected_features)} features: {self.selected_features[:5]}... based on correlation with target. =====")
            return data[self.selected_features].copy() # Return only selected features from original data (before NaN drop for combined)
            
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Error during feature selection: {e}. Stack trace: {traceback.format_exc()} Returning all features. =====")
            self.selected_features = data.columns.tolist() # Fallback to all features from original data
            return data


    def apply_pca(self, data: pd.DataFrame, symbol: Optional[str] = None, fit: bool = False) -> pd.DataFrame:
        """Applies PCA if enabled."""
        if not self.use_pca or self.pca is None:
            logger.info("===== FeatureEngineer: PCA is disabled or PCA model not initialized. Returning data as is. =====")
            return data

        if data.empty:
            logger.warning("===== FeatureEngineer: Input data for PCA is empty. Cannot apply PCA. =====")
            return data
            
        # Ensure data is numeric and has no NaNs for PCA
        data_numeric = data.select_dtypes(include=np.number)
        if data_numeric.isnull().values.any():
            logger.warning("===== FeatureEngineer: Data for PCA contains NaNs. Applying forward-fill and back-fill before PCA. =====")
            data_numeric = data_numeric.ffill().bfill()
            # If NaNs still exist (e.g., all-NaN column), PCA might fail or produce NaNs.
            # PCA itself can handle NaNs if the underlying sklearn version supports it, or one can use an imputer.
            # For simplicity here, we assume ffill/bfill is sufficient or PCA handles it.
            if data_numeric.isnull().values.any():
                 logger.error("===== FeatureEngineer: Data for PCA still contains NaNs after ffill/bfill. PCA might produce unexpected results or fail. =====")
                 # Create list of columns that are still all NaN
                 all_nan_cols = data_numeric.columns[data_numeric.isnull().all()].tolist()
                 if all_nan_cols:
                     logger.warning(f"===== FeatureEngineer: Columns entirely NaN before PCA: {all_nan_cols}. These will likely cause issues or be handled by PCA internally. =====")
                     # Consider dropping all-NaN columns before PCA, though it changes the feature set.
                     # data_numeric = data_numeric.dropna(axis=1, how='all')


        if data_numeric.empty:
            logger.warning("===== FeatureEngineer: No numeric data available after pre-processing for PCA. Cannot apply PCA. =====")
            return data # Return original data (which might include non-numeric columns)


        pca_model_to_use: Optional[PCA] = None
        
        if fit:
            logger.info(f"===== FeatureEngineer: 'fit' is True for PCA. Fitting new PCA model for '{symbol if symbol else 'global'}'. Data shape: {data_numeric.shape} =====")
            # Adjust n_components for PCA if it's larger than number of features
            num_available_features = data_numeric.shape[1]
            current_n_components = self.pca.n_components
            
            if isinstance(current_n_components, int) and current_n_components > num_available_features:
                logger.warning(f"===== FeatureEngineer: Requested n_components={current_n_components} for PCA is more than available features ({num_available_features}). Setting n_components to {num_available_features}. =====")
                self.pca.n_components = num_available_features
            elif isinstance(current_n_components, float) and (0 < current_n_components <= 1.0): # Variance explained
                pass # n_components is a ratio, fine
            elif num_available_features == 0:
                logger.error("===== FeatureEngineer: No features available to fit PCA model. Cannot apply PCA. =====")
                return data


            if self.pca.n_components == 0 and num_available_features > 0 : # If it became 0 but shouldn't
                 logger.warning(f"===== FeatureEngineer: PCA n_components is 0, but {num_available_features} features exist. Defaulting to min(n_samples, n_features). =====")
                 # Let PCA decide n_components if it was set to 0 incorrectly. Or set to num_available_features.
                 self.pca.n_components = None # Or min(data_numeric.shape[0], num_available_features)


            try:
                self.pca.fit(data_numeric)
                if symbol:
                    self.pca_models[symbol] = self.pca
                else:
                    self.global_pca_model = self.pca
                pca_model_to_use = self.pca
                logger.info(f"===== FeatureEngineer: Fitted PCA model for '{symbol if symbol else 'global'}'. Explained variance ratio by {pca_model_to_use.n_components_} components: {np.sum(pca_model_to_use.explained_variance_ratio_):.4f} =====")
            except Exception as e:
                logger.error(f"===== FeatureEngineer: Error fitting PCA for '{symbol if symbol else 'global'}': {e}. Stack trace: {traceback.format_exc()} Returning data without PCA. =====")
                # Restore n_components if it was changed
                if isinstance(current_n_components, int) and self.pca.n_components != current_n_components:
                    self.pca.n_components = current_n_components
                return data # Return original data if PCA fit fails
            
            # Restore n_components if it was changed and not a ratio
            if isinstance(current_n_components, int) and self.pca.n_components != current_n_components:
                self.pca.n_components = current_n_components

        else: # Use existing PCA model
            pca_model_to_use = self.pca_models.get(symbol) if symbol else self.global_pca_model
            if pca_model_to_use is None:
                logger.error(f"===== FeatureEngineer: PCA requested for '{symbol if symbol else 'global'}' but no PCA model is fitted or available, and 'fit' is False. Cannot apply PCA. =====")
                # Fallback or raise error. For now, return data as is.
                return data
            try:
                check_is_fitted(pca_model_to_use)
                logger.info(f"===== FeatureEngineer: Using pre-fitted PCA model for '{symbol if symbol else 'global'}'. Expecting {pca_model_to_use.n_features_in_} input features, producing {pca_model_to_use.n_components_} components. =====")
            except NotFittedError:
                logger.error(f"===== FeatureEngineer: PCA model for '{symbol if symbol else 'global'}' found but reports NotFittedError. Cannot apply PCA. =====")
                return data


        try:
            # Check for feature count mismatch before transform
            if pca_model_to_use.n_features_in_ != data_numeric.shape[1]:
                logger.error(f"===== FeatureEngineer: PCA model expects {pca_model_to_use.n_features_in_} features, but input data has {data_numeric.shape[1]} numeric features. Cannot transform. Symbol: '{symbol if symbol else 'global'}'. =====")
                logger.error(f"PCA model features: {pca_model_to_use.get_feature_names_out() if hasattr(pca_model_to_use, 'get_feature_names_out') else 'N/A'}")
                logger.error(f"Input numeric features: {data_numeric.columns.tolist()}")
                return data # Return original data if feature count mismatch


            pca_result = pca_model_to_use.transform(data_numeric)
            pca_feature_names = [f'pca_comp_{i}' for i in range(pca_result.shape[1])]
            df_pca = pd.DataFrame(pca_result, index=data_numeric.index, columns=pca_feature_names)
            
            # Combine with non-numeric features from original data if any
            df_non_numeric = data.select_dtypes(exclude=np.number)
            result_df = pd.concat([df_pca, df_non_numeric], axis=1)
            
            # Update self.selected_features if PCA is the last step modifying feature set
            # This assumes if PCA is used, its outputs are the 'selected' features.
            self.selected_features = pca_feature_names 

            logger.info(f"===== FeatureEngineer: PCA applied. Data shape changed to {result_df.shape}. Number of PCA components: {pca_result.shape[1]}. =====")
            return result_df
            
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Error applying PCA transformation for '{symbol if symbol else 'global'}': {e}. Stack trace: {traceback.format_exc()} Returning data without PCA. =====")
            return data


    def create_sequences(self, data: pd.DataFrame, symbol: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Creates sequences from the feature-engineered DataFrame.
        Assumes 'data' contains the final set of features (e.g., after selection/PCA if used).
        The features in 'data' should be numeric and scaled.
        """
        logger.info(f"===== FeatureEngineer: Creating final prediction sequence for symbol '{symbol if symbol else 'global'}'. Data shape for sequence creation: {data.shape}. =====")

        if data.empty:
            logger.error(f"===== FeatureEngineer: Data for sequence creation is empty for symbol '{symbol}'. Cannot create sequences. =====")
            return None

        # Determine which features to use for sequencing.
        # If feature selection or PCA was used, self.selected_features should hold the names.
        # Otherwise, use all numeric columns from the input 'data'.
        
        features_for_sequencing: List[str] = []
        
        if self.use_pca and self.selected_features and all(s.startswith('pca_comp_') for s in self.selected_features):
            logger.info(f"===== FeatureEngineer: Using {len(self.selected_features)} PCA components for creating sequences. Symbol: {symbol if symbol else 'global'}. PCA components: {self.selected_features[:5]}... =====")
            features_for_sequencing = [col for col in self.selected_features if col in data.columns]
        elif self.use_feature_selection and self.selected_features:
            logger.info(f"===== FeatureEngineer: Using {len(self.selected_features)} pre-selected features for creating sequences. Symbol: {symbol if symbol else 'global'}. Selected features example: {self.selected_features[:5]}... =====")
            features_for_sequencing = [col for col in self.selected_features if col in data.columns]
        else:
            # Default to using all numeric columns from the provided data if no specific selection/PCA features are set
            # This also implies that self.feature_columns (from TA step) might be broader.
            # The 'data' input to this function should be the one that has undergone normalization.
            numeric_cols_in_data = data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols_in_data:
                 logger.error(f"===== FeatureEngineer: No numeric columns found in data for sequence creation for symbol '{symbol if symbol else 'global'}'. Columns: {data.columns.tolist()}. Cannot create sequences. =====")
                 return None
            
            logger.info(f"===== FeatureEngineer: Using all {len(numeric_cols_in_data)} available numeric features in input data for creating sequences as no specific selection/PCA features were previously finalized. Symbol: {symbol if symbol else 'global'}. Features example: {numeric_cols_in_data[:5]}... =====")
            features_for_sequencing = numeric_cols_in_data

        if not features_for_sequencing:
            logger.error(f"===== FeatureEngineer: No features identified for sequencing for symbol '{symbol if symbol else 'global'}'. Cannot create sequences. self.selected_features: {self.selected_features}, PCA: {self.use_pca}, FeatureSelection: {self.use_feature_selection} =====")
            return None
            
        # Ensure all selected features are actually in the dataframe
        missing_seq_features = [f for f in features_for_sequencing if f not in data.columns]
        if missing_seq_features:
            logger.error(f"===== FeatureEngineer: One or more features intended for sequencing are missing from the input data for symbol '{symbol if symbol else 'global'}': {missing_seq_features}. Available columns: {data.columns.tolist()}. Cannot create sequences. =====")
            return None

        data_for_sequence_creation = data[features_for_sequencing].copy()
        
        logger.info(f"===== FeatureEngineer: Creating sequences using {len(features_for_sequencing)} features for symbol '{symbol if symbol else 'global'}'. Input data shape: {data_for_sequence_creation.shape}, Num features: {len(features_for_sequencing)} =====")

        # Handle NaNs in data for SequencePreprocessor
        # SequencePreprocessor expects data with no NaNs in the features used.
        # NaNs typically occur at the start of TA features due to lookback.
        # If creating sequences for prediction, the *last* sequence is critical and must not have NaNs.
        # If creating sequences for training/fitting, rows with NaNs are typically dropped.
        
        if data_for_sequence_creation.isnull().values.any():
            nan_cols = data_for_sequence_creation.columns[data_for_sequence_creation.isnull().any()].tolist()
            logger.warning(f"===== FeatureEngineer: Data for SequencePreprocessor for symbol '{symbol if symbol else 'global'}' (shape {data_for_sequence_creation.shape}) contains NaNs in columns: {nan_cols}. Attempting ffill then bfill. =====")
            data_for_sequence_creation.ffill(inplace=True)
            data_for_sequence_creation.bfill(inplace=True)

            # Check if any columns are *still* all NaN (e.g., if entire column was NaN initially)
            still_all_nan_cols = data_for_sequence_creation.columns[data_for_sequence_creation.isnull().all()].tolist()
            if still_all_nan_cols:
                logger.warning(f"===== FeatureEngineer: Columns {still_all_nan_cols} are entirely NaN in data for SequencePreprocessor for symbol '{symbol if symbol else 'global'}' AFTER fill. Dropping them. =====")
                data_for_sequence_creation.drop(columns=still_all_nan_cols, inplace=True)
                # Update features_for_sequencing list
                features_for_sequencing = [f for f in features_for_sequencing if f not in still_all_nan_cols]
                if not features_for_sequencing:
                    logger.error(f"===== FeatureEngineer: No features left after dropping all-NaN columns for symbol '{symbol if symbol else 'global'}'. Cannot create sequences. =====")
                    return None
            
            if data_for_sequence_creation.isnull().values.any(): # Should not happen if bfill worked, unless df was too short
                 logger.error(f"===== FeatureEngineer: NaNs STILL PRESENT in data for SequencePreprocessor for symbol '{symbol if symbol else 'global'}' even after ffill/bfill. This indicates insufficient data or persistent all-NaN columns not dropped. Shape: {data_for_sequence_creation.shape}. =====")


        # SequencePreprocessor.create_sequences expects a NumPy array
        try:
            # The target argument in create_sequences is for training. For prediction, it's not strictly needed
            # if we only care about the input sequence X.
            # However, if SequencePreprocessor requires it, we might need to pass a dummy target.
            # Let's assume for now it can handle target=None or a simplified version for prediction path.
            # The current SequencePreprocessor does not seem to use 'target' in its create_sequences method.
            
            sequences_x, _ = self.sequence_preprocessor.create_sequences(
                data_for_sequence_creation.to_numpy(), # Pass only the array of features
                None # No target needed for unsupervised sequence generation for prediction
            )

            if sequences_x is None or sequences_x.shape[0] == 0:
                logger.error(f"===== FeatureEngineer: Sequence creation resulted in no sequences for symbol '{symbol if symbol else 'global'}'. Input data shape was {data_for_sequence_creation.shape}. Sequence length {self.sequence_length}. =====")
                return None

            logger.info(f"===== FeatureEngineer: Sequences created successfully for symbol '{symbol if symbol else 'global'}'. Shape: {sequences_x.shape} (num_sequences, sequence_length, num_features). =====")
            return sequences_x # Return all sequences (e.g., for backtesting or if model needs it)
            
        except Exception as e:
            logger.error(f"===== FeatureEngineer: Error during sequence creation for symbol '{symbol if symbol else 'global'}': {e}. Stack trace: {traceback.format_exc()}. Input data shape to preprocessor: {data_for_sequence_creation.shape} =====")
            return None


    def prepare_features_for_prediction(self, raw_data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Orchestrates the full feature engineering pipeline for a single prediction pass.
        1. Calculates Technical Indicators.
        2. Normalizes features using a pre-fitted scaler.
        3. Optionally applies feature selection or PCA using pre-fitted models/stats.
        4. Creates the final input sequence for the model.

        Args:
            raw_data: DataFrame with raw market data (OHLCV).
            symbol: The stock symbol for which features are being prepared.

        Returns:
            A tuple containing:
                - np.ndarray: The final (single) sequence of features ready for model input,
                              shape (sequence_length, num_features). Or None if error.
                - float: The last actual close price from the raw_data. Or None if error.
        """
        logger.info(f"===== FeatureEngineer: Starting feature preparation pipeline for PREDICTION using input data of shape {raw_data.shape} for symbol '{symbol}'. Goal is to produce a single NumPy array sequence for the model. =====")

        if raw_data.empty:
            logger.error(f"===== FeatureEngineer (predict): Raw data for symbol '{symbol}' is empty. Cannot prepare features. =====")
            return None, None
            
        # Ensure enough data for sequence_length + TA lookbacks
        # This check should ideally be more dynamic based on actual TAs calculated.
        # For now, a simple check against sequence_length. Longest TA lookback is 200.
        min_required_rows = self.sequence_length + 200 # A rough estimate, should be more precise
        if len(raw_data) < self.sequence_length : # Basic check
             logger.warning(f"===== FeatureEngineer (predict): Insufficient raw data rows ({len(raw_data)}) for symbol '{symbol}' to form a sequence of length {self.sequence_length}. Prediction might be compromised. (Min ideally {min_required_rows} for all TAs) =====")
             # If len(raw_data) < self.sequence_length, sequence creation will fail later.

        # 1. Calculate Technical Indicators (and other StockFormatter features)
        logger.info(f"===== FeatureEngineer (predict): Calculating technical indicators for symbol '{symbol}'. Initial raw data shape: {raw_data.shape} =====")
        df_with_ta, last_close_price = self.calculate_technical_indicators(raw_data, symbol=symbol, fit_scalers=False)

        if df_with_ta is None or df_with_ta.empty:
            logger.error(f"===== FeatureEngineer (predict): Failed to calculate technical indicators for symbol '{symbol}'. Raw data shape was {raw_data.shape}. =====")
            return None, last_close_price # last_close_price might still be valid from initial check

        logger.info(f"===== FeatureEngineer (predict): Technical indicators added. Shape after TA for symbol '{symbol}': {df_with_ta.shape} =====")

        # Critical: Handle NaNs in the *last* `sequence_length` rows before normalization.
        # These NaNs can break normalization or lead to NaN sequences.
        # Particularly, NaNs from long-lookback TAs (like SMA200) if not enough history was fetched.
        if df_with_ta.iloc[-self.sequence_length:].isnull().values.any():
            nan_cols_in_seq = df_with_ta.columns[df_with_ta.iloc[-self.sequence_length:].isnull().any()].tolist()
            logger.warning(f"===== FeatureEngineer (predict): CRITICAL - NaNs detected in the last {self.sequence_length} rows for symbol '{symbol}' (cols: {nan_cols_in_seq}) BEFORE normalization. Attempting ffill+bfill on df_with_ta. =====")
            
            # Create a copy for this operation to avoid modifying the original df_with_ta if it's used elsewhere or for debugging
            df_filled_for_seq = df_with_ta.copy()
            df_filled_for_seq.ffill(inplace=True)
            df_filled_for_seq.bfill(inplace=True) # bfill after ffill to handle leading NaNs if any part of sequence was all NaN initially

            if df_filled_for_seq.iloc[-self.sequence_length:].isnull().values.any():
                still_nan_cols = df_filled_for_seq.columns[df_filled_for_seq.iloc[-self.sequence_length:].isnull().any()].tolist()
                logger.error(f"===== FeatureEngineer (predict): NaNs STILL PRESENT in the last {self.sequence_length} rows for symbol '{symbol}' (cols: {still_nan_cols}) after fill. Prediction will likely fail or be inaccurate. =====")
                # This is a critical issue. The model expects complete, scaled sequences.
                # Depending on the model, this could be fatal.
                # For now, we proceed with df_filled_for_seq, but this needs monitoring.
            
            df_to_normalize = df_filled_for_seq # Use the filled DataFrame for subsequent steps
        else:
            df_to_normalize = df_with_ta # No NaNs in the critical part, proceed

        # 2. Normalize Features (using pre-fitted scaler for the symbol or global)
        logger.info(f"===== FeatureEngineer (predict): Normalizing features for symbol '{symbol}'. Data shape before norm: {df_to_normalize.shape} =====")
        df_normalized = self.normalize_features(df_to_normalize, symbol=symbol, fit=False) # fit=False is crucial for prediction

        if df_normalized is None or df_normalized.empty:
            logger.error(f"===== FeatureEngineer (predict): Feature normalization failed for symbol '{symbol}'. =====")
            return None, last_close_price
        
        logger.info(f"===== FeatureEngineer (predict): Features normalized. Shape after normalization for symbol '{symbol}': {df_normalized.shape} =====")

        df_for_processing = df_normalized # This is the dataframe to be used for selection/PCA/sequencing

        # 3. Apply Feature Selection (optional, using pre-defined selected features)
        # For prediction, we should use self.selected_features if it was determined during a 'fit' phase (e.g. on training data)
        # The `apply_feature_selection` method expects a target, which is not available in prediction.
        # So, we just subset the DataFrame if `self.selected_features` is populated.
        
        final_features_for_sequence: List[str] = []

        if self.use_feature_selection:
            if self.selected_features is not None and self.selected_features:
                logger.info(f"===== FeatureEngineer (predict): Using pre-defined selected features ({len(self.selected_features)}) for symbol '{symbol}': {self.selected_features[:5]}... =====")
                # Ensure all selected features are present in the processed dataframe
                missing_selected = [f for f in self.selected_features if f not in df_for_processing.columns]
                if missing_selected:
                    logger.error(f"===== FeatureEngineer (predict): Some pre-selected features are missing from the processed data for symbol '{symbol}': {missing_selected}. Available: {df_for_processing.columns.tolist()}. Will proceed with available intersection. =====")
                    # Use only the intersection
                    final_features_for_sequence = [f for f in self.selected_features if f in df_for_processing.columns]
                    if not final_features_for_sequence:
                         logger.error(f"===== FeatureEngineer (predict): No intersection between pre-selected features and available data columns for symbol '{symbol}'. Cannot proceed with feature selection logic. =====")
                         # Fallback: use all numeric columns from df_for_processing
                         final_features_for_sequence = df_for_processing.select_dtypes(include=np.number).columns.tolist()
                else:
                    final_features_for_sequence = self.selected_features
            else:
                logger.warning(f"===== FeatureEngineer (predict): Feature selection enabled for symbol '{symbol}', but no pre-selected features found (`self.selected_features` is None). Attempting to run selection now (might be inconsistent if not based on training data distribution). =====")
                # This path is problematic for prediction if a target is needed.
                # For now, if self.selected_features is None, it implies all (numeric) features are used.
                # The self.apply_feature_selection needs a target, so we can't call it here directly.
                # We should rely on self.selected_features being set during a training/fitting phase.
                # Fallback: use all numeric columns from df_for_processing.
                logger.warning(f"===== FeatureEngineer (predict): Falling back to using all available numeric features for symbol '{symbol}' as feature selection cannot be run without a target during prediction. =====")
                final_features_for_sequence = df_for_processing.select_dtypes(include=np.number).columns.tolist()

            df_selected = df_for_processing[final_features_for_sequence].copy()
            logger.info(f"===== FeatureEngineer (predict): Shape after applying feature selection rules for symbol '{symbol}': {df_selected.shape} (using {len(final_features_for_sequence)} features) =====")
        else: # No feature selection
            df_selected = df_for_processing.copy()
            final_features_for_sequence = df_selected.select_dtypes(include=np.number).columns.tolist() # All numeric features
            logger.info(f"===== FeatureEngineer (predict): Feature selection disabled. Using all {len(final_features_for_sequence)} numeric features from processed data for symbol '{symbol}'. Shape: {df_selected.shape} =====")
            

        if df_selected.empty or not final_features_for_sequence:
            logger.error(f"===== FeatureEngineer (predict): Dataframe became empty or no features selected after feature selection step for symbol '{symbol}'. Cannot proceed. =====")
            return None, last_close_price

        # 4. Apply PCA (optional, using pre-fitted PCA model)
        if self.use_pca:
            logger.info(f"===== FeatureEngineer (predict): Applying PCA for symbol '{symbol}'. Data shape before PCA: {df_selected.shape} =====")
            # `apply_pca` uses numeric columns. df_selected should already be mostly numeric.
            df_pca = self.apply_pca(df_selected, symbol=symbol, fit=False) # fit=False is crucial
            if df_pca is None or df_pca.empty:
                logger.error(f"===== FeatureEngineer (predict): PCA application failed for symbol '{symbol}'. =====")
                return None, last_close_price
            
            # After PCA, the columns are pca_comp_0, pca_comp_1, etc. These are the features for sequencing.
            final_features_for_sequence = df_pca.select_dtypes(include=np.number).columns.tolist()
            df_final_features = df_pca
            logger.info(f"===== FeatureEngineer (predict): PCA applied. Data shape changed to {df_final_features.shape}. =====")
        else: # No PCA
            df_final_features = df_selected
            # final_features_for_sequence is already set from the feature selection step (or all numerics if no selection)
            logger.info(f"===== FeatureEngineer (predict): PCA disabled. Using features from previous step for symbol '{symbol}'. Shape: {df_final_features.shape} =====")
            

        if df_final_features.empty or not final_features_for_sequence:
            logger.error(f"===== FeatureEngineer (predict): Dataframe became empty or no features selected after PCA/selection step for symbol '{symbol}'. Cannot proceed. =====")
            return None, last_close_price
            
        # Ensure only the final set of features are passed for sequencing
        df_for_sequence_input = df_final_features[final_features_for_sequence].copy()


        # 5. Create Sequences
        # This will generate all possible sequences from df_for_sequence_input.
        # We then need to take the *last* one for prediction.
        logger.info(f"===== FeatureEngineer (predict): Creating sequences from final features for symbol '{symbol}'. Data shape for sequencing: {df_for_sequence_input.shape}, Num features: {len(final_features_for_sequence)} =====")
        
        # Final check for NaNs in the data that goes into sequence creation, especially the last `sequence_length` rows.
        # This should have been handled before normalization, but an extra check on `df_for_sequence_input`
        if df_for_sequence_input.iloc[-self.sequence_length:].isnull().values.any():
            nan_cols_final_seq = df_for_sequence_input.columns[df_for_sequence_input.iloc[-self.sequence_length:].isnull().any()].tolist()
            logger.error(f"===== FeatureEngineer (predict): CRITICAL - NaNs detected in the last {self.sequence_length} rows of data JUST BEFORE sequence creation for symbol '{symbol}' (cols: {nan_cols_final_seq}). This will likely lead to a NaN sequence. Data shape: {df_for_sequence_input.shape}. =====")
            # Attempt one last fill, though this indicates a deeper issue if NaNs persist till here.
            df_for_sequence_input.ffill(inplace=True)
            df_for_sequence_input.bfill(inplace=True)
            if df_for_sequence_input.iloc[-self.sequence_length:].isnull().values.any():
                 logger.error(f"===== FeatureEngineer (predict): CRITICAL - NaNs PERSIST after final fill for symbol '{symbol}'. =====")


        all_sequences = self.create_sequences(df_for_sequence_input, symbol=symbol)

        if all_sequences is None or all_sequences.shape[0] == 0:
            logger.error(f"===== FeatureEngineer (predict): Failed to create any sequences from processed data for symbol '{symbol}'. Shape of data fed to create_sequences: {df_for_sequence_input.shape}. =====")
            return None, last_close_price

        # For prediction, we need only the *last* available sequence
        last_sequence = all_sequences[-1, :, :] # Shape: (sequence_length, num_features)
        
        num_features_in_seq = last_sequence.shape[1]
        expected_num_features = len(final_features_for_sequence)
        
        if num_features_in_seq != expected_num_features:
            logger.warning(f"===== FeatureEngineer (predict): Mismatch in number of features in the final sequence for symbol '{symbol}'. Expected {expected_num_features} (from {final_features_for_sequence}), but got {num_features_in_seq}. This could be due to all-NaN columns being dropped during sequencing. =====")
            # This could be an issue if the model expects a fixed number of features.
            # The tft_predictor.py log showed "Keras expects: (30, 25)".
            # We need to ensure the final sequence here matches that.

        logger.info(f"===== FeatureEngineer (predict): Extracted the last sequence of length {self.sequence_length}. Final prediction input sequence shape for symbol '{symbol}': {last_sequence.shape}. Num features in seq: {num_features_in_seq} =====")
        
        # Final check for NaNs in the very last sequence passed to the model
        if np.isnan(last_sequence).any():
            logger.error(f"===== FeatureEngineer (predict): CRITICAL - The final prepared sequence for symbol '{symbol}' (shape {last_sequence.shape}) CONTAINS NaNs. Model prediction will likely fail or be highly inaccurate. =====")
            # Example: np.argwhere(np.isnan(last_sequence)) could show where NaNs are.

        # Return the DataFrame that was used to create the sequence, plus the last close price
        # The TFTPredictor will be responsible for selecting the final features for the Keras model from this DataFrame.
        logger.info(f"===== FeatureEngineer (predict): Returning final features DataFrame for '{symbol}' (shape {df_for_sequence_input.shape}) and last_close_price {last_close_price}. This DataFrame should contain all necessary scaled features. =====")
        return df_for_sequence_input, last_close_price


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