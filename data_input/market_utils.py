"""Utility functions for market data processing and validation."""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

class MarketDataError(Exception):
    """Exception raised for market data related errors."""
    pass

def validate_market_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 20,
    max_price_change: float = 0.5,  # 50% max price change
    max_volume_change: float = 10.0,  # 10x max volume change
    require_indicators: bool = False
) -> bool:
    """Validate market data DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns (default: standard OHLCV)
        min_rows: Minimum number of rows required
        max_price_change: Maximum allowed price change between consecutive rows
        max_volume_change: Maximum allowed volume change between consecutive rows
        require_indicators: Whether to require technical indicators
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        MarketDataError: If validation fails with specific error message
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if require_indicators:
            required_columns.extend([
                'sma_20', 'sma_50', 'rsi_14', 'macd', 'volume_ma'
            ])
    
    # Check for minimum rows
    if len(df) < min_rows:
        error_msg = f"Insufficient data rows: {len(df)} < {min_rows}"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for missing values in required columns
    null_cols = df[required_columns].columns[df[required_columns].isnull().any()].tolist()
    if null_cols:
        error_msg = f"Columns with missing values: {null_cols}"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Ensure all price columns exist and are not empty
    price_cols = ['open', 'high', 'low', 'close']
    missing_or_empty = [col for col in price_cols if col not in df.columns or df[col].isnull().values.all()]
    if missing_or_empty:
        error_msg = f"Missing or empty required price columns: {missing_or_empty}"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Convert price columns to numpy arrays for faster comparison
    price_data = df[price_cols].values
    
    # Check for negative values in price and volume
    if np.any(price_data < 0):
        error_msg = "Data contains negative values in price columns"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for negative volume using numpy
    volume_data = df['volume'].values
    if np.any(volume_data < 0):
        error_msg = "Data contains negative values in volume"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for high-low consistency using numpy arrays
    high_low_mask = price_data[:, 1] >= price_data[:, 2]  # high >= low
    if not np.all(high_low_mask):
        error_msg = "High prices are not consistently greater than low prices"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for open-close consistency using numpy arrays
    high_open_mask = price_data[:, 1] >= price_data[:, 0]  # high >= open
    high_close_mask = price_data[:, 1] >= price_data[:, 3]  # high >= close
    if not (np.all(high_open_mask) and np.all(high_close_mask)):
        error_msg = "High price is not consistently greater than open and close"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    low_open_mask = price_data[:, 2] <= price_data[:, 0]  # low <= open
    low_close_mask = price_data[:, 2] <= price_data[:, 3]  # low <= close
    if not (np.all(low_open_mask) and np.all(low_close_mask)):
        error_msg = "Low price is not consistently less than open and close"
        logger.error(error_msg)
        raise MarketDataError(error_msg)
    
    # Check for extreme price changes using numpy
    if len(price_data) > 1:
        price_changes = np.abs(np.diff(price_data, axis=0) / price_data[:-1])
        if np.any(price_changes > max_price_change):
            error_msg = f"Extreme price changes detected (> {max_price_change*100}%)"
            logger.error(error_msg)
            raise MarketDataError(error_msg)
    
    # Check for extreme volume changes using numpy
    # Only check if we have more than one row of data
    if len(volume_data) > 1:
        # Handle zero volumes by replacing with a small number
        volume_data_safe = np.where(volume_data == 0, 1e-10, volume_data)
        # Ensure volume_data_safe is 1D array
        volume_data_safe = volume_data_safe.flatten()
        # Calculate volume changes
        volume_changes = np.abs(np.diff(volume_data_safe) / volume_data_safe[:-1])
        if np.any(volume_changes > max_volume_change):
            error_msg = f"Extreme volume changes detected (> {max_volume_change*100}%)"
            logger.error(error_msg)
            raise MarketDataError(error_msg)
    
    return True

def clean_market_data(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    max_gap: int = 3,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0
) -> pd.DataFrame:
    """Clean and preprocess market data.
    
    Args:
        df: DataFrame to clean
        fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
        max_gap: Maximum number of consecutive missing values to fill
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame received in clean_market_data. Returning as is.")
        return df
        
    df = df.copy()
    
    # Handle missing values
    if fill_method == 'ffill':
        df = df.ffill(limit=max_gap)
    elif fill_method == 'bfill':
        df = df.bfill(limit=max_gap)
    elif fill_method == 'interpolate':
        df = df.interpolate(method='linear', limit=max_gap)
    
    # Remove remaining missing values
    df = df.dropna()
    
    # Exit early if we've lost all data
    if df.empty:
        logger.warning("DataFrame became empty after handling missing values. Returning empty DataFrame.")
        return df
    
    if remove_outliers:
        # Create a combined mask for all outliers instead of filtering sequentially
        keep_mask = pd.Series(True, index=df.index)
        
        # Check price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                # Update the mask - only mark as False if it's an outlier
                col_mask = abs(df[col] - mean) <= (outlier_threshold * std)
                keep_mask = keep_mask & col_mask
        
        # Check volume separately
        if 'volume' in df.columns:
            mean_vol = df['volume'].mean()
            std_vol = df['volume'].std()
            vol_mask = abs(df['volume'] - mean_vol) <= (outlier_threshold * std_vol)
            keep_mask = keep_mask & vol_mask
        
        # Apply the combined mask
        df = df[keep_mask]
        
        # If we've filtered out everything, log a warning and keep at least 80% of the original data
        if df.empty:
            logger.warning("Outlier removal filtered out all data. Falling back to keeping top 80% of data by distance from mean.")
            # Re-compute with a more permissive approach - keep the closest 80% of points to the mean
            df_original = df.copy()
            for col in price_cols + ['volume']:
                if col in df_original.columns:
                    mean = df_original[col].mean()
                    # Calculate distance from mean
                    df_original[f'{col}_dist'] = abs(df_original[col] - mean)
            
            # Sort by total distance and keep top 80%
            if len(df_original) > 0:
                total_dist = df_original[[f'{col}_dist' for col in price_cols + ['volume'] if f'{col}_dist' in df_original.columns]].sum(axis=1)
                df_original['total_dist'] = total_dist
                df_original = df_original.sort_values('total_dist')
                keep_count = max(int(len(df_original) * 0.8), 1)  # Keep at least 1 row
                df = df_original.iloc[:keep_count].drop([col for col in df_original.columns if col.endswith('_dist')], axis=1)
    
    # Ensure chronological order
    df = df.sort_index()
    
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Replace infinite values with NaN and drop them
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def resample_market_data(
    df: pd.DataFrame,
    interval: str,
    agg_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Resample market data to a different interval.
    
    Args:
        df: DataFrame with market data
        interval: Target interval (e.g., '1W', '1D', '1H')
        agg_dict: Optional dictionary of aggregation functions
        
    Returns:
        Resampled DataFrame with timestamps aligned to market open (14:30 UTC)
    """
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

    try:
        original_index_names = df.index.names
        df = df.reset_index()

        # Ensure datetime column is present and correctly named
        if 'datetime' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'datetime'})
            elif original_index_names and 'datetime' in original_index_names:
                pass # datetime is already in index names, will be handled
            elif df.index.name == 'datetime': # Covers cases where it was the primary index
                 df = df.reset_index() # Make sure it's a column
            else:
                raise MarketDataError("DataFrame must have a 'datetime' or 'Date' column or a DatetimeIndex.")

        # Ensure datetime is parsed correctly and has UTC timezone
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Check if datetime is already timezone-aware
        if df['datetime'].dt.tz is not None:
            # Convert to UTC if already timezone-aware
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        else:
            # Localize to UTC if timezone-naive
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')

        # Check if we have flattened column names (e.g., 'open_AAPL')
        has_flattened_cols = any('_' in col for col in df.columns if col not in ['symbol', 'datetime'])
        
        # Map columns to OHLCV base names
        base_to_cols = {}
        if has_flattened_cols:
            # Extract base column names from flattened format
            for col in df.columns:
                if col in ['symbol', 'datetime']:
                    continue
                    
                parts = col.split('_', 1)
                if len(parts) == 2:
                    base_col, symbol = parts
                    if base_col not in base_to_cols:
                        base_to_cols[base_col] = []
                    base_to_cols[base_col].append(col)
        else:
            # Simple column mapping for non-flattened data
            for col in df.columns:
                if col in ['symbol', 'datetime']:
                    continue
                if col in agg_dict:
                    if col not in base_to_cols:
                        base_to_cols[col] = []
                    base_to_cols[col].append(col)

        # Identify OHLCV columns using our mapping
        ohlcv_base_cols = [col for col in agg_dict.keys() if col in base_to_cols]
        
        # Get all flattened OHLCV columns
        ohlcv_cols = []
        for base_col in ohlcv_base_cols:
            ohlcv_cols.extend(base_to_cols[base_col])
            
        other_cols = [col for col in df.columns if col not in ohlcv_cols and col not in ['datetime', 'symbol']]

        if not ohlcv_cols:
            # If we still don't have columns, try a more permissive approach for flattened names
            if has_flattened_cols:
                for col in df.columns:
                    for base_col in agg_dict.keys():
                        if col.startswith(f"{base_col}_"):
                            if base_col not in base_to_cols:
                                base_to_cols[base_col] = []
                            base_to_cols[base_col].append(col)
                            ohlcv_cols.append(col)
            
            # If we still don't have columns, raise error
            if not ohlcv_cols:
                raise MarketDataError(f"No OHLCV columns found for resampling. Available: {df.columns.tolist()}")

        # Create aggregation dictionary for flattened columns
        agg_dict_flat = {}
        for base_col, cols in base_to_cols.items():
            if base_col in agg_dict:
                for col in cols:
                    agg_dict_flat[col] = agg_dict[base_col]

        # Separate data for resampling
        if 'symbol' in df.columns:
            resampled_dfs = []
            for symbol_val, group in df.groupby('symbol'):
                # Set datetime as index for resampling
                group = group.set_index('datetime')
                
                # Determine which columns to use for this group
                group_cols = [col for col in ohlcv_cols if col in group.columns]
                
                # Create group-specific aggregation dict
                group_agg = {col: agg_dict_flat[col] for col in group_cols if col in agg_dict_flat}
                
                # Add other columns with 'last' aggregation
                for col in group.columns:
                    if col not in group_agg and col != 'symbol':
                        group_agg[col] = 'last'
                
                if not group_cols:
                    raise MarketDataError(f"No OHLCV columns found for symbol {symbol_val}. Available: {group.columns.tolist()}")
                
                # Resample the data
                resampled = group.resample(interval, offset='14h30min').agg(group_agg)
                
                # Add symbol column back
                resampled['symbol'] = symbol_val
                
                # Reset index for concat
                resampled = resampled.reset_index()
                
                resampled_dfs.append(resampled)
                
            # Combine all resampled data
            final_resampled_df = pd.concat(resampled_dfs, ignore_index=True)
        else:
            # For data without a symbol column
            df = df.set_index('datetime')
            
            # Create aggregation dict
            full_agg = {}
            # Add OHLCV columns
            for col in ohlcv_cols:
                if col in agg_dict_flat:
                    full_agg[col] = agg_dict_flat[col]
            
            # Add other columns with 'last' aggregation
            for col in other_cols:
                full_agg[col] = 'last'
                
            # Resample the data
            final_resampled_df = df.resample(interval, offset='14h30min').agg(full_agg).reset_index()

        # Set the index structure back
        if 'symbol' in final_resampled_df.columns:
            final_resampled_df = final_resampled_df.set_index(['symbol', 'datetime'])
        else:
            final_resampled_df = final_resampled_df.set_index('datetime')
        
        # Ensure original index names are restored if they existed
        if original_index_names and all(name in final_resampled_df.index.names for name in original_index_names):
            final_resampled_df.index.names = original_index_names
        
        return final_resampled_df

    except Exception as e:
        logger.error(f"Error during resampling for interval '{interval}': {str(e)}. DataFrame columns: {df.columns.tolist()}, Index: {df.index.names if hasattr(df.index, 'names') else df.index.name}")
        import traceback
        logger.error(traceback.format_exc())
        raise MarketDataError(f"Error resampling data: {str(e)}")

def calculate_returns(
    df: pd.DataFrame,
    price_col: str = 'close',
    periods: Union[int, List[int]] = [1, 5, 10, 20],
    log_returns: bool = False
) -> pd.DataFrame:
    """Calculate returns for different periods.
    
    Args:
        df: DataFrame with price data
        price_col: Column to use for return calculation
        periods: List of periods to calculate returns for
        log_returns: Whether to calculate log returns
        
    Returns:
        DataFrame with added return columns
    """
    df = df.copy()
    
    if isinstance(periods, int):
        periods = [periods]
    
    for period in periods:
        if log_returns:
            df[f'return_{period}d'] = np.log(df[price_col] / df[price_col].shift(period))
        else:
            df[f'return_{period}d'] = df[price_col].pct_change(periods=period)
    
    return df

def detect_market_regime(
    df: pd.DataFrame,
    window: int = 20,
    volatility_threshold: float = 0.2,
    trend_threshold: float = 0.6
) -> pd.DataFrame:
    """Detect market regime based on volatility and trend.
    
    Args:
        df: DataFrame with price data
        window: Window size for calculations
        volatility_threshold: Threshold for high volatility
        trend_threshold: Threshold for trend strength
        
    Returns:
        DataFrame with added regime columns
    """
    df = df.copy()
    
    # Calculate volatility
    returns = df['close'].pct_change()
    df['volatility'] = returns.rolling(window=window).std()
    
    # Calculate trend strength using ADX
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=window).mean()
    
    # Calculate trend direction
    df['trend'] = df['close'].rolling(window=window).mean().diff()
    df['trend_strength'] = abs(df['trend']) / df['atr']
    
    # Determine regime
    df['regime'] = 'normal'
    df.loc[df['volatility'] > volatility_threshold, 'regime'] = 'volatile'
    df.loc[df['trend_strength'] > trend_threshold, 'regime'] = 'trending'
    df.loc[
        (df['volatility'] > volatility_threshold) & 
        (df['trend_strength'] > trend_threshold),
        'regime'
    ] = 'trending_volatile'
    
    return df 