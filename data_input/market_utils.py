"""Utility functions for market data processing and validation."""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union
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
    remove_outliers: bool = True,
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
    
    if remove_outliers:
        # Remove price outliers
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[abs(df[col] - mean) <= (outlier_threshold * std)]
        
        # Remove volume outliers
        mean_vol = df['volume'].mean()
        std_vol = df['volume'].std()
        df = df[abs(df['volume'] - mean_vol) <= (outlier_threshold * std_vol)]
    
    # Ensure chronological order
    df = df.sort_index()
    
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Ensure all values are finite
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def resample_market_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample market data to a different interval.
    
    Args:
        df: DataFrame with market data (must have MultiIndex with 'symbol' and datetime)
        interval: Target interval (e.g., '1W', '1M')
        
    Returns:
        Resampled DataFrame
        
    Raises:
        MarketDataError: If resampling fails
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise MarketDataError("DataFrame must have MultiIndex with 'symbol' and datetime levels")
    
    if 'symbol' not in df.index.names or 'datetime' not in df.index.names:
        raise MarketDataError("DataFrame must have 'symbol' and 'datetime' levels in index")
    
    # Debug input data
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input DataFrame index names: {df.index.names}")
    logger.debug(f"Input DataFrame columns: {df.columns.tolist()}")
    logger.debug(f"Input DataFrame first few rows:\n{df.head()}")
    
    # Get original end date for filtering
    original_end_date = pd.Timestamp(df.index.get_level_values('datetime').max())
    logger.debug(f"Original end date: {original_end_date}")
    
    try:
        # Process each symbol separately
        resampled_dfs = []
        for symbol in df.index.get_level_values('symbol').unique():
            logger.debug(f"\nProcessing symbol: {symbol}")
            
            # Get data for this symbol
            symbol_data = df.xs(symbol, level='symbol')
            logger.debug(f"Symbol data shape: {symbol_data.shape}")
            logger.debug(f"Symbol data index: {symbol_data.index}")
            
            # Create resampler
            resampler = symbol_data.resample(interval)
            logger.debug(f"Resampler groups: {list(resampler.groups.keys())[:5]}...")  # Show first 5 groups
            
            # Apply aggregation for each column separately to ensure we get a DataFrame
            resampled = pd.DataFrame(index=resampler.groups.keys())
            logger.debug(f"Empty resampled DataFrame index: {resampled.index[:5]}...")  # Show first 5 indices
            
            # Apply aggregations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                logger.debug(f"Processing column: {col}")
                if col == 'open':
                    resampled[col] = resampler[col].apply(lambda x: x.iloc[0] if not x.empty else np.nan)
                elif col == 'high':
                    resampled[col] = resampler[col].apply(lambda x: x.max() if not x.empty else np.nan)
                elif col == 'low':
                    resampled[col] = resampler[col].apply(lambda x: x.min() if not x.empty else np.nan)
                elif col == 'close':
                    resampled[col] = resampler[col].apply(lambda x: x.iloc[-1] if not x.empty else np.nan)
                elif col == 'volume':
                    resampled[col] = resampler[col].apply(lambda x: x.sum() if not x.empty else np.nan)
                logger.debug(f"After {col} aggregation, sample values:\n{resampled[col].head()}")
            
            # Add symbol column
            resampled['symbol'] = symbol
            logger.debug(f"After adding symbol, columns: {resampled.columns.tolist()}")
            
            # Reset index to make datetime a column
            resampled = resampled.reset_index()
            resampled = resampled.rename(columns={'index': 'datetime'})
            logger.debug(f"After reset_index, columns: {resampled.columns.tolist()}")
            logger.debug(f"After reset_index, datetime sample: {resampled['datetime'].head()}")
            
            # Set multi-index with datetime and symbol
            resampled = resampled.set_index(['datetime', 'symbol'])
            logger.debug(f"After set_index, index names: {resampled.index.names}")
            logger.debug(f"After set_index, index levels: {[level.name for level in resampled.index.levels]}")
            
            # Filter out data points beyond original end date
            resampled = resampled[resampled.index.get_level_values('datetime') <= original_end_date]
            logger.debug(f"After filtering, shape: {resampled.shape}")
            
            resampled_dfs.append(resampled)
        
        if not resampled_dfs:
            raise MarketDataError("No data could be resampled")
        
        # Combine all resampled data
        result = pd.concat(resampled_dfs)
        logger.debug(f"\nFinal result shape: {result.shape}")
        logger.debug(f"Final result index names: {result.index.names}")
        logger.debug(f"Final result columns: {result.columns.tolist()}")
        logger.debug(f"Final result first few rows:\n{result.head()}")
        
        # Reorder index levels to maintain structure
        result = result.reorder_levels(['symbol', 'datetime'])
        logger.debug(f"After reordering levels, index names: {result.index.names}")
        
        return result.sort_index()
        
    except Exception as e:
        logger.error(f"Error during resampling: {str(e)}")
        raise MarketDataError(f"Failed to resample data: {str(e)}")

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