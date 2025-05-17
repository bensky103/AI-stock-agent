"""Technical indicators and feature engineering for stock data.

This module provides functions to calculate various technical indicators
and engineered features from stock price and volume data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    # Moving Averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    
    # RSI
    rsi_period: int = 14
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Volume
    volume_periods: List[int] = field(default_factory=lambda: [5, 10, 20])

def calculate_moving_averages(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Calculate Simple Moving Averages (SMA) for given periods.
    
    Args:
        df: DataFrame with 'close' prices
        periods: List of periods for SMA calculation
        
    Returns:
        DataFrame with added SMA columns
    """
    df = df.copy()
    for period in periods:
        df[f'sma_{period}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
    return df

def calculate_ema(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Calculate Exponential Moving Averages (EMA) for given periods.
    
    Args:
        df: DataFrame with 'close' prices
        periods: List of periods for EMA calculation
        
    Returns:
        DataFrame with added EMA columns
    """
    df = df.copy()
    for period in periods:
        df[f'ema_{period}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=period, min_periods=1).mean()
        )
    return df

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'close' prices
        period: RSI calculation period
        
    Returns:
        DataFrame with added RSI column
    """
    df = df.copy()
    
    def rsi(group):
        delta = group.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = df.groupby('symbol')['close'].transform(rsi)
    return df

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with 'close' prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with added MACD columns
    """
    df = df.copy()
    macd_rows = []
    for symbol, group in df.groupby('symbol'):
        exp1 = group['close'].ewm(span=fast, adjust=False).mean()
        exp2 = group['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        macd_rows.append(pd.DataFrame({
            'symbol': symbol,
            'date': group['date'],
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }))
    macd_df = pd.concat(macd_rows, ignore_index=True)
    df = pd.merge(df, macd_df, on=['symbol', 'date'])
    return df

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'close' prices
        period: Moving average period
        std: Number of standard deviations
        
    Returns:
        DataFrame with added Bollinger Bands columns
    """
    df = df.copy()
    
    def bollinger_bands(group):
        ma = group.rolling(window=period).mean()
        std_dev = group.rolling(window=period).std()
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        return pd.DataFrame({
            'date': group.index,
            'bb_middle': ma,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / ma
        })
    
    bb_rows = []
    for symbol, group in df.groupby('symbol'):
        bb = bollinger_bands(group['close'])
        bb['symbol'] = symbol
        bb['date'] = group['date'].values
        bb_rows.append(bb)
    bb_df = pd.concat(bb_rows, ignore_index=True)
    df = pd.merge(df, bb_df, on=['symbol', 'date'])
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC prices
        period: ATR calculation period
        
    Returns:
        DataFrame with added ATR column
    """
    df = df.copy()
    
    def atr(group):
        high = group['high']
        low = group['low']
        close = group['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    df['atr'] = df.groupby('symbol').apply(atr).reset_index(level=0, drop=True)
    return df

def calculate_volume_indicators(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Calculate volume-based indicators.
    
    Args:
        df: DataFrame with 'volume' column
        periods: List of periods for volume SMA
        
    Returns:
        DataFrame with added volume indicator columns
    """
    df = df.copy()
    
    # Volume SMA
    for period in periods:
        df[f'volume_sma_{period}'] = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Volume trend
    df['volume_trend'] = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window=5).mean().pct_change()
    )
    
    return df

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional price-based features.
    
    Args:
        df: DataFrame with OHLC prices
        
    Returns:
        DataFrame with added price feature columns
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    
    # Volatility
    df['volatility'] = df.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    # Price momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(periods=period)
        )
    
    # Price range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['price_range_ma'] = df.groupby('symbol')['price_range'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    
    return df

def add_all_indicators(data: pd.DataFrame, config: IndicatorConfig, verbose: bool = False) -> pd.DataFrame:
    """Add all technical indicators to the dataset."""
    
    if verbose:
        print(f"Before moving averages: {data.columns}")
    
    # Add all indicators...
    data = calculate_moving_averages(data, config.sma_periods)
    data = calculate_ema(data, config.ema_periods)
    data = calculate_rsi(data, config.rsi_period)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)
    data = calculate_atr(data, config.atr_period)
    data = calculate_volume_indicators(data, config.volume_periods)
    data = calculate_price_features(data)
    
    # Add trend column
    data['trend'] = (data['close'] > data['sma_20']).astype(int)
    
    if verbose:
        print(f"After all indicators: {data.columns}")
    
    return data