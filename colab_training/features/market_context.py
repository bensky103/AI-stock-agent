"""Market context analysis and trading signals.

This module provides functions for analyzing market context and generating
trading signals based on technical indicators and market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Market regime classification."""
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    SIDEWAYS = 'sideways'
    VOLATILE = 'volatile'

@dataclass
class SignalConfig:
    """Configuration for trading signals."""
    # RSI thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # MACD signal
    macd_threshold: float = 0.0
    
    # Bollinger Bands
    bb_upper_threshold: float = 0.95  # Price relative to upper band
    bb_lower_threshold: float = 0.05  # Price relative to lower band
    
    # Volume
    volume_threshold: float = 1.5  # Volume ratio threshold
    
    # Trend
    trend_period: int = 20
    trend_threshold: float = 0.02  # Minimum price change for trend

def detect_market_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Detect market regime based on price action and volatility.
    
    Args:
        df: DataFrame with price data and indicators
        window: Window size for regime detection
        
    Returns:
        DataFrame with market regime classification
    """
    df = df.copy()
    
    def classify_regime(x):
        volatility = x['volatility']
        returns = x['returns']
        # Ensure returns and volatility are Series
        last_vol = volatility.iloc[-1] if hasattr(volatility, 'iloc') and len(volatility) > 0 else np.nan
        last_ret = returns.iloc[-1] if hasattr(returns, 'iloc') and len(returns) > 0 else np.nan
        vol_threshold = volatility.quantile(0.8) if hasattr(volatility, 'quantile') else np.nan
        if pd.isna(last_vol) or pd.isna(last_ret):
            return 'UNKNOWN'
        if last_vol > vol_threshold:
            return 'VOLATILE'
        elif last_ret > 0:
            return 'BULLISH'
        elif last_ret < 0:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    df['market_regime'] = df.groupby('symbol').apply(
        lambda x: classify_regime(x)
    ).reset_index(level=0, drop=True)
    
    return df

def generate_trading_signals(df: pd.DataFrame, config: Optional[SignalConfig] = None) -> pd.DataFrame:
    """Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with price data and indicators
        config: Signal configuration
        
    Returns:
        DataFrame with trading signals
    """
    if config is None:
        config = SignalConfig()
    
    df = df.copy()
    
    # RSI signals
    df['rsi_signal'] = 0
    df.loc[df['rsi'] < config.rsi_oversold, 'rsi_signal'] = 1  # Buy signal
    df.loc[df['rsi'] > config.rsi_overbought, 'rsi_signal'] = -1  # Sell signal
    
    # MACD signals
    df['macd_signal'] = 0
    df.loc[df['macd'] > df['macd_signal'], 'macd_signal'] = 1  # Bullish
    df.loc[df['macd'] < df['macd_signal'], 'macd_signal'] = -1  # Bearish
    
    # Bollinger Bands signals
    df['bb_signal'] = 0
    bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df.loc[bb_position < config.bb_lower_threshold, 'bb_signal'] = 1  # Buy signal
    df.loc[bb_position > config.bb_upper_threshold, 'bb_signal'] = -1  # Sell signal
    
    # Volume signals
    df['volume_signal'] = 0
    df.loc[df['volume_ratio'] > config.volume_threshold, 'volume_signal'] = 1  # High volume
    
    # Trend signals
    df['trend_signal'] = 0
    df['trend'] = df.groupby('symbol')['close'].transform(
        lambda x: x.pct_change(config.trend_period)
    )
    df.loc[df['trend'] > config.trend_threshold, 'trend_signal'] = 1  # Uptrend
    df.loc[df['trend'] < -config.trend_threshold, 'trend_signal'] = -1  # Downtrend
    
    # Combined signal
    df['combined_signal'] = (
        df['rsi_signal'] +
        df['macd_signal'] +
        df['bb_signal'] +
        df['volume_signal'] +
        df['trend_signal']
    )
    
    # Normalize combined signal
    df['combined_signal'] = df['combined_signal'] / 5.0
    
    return df

def analyze_market_context(df: pd.DataFrame) -> Dict:
    """Analyze current market context.
    
    Args:
        df: DataFrame with price data and indicators
        
    Returns:
        Dictionary with market context analysis
    """
    # Get latest data for each symbol
    latest = df.groupby('symbol').last()
    
    # Market regime distribution
    regime_dist = latest['market_regime'].value_counts(normalize=True)
    
    # Average indicators
    avg_indicators = {
        'rsi': latest['rsi'].mean(),
        'volatility': latest['volatility'].mean(),
        'volume_ratio': latest['volume_ratio'].mean(),
        'bb_width': latest['bb_width'].mean()
    }
    
    # Signal distribution
    signal_dist = {
        'bullish': (latest['combined_signal'] > 0.2).mean(),
        'bearish': (latest['combined_signal'] < -0.2).mean(),
        'neutral': ((latest['combined_signal'] >= -0.2) & 
                   (latest['combined_signal'] <= 0.2)).mean()
    }
    
    return {
        'regime_distribution': regime_dist.to_dict(),
        'average_indicators': avg_indicators,
        'signal_distribution': signal_dist
    }

def get_trading_recommendations(df: pd.DataFrame, 
                              top_n: int = 5) -> pd.DataFrame:
    """Get trading recommendations based on signals.
    
    Args:
        df: DataFrame with price data and signals
        top_n: Number of top recommendations to return
        
    Returns:
        DataFrame with trading recommendations
    """
    # Get latest data for each symbol
    latest = df.groupby('symbol').last()
    
    # Calculate recommendation score
    latest['recommendation_score'] = (
        latest['combined_signal'] * 
        (1 + latest['volume_ratio']) * 
        (1 - latest['volatility'])
    )
    
    # Get top recommendations
    recommendations = latest.nlargest(top_n, 'recommendation_score')
    
    # Add recommendation reasons
    recommendations['reasons'] = recommendations.apply(
        lambda x: _get_recommendation_reasons(x), axis=1
    )
    
    return recommendations[['recommendation_score', 'reasons', 
                          'combined_signal', 'market_regime']]

def _get_recommendation_reasons(row: pd.Series) -> List[str]:
    """Get reasons for trading recommendation.
    
    Args:
        row: Series with indicator values
        
    Returns:
        List of reasons for recommendation
    """
    reasons = []
    
    # RSI
    if row['rsi'] < 30:
        reasons.append("Oversold (RSI)")
    elif row['rsi'] > 70:
        reasons.append("Overbought (RSI)")
    
    # MACD
    if row['macd'] > row['macd_signal']:
        reasons.append("Bullish MACD")
    elif row['macd'] < row['macd_signal']:
        reasons.append("Bearish MACD")
    
    # Bollinger Bands
    bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
    if bb_position < 0.05:
        reasons.append("Price near lower Bollinger Band")
    elif bb_position > 0.95:
        reasons.append("Price near upper Bollinger Band")
    
    # Volume
    if row['volume_ratio'] > 1.5:
        reasons.append("High volume")
    
    # Trend
    if row['trend_signal'] > 0:
        reasons.append("Uptrend")
    elif row['trend_signal'] < 0:
        reasons.append("Downtrend")
    
    return reasons 