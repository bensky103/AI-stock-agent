"""ML hybrid trading strategy implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

from .base_strategy import TradingStrategy, PositionType

# Configure logging
logger = logging.getLogger(__name__)

class MLHybridStrategy(TradingStrategy):
    """
    Hybrid strategy combining ML predictions with traditional indicators.
    
    This strategy uses:
    1. ML model predictions and uncertainty
    2. Technical indicators (RSI, MACD, etc.)
    3. Market regime detection
    4. Volume analysis
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ML hybrid strategy.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to strategy configuration file
        """
        super().__init__(config_path)
        self.required_indicators = [
            'RSI', 'MACD', 'SMA_20', 'SMA_50',
            'Volume_MA', 'Market_Regime'
        ]
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        prediction: float,
        uncertainty: Optional[float] = None
    ) -> Tuple[PositionType, float]:
        """
        Generate trading signal using hybrid approach.
        
        Strategy rules:
        1. ML Prediction Signal:
           - Strong buy if prediction > current_price * (1 + threshold)
           - Strong sell if prediction < current_price * (1 - threshold)
           - Weight based on uncertainty
        
        2. Technical Signal:
           - RSI overbought/oversold
           - MACD crossover
           - Moving average crossover
           - Volume confirmation
        
        3. Market Regime:
           - Adjust position size based on regime
           - More conservative in high volatility
        """
        # Validate input data
        if not self.validate_data(data):
            logger.error("Invalid input data")
            return PositionType.NEUTRAL, 0.0
        
        try:
            # Get latest data point
            current = data.iloc[-1]
            
            # Handle case sensitivity for price column
            price_col = 'close' if 'close' in current else 'Close'
            current_price = current[price_col]
            
            # Calculate ML signal
            price_change = (prediction - current_price) / current_price
            ml_signal = np.clip(price_change / 0.02, -1, 1)  # Normalize to [-1, 1]
            
            # Adjust ML signal by uncertainty
            if uncertainty is not None:
                confidence = 1 / (1 + uncertainty)
                ml_signal *= confidence
            
            # Calculate technical signals
            tech_signals = self._calculate_technical_signals(current)
            
            # Get market regime
            regime = current.get('Market_Regime', 0.5)  # Default to neutral
            
            # Combine signals
            final_signal = self._combine_signals(ml_signal, tech_signals, regime)
            
            # Convert to position type
            if final_signal > 0.3:
                position_type = PositionType.LONG
            elif final_signal < -0.3:
                position_type = PositionType.SHORT
            else:
                position_type = PositionType.NEUTRAL
            
            return position_type, abs(final_signal)
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return PositionType.NEUTRAL, 0.0
    
    def _calculate_technical_signals(self, data: pd.Series) -> Dict[str, float]:
        """Calculate technical indicator signals."""
        # Initialize all signals with default value of 0
        signals = {
            'rsi': 0,
            'macd': 0,
            'ma': 0,
            'volume': 0
        }
        
        try:
            # Handle case sensitivity for columns
            price_col = 'close' if 'close' in data else 'Close'
            volume_col = 'volume' if 'volume' in data else 'Volume'
            
            # RSI signals
            if 'RSI' in data:
                rsi_value = data['RSI']
                if isinstance(rsi_value, pd.Series):
                    rsi_value = rsi_value.item()
                if rsi_value > 70:
                    signals['rsi'] = -1  # Overbought
                elif rsi_value < 30:
                    signals['rsi'] = 1   # Oversold
            
            # MACD signals - using signal line instead of shift
            if all(x in data for x in ['MACD', 'MACD_Signal']):
                macd_value = data['MACD']
                macd_signal_value = data['MACD_Signal']
                if isinstance(macd_value, pd.Series):
                    macd_value = macd_value.item()
                if isinstance(macd_signal_value, pd.Series):
                    macd_signal_value = macd_signal_value.item()
                
                if macd_value > macd_signal_value:
                    signals['macd'] = 1  # Bullish
                elif macd_value < macd_signal_value:
                    signals['macd'] = -1  # Bearish
            
            # Moving average signals
            if all(x in data for x in ['SMA_20', 'SMA_50']):
                sma20_value = data['SMA_20']
                sma50_value = data['SMA_50']
                if isinstance(sma20_value, pd.Series):
                    sma20_value = sma20_value.item()
                if isinstance(sma50_value, pd.Series):
                    sma50_value = sma50_value.item()
                
                if sma20_value > sma50_value:
                    signals['ma'] = 1  # Bullish
                elif sma20_value < sma50_value:
                    signals['ma'] = -1  # Bearish
            
            # Volume signals
            if 'Volume_MA' in data and volume_col in data:
                volume_value = data[volume_col]
                volume_ma_value = data['Volume_MA']
                if isinstance(volume_value, pd.Series):
                    volume_value = volume_value.item()
                if isinstance(volume_ma_value, pd.Series):
                    volume_ma_value = volume_ma_value.item()
                
                if volume_value > volume_ma_value * 1.5:
                    signals['volume'] = 1  # High volume
                    
        except Exception as e:
            logger.error(f"Error calculating technical signals: {str(e)}")
            # Return default signals on error
            return signals
        
        return signals
    
    def _combine_signals(
        self,
        ml_signal: float,
        tech_signals: Dict[str, float],
        regime: float
    ) -> float:
        """Combine different signals into final trading signal."""
        try:
            # Ensure regime is a scalar value
            if isinstance(regime, pd.Series):
                regime = regime.item()
                
            # Weights for different components
            weights = {
                'ml': 0.4,
                'technical': 0.4,
                'regime': 0.2
            }
            
            # Calculate technical signal average
            tech_signal = np.mean(list(tech_signals.values())) if tech_signals else 0
            
            # Adjust weights based on market regime
            if regime > 0.7:  # High volatility regime
                weights['ml'] *= 0.8
                weights['technical'] *= 0.8
                weights['regime'] *= 1.4
            elif regime < 0.3:  # Low volatility regime
                weights['ml'] *= 1.2
                weights['technical'] *= 1.2
                weights['regime'] *= 0.6
            
            # Combine signals
            final_signal = (
                weights['ml'] * ml_signal +
                weights['technical'] * tech_signal +
                weights['regime'] * (regime - 0.5) * 2  # Center regime around 0
            )
            
            return np.clip(final_signal, -1, 1)
            
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return 0.0

    def calculate_technical_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and signals.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with technical indicators
        """
        # Handle empty DataFrame
        if data.empty:
            return pd.DataFrame(columns=['RSI', 'MACD', 'SMA_20', 'SMA_50', 'Volume_MA', 'Market_Regime'])
        
        # Create a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Calculate RSI (14 periods)
        delta = signals['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = signals['close'].ewm(span=12, adjust=False).mean()
        exp2 = signals['close'].ewm(span=26, adjust=False).mean()
        signals['MACD'] = exp1 - exp2
        signals['MACD_Signal'] = signals['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate moving averages
        signals['SMA_20'] = signals['close'].rolling(window=20).mean()
        signals['SMA_50'] = signals['close'].rolling(window=50).mean()
        
        # Calculate volume moving average
        signals['Volume_MA'] = signals['volume'].rolling(window=20).mean()
        
        # Calculate market regime (simplified version using volatility)
        returns = signals['close'].pct_change()
        signals['Market_Regime'] = returns.rolling(window=20).std()
        
        # Add test-required columns
        signals['sma_4'] = signals['close'].rolling(window=4).mean()
        signals['rsi_14'] = signals['RSI']
        signals['macd'] = signals['MACD']
        
        return signals

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for a series of data points.
    
        Parameters
        ----------
        data : pd.DataFrame
            Market data with technical indicators
    
        Returns
        -------
        pd.DataFrame
            DataFrame with signals and additional information
        """
        # Handle empty DataFrame
        if data.empty:
            raise ValueError("Invalid data format: Empty DataFrame")
            
        # Calculate technical indicators first
        data_with_indicators = self.calculate_technical_signals(data)
        
        # Now validate the data with indicators
        if not self.validate_data(data_with_indicators):
            raise ValueError("Invalid data format")
        
        try:
            # Initialize signals DataFrame with all original data
            signals = data_with_indicators.copy()
            signals['signal'] = 0  # Default to neutral
            signals['signal_strength'] = 0.0
            
            # Generate signals for each data point
            for idx, row in data_with_indicators.iterrows():
                # Get ML prediction (placeholder for now)
                price_col = 'close' if 'close' in row else 'Close'
                prediction = row[price_col] * 1.02  # 2% higher than current price
                uncertainty = 0.1  # Placeholder uncertainty
                
                # Generate signal
                position_type, signal_strength = self.generate_signal(
                    data_with_indicators.loc[:idx],
                    prediction,
                    uncertainty
                )
                
                # Convert position type to signal (-1, 0, 1)
                signals.loc[idx, 'signal'] = position_type.value
                signals.loc[idx, 'signal_strength'] = signal_strength
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise ValueError("Error generating signals") 