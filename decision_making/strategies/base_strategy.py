"""Base classes for trading strategies."""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class PositionType(Enum):
    """Type of trading position."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Position:
    """Trading position information."""
    symbol: str
    type: PositionType
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    is_closed: bool = False
    pnl: Optional[float] = None

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize trading strategy.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to strategy configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.required_indicators = ['open', 'high', 'low', 'close', 'volume']
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load strategy configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading strategy config: {str(e)}")
            return {}
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        prediction: float,
        uncertainty: Optional[float] = None
    ) -> Tuple[PositionType, float]:
        """
        Generate trading signal.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with technical indicators
        prediction : float
            Model's price prediction
        uncertainty : float, optional
            Model's prediction uncertainty
            
        Returns
        -------
        tuple
            (PositionType, confidence)
        """
        raise NotImplementedError
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data to validate
            
        Returns
        -------
        bool
            True if data is valid
        """
        required_columns = getattr(self, 'required_indicators', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
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
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['confidence'] = 0.0
        
        # Calculate technical signals
        tech_signals = self.calculate_technical_signals(data)
        
        # Combine signals for each timestamp
        for idx, row in data.iterrows():
            # Get prediction and uncertainty (placeholder values for base class)
            prediction = row['close'] * 1.01  # 1% increase as placeholder
            uncertainty = 0.1  # 10% uncertainty as placeholder
            
            # Generate signal
            signal_type, confidence = self.generate_signal(
                data.loc[:idx],  # Use data up to current point
                prediction,
                uncertainty
            )
            
            signals.loc[idx, 'signal'] = signal_type.value
            signals.loc[idx, 'confidence'] = confidence
        
        # Add technical indicators
        for col in tech_signals.columns:
            signals[col] = tech_signals[col]
        
        return signals
    
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
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate basic moving averages
        signals['sma_4'] = data['close'].rolling(window=4).mean()
        signals['sma_13'] = data['close'].rolling(window=13).mean()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        signals['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        signals['macd'] = exp1 - exp2
        signals['macd_signal'] = signals['macd'].ewm(span=9, adjust=False).mean()
        
        return signals 