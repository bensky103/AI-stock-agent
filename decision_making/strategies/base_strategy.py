"""Base classes for trading strategies."""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import yaml
import pandas as pd

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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

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