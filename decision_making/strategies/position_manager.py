"""Position management and risk control implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .base_strategy import PositionType, Position

# Configure logging
logger = logging.getLogger(__name__)

class PositionManager:
    """
    Manages trading positions and risk control.
    
    Features:
    1. Position sizing based on volatility and confidence
    2. Dynamic stop-loss and take-profit levels
    3. Position tracking and P&L calculation
    4. Risk management rules
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize position manager.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to position management configuration file
        """
        # Initialize attributes
        self.max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
        self.max_drawdown: float = 0.05      # Maximum drawdown allowed
        self.stop_loss_pct: float = 0.02     # Default stop loss percentage
        self.take_profit_pct: float = 0.04   # Default take profit percentage
        self.trailing_stop: bool = True       # Use trailing stop loss
        self.trailing_stop_pct: float = 0.01  # Trailing stop percentage
        self.portfolio_value: float = 100000.0  # Initial portfolio value
        self.current_drawdown: float = 0.0
        self.positions: Dict[str, Position] = {}
        
        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
    
    def _load_config(self, config_path: Path) -> None:
        """Load position management configuration."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update parameters from config
            for key, value in config.get('position_management', {}).items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            logger.error(f"Error loading position management config: {str(e)}")
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
        volatility: float,
        market_regime: float
    ) -> float:
        """
        Calculate position size based on various factors.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        price : float
            Current price
        signal_strength : float
            Signal strength from strategy (0 to 1)
        volatility : float
            Current volatility
        market_regime : float
            Market regime indicator (0 to 1)
        
        Returns
        -------
        float
            Position size in number of shares
        """
        try:
            # Ensure all inputs are scalar values
            if isinstance(price, pd.Series):
                price = price.item()
            if isinstance(signal_strength, pd.Series):
                signal_strength = signal_strength.item()
            if isinstance(volatility, pd.Series):
                volatility = volatility.item()
            if isinstance(market_regime, pd.Series):
                market_regime = market_regime.item()
                
            # Base position size from signal strength
            base_size = self.max_position_size * signal_strength
            
            # Adjust for volatility
            vol_factor = 1 / (1 + volatility)
            position_size = base_size * vol_factor
            
            # Adjust for market regime
            if market_regime > 0.7:  # High volatility regime
                position_size *= 0.7
            elif market_regime < 0.3:  # Low volatility regime
                position_size *= 1.2
            
            # Calculate number of shares
            max_shares = (self.portfolio_value * position_size) / price
            
            # Round to appropriate lot size
            lot_size = self._get_lot_size(symbol)
            shares = np.floor(max_shares / lot_size) * lot_size
            
            return max(0, shares)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def update_position(
        self,
        symbol: str,
        position_type: PositionType,
        entry_price: float,
        size: float,
        current_price: float,
        timestamp: datetime
    ) -> None:
        """
        Update or create a new position.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        position_type : PositionType
            Type of position (LONG/SHORT)
        entry_price : float
            Entry price
        size : float
            Position size in shares
        current_price : float
            Current market price
        timestamp : datetime
            Current timestamp
        """
        try:
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_levels(
                position_type, entry_price, current_price
            )
            
            # Create or update position
            self.positions[symbol] = Position(
                symbol=symbol,
                type=position_type,
                entry_price=entry_price,
                size=size,
                entry_time=pd.Timestamp(timestamp),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"Updated position for {symbol}: {self.positions[symbol]}")
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
    
    def check_exit_signals(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> Tuple[bool, Optional[PositionType]]:
        """
        Check if position should be exited.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        current_price : float
            Current market price
        timestamp : datetime
            Current timestamp
        
        Returns
        -------
        Tuple[bool, Optional[PositionType]]
            (Should exit, Exit type if applicable)
        """
        if symbol not in self.positions:
            return False, None
        
        try:
            position = self.positions[symbol]
            
            # Update trailing stop if enabled
            if self.trailing_stop:
                self._update_trailing_stop(position, current_price)
            
            # Check stop loss
            if self._check_stop_loss(position, current_price):
                return True, PositionType.NEUTRAL
            
            # Check take profit
            if self._check_take_profit(position, current_price):
                return True, PositionType.NEUTRAL
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking exit signals: {str(e)}")
            return False, None
    
    def _calculate_levels(
        self,
        position_type: PositionType,
        entry_price: float,
        current_price: float
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if position_type == PositionType.LONG:
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        return stop_loss, take_profit
    
    def _update_trailing_stop(self, position: Position, current_price: float) -> None:
        """Update trailing stop loss level."""
        if position.type == PositionType.LONG:
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:  # SHORT
            new_stop = current_price * (1 + self.trailing_stop_pct)
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if position.type == PositionType.LONG:
            return current_price <= position.stop_loss
        else:  # SHORT
            return current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit has been hit."""
        if position.type == PositionType.LONG:
            return current_price >= position.take_profit
        else:  # SHORT
            return current_price <= position.take_profit
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get appropriate lot size for symbol."""
        # Default to 1, override for specific symbols if needed
        return 1
    
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate current P&L for a position."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        if position.type == PositionType.LONG:
            return (current_price - position.entry_price) * position.size
        else:  # SHORT
            return (position.entry_price - current_price) * position.size
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> None:
        """Update portfolio value and drawdown."""
        try:
            # Calculate total P&L
            total_pnl = sum(
                self.calculate_pnl(symbol, price)
                for symbol, price in current_prices.items()
            )
            
            # Update portfolio value
            new_value = self.portfolio_value + total_pnl
            
            # Calculate drawdown (only positive values)
            self.current_drawdown = max(0, (self.portfolio_value - new_value) / self.portfolio_value)
            
            # Check if max drawdown exceeded
            if self.current_drawdown > self.max_drawdown:
                logger.warning(f"Max drawdown exceeded: {self.current_drawdown:.2%}")
                # Close all positions to prevent further drawdown
                for symbol in list(self.positions.keys()):
                    logger.info(f"Closing position for {symbol} due to max drawdown exceeded")
                    del self.positions[symbol]
                # Reset drawdown after closing positions
                self.current_drawdown = 0.0
                self.portfolio_value = new_value
            else:
                self.portfolio_value = new_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {str(e)}")
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all current positions."""
        return {
            symbol: {
                'type': position.type.name,
                'size': position.size,
                'entry_price': position.entry_price,
                'pnl': self.calculate_pnl(symbol, position.entry_price),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
            for symbol, position in self.positions.items()
        }
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        position_type: Union[str, PositionType]
    ) -> Position:
        """
        Open a new trading position.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        entry_price : float
            Entry price
        size : float
            Position size in shares
        position_type : Union[str, PositionType]
            Type of position ('long'/'short' or PositionType)
            
        Returns
        -------
        Position
            Created position object
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        try:
            # Ensure inputs are scalar values
            if isinstance(entry_price, pd.Series):
                entry_price = entry_price.item()
            if isinstance(size, pd.Series):
                size = size.item()
                
            # Validate parameters
            if entry_price <= 0:
                raise ValueError("Entry price must be positive")
            if size <= 0:
                raise ValueError("Position size must be positive")
            
            # Convert string position type to enum
            if isinstance(position_type, str):
                position_type = position_type.lower()
                if position_type == 'long':
                    position_type = PositionType.LONG
                elif position_type == 'short':
                    position_type = PositionType.SHORT
                else:
                    raise ValueError(f"Invalid position type: {position_type}. Must be 'long' or 'short'")
            
            if not isinstance(position_type, PositionType):
                raise ValueError(f"Invalid position type: {position_type}. Must be PositionType enum")
            
            # Create position
            position = Position(
                symbol=symbol,
                type=position_type,
                entry_price=entry_price,
                size=size,
                entry_time=pd.Timestamp.now(),
                stop_loss=self._calculate_levels(position_type, entry_price, entry_price)[0],
                take_profit=self._calculate_levels(position_type, entry_price, entry_price)[1]
            )
            
            # Store position
            self.positions[symbol] = position
            
            logger.info(f"Opened {position_type.name} position for {symbol}: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            raise
    
    def close_position(
        self,
        position_id: str,
        exit_price: float
    ) -> Position:
        """
        Close an existing position.
        
        Parameters
        ----------
        position_id : str
            Position identifier (symbol)
        exit_price : float
            Exit price
            
        Returns
        -------
        Position
            Closed position object
            
        Raises
        ------
        ValueError
            If position doesn't exist or parameters are invalid
        """
        try:
            if position_id not in self.positions:
                raise ValueError(f"Position {position_id} not found")
            if exit_price <= 0:
                raise ValueError("Exit price must be positive")
            
            position = self.positions[position_id]
            position.exit_price = exit_price
            position.exit_time = pd.Timestamp.now()
            position.is_closed = True
            
            # Calculate final P&L
            position.pnl = self.calculate_pnl(position_id, exit_price)
            
            # Remove from active positions
            del self.positions[position_id]
            
            logger.info(f"Closed position for {position_id}: {position}")
            return position
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise
    
    def execute_trade(
        self,
        symbol: str,
        signal: int,
        price: float,
        timestamp: datetime
    ) -> Optional[Position]:
        """
        Execute a trade based on signal.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        signal : int
            Trading signal (-1 for short, 0 for neutral, 1 for long)
        price : float
            Current price
        timestamp : datetime
            Current timestamp
            
        Returns
        -------
        Optional[Position]
            Created position if trade executed, None otherwise
        """
        try:
            # Ensure inputs are scalar values
            if isinstance(signal, pd.Series):
                signal = signal.item()
            if isinstance(price, pd.Series):
                price = price.item()
                
            # Convert signal to position type
            if signal == 0:
                return None
            
            position_type = PositionType.LONG if signal > 0 else PositionType.SHORT
            
            # Check if we already have a position
            if symbol in self.positions:
                current_position = self.positions[symbol]
                
                # If signal is opposite to current position, close it
                if current_position.type != position_type:
                    self.close_position(symbol, price)
                else:
                    # Update existing position
                    self.update_position(
                        symbol=symbol,
                        position_type=position_type,
                        entry_price=current_position.entry_price,
                        size=current_position.size,
                        current_price=price,
                        timestamp=timestamp
                    )
                    return current_position
            
            # Calculate position size (simplified for base implementation)
            size = self.calculate_position_size(
                symbol=symbol,
                price=price,
                signal_strength=abs(signal),
                volatility=0.1,  # Placeholder
                market_regime=0.5  # Placeholder
            )
            
            # Open new position
            return self.open_position(
                symbol=symbol,
                entry_price=price,
                size=size,
                position_type=position_type
            )
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None 