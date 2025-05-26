"""Test suite for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from decision_making.strategies.base_strategy import (
    TradingStrategy, PositionType, Position
)
from decision_making.strategies.ml_hybrid_strategy import MLHybridStrategy
from decision_making.strategies.position_manager import PositionManager

# Test data fixtures
@pytest.fixture
def sample_market_data():
    """Create sample market data with technical indicators."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 2, 100),
        'High': np.random.normal(102, 2, 100),
        'Low': np.random.normal(98, 2, 100),
        'Close': np.random.normal(100, 2, 100),
        'Volume': np.random.normal(1000000, 200000, 100),
    }, index=dates)
    
    # Add technical indicators
    data['RSI'] = np.random.uniform(0, 100, 100)
    data['MACD'] = np.random.normal(0, 1, 100)
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Market_Regime'] = np.random.uniform(0, 1, 100)
    
    return data

@pytest.fixture
def strategy_config():
    """Create sample strategy configuration."""
    config = {
        'strategy': {
            'name': 'ml_hybrid',
            'parameters': {
                'signal_threshold': 0.3,
                'weights': {
                    'ml': 0.4,
                    'technical': 0.4,
                    'regime': 0.2
                }
            }
        },
        'position_management': {
            'max_position_size': 0.1,
            'max_drawdown': 0.05,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'trailing_stop': True,
            'trailing_stop_pct': 0.01
        }
    }
    
    config_path = Path('tests/test_strategy_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

# Base Strategy Tests
class TestBaseStrategy:
    @pytest.mark.strategy
    def test_position_type_enum(self):
        """Test PositionType enumeration."""
        assert PositionType.LONG.value == 1
        assert PositionType.SHORT.value == -1
        assert PositionType.NEUTRAL.value == 0
    
    @pytest.mark.strategy
    def test_position_dataclass(self):
        """Test Position dataclass creation and attributes."""
        position = Position(
            symbol='AAPL',
            type=PositionType.LONG,
            entry_price=150.0,
            entry_time=pd.Timestamp('2024-01-01'),
            size=100,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        assert position.symbol == 'AAPL'
        assert position.type == PositionType.LONG
        assert position.entry_price == 150.0
        assert position.size == 100
        assert position.stop_loss == 145.0
        assert position.take_profit == 160.0

# ML Hybrid Strategy Tests
class TestMLHybridStrategy:
    @pytest.mark.strategy
    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization with config."""
        strategy = MLHybridStrategy(strategy_config)
        assert strategy.required_indicators == [
            'RSI', 'MACD', 'SMA_20', 'SMA_50',
            'Volume_MA', 'Market_Regime'
        ]
    
    @pytest.mark.strategy
    def test_data_validation(self, sample_market_data):
        """Test input data validation."""
        strategy = MLHybridStrategy()
        
        # Test with valid data
        assert strategy.validate_data(sample_market_data)
        
        # Test with missing columns
        invalid_data = sample_market_data.drop('RSI', axis=1)
        assert not strategy.validate_data(invalid_data)
    
    @pytest.mark.strategy
    def test_signal_generation(self, sample_market_data):
        """Test trading signal generation."""
        strategy = MLHybridStrategy()
        
        # Test with bullish prediction
        signal, confidence = strategy.generate_signal(
            sample_market_data,
            prediction=sample_market_data['Close'].iloc[-1] * 1.05,  # 5% higher
            uncertainty=0.1
        )
        assert signal in [PositionType.LONG, PositionType.SHORT, PositionType.NEUTRAL]
        assert 0 <= confidence <= 1
        
        # Test with bearish prediction
        signal, confidence = strategy.generate_signal(
            sample_market_data,
            prediction=sample_market_data['Close'].iloc[-1] * 0.95,  # 5% lower
            uncertainty=0.1
        )
        assert signal in [PositionType.LONG, PositionType.SHORT, PositionType.NEUTRAL]
        assert 0 <= confidence <= 1
    
    @pytest.mark.strategy
    def test_technical_signals(self, sample_market_data):
        """Test technical indicator signal calculation."""
        strategy = MLHybridStrategy()
        signals = strategy._calculate_technical_signals(sample_market_data.iloc[-1])
        
        assert isinstance(signals, dict)
        assert all(k in signals for k in ['rsi', 'macd', 'ma', 'volume'])
        assert all(-1 <= v <= 1 for v in signals.values())
    
    @pytest.mark.strategy
    def test_signal_combination(self, sample_market_data):
        """Test signal combination logic."""
        strategy = MLHybridStrategy()
        
        # Test with different market regimes
        for regime in [0.2, 0.5, 0.8]:  # Low, neutral, high volatility
            signal = strategy._combine_signals(
                ml_signal=0.5,
                tech_signals={'rsi': 0.3, 'macd': 0.4, 'ma': 0.2, 'volume': 0.1},
                regime=regime
            )
            assert -1 <= signal <= 1

# Position Manager Tests
class TestPositionManager:
    @pytest.mark.position
    def test_position_manager_initialization(self, strategy_config):
        """Test position manager initialization with config."""
        manager = PositionManager(strategy_config)
        assert manager.max_position_size == 0.1
        assert manager.max_drawdown == 0.05
        assert manager.stop_loss_pct == 0.02
        assert manager.take_profit_pct == 0.04
    
    @pytest.mark.position
    def test_position_sizing(self):
        """Test position size calculation."""
        manager = PositionManager()
        
        # Test with different scenarios
        size = manager.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=0.8,
            volatility=0.2,
            market_regime=0.5
        )
        assert size > 0
        
        # Test with high volatility
        size_high_vol = manager.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=0.8,
            volatility=0.5,
            market_regime=0.8
        )
        assert size_high_vol < size  # Should be smaller due to high volatility
    
    @pytest.mark.position
    def test_position_update(self):
        """Test position creation and update."""
        manager = PositionManager()
        timestamp = datetime.now()
        
        # Create new position
        manager.update_position(
            symbol='AAPL',
            position_type=PositionType.LONG,
            entry_price=150.0,
            size=100,
            current_price=155.0,
            timestamp=timestamp
        )
        
        assert 'AAPL' in manager.positions
        position = manager.positions['AAPL']
        assert position.entry_price == 150.0
        assert position.size == 100
        assert position.stop_loss is not None
        assert position.take_profit is not None
    
    @pytest.mark.position
    def test_exit_signals(self):
        """Test position exit signal generation."""
        manager = PositionManager()
        timestamp = datetime.now()
        
        # Create test position
        manager.update_position(
            symbol='AAPL',
            position_type=PositionType.LONG,
            entry_price=150.0,
            size=100,
            current_price=155.0,
            timestamp=timestamp
        )
        
        # Test stop loss
        should_exit, exit_type = manager.check_exit_signals(
            'AAPL',
            current_price=145.0,  # Below stop loss
            timestamp=timestamp + timedelta(hours=1)
        )
        assert should_exit
        assert exit_type == PositionType.NEUTRAL
        
        # Test take profit
        should_exit, exit_type = manager.check_exit_signals(
            'AAPL',
            current_price=160.0,  # Above take profit
            timestamp=timestamp + timedelta(hours=1)
        )
        assert should_exit
        assert exit_type == PositionType.NEUTRAL
    
    @pytest.mark.risk
    def test_portfolio_tracking(self):
        """Test portfolio value and drawdown tracking."""
        manager = PositionManager()
        timestamp = datetime.now()
        
        # Create test positions
        manager.update_position(
            symbol='AAPL',
            position_type=PositionType.LONG,
            entry_price=150.0,
            size=100,
            current_price=155.0,
            timestamp=timestamp
        )
        
        manager.update_position(
            symbol='MSFT',
            position_type=PositionType.SHORT,
            entry_price=300.0,
            size=50,
            current_price=290.0,
            timestamp=timestamp
        )
        
        # Update portfolio value
        current_prices = {'AAPL': 155.0, 'MSFT': 290.0}
        manager.update_portfolio_value(current_prices)
        
        assert manager.portfolio_value > 100000.0  # Should have made profit
        assert manager.current_drawdown >= 0  # Should be non-negative
    
    @pytest.mark.position
    def test_position_summary(self):
        """Test position summary generation."""
        manager = PositionManager()
        timestamp = datetime.now()
        
        # Create test position
        manager.update_position(
            symbol='AAPL',
            position_type=PositionType.LONG,
            entry_price=150.0,
            size=100,
            current_price=155.0,
            timestamp=timestamp
        )
        
        summary = manager.get_position_summary()
        assert 'AAPL' in summary
        assert all(k in summary['AAPL'] for k in [
            'type', 'size', 'entry_price',
            'pnl', 'stop_loss', 'take_profit'
        ])
        
        # Verify summary values
        position_summary = summary['AAPL']
        assert position_summary['type'] == 'LONG'
        assert position_summary['size'] == 100
        assert position_summary['entry_price'] == 150.0
        assert position_summary['stop_loss'] == 147.0  # 150 * (1 - 0.02)
        assert position_summary['take_profit'] == 156.0  # 150 * (1 + 0.04)
    
    @pytest.mark.risk
    def test_risk_limits(self, test_symbols, test_timeframe):
        """Test risk management limits."""
        manager = PositionManager()
        timestamp = datetime.now()
        
        # Try to exceed max position size
        size = manager.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=1.0,  # Maximum signal strength
            volatility=0.1,      # Low volatility
            market_regime=0.3    # Low volatility regime
        )
        assert size <= manager.max_position_size * manager.portfolio_value / 150.0
        
        # Try to exceed max drawdown
        manager.update_position(
            symbol='AAPL',
            position_type=PositionType.LONG,
            entry_price=150.0,
            size=1000,  # Large position
            current_price=140.0,  # Significant loss
            timestamp=timestamp
        )
        
        current_prices = {'AAPL': 140.0}
        manager.update_portfolio_value(current_prices)
        assert manager.current_drawdown <= manager.max_drawdown 