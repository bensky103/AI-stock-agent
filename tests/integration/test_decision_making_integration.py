"""Integration tests for decision making module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decision_making.strategies.base_strategy import PositionType
from decision_making.strategies.ml_hybrid_strategy import MLHybridStrategy
from decision_making.strategies.position_manager import PositionManager
from data_input.market_feed import MarketFeed
from pathlib import Path
import pytz

class TestDecisionMakingIntegration:
    @pytest.fixture
    def setup_decision_components(self, test_config):
        """Setup decision-making components"""
        # Create a temporary config file for MarketFeed and TradingStrategy
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        # Initialize components with config path and weekly data
        market_feed = MarketFeed(config_path=config_path, default_interval='1W')
        strategy = MLHybridStrategy(config_path=Path(config_path))
        position_manager = PositionManager(config_path=Path(config_path))
        
        return {
            'market_feed': market_feed,
            'strategy': strategy,
            'position_manager': position_manager,
            'config_path': config_path  # Store path for cleanup
        }
    
    def teardown_method(self, method):
        """Clean up temporary files after each test"""
        if hasattr(self, 'config_path'):
            import os
            try:
                os.unlink(self.config_path)
            except Exception as e:
                print(f"Error cleaning up config file: {e}")
    
    def test_strategy_flow(self, setup_decision_components):
        """Test the complete strategy decision flow"""
        components = setup_decision_components
        
        # 1. Get market data - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert len(market_data) >= 8  # Ensure we have enough weekly data points
        
        # 2. Generate trading signals
        signals = components['strategy'].generate_signals(market_data)
        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty
        assert 'signal' in signals.columns
        
        # 3. Execute trades based on signals
        for timestamp, row in signals.iterrows():
            if row['signal'].item() != 0:  # If there's a signal
                position = components['position_manager'].execute_trade(
                    symbol='AAPL',
                    signal=row['signal'].item(),
                    price=row['close'],
                    timestamp=timestamp
                )
                assert position is not None
                assert position.symbol == 'AAPL'
                assert position.entry_price == row['close']
    
    def test_position_management(self, setup_decision_components):
        """Test position management functionality"""
        components = setup_decision_components
        
        # 1. Open a position
        position = components['position_manager'].open_position(
            symbol='AAPL',
            entry_price=150.0,
            size=100,
            position_type='long'
        )
        assert position is not None
        assert position.symbol == 'AAPL'
        assert position.entry_price == 150.0
        assert position.size == 100
        assert position.type == PositionType.LONG
        
        # 2. Update position
        components['position_manager'].update_position(
            symbol='AAPL',
            position_type=position.type,
            entry_price=position.entry_price,
            size=position.size,
            current_price=155.0,
            timestamp=datetime.now()
        )
        
        # 3. Close position
        closed_position = components['position_manager'].close_position(
            position_id=position.symbol,  # Use symbol as position_id
            exit_price=160.0
        )
        assert closed_position.is_closed
        assert closed_position.exit_price == 160.0
        assert closed_position.pnl == 1000.0  # (160 - 150) * 100
    
    def test_strategy_components(self, setup_decision_components):
        """Test individual strategy components"""
        components = setup_decision_components
        
        # Get market data - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        # Test technical analysis
        technical_signals = components['strategy'].calculate_technical_signals(market_data)
        assert isinstance(technical_signals, pd.DataFrame)
        assert not technical_signals.empty
        assert all(col in technical_signals.columns for col in ['sma_4', 'rsi_14', 'macd'])  # Updated for weekly data
        
        # Test signal generation
        signals = components['strategy'].generate_signals(market_data)
        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty
        assert 'signal' in signals.columns
        assert all(signal in [-1, 0, 1] for signal in signals['signal'])
    
    def test_error_handling(self, setup_decision_components):
        """Test error handling in decision making"""
        components = setup_decision_components
        
        # Test invalid market data
        with pytest.raises(ValueError, match="Invalid data format"):
            components['strategy'].generate_signals(pd.DataFrame())
        
        # Test invalid position parameters
        with pytest.raises(ValueError, match="Entry price must be positive"):
            components['position_manager'].open_position(
                symbol='AAPL',
                entry_price=-150.0,  # Invalid price
                size=100,
                position_type='long'
            )
        
        # Test invalid position type
        with pytest.raises(ValueError, match="Invalid position type"):
            components['position_manager'].open_position(
                symbol='AAPL',
                entry_price=150.0,
                size=100,
                position_type='invalid_type'  # Clearly invalid type
            )
    
    def test_data_consistency(self, setup_decision_components):
        """Test data consistency across decision making components"""
        components = setup_decision_components
        
        # Get market data - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        # Generate signals
        signals = components['strategy'].generate_signals(market_data)
        
        # Ensure at least one signal is non-zero for testing
        # Use the last data point to create a test signal if no signals exist
        if all(signals['signal'].apply(lambda x: x == 0)):
            last_idx = signals.index[-1]
            signals.loc[last_idx, 'signal'] = 1  # Create a buy signal for testing
        
        # Open and track positions
        positions = []
        for timestamp, row in signals.iterrows():
            if row['signal'].item() != 0:
                position = components['position_manager'].execute_trade(
                    symbol='AAPL',
                    signal=row['signal'].item(),
                    price=row['close'],
                    timestamp=timestamp
                )
                if position is not None:
                    positions.append(position)
        
        # Verify position consistency
        assert len(positions) > 0
        for position in positions:
            assert position.symbol == 'AAPL'
            assert position.entry_price > 0
            assert position.size > 0
            assert position.type in [PositionType.LONG, PositionType.SHORT]
            
            # Verify position exists in manager
            assert position.symbol in components['position_manager'].positions
            stored_position = components['position_manager'].positions[position.symbol]
            assert stored_position.symbol == position.symbol
            assert stored_position.entry_price == position.entry_price 