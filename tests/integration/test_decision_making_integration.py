"""Integration tests for decision making module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.strategies import TradingStrategy
from trading.position_manager import PositionManager
from data_input.market_feed import MarketFeed

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
        
        # Initialize components with config path
        market_feed = MarketFeed(config_path=config_path)
        base_strategy = TradingStrategy(config_path=config_path)
        position_manager = PositionManager(config_path=config_path)
        
        return {
            'market_feed': market_feed,
            'strategy': base_strategy,
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
        
        # 1. Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        
        # 2. Generate trading signals
        signals = components['strategy'].generate_signals(market_data)
        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty
        assert 'signal' in signals.columns
        
        # 3. Execute trades based on signals
        for timestamp, row in signals.iterrows():
            if row['signal'] != 0:  # If there's a signal
                position = components['position_manager'].execute_trade(
                    symbol='AAPL',
                    signal=row['signal'],
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
        assert position.position_type == 'long'
        
        # 2. Update position
        updated_position = components['position_manager'].update_position(
            position_id=position.id,
            current_price=155.0
        )
        assert updated_position.current_price == 155.0
        assert updated_position.pnl == 500.0  # (155 - 150) * 100
        
        # 3. Close position
        closed_position = components['position_manager'].close_position(
            position_id=position.id,
            exit_price=160.0
        )
        assert closed_position.is_closed
        assert closed_position.exit_price == 160.0
        assert closed_position.pnl == 1000.0  # (160 - 150) * 100
    
    def test_strategy_components(self, setup_decision_components):
        """Test individual strategy components"""
        components = setup_decision_components
        
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        # Test technical analysis
        technical_signals = components['strategy'].calculate_technical_signals(market_data)
        assert isinstance(technical_signals, pd.DataFrame)
        assert not technical_signals.empty
        assert all(col in technical_signals.columns for col in ['sma_20', 'rsi_14', 'macd'])
        
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
        with pytest.raises(Exception):
            components['strategy'].generate_signals(pd.DataFrame())
        
        # Test invalid position parameters
        with pytest.raises(Exception):
            components['position_manager'].open_position(
                symbol='AAPL',
                entry_price=-150.0,  # Invalid price
                size=100,
                position_type='long'
            )
        
        # Test invalid position type
        with pytest.raises(Exception):
            components['position_manager'].open_position(
                symbol='AAPL',
                entry_price=150.0,
                size=100,
                position_type='invalid_type'  # Invalid position type
            )
    
    def test_data_consistency(self, setup_decision_components):
        """Test data consistency across decision making components"""
        components = setup_decision_components
        
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_feed'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate signals
        signals = components['strategy'].generate_signals(market_data)
        
        # Open and track positions
        positions = []
        for timestamp, row in signals.iterrows():
            if row['signal'] != 0:
                position = components['position_manager'].execute_trade(
                    symbol='AAPL',
                    signal=row['signal'],
                    price=row['close'],
                    timestamp=timestamp
                )
                positions.append(position)
        
        # Verify position consistency
        assert len(positions) > 0
        for position in positions:
            assert position.symbol == 'AAPL'
            assert position.entry_price > 0
            assert position.size > 0
            assert position.position_type in ['long', 'short']
            
            # Verify position updates
            updated_position = components['position_manager'].get_position(position.id)
            assert updated_position.id == position.id
            assert updated_position.symbol == position.symbol
            assert updated_position.entry_price == position.entry_price 