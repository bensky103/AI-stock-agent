"""Integration tests for decision making module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decision_making.strategies.base_strategy import TradingStrategy
from decision_making.strategies.ml_hybrid_strategy import MLHybridStrategy
from decision_making.strategies.position_manager import PositionManager

class TestDecisionMakingIntegration:
    @pytest.fixture
    def setup_decision_components(self, test_config):
        """Setup decision-making components"""
        base_strategy = TradingStrategy(config=test_config)
        ml_strategy = MLHybridStrategy(config=test_config)
        position_manager = PositionManager(config=test_config)
        
        return {
            'base_strategy': base_strategy,
            'ml_strategy': ml_strategy,
            'position_manager': position_manager
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, 100),
            'High': np.random.normal(102, 2, 100),
            'Low': np.random.normal(98, 2, 100),
            'Close': np.random.normal(100, 2, 100),
            'Volume': np.random.normal(1000000, 200000, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.normal(0, 1, 100),
            'prediction': np.random.normal(0, 1, 100),  # Mock ML prediction
        }, index=dates)
        return data
    
    def test_strategy_flow(self, setup_decision_components, sample_market_data):
        """Test the complete strategy decision flow"""
        components = setup_decision_components
        
        # 1. Generate signals from base strategy
        base_signals = components['base_strategy'].generate_signals(sample_market_data)
        assert isinstance(base_signals, pd.Series)
        assert not base_signals.empty
        assert all(signal in [-1, 0, 1] for signal in base_signals)
        
        # 2. Generate signals from ML strategy
        ml_signals = components['ml_strategy'].generate_signals(sample_market_data)
        assert isinstance(ml_signals, pd.Series)
        assert not ml_signals.empty
        assert all(signal in [-1, 0, 1] for signal in ml_signals)
        
        # 3. Combine signals and manage positions
        combined_signals = components['ml_strategy'].combine_signals(
            base_signals=base_signals,
            ml_signals=ml_signals
        )
        assert isinstance(combined_signals, pd.Series)
        assert not combined_signals.empty
        
        # 4. Generate position decisions
        position_decisions = components['position_manager'].generate_decisions(
            signals=combined_signals,
            market_data=sample_market_data
        )
        assert isinstance(position_decisions, pd.DataFrame)
        assert not position_decisions.empty
        assert 'position' in position_decisions.columns
        assert 'size' in position_decisions.columns
    
    def test_position_management(self, setup_decision_components, sample_market_data):
        """Test position management functions"""
        components = setup_decision_components
        
        # Generate sample signals
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(sample_market_data)),
            index=sample_market_data.index
        )
        
        # Test position sizing
        position_sizes = components['position_manager'].calculate_position_sizes(
            signals=signals,
            market_data=sample_market_data
        )
        assert isinstance(position_sizes, pd.Series)
        assert not position_sizes.empty
        assert all(0 <= size <= 1 for size in position_sizes)
        
        # Test risk management
        risk_metrics = components['position_manager'].calculate_risk_metrics(
            positions=position_sizes,
            market_data=sample_market_data
        )
        assert isinstance(risk_metrics, dict)
        assert 'exposure' in risk_metrics
        assert 'drawdown' in risk_metrics
        
        # Test stop loss calculation
        stop_losses = components['position_manager'].calculate_stop_losses(
            positions=position_sizes,
            market_data=sample_market_data
        )
        assert isinstance(stop_losses, pd.Series)
        assert not stop_losses.empty
    
    def test_strategy_components(self, setup_decision_components, sample_market_data):
        """Test individual strategy components"""
        components = setup_decision_components
        
        # Test technical analysis in base strategy
        tech_signals = components['base_strategy'].calculate_technical_signals(
            sample_market_data
        )
        assert isinstance(tech_signals, pd.Series)
        assert not tech_signals.empty
        
        # Test ML signal processing
        ml_processed = components['ml_strategy'].process_ml_signals(
            sample_market_data['prediction']
        )
        assert isinstance(ml_processed, pd.Series)
        assert not ml_processed.empty
        
        # Test position validation
        positions = pd.Series(
            np.random.uniform(-1, 1, size=len(sample_market_data)),
            index=sample_market_data.index
        )
        is_valid = components['position_manager'].validate_positions(
            positions=positions,
            market_data=sample_market_data
        )
        assert isinstance(is_valid, bool)
    
    def test_error_handling(self, setup_decision_components):
        """Test error handling in decision components"""
        components = setup_decision_components
        
        # Test with empty data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            components['base_strategy'].generate_signals(empty_df)
        
        # Test with invalid signals
        invalid_signals = pd.Series([2, -2, 3])  # Invalid signal values
        with pytest.raises(Exception):
            components['position_manager'].generate_decisions(
                signals=invalid_signals,
                market_data=pd.DataFrame({'Close': [100, 101, 102]})
            )
        
        # Test with missing required columns
        invalid_data = pd.DataFrame({'Close': [100, 101, 102]})
        with pytest.raises(Exception):
            components['ml_strategy'].generate_signals(invalid_data)
    
    def test_data_consistency(self, setup_decision_components, sample_market_data):
        """Test data consistency across decision components"""
        components = setup_decision_components
        
        # Generate signals from both strategies
        base_signals = components['base_strategy'].generate_signals(sample_market_data)
        ml_signals = components['ml_strategy'].generate_signals(sample_market_data)
        
        # Combine signals
        combined_signals = components['ml_strategy'].combine_signals(
            base_signals=base_signals,
            ml_signals=ml_signals
        )
        
        # Generate position decisions
        decisions = components['position_manager'].generate_decisions(
            signals=combined_signals,
            market_data=sample_market_data
        )
        
        # Verify data consistency
        assert len(decisions) == len(sample_market_data)
        assert all(col in decisions.columns for col in ['position', 'size', 'stop_loss'])
        assert not decisions.isnull().any().any()
        
        # Verify position constraints
        assert all(-1 <= pos <= 1 for pos in decisions['position'])
        assert all(0 <= size <= 1 for size in decisions['size'])
        assert all(stop >= 0 for stop in decisions['stop_loss']) 