"""Integration tests for market data module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketFeed
from data_input.market_data_manager import MarketDataManager
from data_input import market_utils

class TestMarketDataIntegration:
    @pytest.fixture
    def setup_market_components(self, test_config):
        """Setup market data components"""
        # Create a temporary config file for MarketFeed
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        market_manager = MarketFeed(config_path=config_path)
        enhanced_manager = MarketDataManager(config_path=config_path)
        
        return {
            'market_manager': market_manager,
            'enhanced_manager': enhanced_manager,
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
    
    def test_market_data_flow(self, setup_market_components):
        """Test the complete market data flow"""
        components = setup_market_components
        
        # 1. Fetch basic market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        
        # 2. Enhance with additional data
        enhanced_data = components['enhanced_manager'].get_market_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        assert isinstance(enhanced_data, pd.DataFrame)
        assert not enhanced_data.empty
        assert len(enhanced_data.columns) >= len(market_data.columns)
    
    def test_multi_symbol_handling(self, setup_market_components):
        """Test handling of multiple symbols"""
        components = setup_market_components
        
        # Test fetching data for multiple symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        market_data = components['market_manager'].fetch_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert all(symbol in market_data.index.get_level_values('symbol') for symbol in symbols)
    
    def test_market_utils_functions(self, setup_market_components):
        """Test market utility functions"""
        components = setup_market_components
        
        # Get some market data first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        # Test data cleaning
        cleaned_data = market_utils.clean_market_data(market_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.empty
        assert cleaned_data.isnull().sum().sum() == 0
    
    def test_error_handling(self, setup_market_components):
        """Test error handling in market data operations"""
        components = setup_market_components
        
        # Test invalid symbol
        with pytest.raises(Exception):
            components['market_manager'].fetch_data(
                symbols='INVALID_SYMBOL',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
        
        # Test invalid date range
        with pytest.raises(Exception):
            components['market_manager'].fetch_data(
                symbols='AAPL',
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)
            )
    
    def test_data_consistency(self, setup_market_components):
        """Test data consistency across different operations"""
        components = setup_market_components
        
        # Get data for a specific period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Fetch from both managers
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        enhanced_data = components['enhanced_manager'].get_market_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify data consistency
        assert isinstance(market_data, pd.DataFrame)
        assert isinstance(enhanced_data, pd.DataFrame)
        assert not market_data.empty
        assert not enhanced_data.empty
        
        # Verify date ranges
        market_dates = market_data.index.get_level_values('datetime')
        enhanced_dates = enhanced_data.index.get_level_values('datetime')
        
        assert market_dates.min() >= start_date
        assert market_dates.max() <= end_date
        assert enhanced_dates.min() >= start_date
        assert enhanced_dates.max() <= end_date 