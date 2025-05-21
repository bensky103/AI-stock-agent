"""Integration tests for market data module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketFeed
from data_input.market_data_manager import MarketDataManager
from data_input import market_utils
import pytz
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
        
        market_manager = MarketFeed(config_path=config_path, default_interval='1W')  # Set weekly as default
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
        
        # 1. Fetch basic market data - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert len(market_data) >= 8  # Ensure we have enough weekly data points
        
        # 2. Enhance with additional data
        enhanced_data = components['enhanced_manager'].get_market_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            interval='1W'  # Use weekly data
        )
        assert isinstance(enhanced_data, pd.DataFrame)
        assert not enhanced_data.empty
        assert len(enhanced_data.columns) >= len(market_data.columns)
    
    def test_multi_symbol_handling(self, setup_market_components):
        """Test handling of multiple symbols"""
        components = setup_market_components
        
        # Test fetching data for multiple symbols - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        market_data = components['market_manager'].fetch_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert all(symbol in market_data.index.get_level_values('symbol') for symbol in symbols)
        assert len(market_data) >= 8  # Ensure enough weekly data points for each symbol
    
    def test_market_utils_functions(self, setup_market_components):
        """Test market utility functions"""
        components = setup_market_components
        
        # Get some market data first - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        # Test data cleaning
        cleaned_data = market_utils.clean_market_data(market_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.empty
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) >= 8  # Ensure enough weekly data points
    
    def test_error_handling(self, setup_market_components):
        """Test error handling in market data operations"""
        components = setup_market_components
        
        # Test invalid symbol
        with pytest.raises(Exception):
            components['market_manager'].fetch_data(
                symbols='INVALID_SYMBOL',
                start_date=datetime.now(pytz.UTC) - timedelta(days=2),
                end_date=datetime.now(pytz.UTC) - timedelta(days=1)
            )
        
        # Test invalid date range
        with pytest.raises(Exception):
            components['market_manager'].fetch_data(
                symbols='AAPL',
                start_date=datetime.now(pytz.UTC) - timedelta(days=1),
                end_date=datetime.now(pytz.UTC) - timedelta(days=2)
            )
    
    def test_data_consistency(self, setup_market_components):
        """Test data consistency across different operations"""
        components = setup_market_components
        
        # Get data for a specific period - fetch 26 weeks for weekly data
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        start_date = end_date - timedelta(weeks=26)  # Changed to 26 weeks
        
        logger.debug(f"Fetching data from {start_date} to {end_date}")

        # Fetch from both managers
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        logger.debug(f"Market data shape: {market_data.shape}")
        logger.debug(f"Market data index levels: {market_data.index.names}")
        logger.debug(f"Market data first few dates: {market_data.index.get_level_values('datetime')[:5]}")

        enhanced_data = components['enhanced_manager'].get_market_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            interval='1W'  # Use weekly data
        )
        
        logger.debug(f"Enhanced data shape: {enhanced_data.shape}")
        logger.debug(f"Enhanced data index levels: {enhanced_data.index.names}")
        logger.debug(f"Enhanced data first few dates: {enhanced_data.index.get_level_values('datetime')[:5]}")

        # Verify data consistency
        assert isinstance(market_data, pd.DataFrame)
        assert isinstance(enhanced_data, pd.DataFrame)
        assert not market_data.empty
        assert not enhanced_data.empty
        assert len(market_data) >= 8  # Ensure enough weekly data points
        assert len(enhanced_data) >= 8  # Ensure enough weekly data points

        # Get dates and convert to UTC if needed
        market_dates = market_data.index.get_level_values('datetime')
        enhanced_dates = enhanced_data.index.get_level_values('datetime')
        
        logger.debug(f"Market dates timezone: {market_dates.tz}")
        logger.debug(f"Enhanced dates timezone: {enhanced_dates.tz}")
        
        # Convert to UTC if not already in UTC
        if market_dates.tz is None:
            logger.debug("Converting market dates from naive to UTC")
            market_dates = market_dates.tz_localize(pytz.UTC)
        elif market_dates.tz != pytz.UTC:
            logger.debug(f"Converting market dates from {market_dates.tz} to UTC")
            market_dates = market_dates.tz_convert(pytz.UTC)
            
        if enhanced_dates.tz is None:
            logger.debug("Converting enhanced dates from naive to UTC")
            enhanced_dates = enhanced_dates.tz_localize(pytz.UTC)
        elif enhanced_dates.tz != pytz.UTC:
            logger.debug(f"Converting enhanced dates from {enhanced_dates.tz} to UTC")
            enhanced_dates = enhanced_dates.tz_convert(pytz.UTC)

        # Normalize timestamps to midnight UTC
        logger.debug("Normalizing timestamps to midnight UTC")
        market_dates = pd.DatetimeIndex([d.replace(hour=0, minute=0, second=0, microsecond=0) for d in market_dates])
        enhanced_dates = pd.DatetimeIndex([d.replace(hour=0, minute=0, second=0, microsecond=0) for d in enhanced_dates])
        
        logger.debug(f"Normalized market dates first few: {market_dates[:5]}")
        logger.debug(f"Normalized enhanced dates first few: {enhanced_dates[:5]}")

        # Compare dates
        assert market_dates.equals(enhanced_dates), "Date indices should match after normalization"

        # Compare data values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            market_values = market_data[col].values
            enhanced_values = enhanced_data[col].values
            logger.debug(f"Comparing {col} values - shapes: {market_values.shape} vs {enhanced_values.shape}")
            logger.debug(f"First few {col} values - Market: {market_values[:5]} vs Enhanced: {enhanced_values[:5]}")
            np.testing.assert_array_almost_equal(market_values, enhanced_values, decimal=2) 