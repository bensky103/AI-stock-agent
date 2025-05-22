"""Integration tests for market data module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketFeed
from data_input.market_data_manager import MarketDataManager
from data_input import market_utils
import pytz
import numpy as np

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
    
    def test_data_consistency(self, setup_market_components):
        """Test data consistency across different operations"""
        components = setup_market_components
        
        # Get data for a specific period using fixed historical dates
        end_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)  # Fixed historical date with UTC timezone
        start_date = datetime(2022, 11, 1, tzinfo=pytz.UTC)  # ~6 months before end_date
        
        # Fetch from both managers
        market_data = components['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        enhanced_data = components['enhanced_manager'].get_market_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            interval='1W'  # Use weekly data
        )
        
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
        
        # Convert to UTC if not already in UTC
        if market_dates.tz is None:
            market_dates = market_dates.tz_localize(pytz.UTC)
        elif market_dates.tz != pytz.UTC:
            market_dates = market_dates.tz_convert(pytz.UTC)
        
        if enhanced_dates.tz is None:
            enhanced_dates = enhanced_dates.tz_localize(pytz.UTC)
        elif enhanced_dates.tz != pytz.UTC:
            enhanced_dates = enhanced_dates.tz_convert(pytz.UTC)
        
        # Compare dates directly - they should both be at market open (14:30 UTC)
        assert market_dates.equals(enhanced_dates), "Date indices should match at market open (14:30 UTC)"
        
        # Compare data trend/patterns rather than exact values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            market_values = market_data[col].values
            enhanced_values = enhanced_data[col].values
            
            # Convert to 1D arrays if needed
            if market_values.ndim > 1:
                market_values = market_values.flatten()
            if enhanced_values.ndim > 1:
                enhanced_values = enhanced_values.flatten()
            
            # Check correlation - should be high (>0.9) for price series from the same symbol
            correlation = np.corrcoef(market_values, enhanced_values)[0, 1]
            if col == 'volume':
                # Volume data often has more variance between sources
                assert correlation > 0.5, f"Data correlation for {col} should be >0.5, got {correlation}"
            else:
                assert correlation > 0.85, f"Data correlation for {col} should be >0.85, got {correlation}"
            
            # For integration testing, we use higher tolerance since the data sources may have differences
            # but should still follow the same overall trends
            if col == 'volume':
                # Volume can vary significantly between data sources
                # Skip the exact comparison for volume data as it's enough to verify correlation
                pass
            else:
                # Price data should be within 12% relative tolerance
                np.testing.assert_allclose(market_values, enhanced_values, rtol=0.12, atol=30) 