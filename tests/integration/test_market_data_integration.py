import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketDataManager
from data_input.market_data_manager import MarketDataManager as EnhancedMarketManager
from data_input.market_utils import MarketUtils

class TestMarketDataIntegration:
    @pytest.fixture
    def setup_market_components(self, test_config):
        """Setup market data components"""
        market_manager = MarketDataManager(config=test_config)
        enhanced_manager = EnhancedMarketManager(config=test_config)
        market_utils = MarketUtils(config=test_config)
        
        return {
            'market_manager': market_manager,
            'enhanced_manager': enhanced_manager,
            'market_utils': market_utils
        }
    
    def test_market_data_flow(self, setup_market_components):
        """Test the complete market data flow"""
        components = setup_market_components
        
        # 1. Fetch basic market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = components['market_manager'].fetch_historical_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        
        # 2. Enhance with additional data
        enhanced_data = components['enhanced_manager'].enhance_market_data(market_data)
        assert isinstance(enhanced_data, pd.DataFrame)
        assert not enhanced_data.empty
        assert len(enhanced_data.columns) >= len(market_data.columns)
        
        # 3. Process with market utils
        processed_data = components['market_utils'].process_market_data(enhanced_data)
        assert isinstance(processed_data, pd.DataFrame)
        assert not processed_data.empty
    
    def test_multi_symbol_handling(self, setup_market_components):
        """Test handling of multiple symbols"""
        components = setup_market_components
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Fetch data for multiple symbols
        multi_data = components['enhanced_manager'].fetch_multiple_symbols(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert isinstance(multi_data, dict)
        assert all(symbol in multi_data for symbol in symbols)
        assert all(isinstance(data, pd.DataFrame) for data in multi_data.values())
    
    def test_market_utils_functions(self, setup_market_components):
        """Test market utility functions"""
        components = setup_market_components
        
        # Get sample data
        market_data = components['market_manager'].fetch_historical_data(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now()
        )
        
        # Test various utility functions
        volatility = components['market_utils'].calculate_volatility(market_data)
        assert isinstance(volatility, float)
        assert volatility >= 0
        
        returns = components['market_utils'].calculate_returns(market_data)
        assert isinstance(returns, pd.Series)
        assert not returns.empty
        
        correlation = components['market_utils'].calculate_correlation(
            market_data,
            components['market_manager'].fetch_historical_data(
                symbol='MSFT',
                start_date=datetime.now() - timedelta(days=5),
                end_date=datetime.now()
            )
        )
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1
    
    def test_error_handling(self, setup_market_components):
        """Test error handling in market components"""
        components = setup_market_components
        
        # Test invalid symbol
        with pytest.raises(Exception):
            components['market_manager'].fetch_historical_data(
                symbol='INVALID_SYMBOL',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
        
        # Test invalid date range
        with pytest.raises(Exception):
            components['market_manager'].fetch_historical_data(
                symbol='AAPL',
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)
            )
        
        # Test empty data handling
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            components['market_utils'].process_market_data(empty_df)
    
    def test_data_consistency(self, setup_market_components):
        """Test data consistency across market components"""
        components = setup_market_components
        
        # Get data from different components
        market_data = components['market_manager'].fetch_historical_data(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now()
        )
        
        enhanced_data = components['enhanced_manager'].enhance_market_data(market_data)
        processed_data = components['market_utils'].process_market_data(enhanced_data)
        
        # Verify data consistency
        assert len(processed_data) == len(market_data)
        assert all(col in processed_data.columns for col in market_data.columns)
        assert not processed_data.isnull().any().any() 