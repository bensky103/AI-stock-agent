"""Tests for the market feed module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from unittest.mock import patch, MagicMock
import yfinance as yf

from data_input.market_feed import (
    get_market_data,
    load_config,
    add_technical_indicators,
    MarketFeed
)
from data_input.market_utils import validate_market_data, MarketDataError

# Test fixtures
@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration file."""
    config = {
        'market_data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'intervals': ['1d', '1h'],
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_indicators': {
                'sma': [20, 50],
                'ema': [12, 26],
                'rsi': 14,
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger_bands': {'window': 20, 'std': 2}
            }
        }
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-02', end='2024-01-31', freq='D')
    
    # Generate base prices with a slight upward trend
    base_prices = np.linspace(100, 105, len(dates))
    
    # Generate random daily variations
    daily_variations = np.random.uniform(-1, 1, len(dates))
    
    # Calculate OHLC prices ensuring valid relationships
    close_prices = base_prices + daily_variations
    open_prices = close_prices + np.random.uniform(-0.5, 0.5, len(dates))
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, len(dates))
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, len(dates))
    
    # Ensure high is always highest and low is always lowest
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    data = {
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Adj Close': close_prices,  # Use close prices for adjusted close
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    # Verify price relationships
    assert (df['High'] >= df['Low']).all(), "High prices must be greater than low prices"
    assert (df['High'] >= df['Open']).all(), "High prices must be greater than open prices"
    assert (df['High'] >= df['Close']).all(), "High prices must be greater than close prices"
    assert (df['Low'] <= df['Open']).all(), "Low prices must be less than open prices"
    assert (df['Low'] <= df['Close']).all(), "Low prices must be less than close prices"
    
    return df

# Configuration tests
def test_load_config(sample_config):
    """Test loading configuration file."""
    config = load_config(sample_config)
    assert 'market_data' in config
    assert 'symbols' in config['market_data']
    assert 'AAPL' in config['market_data']['symbols']
    assert 'technical_indicators' in config['market_data']

def test_load_config_invalid_path(tmp_path):
    """Test loading invalid configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent_config.yaml")

def test_load_config_invalid_yaml(tmp_path):
    """Test loading invalid YAML configuration."""
    config_path = tmp_path / "invalid_config.yaml"
    config_path.write_text("invalid: yaml: content: [")
    
    with pytest.raises(Exception):
        load_config(config_path)

# Market data fetching tests
@patch('yfinance.download')
def test_get_market_data_single_symbol(mock_download, sample_config, sample_market_data):
    """Test fetching market data for a single symbol."""
    mock_download.return_value = sample_market_data
    
    df = get_market_data(
        symbols=['AAPL'],
        start_date='2024-01-02',
        end_date='2024-01-31',
        interval='1d',
        config_path=sample_config
    )
    
    assert isinstance(df, pd.DataFrame)
    assert ('AAPL', pd.Timestamp('2024-01-02')) in df.index
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume'])
    mock_download.assert_called_once()

@patch('yfinance.download')
def test_get_market_data_multiple_symbols(mock_download, sample_config, sample_market_data):
    """Test fetching market data for multiple symbols."""
    # Simulate yf.download output for multiple symbols (MultiIndex columns)
    aapl_data = sample_market_data.copy()
    msft_data = sample_market_data.copy()

    # yfinance typically returns columns like: Open, High, Low, Close, Adj Close, Volume
    # We need to create a MultiIndex for columns: (Field, Symbol)
    aapl_data_multi = pd.concat({'AAPL': aapl_data}, axis=1)
    msft_data_multi = pd.concat({'MSFT': msft_data}, axis=1)
    
    # Combine them into a single DataFrame as yfinance would for multiple symbols
    # The column levels would be ['Open', 'High', ..., 'Symbol']
    # yfinance actually returns columns as [(Level0_value, Level1_value), ...]
    # e.g. [ ('Adj Close', 'AAPL'), ('Adj Close', 'MSFT'), ('Close', 'AAPL'), ... ]
    # The order of fields (Open, High, etc.) is usually grouped first, then symbols.
    # Let's construct this structure.
    
    # Get common columns (Open, High, etc.)
    ohlc_cols = sample_market_data.columns.tolist()
    
    # Create DataFrames for each symbol, with original column names
    aapl_for_concat = aapl_data.copy()
    msft_for_concat = msft_data.copy()

    # Concatenate them, resulting in columns like ('AAPL', 'Open'), ('MSFT', 'Open') if keys were on level 0
    # yfinance.download(tickers=['AAPL', 'MSFT']) results in columns: MultiIndex
    # Levels: ['Attributes', 'Symbols'] e.g. Attributes: Adj Close, Close, High, Low, Open, Volume
    #                                     Symbols:    AAPL, MSFT
    # So, ('Adj Close', 'AAPL'), ('Adj Close', 'MSFT'), ('Close', 'AAPL'), etc.

    # Create the expected yfinance multi-symbol DataFrame structure
    mock_return_df = pd.DataFrame(
        index=sample_market_data.index, 
        columns=pd.MultiIndex.from_product([ohlc_cols, ['AAPL', 'MSFT']], names=['Attributes', 'Symbols'])
    )
    
    for col in ohlc_cols:
        mock_return_df[(col, 'AAPL')] = aapl_for_concat[col]
        mock_return_df[(col, 'MSFT')] = msft_for_concat[col]

    # Ensure the mock returns this structured DataFrame when yf.download is called with a list
    def mock_download_side_effect(symbols_arg, *args, **kwargs):
        if isinstance(symbols_arg, list) and sorted(symbols_arg) == sorted(['AAPL', 'MSFT']):
            return mock_return_df
        elif symbols_arg == 'AAPL': # For other tests that might call with single symbol
             return aapl_for_concat
        elif symbols_arg == 'MSFT':
             return msft_for_concat
        return pd.DataFrame() # Default for other unexpected calls

    mock_download.side_effect = mock_download_side_effect
    
    df = get_market_data(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-02',
        end_date='2024-01-31',
        interval='1d',
        config_path=sample_config
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert ('AAPL', pd.Timestamp('2024-01-02')) in df.index
    assert ('MSFT', pd.Timestamp('2024-01-02')) in df.index
    # After processing by get_market_data, columns should be flat and lowercase
    assert 'open' in df.columns 
    assert 'adj_close' in df.columns
    # And the data for each symbol should be present
    assert not df.loc['AAPL'].empty
    assert not df.loc['MSFT'].empty

@patch('yfinance.download')
def test_get_market_data_error_handling(mock_download, sample_config):
    """Test error handling in market data fetching."""
    # Test network error
    mock_download.side_effect = Exception("Network error")
    
    with pytest.raises(Exception):
        get_market_data(
            symbols=['AAPL'],
            start_date='2024-01-02',
            end_date='2024-01-31',
            config_path=sample_config
        )
    
    # Test invalid date range
    with pytest.raises(ValueError):
        get_market_data(
            symbols=['AAPL'],
            start_date='2024-01-31',
            end_date='2024-01-01',  # End date before start date
            config_path=sample_config
        )

# Technical indicators tests
def test_add_technical_indicators(sample_market_data):
    """Test adding technical indicators to market data."""
    # Create a multi-symbol DataFrame
    df = pd.concat([sample_market_data], keys=['AAPL'], axis=1)
    df.columns = pd.MultiIndex.from_product([['open', 'high', 'low', 'close', 'volume', 'adj_close'], ['AAPL']])
    
    df = add_technical_indicators(df)
    
    # Check basic indicators for AAPL
    assert ('sma_20', 'AAPL') in df.columns
    assert ('sma_50', 'AAPL') in df.columns
    assert ('rsi', 'AAPL') in df.columns
    assert ('macd', 'AAPL') in df.columns
    assert ('macd_signal', 'AAPL') in df.columns
    assert ('macd_hist', 'AAPL') in df.columns
    
    # Check indicator values
    assert not df[('sma_20', 'AAPL')].isna().all()
    assert not df[('rsi', 'AAPL')].isna().all()
    assert not df[('macd', 'AAPL')].isna().all()
    
    # Check indicator relationships
    assert (df[('high', 'AAPL')] >= df[('low', 'AAPL')]).all()
    assert (df[('high', 'AAPL')] >= df[('close', 'AAPL')]).all()
    assert (df[('high', 'AAPL')] >= df[('open', 'AAPL')]).all()
    assert (df[('low', 'AAPL')] <= df[('close', 'AAPL')]).all()
    assert (df[('low', 'AAPL')] <= df[('open', 'AAPL')]).all()

def test_add_technical_indicators_custom_config(sample_market_data):
    """Test adding technical indicators with custom configuration."""
    # Create a multi-symbol DataFrame
    df = pd.concat([sample_market_data], keys=['AAPL'], axis=1)
    df.columns = pd.MultiIndex.from_product([['open', 'high', 'low', 'close', 'volume', 'adj_close'], ['AAPL']])
    
    # Add custom indicators
    custom_config = {
        'sma_periods': [10, 30],
        'ema_periods': [8, 21],
        'rsi_period': 7,
        'macd_fast': 8,
        'macd_slow': 21,
        'macd_signal': 5
    }
    
    df = add_technical_indicators(df, custom_config)
    
    # Check custom indicators
    assert ('sma_10', 'AAPL') in df.columns
    assert ('sma_30', 'AAPL') in df.columns
    assert ('ema_8', 'AAPL') in df.columns
    assert ('ema_21', 'AAPL') in df.columns
    assert ('rsi', 'AAPL') in df.columns
    assert ('macd', 'AAPL') in df.columns
    assert ('macd_signal', 'AAPL') in df.columns
    assert ('macd_hist', 'AAPL') in df.columns
    
    # Check indicator values
    assert not df[('sma_10', 'AAPL')].isna().all()
    assert not df[('rsi', 'AAPL')].isna().all()
    assert not df[('macd', 'AAPL')].isna().all()

# Market feed class tests
def test_market_feed_initialization(sample_config):
    """Test MarketFeed class initialization."""
    feed = MarketFeed(config_path=sample_config)
    
    assert feed.config is not None
    assert 'market_data' in feed.config
    assert feed.symbols == ['AAPL', 'MSFT', 'GOOGL']

@patch('yfinance.Ticker')
def test_market_feed_fetch_data(mock_ticker, sample_config, sample_market_data):
    """Test MarketFeed data fetching."""
    # Set up the mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_market_data
    mock_ticker.return_value = mock_ticker_instance
    
    feed = MarketFeed(config_path=sample_config)
    df = feed.fetch_data(
        symbols=['AAPL'],
        start_date='2024-01-02',
        end_date='2024-01-31'
    )
    
    assert isinstance(df, pd.DataFrame)
    assert ('AAPL', pd.Timestamp('2024-01-02')) in df.index
    mock_ticker_instance.history.assert_called_once()

def test_market_feed_data_validation(sample_config, sample_market_data):
    """Test MarketFeed data validation."""
    # If required columns are missing, expect MarketDataError
    with pytest.raises(MarketDataError):
        validate_market_data(sample_market_data)

# Integration tests
def test_market_feed_integration(sample_config):
    """Test market feed integration with multiple symbols and indicators using real yfinance data."""
    feed = MarketFeed(config_path=sample_config)
    
    # Use dates from last year to ensure data availability
    end_date = datetime(2023, 12, 31)  # December 31, 2023
    start_date = datetime(2023, 12, 1)  # December 1, 2023
    
    print(f"\nFetching data from {start_date} to {end_date}")
    
    # Test data fetching with multiple symbols
    symbols = ['AAPL', 'MSFT']
    try:
        # First, try to get raw data directly from yfinance to verify data availability
        print("\nChecking raw data availability:")
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            raw_data = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                prepost=False
            )
            print(f"\nRaw data for {symbol}:")
            print(f"Shape: {raw_data.shape}")
            print(f"Columns: {raw_data.columns.tolist()}")
            print(f"Index type: {type(raw_data.index)}")
            print(f"First few rows:\n{raw_data.head()}")
            print(f"NaN counts:\n{raw_data.isna().sum()}")
        
        # Now try the full fetch_data call
        print("\nAttempting full data fetch with indicators...")
        df = feed.fetch_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            add_indicators=True
        )
        
        print("\nDataFrame Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Index levels: {df.index.names}")
        print(f"Index values: {df.index.get_level_values('symbol').unique()}")
        
        # Print sample data for each symbol
        for symbol in symbols:
            symbol_data = df.xs(symbol, level='symbol')
            print(f"\nSample data for {symbol}:")
            print(symbol_data.head())
            print(f"\nNaN counts for {symbol}:")
            print(symbol_data.isna().sum())
            print(f"\nFirst few dates for {symbol}:")
            print(symbol_data.index[:5])
        
        # Verify the data structure and content
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Check that we have data for both symbols
        assert len(df.index.get_level_values('symbol').unique()) == 2
        assert 'AAPL' in df.index.get_level_values('symbol')
        assert 'MSFT' in df.index.get_level_values('symbol')
        
        # Check basic columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for symbol in symbols:
            for col in required_columns:
                assert (col, symbol) in df.columns, f"Missing {col} column for {symbol}"
                # Verify we have actual data
                symbol_data = df.xs(symbol, level='symbol')
                assert not symbol_data[(col, symbol)].isna().all(), f"All values are NaN for {col} in {symbol}"
                assert len(symbol_data) > 0, f"No data rows for {symbol}"
        
        # Check technical indicators
        indicator_columns = ['sma_20', 'rsi']
        for symbol in symbols:
            for col in indicator_columns:
                assert (col, symbol) in df.columns, f"Missing {col} indicator for {symbol}"
                # Verify indicators are calculated
                symbol_data = df.xs(symbol, level='symbol')
                assert not symbol_data[(col, symbol)].isna().all(), f"All values are NaN for {col} in {symbol}"
        
        # Verify price relationships
        for symbol in symbols:
            symbol_data = df.xs(symbol, level='symbol')
            # High should be highest
            assert (symbol_data[('high', symbol)] >= symbol_data[('low', symbol)]).all(), f"High prices not always >= Low prices for {symbol}"
            assert (symbol_data[('high', symbol)] >= symbol_data[('open', symbol)]).all(), f"High prices not always >= Open prices for {symbol}"
            assert (symbol_data[('high', symbol)] >= symbol_data[('close', symbol)]).all(), f"High prices not always >= Close prices for {symbol}"
            # Low should be lowest
            assert (symbol_data[('low', symbol)] <= symbol_data[('open', symbol)]).all(), f"Low prices not always <= Open prices for {symbol}"
            assert (symbol_data[('low', symbol)] <= symbol_data[('close', symbol)]).all(), f"Low prices not always <= Close prices for {symbol}"
            # Volume should be positive
            assert (symbol_data[('volume', symbol)] >= 0).all(), f"Negative volume found for {symbol}"
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
        raise

