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
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
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
        start_date='2024-01-01',
        end_date='2024-01-31',
        interval='1d',
        config_path=sample_config
    )
    
    assert isinstance(df, pd.DataFrame)
    assert ('AAPL', pd.Timestamp('2024-01-01')) in df.index
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume'])
    mock_download.assert_called_once()

@patch('yfinance.download')
def test_get_market_data_multiple_symbols(mock_download, sample_config, sample_market_data):
    """Test fetching market data for multiple symbols."""
    # Create separate data for each symbol
    aapl_data = sample_market_data.copy()
    msft_data = sample_market_data.copy()
    
    # Mock download to return different data for each symbol
    def mock_download_side_effect(symbol, *args, **kwargs):
        if symbol == 'AAPL':
            return aapl_data
        elif symbol == 'MSFT':
            return msft_data
        return pd.DataFrame()
    
    mock_download.side_effect = mock_download_side_effect
    
    df = get_market_data(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        interval='1d',
        config_path=sample_config
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert ('AAPL', pd.Timestamp('2024-01-01')) in df.index
    assert ('MSFT', pd.Timestamp('2024-01-01')) in df.index
    assert ('close', 'AAPL') in df.columns
    assert ('close', 'MSFT') in df.columns

@patch('yfinance.download')
def test_get_market_data_error_handling(mock_download, sample_config):
    """Test error handling in market data fetching."""
    # Test network error
    mock_download.side_effect = Exception("Network error")
    
    with pytest.raises(Exception):
        get_market_data(
            symbols=['AAPL'],
            start_date='2024-01-01',
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
        start_date='2024-01-01',
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
@patch('yfinance.Ticker')
def test_market_feed_integration(mock_ticker, sample_config, sample_market_data):
    """Test market feed integration with multiple symbols and indicators."""
    # Create separate data for each symbol and ensure lowercase column names
    aapl_data = sample_market_data.copy()
    msft_data = sample_market_data.copy()
    
    # Convert column names to match yfinance.Ticker.history() output exactly
    column_mapping = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Adj Close': 'Adj Close',
        'Volume': 'Volume'
    }
    
    # Ensure both datasets have the exact same structure as yfinance output
    for df in [aapl_data, msft_data]:
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        df.index.name = 'Date'
    
    # Mock Ticker.history to return different data for each symbol
    def mock_history_side_effect(*args, **kwargs):
        symbol = mock_ticker.call_args[0][0]  # Get the symbol from Ticker constructor
        if symbol == 'AAPL':
            return aapl_data.copy()  # Return a copy to prevent modification
        elif symbol == 'MSFT':
            return msft_data.copy()  # Return a copy to prevent modification
        return pd.DataFrame()
    
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = mock_history_side_effect
    mock_ticker.return_value = mock_ticker_instance
    
    feed = MarketFeed(config_path=sample_config)
    
    # Test data fetching with multiple symbols
    symbols = ['AAPL', 'MSFT']
    df = feed.fetch_data(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2024-01-31',
        add_indicators=True
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert ('AAPL', pd.Timestamp('2024-01-02')) in df.index
    assert ('MSFT', pd.Timestamp('2024-01-02')) in df.index
    assert ('close', 'AAPL') in df.columns
    assert ('close', 'MSFT') in df.columns
    assert ('sma_20', 'AAPL') in df.columns
    assert ('sma_20', 'MSFT') in df.columns
    assert ('rsi', 'AAPL') in df.columns
    assert ('rsi', 'MSFT') in df.columns

