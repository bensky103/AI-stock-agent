"""Tests for the market data manager."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data_input.market_data_manager import (
    MarketDataManager,
    MarketDataError,
    YFinanceSource
)

def test_market_data_manager_initialization(test_config):
    """Test market data manager initialization."""
    manager = MarketDataManager(config_path="tests/test_config.yaml")
    assert manager.config == test_config
    assert isinstance(manager.source, YFinanceSource)

def test_yfinance_source_initialization():
    """Test YFinance source initialization."""
    source = YFinanceSource()
    assert source.rate_limit == (2000, 3600)  # 2000 requests per hour

def test_fetch_market_data(mock_market_data, test_config):
    """Test fetching market data."""
    with patch("yfinance.download") as mock_download:
        # Mock the download function to return our test data
        mock_download.return_value = mock_market_data["AAPL"]
        
        manager = MarketDataManager(config_path="tests/test_config.yaml")
        
        # Test fetching single symbol
        df = manager.get_market_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert not df.empty
        assert isinstance(df.index, pd.MultiIndex)
        assert "AAPL" in df.index.get_level_values("symbol")
        assert all(col in df.columns for col in [
            "open", "high", "low", "close", "volume",
            "sma_20", "sma_50", "rsi_14", "macd"
        ])
        
        # Test fetching multiple symbols
        mock_download.side_effect = [
            mock_market_data["AAPL"],
            mock_market_data["MSFT"]
        ]
        
        df = manager.get_market_data(
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert not df.empty
        assert set(df.index.get_level_values("symbol")) == {"AAPL", "MSFT"}

def test_data_validation(mock_market_data, test_config):
    """Test market data validation."""
    with patch("data_input.market_data_manager.YFinanceSource.get_data") as mock_get_data:
        # Create invalid data (missing required columns)
        invalid_data = mock_market_data["AAPL"].copy()
        invalid_data = invalid_data.drop(columns=["volume"])
        mock_get_data.return_value = invalid_data
        
        manager = MarketDataManager(config_path="tests/test_config.yaml")
        
        # Test validation of fetched data
        with pytest.raises(MarketDataError) as exc_info:
            manager.get_market_data(
                symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-01-31",
                use_cache=False  # Disable cache to ensure we use the mocked data
            )
        assert "Missing required columns" in str(exc_info.value)
        
        # Test validation of invalid price data
        invalid_price_data = mock_market_data["AAPL"].copy()
        invalid_price_data.loc[invalid_price_data.index[0], "close"] = -1
        mock_get_data.return_value = invalid_price_data
        
        with pytest.raises(MarketDataError) as exc_info:
            manager.get_market_data(
                symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-01-31",
                use_cache=False  # Disable cache to ensure we use the mocked data
            )
        assert "Data contains negative values in price columns" in str(exc_info.value)
        
        # Test validation of invalid volume data
        invalid_volume_data = mock_market_data["AAPL"].copy()
        invalid_volume_data.loc[invalid_volume_data.index[0], "volume"] = -1
        mock_get_data.return_value = invalid_volume_data
        
        with pytest.raises(MarketDataError) as exc_info:
            manager.get_market_data(
                symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-01-31",
                use_cache=False  # Disable cache to ensure we use the mocked data
            )
        assert "Data contains negative values in volume" in str(exc_info.value)

def test_rate_limiting(test_config):
    """Test rate limiting functionality."""
    manager = MarketDataManager(config_path="tests/test_config.yaml")
    
    # Test rate limiter
    assert manager._check_rate_limit()
    for _ in range(2000):  # Max requests
        manager._update_rate_limit()
    assert not manager._check_rate_limit()

def test_technical_indicators(mock_market_data, test_config):
    """Test technical indicator calculation."""
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = mock_market_data["AAPL"]
        
        manager = MarketDataManager(config_path="tests/test_config.yaml")
        df = manager.get_market_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        # Check technical indicators
        assert "sma_20" in df.columns
        assert "sma_50" in df.columns
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
        
        # Verify calculations
        aapl_data = df.loc["AAPL"]
        assert not aapl_data["sma_20"].isna().all()
        assert not aapl_data["sma_50"].isna().all()
        assert not aapl_data["rsi_14"].isna().all()
        assert not aapl_data["macd"].isna().all()

def test_error_handling(test_config):
    """Test error handling."""
    manager = MarketDataManager(config_path="tests/test_config.yaml")
    
    # Test with invalid date range
    with pytest.raises(ValueError):
        manager.get_market_data(
            symbols=["AAPL"],
            start_date="2023-12-31",
            end_date="2023-01-01"
        )
    
    # Test with invalid symbols
    with pytest.raises(MarketDataError):
        manager.get_market_data(symbols=[])
    
    # Test with future dates
    future_date = datetime.now(pytz.UTC) + timedelta(days=1)
    with pytest.raises(ValueError):
        manager.get_market_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date=future_date.strftime("%Y-%m-%d")
        )

def test_parallel_processing(mock_market_data, test_config):
    """Test parallel data fetching."""
    with patch("yfinance.download") as mock_download:
        mock_download.side_effect = [
            mock_market_data["AAPL"],
            mock_market_data["MSFT"]
        ]

        manager = MarketDataManager(config_path="tests/test_config.yaml")

        # Test fetching multiple symbols
        df = manager.get_market_data(
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )

        # Verify data for both symbols
        assert not df.empty
        assert isinstance(df.index, pd.MultiIndex)
        assert "AAPL" in df.index.get_level_values("symbol")
        assert "MSFT" in df.index.get_level_values("symbol")
        
        # Verify technical indicators
        assert all(col in df.columns for col in [
            "open", "high", "low", "close", "volume",
            "sma_20", "sma_50", "rsi_14", "macd"
        ])

def test_data_cleaning(mock_market_data, test_config):
    """Test data cleaning and preprocessing."""
    with patch("yfinance.download") as mock_download:
        # Create data with some issues
        data = mock_market_data["AAPL"].copy()
        data.loc[data.index[0], "volume"] = -1  # Invalid volume
        data.loc[data.index[1], "close"] = 0   # Invalid price
        mock_download.return_value = data
        
        manager = MarketDataManager(config_path="tests/test_config.yaml")
        df = manager.get_market_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        # Verify data cleaning
        assert (df["volume"] > 0).all()
        assert (df["close"] > 0).all()
        assert not df.isna().any().any()

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 