"""Shared test fixtures for the trading system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import yaml
from unittest.mock import Mock, patch
import os
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "test_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(
        start="2023-01-01",
        end="2023-12-31",
        freq="D",
        tz=pytz.UTC
    )
    
    # Create mock data for multiple symbols
    data = {}
    for symbol in ["AAPL", "MSFT"]:
        # Generate random price data
        np.random.seed(42)  # For reproducibility
        base_price = 100 if symbol == "AAPL" else 200
        prices = base_price + np.random.normal(0, 2, len(dates)).cumsum()
        
        df = pd.DataFrame({
            "open": prices * (1 + np.random.normal(0, 0.01, len(dates))),
            "high": prices * (1 + np.random.normal(0.02, 0.01, len(dates))),
            "low": prices * (1 + np.random.normal(-0.02, 0.01, len(dates))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Add technical indicators
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["rsi_14"] = 50 + np.random.normal(0, 10, len(dates))  # Mock RSI
        df["macd"] = df["sma_20"] - df["sma_50"]  # Mock MACD
        
        data[symbol] = df
    
    return data

@pytest.fixture(scope="session")
def mock_sentiment_data():
    """Create mock sentiment data for testing."""
    dates = pd.date_range(
        start="2023-01-01",
        end="2023-12-31",
        freq="D",
        tz=pytz.UTC
    )
    
    # Create mock sentiment data
    data = {}
    for symbol in ["AAPL", "MSFT"]:
        # Generate random sentiment scores
        np.random.seed(42)  # For reproducibility
        sentiment = np.random.normal(0, 0.5, len(dates))
        engagement = np.random.randint(100, 1000, len(dates))
        
        df = pd.DataFrame({
            "sentiment_score": sentiment,
            "engagement_score": engagement
        }, index=dates)
        
        data[symbol] = df
    
    return data

@pytest.fixture(scope="session")
def mock_twitter_api():
    """Create mock Twitter API for testing."""
    with patch("tweepy.API") as mock_api:
        mock_cursor = Mock()
        mock_cursor.items.return_value = [
            Mock(
                created_at=datetime.now(pytz.UTC),
                full_text=f"Test tweet about $AAPL",
                retweet_count=100,
                favorite_count=500,
                user=Mock(followers_count=10000, verified=True)
            )
        ]
        mock_api.return_value.search_tweets.return_value = mock_cursor
        yield mock_api

@pytest.fixture(scope="session")
def mock_reddit_client():
    """Create mock Reddit client for testing."""
    with patch("praw.Reddit") as mock_reddit:
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = [
            Mock(
                created_utc=datetime.now(pytz.UTC).timestamp(),
                title="Test post about AAPL",
                selftext="Test content",
                score=100,
                num_comments=50,
                comments=Mock(
                    list=Mock(return_value=[
                        Mock(
                            body="Test comment",
                            score=20,
                            created_utc=datetime.now(pytz.UTC).timestamp()
                        )
                    ])
                )
            )
        ]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        yield mock_reddit

@pytest.fixture(scope="session")
def mock_model():
    """Create mock model for testing."""
    with patch("torch.nn.Module") as mock_model:
        mock_model.return_value.forward.return_value = (
            np.random.random((32, 1)),  # predictions
            np.random.random((32, 1))   # confidence scores
        )
        yield mock_model

@pytest.fixture(scope="session")
def test_logger():
    """Create test logger."""
    logger = logging.getLogger("test_trading")
    logger.setLevel(logging.WARNING)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add file handler
    handler = logging.FileHandler("logs/test_trading.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    
    return logger

# Global test fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return test data directory."""
    data_dir = Path("tests/data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def sample_market_data():
    """Create sample market data with technical indicators."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 2, 100),
        'High': np.random.normal(102, 2, 100),
        'Low': np.random.normal(98, 2, 100),
        'Close': np.random.normal(100, 2, 100),
        'Volume': np.random.normal(1000000, 200000, 100),
    }, index=dates)
    
    # Add technical indicators
    data['RSI'] = np.random.uniform(0, 100, 100)
    data['MACD'] = np.random.normal(0, 1, 100)
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Market_Regime'] = np.random.uniform(0, 1, 100)
    
    return data

@pytest.fixture(scope="session")
def strategy_config():
    """Create sample strategy configuration."""
    config = {
        'strategy': {
            'name': 'ml_hybrid',
            'parameters': {
                'signal_threshold': 0.3,
                'weights': {
                    'ml': 0.4,
                    'technical': 0.4,
                    'regime': 0.2
                }
            }
        },
        'position_management': {
            'max_position_size': 0.1,
            'max_drawdown': 0.05,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'trailing_stop': True,
            'trailing_stop_pct': 0.01
        }
    }
    
    config_path = Path('tests/test_strategy_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture(scope="session")
def test_symbols():
    """Return list of test symbols."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

@pytest.fixture(scope="session")
def test_timeframe():
    """Return test timeframe."""
    return {
        'start': datetime(2024, 1, 1),
        'end': datetime(2024, 1, 31)
    }

# Test data generation helpers
def generate_test_market_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    freq: str = 'D'
) -> pd.DataFrame:
    """Generate test market data for a symbol."""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = pd.DataFrame({
        'Open': np.random.normal(100, 2, len(dates)),
        'High': np.random.normal(102, 2, len(dates)),
        'Low': np.random.normal(98, 2, len(dates)),
        'Close': np.random.normal(100, 2, len(dates)),
        'Volume': np.random.normal(1000000, 200000, len(dates)),
    }, index=dates)
    
    # Add technical indicators
    data['RSI'] = np.random.uniform(0, 100, len(dates))
    data['MACD'] = np.random.normal(0, 1, len(dates))
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Market_Regime'] = np.random.uniform(0, 1, len(dates))
    
    return data

def generate_test_portfolio(
    symbols: list,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """Generate test portfolio data."""
    portfolio = {}
    for symbol in symbols:
        portfolio[symbol] = generate_test_market_data(
            symbol, start_date, end_date
        )
    return portfolio

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "strategy: mark test as strategy test"
    )
    config.addinivalue_line(
        "markers",
        "position: mark test as position management test"
    )
    config.addinivalue_line(
        "markers",
        "risk: mark test as risk management test"
    )

def pytest_collection_modifyitems(items):
    """Modify test collection."""
    # Sort tests by type
    strategy_tests = []
    position_tests = []
    risk_tests = []
    other_tests = []
    
    for item in items:
        if item.get_closest_marker('strategy'):
            strategy_tests.append(item)
        elif item.get_closest_marker('position'):
            position_tests.append(item)
        elif item.get_closest_marker('risk'):
            risk_tests.append(item)
        else:
            other_tests.append(item)
    
    # Reorder tests
    items[:] = strategy_tests + position_tests + risk_tests + other_tests 