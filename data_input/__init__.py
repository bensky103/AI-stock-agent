"""Data input package for market data and sentiment analysis."""

# Import directly using absolute imports instead of relative imports
import sys
from pathlib import Path

# Add the necessary path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import market_data_manager directly
from data_input.market_feed import (
    get_market_data,
    load_config,
    validate_market_data,
    add_technical_indicators
)

# Try direct import first
try:
    from data_input.market_data_manager import MarketDataManager, MarketDataError, YFinanceSource
except ImportError:
    # Fallback to importlib for direct file loading
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "market_data_manager", 
        str(Path(__file__).parent / "market_data_manager.py")
    )
    market_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(market_data_module)
    MarketDataManager = market_data_module.MarketDataManager
    MarketDataError = market_data_module.MarketDataError
    YFinanceSource = market_data_module.YFinanceSource

from data_input.news_sentiment import (
    NewsSentimentAnalyzer,
    NewsSentimentError,
    get_news_sentiment,
    get_aggregate_sentiment,
    _preprocess_text,
    _calculate_credibility_score,
    _get_cached_sentiment,
    _get_twitter_client,
    _get_reddit_client,
    _fetch_twitter_news,
    _fetch_reddit_news
)
from data_input.sentiment_manager import (
    SentimentManager,
    SentimentError,
    TwitterSource,
    RedditSource
)

__all__ = [
    # Market feed
    'get_market_data',
    'load_config',
    'validate_market_data',
    'add_technical_indicators',
    
    # Market data manager
    'MarketDataManager',
    'MarketDataError',
    'YFinanceSource',
    
    # News sentiment
    'NewsSentimentAnalyzer',
    'NewsSentimentError',
    'get_news_sentiment',
    'get_aggregate_sentiment',
    '_preprocess_text',
    '_calculate_credibility_score',
    '_get_cached_sentiment',
    '_get_twitter_client',
    '_get_reddit_client',
    '_fetch_twitter_news',
    '_fetch_reddit_news',
    
    # Sentiment manager
    'SentimentManager',
    'SentimentError',
    'TwitterSource',
    'RedditSource'
] 