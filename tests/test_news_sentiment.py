import pytest
from datetime import datetime, timedelta
import pytz
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from data_input.news_sentiment import (
    get_news_sentiment,
    _preprocess_text,
    _calculate_credibility_score,
    _get_cached_sentiment,
    NewsSentimentError
)

# Test data
SAMPLE_TEXT = "Tesla's stock price increased by 10% today! Check out https://example.com for more info."
SAMPLE_TEXT_CLEAN = "teslas stock price increased by today check out for more info"

@pytest.fixture
def mock_yfinance_news():
    return [
        {
            'title': 'Tesla Q4 Earnings Beat Expectations',
            'link': 'https://example.com/tesla-earnings',
            'providerPublishTime': int(datetime.now(pytz.UTC).timestamp())
        }
    ]

@pytest.fixture
def mock_twitter_data():
    return [
        {
            'text': 'Tesla stock looking bullish!',
            'created_at': datetime.now(pytz.UTC),
            'public_metrics': {
                'followers_count': 5000,
                'retweet_count': 100,
                'like_count': 500
            }
        }
    ]

@pytest.fixture
def mock_reddit_data():
    return [
        {
            'title': 'Tesla Stock Analysis',
            'selftext': 'Great potential for growth',
            'created_utc': datetime.now(pytz.UTC).timestamp(),
            'score': 100,
            'num_comments': 50
        }
    ]

def test_preprocess_text():
    """Test text preprocessing functionality."""
    # Test basic preprocessing
    assert _preprocess_text(SAMPLE_TEXT) == SAMPLE_TEXT_CLEAN
    
    # Test empty string
    assert _preprocess_text("") == ""
    
    # Test None input
    assert _preprocess_text(None) == ""
    
    # Test with special characters
    assert _preprocess_text("Hello! @World #123") == "hello world"
    
    # Test with multiple spaces
    assert _preprocess_text("  multiple   spaces  ") == "multiple spaces"

def test_calculate_credibility_score():
    """Test credibility score calculation."""
    # Test yfinance source
    assert _calculate_credibility_score('yfinance', {}) == 1.0
    
    # Test Twitter source with metrics
    twitter_metrics = {
        'followers_count': 10000,
        'retweet_count': 1000,
        'like_count': 5000
    }
    twitter_score = _calculate_credibility_score('twitter', twitter_metrics)
    assert 0 <= twitter_score <= 1.0
    
    # Test Reddit source with metrics
    reddit_metrics = {
        'score': 1000,
        'num_comments': 500
    }
    reddit_score = _calculate_credibility_score('reddit_stocks', reddit_metrics)
    assert 0 <= reddit_score <= 1.0
    
    # Test unknown source
    assert _calculate_credibility_score('unknown', {}) == 0.5

def test_cached_sentiment():
    """Test sentiment caching functionality."""
    # Test first call (not cached)
    result1 = _get_cached_sentiment("Tesla stock is doing great!")
    assert 'vader' in result1
    assert 'textblob' in result1
    
    # Test second call (should be cached)
    result2 = _get_cached_sentiment("Tesla stock is doing great!")
    assert result1 == result2
    
    # Test different text
    result3 = _get_cached_sentiment("Tesla stock is doing poorly!")
    assert result1 != result3

@patch('yfinance.Ticker')
def test_get_news_sentiment_yfinance(mock_ticker, mock_yfinance_news):
    """Test news sentiment analysis with yfinance data."""
    # Setup mock
    mock_ticker.return_value.news = mock_yfinance_news
    
    # Test basic functionality
    df = get_news_sentiment(
        symbols="TSLA",
        start_date="2024-01-01",
        end_date="2024-01-31",
        sources=['news']
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'sentiment_score' in df.columns
    assert 'credibility_score' in df.columns
    assert 'weighted_sentiment' in df.columns

@patch('tweepy.Client')
def test_get_news_sentiment_twitter(mock_twitter_client, mock_twitter_data):
    """Test news sentiment analysis with Twitter data."""
    # Setup mock
    mock_twitter_client.return_value.search_recent_tweets.return_value.data = mock_twitter_data
    
    # Test basic functionality
    df = get_news_sentiment(
        symbols="TSLA",
        start_date="2024-01-01",
        end_date="2024-01-31",
        sources=['twitter']
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'sentiment_score' in df.columns
    assert 'credibility_score' in df.columns
    assert 'weighted_sentiment' in df.columns

@patch('praw.Reddit')
def test_get_news_sentiment_reddit(mock_reddit_client, mock_reddit_data):
    """Test news sentiment analysis with Reddit data."""
    # Setup mock
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = mock_reddit_data
    mock_reddit_client.return_value.subreddit.return_value = mock_subreddit
    
    # Test basic functionality
    df = get_news_sentiment(
        symbols="TSLA",
        start_date="2024-01-01",
        end_date="2024-01-31",
        sources=['reddit']
    )
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'sentiment_score' in df.columns
    assert 'credibility_score' in df.columns
    assert 'weighted_sentiment' in df.columns

def test_get_news_sentiment_error_handling():
    """Test error handling in news sentiment analysis."""
    # Test with invalid date range
    with pytest.raises(NewsSentimentError):
        get_news_sentiment(
            symbols="TSLA",
            start_date="2024-01-31",
            end_date="2024-01-01"  # Invalid date range
        )
    
    # Test with empty symbols list
    with pytest.raises(NewsSentimentError):
        get_news_sentiment(symbols=[])

def test_get_news_sentiment_combined():
    """Test news sentiment analysis with multiple sources."""
    with patch('yfinance.Ticker') as mock_ticker, \
         patch('tweepy.Client') as mock_twitter, \
         patch('praw.Reddit') as mock_reddit:
        
        # Setup mocks
        mock_ticker.return_value.news = mock_yfinance_news
        mock_twitter.return_value.search_recent_tweets.return_value.data = mock_twitter_data
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = mock_reddit_data
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        
        # Test combined functionality
        df = get_news_sentiment(
            symbols="TSLA",
            start_date="2024-01-01",
            end_date="2024-01-31",
            sources=['news', 'twitter', 'reddit']
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df['source'].unique()) == 3  # Should have data from all sources
        assert 'sentiment_score' in df.columns
        assert 'credibility_score' in df.columns
        assert 'weighted_sentiment' in df.columns
        assert 'sentiment_category' in df.columns

def test_get_news_sentiment_caching():
    """Test sentiment caching functionality."""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.news = mock_yfinance_news
        
        # Test with caching enabled
        df1 = get_news_sentiment(
            symbols="TSLA",
            start_date="2024-01-01",
            end_date="2024-01-31",
            sources=['news'],
            use_cache=True
        )
        
        # Test with caching disabled
        df2 = get_news_sentiment(
            symbols="TSLA",
            start_date="2024-01-01",
            end_date="2024-01-31",
            sources=['news'],
            use_cache=False
        )
        
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert not df1.empty
        assert not df2.empty 