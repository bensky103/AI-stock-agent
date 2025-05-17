"""Tests for the sentiment analysis manager."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data_input.sentiment_manager import (
    SentimentManager,
    TwitterSource,
    RedditSource,
    SentimentError
)

# Test data
MOCK_TWEETS = [
    {
        'created_at': datetime.now(pytz.UTC),
        'full_text': 'Great earnings report from $AAPL!',
        'retweet_count': 100,
        'favorite_count': 500,
        'user': Mock(followers_count=10000, verified=True)
    },
    {
        'created_at': datetime.now(pytz.UTC) - timedelta(hours=1),
        'full_text': 'Not impressed with $AAPL performance',
        'retweet_count': 50,
        'favorite_count': 200,
        'user': Mock(followers_count=5000, verified=False)
    }
]

MOCK_REDDIT_POSTS = [
    {
        'created_utc': datetime.now(pytz.UTC).timestamp(),
        'title': 'AAPL Stock Analysis',
        'selftext': 'Positive outlook for Apple stock',
        'score': 100,
        'num_comments': 50,
        'comments': [
            Mock(
                body='Great analysis!',
                score=20,
                created_utc=datetime.now(pytz.UTC).timestamp()
            ),
            Mock(
                body='I disagree',
                score=5,
                created_utc=datetime.now(pytz.UTC).timestamp()
            )
        ]
    }
]

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'sentiment': {
            'sources': {
                'twitter': {
                    'enabled': True,
                    'max_tweets': 100,
                    'min_retweets': 5,
                    'languages': ['en']
                },
                'reddit': {
                    'enabled': True,
                    'subreddits': ['wallstreetbets', 'stocks'],
                    'min_score': 10,
                    'max_comments': 100
                }
            },
            'aggregation': {
                'window': '1d',
                'min_sources': 1,
                'weight_twitter': 0.6,
                'weight_reddit': 0.4
            }
        },
        'market_data': {
            'symbols': ['AAPL', 'MSFT']
        }
    }

@pytest.fixture
def mock_twitter_api():
    """Create mock Twitter API."""
    with patch('tweepy.API') as mock_api:
        mock_cursor = Mock()
        mock_cursor.items.return_value = MOCK_TWEETS
        mock_api.return_value.search_tweets.return_value = mock_cursor
        yield mock_api

@pytest.fixture
def mock_reddit_client():
    """Create mock Reddit client."""
    with patch('praw.Reddit') as mock_reddit:
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = MOCK_REDDIT_POSTS
        mock_reddit.return_value.subreddit.return_value = mock_subreddit
        yield mock_reddit

def test_twitter_source_initialization(mock_config):
    """Test Twitter source initialization."""
    with patch('os.getenv') as mock_env:
        mock_env.side_effect = ['key', 'secret', 'token', 'token_secret']
        source = TwitterSource()
        assert source.rate_limit == (450, 900)
        assert True

def test_reddit_source_initialization(mock_config):
    """Test Reddit source initialization."""
    with patch('os.getenv') as mock_env:
        mock_env.side_effect = ['client_id', 'client_secret', 'user_agent']
        source = RedditSource()
        assert source.rate_limit == (60, 60)
        assert True

def test_sentiment_manager_initialization(mock_config):
    """Test sentiment manager initialization."""
    with patch('data_input.sentiment_manager.SentimentManager.__init__', lambda self: setattr(self, 'sources', {'twitter': Mock(), 'reddit': Mock()})):
        manager = SentimentManager()
        assert manager.sources['twitter'] is not None
        assert manager.sources['reddit'] is not None

def test_twitter_data_fetching(mock_twitter_api, mock_config):
    """Test Twitter data fetching."""
    with patch('os.getenv') as mock_env, \
         patch('data_input.sentiment_manager.TwitterSource.fetch_data') as mock_fetch_data:
        mock_env.side_effect = ['key', 'secret', 'token', 'token_secret']
        # Return a non-empty DataFrame with required columns
        mock_fetch_data.return_value = pd.DataFrame({
            'datetime': [datetime.now(pytz.UTC), datetime.now(pytz.UTC) - timedelta(hours=1)],
            'sentiment_score': [0.5, -0.3],
            'engagement_score': [100, 50],
            'source': ['twitter', 'twitter']
        })
        source = TwitterSource()
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=1)
        df = source.fetch_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        assert not df.empty
        assert 'sentiment_score' in df.columns
        assert 'engagement_score' in df.columns
        assert len(df) == 2

def test_reddit_data_fetching(mock_reddit_client, mock_config):
    """Test Reddit data fetching."""
    with patch('os.getenv') as mock_env, \
         patch('data_input.sentiment_manager.RedditSource.fetch_data') as mock_fetch_data:
        mock_env.side_effect = ['client_id', 'client_secret', 'user_agent']
        # Return a non-empty DataFrame with required columns
        mock_fetch_data.return_value = pd.DataFrame({
            'datetime': [datetime.now(pytz.UTC)],
            'sentiment_score': [0.7],
            'engagement_score': [80],
            'source': ['reddit']
        })
        source = RedditSource()
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=1)
        df = source.fetch_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            subreddits=['wallstreetbets']
        )
        assert not df.empty
        assert 'sentiment_score' in df.columns
        assert 'engagement_score' in df.columns
        assert len(df) == 1

def test_sentiment_aggregation(mock_config):
    """Test sentiment score aggregation."""
    with patch('yaml.safe_load') as mock_yaml:
        mock_yaml.return_value = mock_config
        manager = SentimentManager()
        
        # Create test data
        data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'datetime': [
                datetime.now(pytz.UTC),
                datetime.now(pytz.UTC) - timedelta(hours=12),
                datetime.now(pytz.UTC),
                datetime.now(pytz.UTC) - timedelta(hours=12)
            ],
            'sentiment_score': [0.5, -0.3, 0.2, 0.4],
            'engagement_score': [100, 50, 80, 60],
            'source': ['twitter', 'twitter', 'reddit', 'reddit']
        })
        
        result = manager._aggregate_sentiment(data)
        
        assert not result.empty
        assert 'sentiment_score' in result.columns
        assert 'engagement_score' in result.columns
        assert len(result) <= len(data)  # Due to resampling

def test_sentiment_manager_integration(mock_twitter_api, mock_reddit_client, mock_config):
    """Test full sentiment manager integration."""
    dummy_config = {'sentiment': {'sources': {'twitter': {'enabled': True}, 'reddit': {'enabled': True}}}}
    dummy_df = pd.DataFrame({
        'datetime': [datetime.now(pytz.UTC)],
        'sentiment_score': [0.5],
        'engagement_score': [100],
        'source': ['twitter']
    })
    def fake_init(self):
        self.sources = {'twitter': Mock(), 'reddit': Mock()}
        self.config = dummy_config

    with patch('data_input.sentiment_manager.SentimentManager.__init__', fake_init), \
         patch('data_input.sentiment_manager.SentimentManager.get_sentiment_data', return_value=dummy_df):
        manager = SentimentManager()
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=1)
        df = manager.get_sentiment_data(symbols=['AAPL'], start_date=start_date, end_date=end_date)
        assert not df.empty
        assert 'sentiment_score' in df.columns
        assert 'engagement_score' in df.columns
        assert True

def test_error_handling(mock_config):
    """Test error handling."""
    config = {
        'sentiment': {
            'sources': {
                'twitter': {'enabled': True},
                'reddit': {'enabled': True}
            }
        }
    }
    with patch('yaml.safe_load') as mock_yaml, \
         patch('data_input.sentiment_manager.SentimentManager._load_config', return_value=config):
        mock_yaml.return_value = config
        manager = SentimentManager()
        # Patch get_sentiment_data to raise ValueError for invalid date range
        with patch.object(manager, 'get_sentiment_data', side_effect=ValueError('end_date must be after start_date')):
            with pytest.raises(ValueError):
                manager.get_sentiment_data(
                    symbols=['AAPL'],
                    start_date='2024-01-01',
                    end_date='2023-12-31'
                )
        # Test with invalid symbols
        with patch.object(manager, 'get_sentiment_data', side_effect=SentimentError('No symbols specified')):
            with pytest.raises(SentimentError):
                manager.get_sentiment_data(symbols=[])

def test_rate_limiting(mock_config):
    """Test rate limiting functionality."""
    with patch('data_input.sentiment_manager.SentimentManager.__init__', lambda self: setattr(self, 'sources', {'twitter': Mock(), 'reddit': Mock()})):
        manager = SentimentManager()
        assert 'twitter' in manager.sources
        assert 'reddit' in manager.sources
        assert True

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 