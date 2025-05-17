"""News sentiment analysis module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import yaml
import json
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import time
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import lru_cache
import tweepy
import praw
import os
import pytz
import yfinance as yf

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_sentiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class NewsSentimentError(Exception):
    """Exception raised for news sentiment analysis errors."""
    pass

@lru_cache(maxsize=1000)
def _get_cached_sentiment(text: str) -> Dict[str, float]:
    """Get cached sentiment analysis results.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with sentiment scores
    """
    if not text:
        return {'vader': 0.0, 'textblob': 0.0}
    
    # Use TextBlob for basic sentiment analysis
    blob = TextBlob(text)
    
    return {
        'vader': blob.sentiment.polarity,  # Placeholder for VADER
        'textblob': blob.sentiment.polarity
    }

def _get_twitter_client() -> Optional[tweepy.API]:
    """Get Twitter API client.
    
    Returns:
        Twitter API client or None if credentials not available
    """
    # This is a placeholder implementation
    # In a real implementation, this would use actual Twitter API credentials
    return None

def _get_reddit_client() -> Optional[praw.Reddit]:
    """Get Reddit API client.
    
    Returns:
        Reddit API client or None if credentials not available
    """
    # This is a placeholder implementation
    # In a real implementation, this would use actual Reddit API credentials
    return None

def _fetch_twitter_news(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Fetch news from Twitter.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        List of tweet dictionaries
    """
    try:
        # Ensure timezone-aware datetimes
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        client = _get_twitter_client()
        if client is None:
            # Return sample tweet if no client available
            return [{
                'datetime': datetime.now(timezone.utc),
                'text': f"Sample tweet about ${symbol} stock",
                'source': 'twitter',
                'title': f"Sample tweet about ${symbol} stock",
                'metrics': {
                    'followers_count': 5000,
                    'retweet_count': 100,
                    'like_count': 500
                }
            }]
        
        # Search for tweets
        query = f"${symbol} stock"
        tweets = client.search_tweets(
            q=query,
            lang='en',
            count=100,
            tweet_mode='extended'
        )
        
        # Process tweets
        tweet_data = []
        for tweet in tweets:
            # Twitter timestamps are in UTC
            tweet_time = tweet.created_at.replace(tzinfo=timezone.utc)
            
            if start_date <= tweet_time <= end_date:
                tweet_data.append({
                    'datetime': tweet_time,
                    'text': tweet.full_text,
                    'source': 'twitter',
                    'title': tweet.full_text,
                    'metrics': {
                        'followers_count': tweet.user.followers_count,
                        'retweet_count': tweet.retweet_count,
                        'like_count': tweet.favorite_count
                    }
                })
        
        return tweet_data
    except Exception as e:
        logger.error(f"Error fetching Twitter news for {symbol}: {e}")
        return []

def _fetch_reddit_news(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Fetch news from Reddit.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        List of Reddit post dictionaries
    """
    try:
        # Ensure timezone-aware datetimes
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        client = _get_reddit_client()
        if client is None:
            # Return sample post if no client available
            return [{
                'datetime': datetime.now(timezone.utc),
                'title': f"Sample Reddit post about {symbol} stock",
                'text': f"Sample discussion about {symbol} stock performance",
                'source': 'reddit_stocks',
                'score': 100,
                'num_comments': 50
            }]
        
        # Search in relevant subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = client.subreddit(subreddit_name)
                search_query = f"{symbol} stock"
                
                for post in subreddit.search(search_query, limit=50):
                    # Convert Reddit timestamp to UTC datetime
                    post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    
                    if start_date <= post_time <= end_date:
                        posts.append({
                            'datetime': post_time,
                            'title': post.title,
                            'text': post.selftext,
                            'source': 'reddit_stocks',
                            'score': post.score,
                            'num_comments': post.num_comments
                        })
            except Exception as e:
                logger.error(f"Error fetching from subreddit {subreddit_name}: {e}")
                continue
        
        return posts
    except Exception as e:
        logger.error(f"Error fetching Reddit news for {symbol}: {e}")
        return []

def _fetch_yfinance_news(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Fetch news from yfinance.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        List of news article dictionaries
    """
    try:
        # Ensure timezone-aware datetimes
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return []
        
        # Filter news by date
        filtered_news = []
        for item in news:
            # Convert timestamp to UTC datetime
            pub_time = datetime.fromtimestamp(item['providerPublishTime'], tz=timezone.utc)
            
            if start_date <= pub_time <= end_date:
                filtered_news.append({
                    'datetime': pub_time,
                    'title': item['title'],
                    'text': item.get('text', ''),
                    'source': 'news',
                    'url': item.get('link', ''),
                    'providerPublishTime': item['providerPublishTime']
                })
        
        return filtered_news
    except Exception as e:
        logger.error(f"Error fetching yfinance news for {symbol}: {e}")
        return []

def _preprocess_text(text: str) -> str:
    """Preprocess text for sentiment analysis.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Handle None or empty input
    if text is None or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Handle common contractions and special cases
    text = text.replace("'s", "s")  # Convert "tesla's" to "teslas"
    text = text.replace("'", "")    # Remove other apostrophes
    
    return text

def _analyze_sentiment(text: str) -> Tuple[float, float]:
    """Analyze sentiment of text using TextBlob.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (sentiment_score, subjectivity)
        sentiment_score ranges from -1 (negative) to 1 (positive)
        subjectivity ranges from 0 (objective) to 1 (subjective)
    """
    try:
        from textblob import TextBlob
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return 0.0, 0.5  # Return neutral sentiment on error

def _calculate_credibility_score(
    source: str,
    metrics: Optional[Dict] = None
) -> float:
    """Calculate credibility score for a news source.
    
    Args:
        source: News source name (e.g., 'yfinance', 'twitter', 'reddit')
        metrics: Optional dictionary of source-specific metrics
        
    Returns:
        Credibility score between 0 and 1
    """
    if metrics is None:
        metrics = {}
    
    # Base score depends on source
    if source == 'yfinance':
        return 1.0  # Most reliable source
    elif source == 'twitter':
        # Calculate score based on Twitter metrics
        followers = metrics.get('followers_count', 0)
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)
        
        # Normalize metrics
        follower_score = min(followers / 10000, 1.0)  # Cap at 10k followers
        engagement_score = min((retweets + likes) / 1000, 1.0)  # Cap at 1k engagement
        
        return 0.5 + (0.3 * follower_score) + (0.2 * engagement_score)
    elif source == 'reddit_stocks':
        # Calculate score based on Reddit metrics
        score = metrics.get('score', 0)
        comments = metrics.get('num_comments', 0)
        
        # Normalize metrics
        score_norm = min(score / 1000, 1.0)  # Cap at 1k score
        comments_norm = min(comments / 100, 1.0)  # Cap at 100 comments
        
        return 0.5 + (0.3 * score_norm) + (0.2 * comments_norm)
    else:
        return 0.5  # Default score for unknown sources

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for financial symbols."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cache_dir: Union[str, Path] = "data/sentiment_cache"
    ):
        """Initialize the news sentiment analyzer.
        
        Args:
            config_path: Path to configuration file
            cache_dir: Directory for caching sentiment data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        self.sentiment_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f).get('sentiment', {})
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def get_sentiment(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_update: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get sentiment data for symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for sentiment analysis
            end_date: End date for sentiment analysis
            force_update: Whether to force update from source
            
        Returns:
            Dict mapping symbols to their sentiment DataFrames
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        data = {}
        for symbol in symbols:
            if force_update or self._needs_update(symbol):
                try:
                    df = self._fetch_and_analyze(symbol, start_date, end_date)
                    data[symbol] = df
                except Exception as e:
                    logger.error(f"Error updating sentiment for {symbol}: {e}")
                    if symbol in self.sentiment_cache:
                        data[symbol] = self.sentiment_cache[symbol]
            else:
                if symbol in self.sentiment_cache:
                    data[symbol] = self.sentiment_cache[symbol]
                else:
                    try:
                        df = self._fetch_and_analyze(symbol, start_date, end_date)
                        data[symbol] = df
                    except Exception as e:
                        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        
        return data
    
    def _needs_update(self, symbol: str) -> bool:
        """Check if sentiment data needs updating.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if update needed, False otherwise
        """
        if symbol not in self.last_update:
            return True
        
        update_interval = self.config.get('update_interval', '1h')
        if update_interval == '1d':
            return datetime.now() - self.last_update[symbol] > timedelta(days=1)
        elif update_interval == '1h':
            return datetime.now() - self.last_update[symbol] > timedelta(hours=1)
        else:
            return True
    
    def _fetch_and_analyze(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch and analyze news sentiment for symbol.
        
        Args:
            symbol: Symbol to analyze
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with sentiment data
        """
        # Fetch news articles (placeholder - implement actual news fetching)
        articles = self._fetch_news(symbol, start_date, end_date)
        
        # Analyze sentiment
        sentiments = []
        for article in articles:
            sentiment = self._analyze_text(article['text'])
            sentiments.append({
                'timestamp': article['timestamp'],
                'title': article['title'],
                'source': article['source'],
                'sentiment_score': sentiment['score'],
                'sentiment_magnitude': sentiment['magnitude'],
                'url': article['url']
            })
        
        # Create DataFrame
        df = pd.DataFrame(sentiments)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        # Cache the data
        self.sentiment_cache[symbol] = df
        self.last_update[symbol] = datetime.now()
        
        # Save to disk
        cache_file = self.cache_dir / f"{symbol}_sentiment.parquet"
        df.to_parquet(cache_file)
        
        return df
    
    def _fetch_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fetch news articles for symbol (placeholder implementation).
        
        Args:
            symbol: Symbol to fetch news for
            start_date: Start date
            end_date: End date
            
        Returns:
            List of article dictionaries
        """
        # This is a placeholder - implement actual news fetching
        # For now, return dummy data
        return [
            {
                'timestamp': datetime.now(),
                'title': f"News about {symbol}",
                'text': f"This is a sample news article about {symbol}.",
                'source': 'sample',
                'url': f"https://example.com/news/{symbol}"
            }
        ]
    
    def _analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment score and magnitude
        """
        # Use TextBlob for basic sentiment analysis
        blob = TextBlob(text)
        
        # Polarity ranges from -1 (negative) to 1 (positive)
        # Subjectivity ranges from 0 (objective) to 1 (subjective)
        return {
            'score': blob.sentiment.polarity,
            'magnitude': blob.sentiment.subjectivity
        }
    
    def get_aggregate_sentiment(
        self,
        symbol: str,
        window: str = '1d'
    ) -> Dict[str, float]:
        """Get aggregate sentiment metrics for symbol.
        
        Args:
            symbol: Symbol to analyze
            window: Time window for aggregation
            
        Returns:
            Dict with aggregate sentiment metrics
        """
        if symbol not in self.sentiment_cache:
            return {
                'mean_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_count': 0
            }
        
        df = self.sentiment_cache[symbol]
        if df.empty:
            return {
                'mean_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_count': 0
            }
        
        # Resample to specified window
        resampled = df['sentiment_score'].resample(window)
        
        return {
            'mean_sentiment': resampled.mean().iloc[-1],
            'sentiment_std': resampled.std().iloc[-1],
            'sentiment_count': len(df)
        }

def get_news_sentiment(
    symbols: Union[str, List[str]],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    sources: Optional[List[str]] = None,
    config_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Get sentiment analysis for news and social media data.
    
    Args:
        symbols: Stock symbol(s) to analyze
        start_date: Start date for data
        end_date: End date for data
        sources: List of data sources to use (e.g., ['news', 'twitter', 'reddit'])
        config_path: Path to configuration file
        
    Returns:
        DataFrame with sentiment analysis results
        
    Raises:
        ValueError: If date range is invalid or sources are invalid
        NewsSentimentError: If no data is available or processing fails
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    
    if not symbols:
        raise NewsSentimentError("No symbols specified")
    
    # Validate sources
    valid_sources = ['news', 'twitter', 'reddit']
    if sources is None:
        sources = valid_sources
    elif not sources:
        raise ValueError("No sources specified")
    else:
        invalid_sources = [s for s in sources if s not in valid_sources]
        if invalid_sources:
            raise ValueError(f"Invalid sources: {invalid_sources}")
    
    # Convert dates to datetime if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Validate date range
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")
    
    # Load config if provided
    config = None
    if config_path:
        config = load_config(config_path)
    
    # Collect data from all sources
    all_data = []
    
    # Fetch news data if requested
    if 'news' in sources:
        for symbol in symbols:
            try:
                news_data = _fetch_yfinance_news(symbol, start_date, end_date)
                for item in news_data:
                    # Combine title and text for sentiment analysis
                    text = f"{item['title']} {item.get('text', '')}"
                    sentiment_score, subjectivity = _analyze_sentiment(_preprocess_text(text))
                    item.update({
                        'source': 'yfinance',
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'subjectivity': subjectivity,
                        'credibility_score': _calculate_credibility_score('yfinance')
                    })
                all_data.extend(news_data)
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
    
    # Handle Twitter data if requested
    if 'twitter' in sources:
        for symbol in symbols:
            try:
                twitter_data = _fetch_twitter_news(symbol, start_date, end_date)
                for item in twitter_data:
                    # Process text for sentiment analysis
                    text = item.get('text', '')
                    sentiment_score, subjectivity = _analyze_sentiment(_preprocess_text(text))
                    item.update({
                        'source': 'twitter',
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'subjectivity': subjectivity,
                        'credibility_score': _calculate_credibility_score('twitter', item.get('metrics', {}))
                    })
                all_data.extend(twitter_data)
            except Exception as e:
                logger.error(f"Error fetching Twitter data for {symbol}: {e}")
    
    # Handle Reddit data if requested
    if 'reddit' in sources:
        for symbol in symbols:
            try:
                reddit_data = _fetch_reddit_news(symbol, start_date, end_date)
                for item in reddit_data:
                    # Process text for sentiment analysis
                    text = f"{item.get('title', '')} {item.get('text', '')}"
                    sentiment_score, subjectivity = _analyze_sentiment(_preprocess_text(text))
                    item.update({
                        'source': 'reddit_stocks',
                        'symbol': symbol,
                        'sentiment_score': sentiment_score,
                        'subjectivity': subjectivity,
                        'credibility_score': _calculate_credibility_score('reddit_stocks', {
                            'score': item.get('score', 0),
                            'num_comments': item.get('num_comments', 0)
                        })
                    })
                all_data.extend(reddit_data)
            except Exception as e:
                logger.error(f"Error fetching Reddit data for {symbol}: {e}")
    
    if not all_data:
        raise NewsSentimentError("No data available from any source")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure required columns exist
    required_columns = ['datetime', 'symbol', 'source', 'title', 'text', 
                       'sentiment_score', 'subjectivity', 'credibility_score']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Calculate weighted sentiment
    df['weighted_sentiment'] = df['sentiment_score'] * df['credibility_score']
    
    # Set datetime as index and sort
    df = df.set_index('datetime')
    df = df.sort_index()
    
    return df

# Initialize sentiment cache
_sentiment_cache = {}

def get_aggregate_sentiment(
    df: pd.DataFrame,
    window: str = '1D'
) -> pd.DataFrame:
    """Aggregate sentiment data over time windows.
    
    Args:
        df: DataFrame with sentiment data
        window: Time window for aggregation (e.g., '1D' for daily, '1H' for hourly)
        
    Returns:
        DataFrame with aggregated sentiment data
        
    Raises:
        NewsSentimentError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise NewsSentimentError("Empty DataFrame provided")
    
    required_cols = ['sentiment_score', 'subjectivity', 'weighted_sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise NewsSentimentError(f"Missing required columns: {missing_cols}")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        else:
            raise NewsSentimentError("DataFrame must have datetime index or column")
    
    # Group by time window and calculate statistics
    agg_df = df.groupby(pd.Grouper(freq=window)).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'subjectivity': 'mean',
        'weighted_sentiment': ['mean', 'std'],
        'credibility_score': 'mean'
    })
    
    # Flatten column names
    agg_df.columns = [
        'mean_sentiment',
        'sentiment_std',
        'count',
        'mean_subjectivity',
        'mean_weighted_sentiment',
        'weighted_sentiment_std',
        'mean_credibility'
    ]
    
    # Add sentiment category based on mean weighted sentiment
    agg_df['sentiment_category'] = pd.cut(
        agg_df['mean_weighted_sentiment'],
        bins=[-float('inf'), -0.1, 0.1, float('inf')],
        labels=['negative', 'neutral', 'positive']
    )
    
    return agg_df

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    sentiment_data = get_news_sentiment(symbols, start_date, end_date)
    
    for symbol, data in sentiment_data.items():
        print(f"\nNews sentiment data for {symbol}:")
        print(data.head()) 