<<<<<<< HEAD
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
=======
import logging
from typing import List, Dict, Optional, Union
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
import tweepy
import praw
import re
from functools import lru_cache
import json
from pathlib import Path
import time

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Source credibility weights (can be adjusted based on domain expertise)
SOURCE_WEIGHTS = {
    'yfinance': 1.0,
    'twitter': 0.7,
    'reddit_stocks': 0.8,
    'reddit_investing': 0.8,
    'reddit_wallstreetbets': 0.6
}

class NewsSentimentError(Exception):
    """Custom exception for news sentiment related errors."""
    pass

def _preprocess_text(text: str) -> str:
    """
    Preprocess text for better sentiment analysis.
    
    Parameters
    ----------
    text : str
        Input text to preprocess
        
    Returns
    -------
    str
        Preprocessed text
    """
    if not isinstance(text, str):
>>>>>>> e719f64 (news_sentiment, support for REDDIT+TWITTER news + tests)
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
    
<<<<<<< HEAD
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
=======
    return text

def _calculate_credibility_score(source: str, metrics: Dict) -> float:
    """
    Calculate credibility score based on source and metrics.
    
    Parameters
    ----------
    source : str
        News source
    metrics : Dict
        Source-specific metrics (e.g., likes, comments, etc.)
        
    Returns
    -------
    float
        Credibility score between 0 and 1
    """
    base_score = SOURCE_WEIGHTS.get(source, 0.5)
    
    if source == 'twitter':
        # Consider engagement metrics
>>>>>>> e719f64 (news_sentiment, support for REDDIT+TWITTER news + tests)
        followers = metrics.get('followers_count', 0)
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)
        
<<<<<<< HEAD
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
=======
        engagement_score = min(1.0, (retweets + likes) / 1000)
        follower_score = min(1.0, followers / 10000)
        
        return base_score * (0.7 * engagement_score + 0.3 * follower_score)
    
    elif source.startswith('reddit_'):
        # Consider Reddit-specific metrics
        score = metrics.get('score', 0)
        num_comments = metrics.get('num_comments', 0)
        
        engagement_score = min(1.0, (score + num_comments) / 1000)
        return base_score * engagement_score
    
    return base_score

@lru_cache(maxsize=1000)
def _get_cached_sentiment(text: str) -> Dict:
    """
    Get cached sentiment scores for text to improve performance.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    Dict
        Dictionary containing sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    text_blob = TextBlob(text)
    
    return {
        'vader': sia.polarity_scores(text),
        'textblob': {
            'polarity': text_blob.sentiment.polarity,
            'subjectivity': text_blob.sentiment.subjectivity
        }
    }

def _get_twitter_client() -> Optional[tweepy.Client]:
    """Initialize and return Twitter API client."""
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Twitter client: {str(e)}")
        return None

def _get_reddit_client() -> Optional[praw.Reddit]:
    """Initialize and return Reddit API client."""
    try:
        client = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'stock_sentiment_analyzer')
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Reddit client: {str(e)}")
        return None

def _fetch_twitter_news(symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Fetch relevant tweets for a given symbol."""
    tweets = []
    client = _get_twitter_client()
    
    if not client:
        return tweets
    
    try:
        # Search for tweets containing the symbol
        query = f"${symbol} OR #{symbol} -is:retweet"
        response = client.search_recent_tweets(
            query=query,
            max_results=100,
            start_time=start_date,
            end_time=end_date,
            tweet_fields=['created_at', 'text', 'public_metrics']
        )
        
        if response.data:
            for tweet in response.data:
                tweets.append({
                    'symbol': symbol,
                    'datetime': tweet.created_at,
                    'source': 'twitter',
                    'title': tweet.text[:100],  # Use first 100 chars as title
                    'text': tweet.text,
                    'metrics': tweet.public_metrics
                })
    except Exception as e:
        logger.error(f"Error fetching Twitter data for {symbol}: {str(e)}")
    
    return tweets

def _fetch_reddit_news(symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Fetch relevant Reddit posts for a given symbol."""
    posts = []
    client = _get_reddit_client()
    
    if not client:
        return posts
    
    try:
        # Search in relevant subreddits
        subreddits = ['stocks', 'investing', 'wallstreetbets']
        for subreddit_name in subreddits:
            subreddit = client.subreddit(subreddit_name)
            search_query = f"{symbol} OR ${symbol}"
            
            for post in subreddit.search(search_query, limit=50, time_filter='week'):
                post_date = datetime.fromtimestamp(post.created_utc, tz=pytz.UTC)
                if start_date <= post_date <= end_date:
                    posts.append({
                        'symbol': symbol,
                        'datetime': post_date,
                        'source': f'reddit_{subreddit_name}',
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments
                    })
    except Exception as e:
        logger.error(f"Error fetching Reddit data for {symbol}: {str(e)}")
    
    return posts

def get_news_sentiment(
    symbols: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sources: Optional[List[str]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch and analyze news sentiment for given symbols.

    Parameters
    ----------
    symbols : str or list of str
        Stock symbol(s) to fetch news for
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 7 days ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    sources : list of str, optional
        News sources to fetch from. Defaults to ['news', 'twitter', 'reddit']
    use_cache : bool, optional
        Whether to use sentiment caching. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentiment scores and metadata
    """
    try:
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now(pytz.UTC)
        else:
            end_date = pd.to_datetime(end_date, utc=True)

        if start_date is None:
            start_date = end_date - timedelta(days=7)
        else:
            start_date = pd.to_datetime(start_date, utc=True)

        # Set default sources
        if sources is None:
            sources = ['news', 'twitter', 'reddit']

        # Collect news from all sources
        all_news = []
        for symbol in symbols:
            logger.info(f"Fetching news for {symbol}")
            
            # Fetch news from yfinance
            if 'news' in sources:
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    if news:
                        for article in news:
                            # Preprocess text
                            title = _preprocess_text(article.get('title', ''))
                            
                            # Get sentiment scores
                            if use_cache:
                                sentiment = _get_cached_sentiment(title)
                                vader_score = sentiment['vader']['compound']
                                textblob_scores = sentiment['textblob']
                            else:
                                sia = SentimentIntensityAnalyzer()
                                text_blob = TextBlob(title)
                                vader_score = sia.polarity_scores(title)['compound']
                                textblob_scores = {
                                    'polarity': text_blob.sentiment.polarity,
                                    'subjectivity': text_blob.sentiment.subjectivity
                                }
                            
                            # Calculate credibility score
                            credibility = _calculate_credibility_score('yfinance', {})
                            
                            all_news.append({
                                'symbol': symbol,
                                'datetime': pd.to_datetime(article.get('providerPublishTime', 0), unit='s', utc=True),
                                'source': 'yfinance',
                                'title': article.get('title', ''),
                                'text': article.get('link', ''),
                                'sentiment_score': vader_score,
                                'subjectivity': textblob_scores['subjectivity'],
                                'polarity': textblob_scores['polarity'],
                                'credibility_score': credibility
                            })
                except Exception as e:
                    logger.error(f"Error fetching yfinance news for {symbol}: {str(e)}")

            # Fetch Twitter data
            if 'twitter' in sources:
                twitter_news = _fetch_twitter_news(symbol, start_date, end_date)
                for tweet in twitter_news:
                    # Preprocess text
                    text = _preprocess_text(tweet['text'])
                    
                    # Get sentiment scores
                    if use_cache:
                        sentiment = _get_cached_sentiment(text)
                        vader_score = sentiment['vader']['compound']
                        textblob_scores = sentiment['textblob']
                    else:
                        sia = SentimentIntensityAnalyzer()
                        text_blob = TextBlob(text)
                        vader_score = sia.polarity_scores(text)['compound']
                        textblob_scores = {
                            'polarity': text_blob.sentiment.polarity,
                            'subjectivity': text_blob.sentiment.subjectivity
                        }
                    
                    # Calculate credibility score
                    credibility = _calculate_credibility_score('twitter', tweet.get('metrics', {}))
                    
                    all_news.append({
                        'symbol': symbol,
                        'datetime': tweet['datetime'],
                        'source': tweet['source'],
                        'title': tweet['title'],
                        'text': tweet['text'],
                        'sentiment_score': vader_score,
                        'subjectivity': textblob_scores['subjectivity'],
                        'polarity': textblob_scores['polarity'],
                        'credibility_score': credibility,
                        'metrics': tweet.get('metrics', {})
                    })

            # Fetch Reddit data
            if 'reddit' in sources:
                reddit_news = _fetch_reddit_news(symbol, start_date, end_date)
                for post in reddit_news:
                    # Preprocess text
                    title = _preprocess_text(post['title'])
                    text = _preprocess_text(post['text'])
                    
                    # Get sentiment scores
                    if use_cache:
                        title_sentiment = _get_cached_sentiment(title)
                        text_sentiment = _get_cached_sentiment(text)
                        vader_score = (title_sentiment['vader']['compound'] + text_sentiment['vader']['compound']) / 2
                        textblob_scores = {
                            'polarity': (title_sentiment['textblob']['polarity'] + text_sentiment['textblob']['polarity']) / 2,
                            'subjectivity': (title_sentiment['textblob']['subjectivity'] + text_sentiment['textblob']['subjectivity']) / 2
                        }
                    else:
                        sia = SentimentIntensityAnalyzer()
                        title_blob = TextBlob(title)
                        text_blob = TextBlob(text)
                        vader_score = (sia.polarity_scores(title)['compound'] + sia.polarity_scores(text)['compound']) / 2
                        textblob_scores = {
                            'polarity': (title_blob.sentiment.polarity + text_blob.sentiment.polarity) / 2,
                            'subjectivity': (title_blob.sentiment.subjectivity + text_blob.sentiment.subjectivity) / 2
                        }
                    
                    # Calculate credibility score
                    metrics = {
                        'score': post['score'],
                        'num_comments': post['num_comments']
                    }
                    credibility = _calculate_credibility_score(post['source'], metrics)
                    
                    all_news.append({
                        'symbol': symbol,
                        'datetime': post['datetime'],
                        'source': post['source'],
                        'title': post['title'],
                        'text': post['text'],
                        'sentiment_score': vader_score,
                        'subjectivity': textblob_scores['subjectivity'],
                        'polarity': textblob_scores['polarity'],
                        'credibility_score': credibility,
                        'score': post['score'],
                        'num_comments': post['num_comments']
                    })

        if not all_news:
            logger.warning("No news found for any symbol")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        
        # Filter by date range
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        # Sort by datetime
        df.sort_values('datetime', inplace=True)
        
        # Calculate weighted sentiment scores
        df['weighted_sentiment'] = df['sentiment_score'] * df['credibility_score']
        
        # Calculate aggregate sentiment metrics
        df['sentiment_category'] = pd.cut(
            df['weighted_sentiment'],
            bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
            labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        )

        logger.info(f"Successfully processed {len(df)} news items")
        return df

    except Exception as e:
        logger.exception("Failed to fetch news sentiment")
        raise NewsSentimentError(f"News sentiment analysis failed: {str(e)}")

def get_aggregate_sentiment(
    sentiment_df: pd.DataFrame,
    window: str = '1D'
) -> pd.DataFrame:
    """
    Calculate aggregate sentiment metrics over time.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        DataFrame from get_news_sentiment
    window : str
        Time window for aggregation (e.g., '1D', '1H', '1W')

    Returns
    -------
    pd.DataFrame
        Aggregated sentiment metrics:
        - mean_sentiment: Average sentiment score
        - sentiment_std: Standard deviation of sentiment
        - mean_subjectivity: Average subjectivity
        - count: Number of news items
    """
    try:
        # Group by symbol and time window
        grouped = sentiment_df.groupby(['symbol', pd.Grouper(key='datetime', freq=window)])
        
        # Calculate aggregate metrics
        agg_df = grouped.agg({
            'sentiment_score': ['mean', 'std'],
            'subjectivity': 'mean',
            'title': 'count'
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['symbol', 'datetime', 'mean_sentiment', 'sentiment_std', 
                         'mean_subjectivity', 'count']
        
        return agg_df

    except Exception as e:
        logger.error(f"Error calculating aggregate sentiment: {str(e)}")
        raise NewsSentimentError(f"Aggregate sentiment calculation failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        # Get news sentiment for Tesla
        df = get_news_sentiment(
            symbols="TSLA",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Calculate daily aggregate sentiment
        agg_df = get_aggregate_sentiment(df, window='1D')
        
        print("\nNews Sentiment Summary:")
        print(f"Total news items: {len(df)}")
        print("\nSentiment Distribution:")
        print(df['sentiment_category'].value_counts())
        
        print("\nDaily Aggregate Sentiment:")
        print(agg_df.head())
        
    except Exception as ex:
        logger.error(f"Error in main: {ex}") 
>>>>>>> e719f64 (news_sentiment, support for REDDIT+TWITTER news + tests)
