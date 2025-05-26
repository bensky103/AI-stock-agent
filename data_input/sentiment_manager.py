"""Sentiment data management for stock prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from textblob import TextBlob
import nltk
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import yaml
import tweepy
import praw
import pytz

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentError(Exception):
    """Exception raised for sentiment analysis related errors."""
    pass

class SentimentManager:
    """Manages sentiment analysis from multiple sources."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize sentiment manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.sources = {}
        
        # Initialize Twitter source if enabled
        if self.config.get('sentiment', {}).get('sources', {}).get('twitter', {}).get('enabled', False):
            try:
                self.sources['twitter'] = TwitterSource()
            except Exception as e:
                logger.error(f"Failed to initialize Twitter source: {e}")
                self.sources['twitter'] = None
        
        # Initialize Reddit source if enabled
        if self.config.get('sentiment', {}).get('sources', {}).get('reddit', {}).get('enabled', False):
            try:
                self.sources['reddit'] = RedditSource()
            except Exception as e:
                logger.error(f"Failed to initialize Reddit source: {e}")
                self.sources['reddit'] = None
    
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_sentiment_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get sentiment data from all available sources.
        
        Args:
            symbols: Stock symbol(s) to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            sources: List of sources to use (twitter, reddit)
            
        Returns:
            DataFrame with sentiment data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not sources:
            sources = list(self.sources.keys())
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        all_data = []
        
        for symbol in symbols:
            # Get data from each source
            for source_name in sources:
                source = self.sources.get(source_name)
                if source is None:
                    continue
                
                try:
                    if source_name == 'twitter':
                        df = source.fetch_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date
                        )
                    elif source_name == 'reddit':
                        df = source.fetch_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            subreddits=self.config.get('sentiment', {}).get('sources', {}).get('reddit', {}).get('subreddits', ['wallstreetbets'])
                        )
                    else:
                        continue
                    
                    if not df.empty:
                        all_data.append(df)
                
                except Exception as e:
                    logger.error(f"Error fetching {source_name} data for {symbol}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Aggregate sentiment scores
        return self._aggregate_sentiment(combined_df)
    
    def _aggregate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment scores from multiple sources.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        if df.empty:
            return df
        
        # Get aggregation settings from config
        agg_config = self.config.get('sentiment', {}).get('aggregation', {})
        window = agg_config.get('window', '1d')
        min_sources = agg_config.get('min_sources', 1)
        weights = {
            'twitter': agg_config.get('weight_twitter', 0.6),
            'reddit': agg_config.get('weight_reddit', 0.4)
        }
        
        # Ensure datetime column is datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Group by symbol and time window
        grouped = df.groupby(['symbol', pd.Grouper(key='datetime', freq=window)])
        
        # Aggregate sentiment scores
        agg_data = []
        for (symbol, time_group), group_df in grouped:
            # Count unique sources
            source_count = group_df['source'].nunique()
            if source_count < min_sources:
                continue
            
            # Calculate weighted sentiment
            weighted_sentiment = 0
            total_weight = 0
            
            for source, group in group_df.groupby('source'):
                weight = weights.get(source, 1.0)
                weighted_sentiment += group['sentiment_score'].mean() * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_sentiment /= total_weight
            
            agg_data.append({
                'symbol': symbol,
                'datetime': time_group,
                'sentiment_score': weighted_sentiment,
                'source_count': source_count,
                'engagement_score': group_df['engagement_score'].mean() if 'engagement_score' in group_df.columns else 0.0
            })
        
        return pd.DataFrame(agg_data)
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if source is within rate limits.
        
        Args:
            source: Source name (twitter, reddit)
            
        Returns:
            bool: True if within limits, False otherwise
        """
        if source not in self.sources:
            return False
        
        return self.sources[source]._check_rate_limit()
    
    def _update_rate_limit(self, source: str) -> None:
        """Update rate limit counter for source.
        
        Args:
            source: Source name (twitter, reddit)
        """
        if source in self.sources:
            self.sources[source]._update_rate_limit()

class TwitterSource:
    """Twitter data source for sentiment analysis."""
    
    def __init__(self):
        """Initialize Twitter source."""
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise SentimentError("Twitter API credentials not found")
        
        self.client = self._initialize_client()
        self.rate_limit = (450, 900)  # 450 requests per 15 minutes
    
    def _initialize_client(self):
        """Initialize Twitter API client."""
        try:
            auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            return tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            return None
    
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch Twitter data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with Twitter data
        """
        if not self.client:
            # Return sample data if client not available
            return pd.DataFrame([{
                'datetime': datetime.now(pytz.UTC),
                'text': f"Sample tweet about ${symbol} stock",
                'sentiment_score': 0.5,
                'engagement_score': 0.7
            }])
        
        try:
            # Search for tweets
            query = f"${symbol} stock"
            tweets = self.client.search_tweets(
                q=query,
                lang='en',
                count=100,
                tweet_mode='extended'
            )
            
            # Process tweets
            tweet_data = []
            for tweet in tweets:
                if start_date <= tweet.created_at <= end_date:
                    # Calculate engagement score
                    engagement = (
                        tweet.retweet_count * 0.4 +
                        tweet.favorite_count * 0.4 +
                        (1 if tweet.user.verified else 0) * 0.2
                    ) / 1000  # Normalize to 0-1 range
                    
                    tweet_data.append({
                        'datetime': tweet.created_at,
                        'text': tweet.full_text,
                        'sentiment_score': 0.5,  # Placeholder, would use sentiment analysis
                        'engagement_score': min(engagement, 1.0)
                    })
            
            return pd.DataFrame(tweet_data)
        
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
            return pd.DataFrame()

class RedditSource:
    """Reddit data source for sentiment analysis."""
    
    def __init__(self):
        """Initialize Reddit source."""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT')
        
        if not all([self.client_id, self.client_secret, self.user_agent]):
            raise SentimentError("Reddit API credentials not found")
        
        self.client = self._initialize_client()
        self.rate_limit = (60, 60)  # 60 requests per minute
    
    def _initialize_client(self):
        """Initialize Reddit API client."""
        try:
            return praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            return None
    
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        subreddits: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch Reddit data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            subreddits: List of subreddits to search (optional)
            
        Returns:
            DataFrame with Reddit data
        """
        if not self.client:
            # Return sample data if client not available
            return pd.DataFrame([{
                'datetime': datetime.now(pytz.UTC),
                'text': f"Sample Reddit post about {symbol} stock",
                'sentiment_score': 0.5,
                'engagement_score': 0.7
            }])
        
        if not subreddits:
            subreddits = ['wallstreetbets', 'stocks', 'investing']
        
        try:
            posts = []
            for subreddit_name in subreddits:
                subreddit = self.client.subreddit(subreddit_name)
                search_query = f"{symbol} stock"
                
                for post in subreddit.search(search_query, limit=50):
                    post_time = datetime.fromtimestamp(post.created_utc, tz=pytz.UTC)
                    if start_date <= post_time <= end_date:
                        # Calculate engagement score
                        engagement = (
                            post.score * 0.6 +
                            post.num_comments * 0.4
                        ) / 1000  # Normalize to 0-1 range
                        
                        posts.append({
                            'datetime': post_time,
                            'text': f"{post.title}\n{post.selftext}",
                            'sentiment_score': 0.5,  # Placeholder, would use sentiment analysis
                            'engagement_score': min(engagement, 1.0)
                        })
            
            return pd.DataFrame(posts)
        
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return pd.DataFrame()

def get_sentiment_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    window_size: int = 5
) -> Dict[str, pd.DataFrame]:
    """Get sentiment data for the given symbols and date range.
    
    This is a convenience function that creates a SentimentManager instance
    and calls its get_sentiment_data method.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        window_size: Size of the rolling window for sentiment aggregation
        
    Returns:
        Dictionary mapping symbols to sentiment DataFrames
    """
    manager = SentimentManager()
    return manager.get_sentiment_data(symbols, start_date, end_date, window_size)

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    sentiment_data = get_sentiment_data(symbols, start_date, end_date)
    
    for symbol, data in sentiment_data.items():
        print(f"\nSentiment data for {symbol}:")
        print(data.head()) 