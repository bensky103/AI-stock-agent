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
        followers = metrics.get('followers_count', 0)
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)
        
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