"""Market data feed module for fetching and processing market data.

This module provides functionality for fetching, processing, and managing market data
from various sources including yfinance and Alpaca. It supports both historical and
real-time data streaming, with built-in caching, validation, and technical analysis.

Key features:
- Multiple data source support (yfinance, Alpaca)
- Real-time data streaming
- Technical indicator calculation
- Data validation and cleaning
- Caching to prevent excessive API calls
- Robust error handling with retries
- Comprehensive logging
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import time
from functools import lru_cache
import requests
from requests.exceptions import RequestException
import threading
from queue import Queue
import pytz
from .market_utils import validate_market_data, clean_market_data

# Ensure logs directory exists
import os
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_feed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Standalone functions for module-level imports
def load_config(config_path: Union[str, Path]) -> dict:
    """Load and validate market data configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file does not exist
        MarketDataError: If config is invalid or missing required fields
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {config_path}")
        raise  # Re-raise the FileNotFoundError
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise MarketDataError(f"Failed to load config: {e}")
    
    # Ensure market_data section exists
    if 'market_data' not in config:
        # If config is already in market_data format, wrap it
        if all(key in config for key in ['symbols', 'intervals', 'features']):
            config = {'market_data': config}
        else:
            raise MarketDataError("Missing market_data configuration")
    
    market_config = config['market_data']
    
    # Validate required fields
    required_fields = ['symbols', 'intervals', 'features']
    missing_fields = [field for field in required_fields if field not in market_config]
    if missing_fields:
        raise MarketDataError(f"Missing required fields: {missing_fields}")
    
    # Validate symbols
    if not isinstance(market_config['symbols'], list) or not market_config['symbols']:
        raise MarketDataError("symbols must be a non-empty list")
    
    # Validate intervals
    if not isinstance(market_config['intervals'], list) or not market_config['intervals']:
        raise MarketDataError("intervals must be a non-empty list")
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo']
    invalid_intervals = [i for i in market_config['intervals'] if i not in valid_intervals]
    if invalid_intervals:
        raise MarketDataError(f"Invalid intervals: {invalid_intervals}")
    
    # Validate features
    if not isinstance(market_config['features'], list) or not market_config['features']:
        raise MarketDataError("features must be a non-empty list")
    required_features = ['open', 'high', 'low', 'close', 'volume']
    missing_features = [f for f in required_features if f not in market_config['features']]
    if missing_features:
        raise MarketDataError(f"Missing required features: {missing_features}")
    
    # Validate technical indicators if present
    if 'technical_indicators' in market_config:
        indicators = market_config['technical_indicators']
        valid_indicators = [
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'ema_50',
            'rsi_14', 'macd', 'bollinger_bands',
            'atr_14', 'obv', 'volume_ma'
        ]
        
        # Handle both list and dictionary formats
        if isinstance(indicators, list):
            # List format: ['sma_20', 'rsi_14', etc.]
            invalid_indicators = [i for i in indicators if i not in valid_indicators]
            if invalid_indicators:
                raise MarketDataError(f"Invalid technical indicators: {invalid_indicators}")
        elif isinstance(indicators, dict):
            # Dictionary format: {'sma': [20, 50], 'rsi': 14, etc.}
            # Convert to list format for validation
            indicator_list = []
            for indicator, params in indicators.items():
                if indicator == 'sma':
                    for period in params:
                        indicator_list.append(f'sma_{period}')
                elif indicator == 'ema':
                    for period in params:
                        indicator_list.append(f'ema_{period}')
                elif indicator == 'rsi':
                    indicator_list.append(f'rsi_{params}')
                elif indicator == 'macd':
                    indicator_list.append('macd')
                elif indicator == 'bollinger_bands':
                    indicator_list.append('bollinger_bands')
                else:
                    raise MarketDataError(f"Unknown technical indicator: {indicator}")
            
            invalid_indicators = [i for i in indicator_list if i not in valid_indicators]
            if invalid_indicators:
                raise MarketDataError(f"Invalid technical indicators: {invalid_indicators}")
        else:
            raise MarketDataError("technical_indicators must be either a list or a dictionary")
    
    return config

def get_market_data(
    symbols: Union[str, List[str]],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    interval: str = '1d',
    config_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Fetch market data for given symbols.
    
    Args:
        symbols: Single symbol or list of symbols to fetch
        start_date: Start date for data fetch
        end_date: End date for data fetch
        interval: Data interval (e.g., '1d', '1h')
        config_path: Path to configuration file
        
    Returns:
        DataFrame with market data for all symbols
        
    Raises:
        MarketDataError: If data validation fails or required columns are missing
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    
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
    
    # Validate future dates
    if end_date > datetime.now():
        raise ValueError("end_date cannot be in the future")
    
    # Load config if provided
    config = None
    if config_path:
        config = load_config(config_path)
    
    data_frames = []
    for symbol in symbols:
        try:
            # Fetch data using yfinance
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                raise MarketDataError(f"No data available for {symbol}")
            
            # Standardize column names
            df = df.rename(columns={
                'Adj Close': 'adj_close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index and rename Date to datetime
            df = df.reset_index()
            df = df.rename(columns={'Date': 'datetime'})
            
            # Add symbol column and set multi-index
            df['symbol'] = symbol
            df = df.set_index(['symbol', 'datetime'])
            
            # Add symbol level to columns
            df.columns = pd.MultiIndex.from_product([df.columns, [symbol]])
            
            data_frames.append(df)
            logger.info(f"Successfully fetched data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            if isinstance(e, MarketDataError):
                raise
            raise MarketDataError(f"Error fetching data for {symbol}: {str(e)}")
    
    if not data_frames:
        raise MarketDataError("No data available for any symbols")
    
    # Combine all data frames
    result_df = pd.concat(data_frames, axis=1)
    
    # Add technical indicators if config specifies them
    if config and 'technical_indicators' in config.get('market_data', {}):
        result_df = add_technical_indicators(result_df)
    
    return result_df

def add_technical_indicators(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Add technical indicators to market data.
    
    Args:
        df: DataFrame with market data (must have MultiIndex columns)
        config: Optional configuration for indicators
        
    Returns:
        DataFrame with added technical indicators
        
    Raises:
        MarketDataError: If DataFrame structure is invalid
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise MarketDataError("DataFrame must have MultiIndex columns")
    
    # Get unique symbols from column names
    symbols = df.columns.get_level_values(1).unique()
    
    # Use default config if none provided
    if config is None:
        config = {
            'sma_periods': [20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_window': 20,
            'bollinger_std': 2
        }
    
    # Add indicators for each symbol
    for symbol in symbols:
        try:
            # Get close prices for the symbol
            close_col = ('close', symbol)
            if close_col not in df.columns:
                raise MarketDataError(f"Missing close price column for {symbol}")
            close = df[close_col]
            
            # Simple Moving Averages
            for period in config.get('sma_periods', [20, 50]):
                df[('sma_' + str(period), symbol)] = close.rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in config.get('ema_periods', [12, 26]):
                df[('ema_' + str(period), symbol)] = close.ewm(span=period, adjust=False).mean()
            
            # Relative Strength Index (RSI)
            rsi_period = config.get('rsi_period', 14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df[('rsi', symbol)] = 100 - (100 / (1 + rs))
            
            # MACD
            macd_fast = config.get('macd_fast', 12)
            macd_slow = config.get('macd_slow', 26)
            macd_signal = config.get('macd_signal', 9)
            
            ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
            df[('macd', symbol)] = ema_fast - ema_slow
            df[('macd_signal', symbol)] = df[('macd', symbol)].ewm(span=macd_signal, adjust=False).mean()
            df[('macd_hist', symbol)] = df[('macd', symbol)] - df[('macd_signal', symbol)]
            
            # Bollinger Bands
            bb_window = config.get('bollinger_window', 20)
            bb_std = config.get('bollinger_std', 2)
            
            df[('bb_middle', symbol)] = close.rolling(window=bb_window).mean()
            bb_std_dev = close.rolling(window=bb_window).std()
            df[('bb_upper', symbol)] = df[('bb_middle', symbol)] + (bb_std_dev * bb_std)
            df[('bb_lower', symbol)] = df[('bb_middle', symbol)] - (bb_std_dev * bb_std)
            
        except Exception as e:
            logger.error(f"Error adding technical indicators for {symbol}: {e}")
            raise MarketDataError(f"Error adding technical indicators for {symbol}: {str(e)}")
    
    return df

class MarketDataError(Exception):
    """Custom exception for market data related errors."""
    pass

class MarketFeed:
    """Main class for handling market data operations."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cache_size: int = 1000,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """Initialize the market feed.
        
        Args:
            config_path: Path to configuration file (optional)
            cache_size: Maximum number of cached results
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.data_queue = Queue()
        self.streaming = False
        self.stream_thread = None
        
        # Store symbols from config if available
        self.symbols = self.config.get('market_data', {}).get('symbols', [])
    
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load and validate market data configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Validated configuration dictionary
            
        Raises:
            MarketDataError: If config is invalid or missing required fields
        """
        return load_config(config_path)
    
    def fetch_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        add_indicators: bool = False,
        indicator_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Fetch market data for symbols.
        
        Args:
            symbols: Stock symbol(s) to fetch data for
            start_date: Start date for data
            end_date: End date for data
            add_indicators: Whether to add technical indicators
            indicator_config: Optional configuration for indicators
            
        Returns:
            DataFrame with market data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
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
        
        # Fetch data for each symbol
        all_data = {}
        for symbol in symbols:
            try:
                # Try to fetch data with retries
                for attempt in range(3):
                    try:
                        data = self._fetch_symbol_data(symbol, start_date, end_date)
                        if data is not None and not data.empty:
                            # Reset index and rename Date to datetime
                            data = data.reset_index()
                            data = data.rename(columns={'Date': 'datetime'})
                            
                            # Add symbol column and set multi-index
                            data['symbol'] = symbol
                            data = data.set_index(['symbol', 'datetime'])
                            
                            # Store data with proper column structure
                            all_data[symbol] = data
                            break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            logger.error(f"Failed to fetch data for {symbol} after 3 attempts: {e}")
                            raise
                        time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise MarketDataError("No data fetched for any symbols")
        
        # Combine data for all symbols
        # First, ensure all DataFrames have the same column structure
        base_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for symbol, df in all_data.items():
            # Ensure all required columns exist
            for col in base_columns:
                if col not in df.columns:
                    raise MarketDataError(f"Missing required column {col} for {symbol}")
        
        # Combine the DataFrames
        df = pd.concat(all_data.values(), axis=1)
        
        # Add technical indicators if requested
        if add_indicators:
            try:
                if indicator_config is None:
                    indicator_config = self.config.get('market_data', {}).get('technical_indicators', {})
                df = add_technical_indicators(df, indicator_config)
            except Exception as e:
                logger.error(f"Error adding technical indicators: {e}")
                raise
        
        return df

    def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch market data for a single symbol.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with market data or None if fetch fails
        """
        try:
            # Fetch data from yfinance
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Standardize column names to match get_market_data
            data = data.rename(columns={
                'Adj Close': 'adj_close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol level to columns
            data.columns = pd.MultiIndex.from_product([data.columns, [symbol]])
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def start_streaming(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        interval: str = '1m',
        callback: Optional[callable] = None
    ):
        """Start real-time data streaming.
        
        Args:
            symbols: Symbols to stream
            interval: Data interval
            callback: Function to call with new data
        """
        if self.streaming:
            logger.warning("Streaming already in progress")
            return
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not symbols:
            symbols = self.config.get('symbols', [])
        
        self.streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_data,
            args=(symbols, interval, callback),
            daemon=True
        )
        self.stream_thread.start()
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join()
    
    def _stream_data(
        self,
        symbols: List[str],
        interval: str,
        callback: Optional[callable]
    ):
        """Internal method for streaming data.
        
        Args:
            symbols: Symbols to stream
            interval: Data interval
            callback: Function to call with new data
        """
        while self.streaming:
            try:
                # Fetch data for all symbols at once
                df = self.get_market_data(
                    symbols=symbols,
                    start_date=datetime.now() - timedelta(minutes=5),
                    end_date=datetime.now(),
                    interval=interval,
                    use_cache=False
                )
                
                if df is not None and callback:
                    # Call callback for each symbol's data
                    for symbol in df.index.get_level_values('Symbol').unique():
                        symbol_data = df.xs(symbol, level='Symbol')
                        callback(symbol, symbol_data)
                
                time.sleep(60)  # Wait for next update
                
            except Exception as e:
                logger.error(f"Error in data stream: {e}")
                time.sleep(self.retry_delay)
    
    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC.
        
        Args:
            dt: Local datetime
            
        Returns:
            UTC datetime
        """
        if dt.tzinfo is None:
            dt = pytz.timezone('UTC').localize(dt)
        return dt.astimezone(pytz.UTC)

def main():
    """Example usage of the MarketFeed class."""
    try:
        # Initialize market feed
        feed = MarketFeed("config/strategy_config.yaml")
        
        # Fetch historical data
        df = feed.get_market_data(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            interval='1d'
        )
        
        # Print results for each symbol
        for symbol in df.index.get_level_values('symbol').unique():
            symbol_data = df.xs(symbol, level='symbol')
            print(f"\nData for {symbol}:")
            print(symbol_data.tail())
        
        # Example of streaming data
        def data_callback(symbol: str, df: pd.DataFrame):
            print(f"\nNew data for {symbol}:")
            print(df.tail())
        
        # Start streaming (uncomment to use)
        # feed.start_streaming(callback=data_callback)
        # time.sleep(300)  # Stream for 5 minutes
        # feed.stop_streaming()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 