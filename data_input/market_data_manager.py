"""Market data manager for handling data caching and updates."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import yaml
import time
import os
from .market_utils import validate_market_data, MarketDataError, resample_market_data
import yfinance as yf
import pytz

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_data_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set yfinance logger to WARNING
logging.getLogger('yfinance').setLevel(logging.WARNING)

# Define what should be exported from this module
__all__ = ['MarketDataManager', 'MarketDataError', 'YFinanceSource']

class YFinanceSource:
    """Yahoo Finance data source for market data."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """Initialize Yahoo Finance data source.
        
        Args:
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit = (2000, 3600)  # 2000 requests per hour
        self.last_request_time = 0
        self.request_count = 0
        logger.info("Initialized YFinanceSource")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits.
        
        Returns:
            bool: True if within rate limits, False if rate limit exceeded
        """
        current_time = time.time()
        requests_per_hour, time_window = self.rate_limit
        
        # Reset request count if time window has passed
        if current_time - self.last_request_time > time_window:
            self.request_count = 0
            self.last_request_time = current_time
            return True
        
        # Check if we've exceeded the rate limit
        return self.request_count < requests_per_hour
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exceeded."""
        if not self._check_rate_limit():
            sleep_time = self.rate_limit[1] - (time.time() - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
    
    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get market data for symbol from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with market data
            
        Raises:
            MarketDataError: If data validation fails or no data is available
        """
        self._wait_for_rate_limit()  # Wait if needed
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Use yfinance to fetch real data
        ticker = yf.Ticker(symbol)
        try:
            # Fetch data with auto_adjust=False to get raw data
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                raise MarketDataError(f"No data available for {symbol}")
            
            # Log original timestamps
            logger.info(f"Original timestamps from Ticker.history for {symbol} (first 3):")
            logger.info(f"{df.index[:3]}")
            
            # Log original columns
            logger.info(f"Original columns from yfinance: {df.columns.tolist()}")
            
            # Convert column names to lowercase for consistent handling
            df.columns = df.columns.str.lower()
            logger.info(f"Columns after lowercase conversion: {df.columns.tolist()}")
            
            # Map required columns (case-insensitive)
            required_cols = {
                'open': ['open'],
                'high': ['high'],
                'low': ['low'],
                'close': ['close'],
                'volume': ['volume']
            }
            
            # Check for required columns (case-insensitive)
            missing_cols = []
            for req_col, possible_names in required_cols.items():
                if not any(name in df.columns for name in possible_names):
                    missing_cols.append(req_col)
                    logger.error(f"Missing column {req_col}. Available columns: {df.columns.tolist()}")
            
            if missing_cols:
                raise MarketDataError(f"Missing required columns from yfinance: {missing_cols}")
            
            # Ensure we have all required columns with correct names
            df = df.rename(columns={
                col: req_col for req_col, possible_names in required_cols.items()
                for col in possible_names if col in df.columns
            })
            logger.info(f"Final columns after renaming: {df.columns.tolist()}")
            
            # Log final timestamps
            logger.info(f"Final timestamps for {symbol} (first 3):")
            logger.info(f"{df.index[:3]}")
            
            # Validate raw data immediately after fetching
            try:
                validate_market_data(
                    df,
                    min_rows=1,  # Allow any number of rows for initial validation
                    require_indicators=False  # Don't require indicators for raw data
                )
            except MarketDataError as e:
                logger.error(f"Validation failed for {symbol}: {str(e)}")
                raise MarketDataError(f"Invalid data for {symbol}: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {symbol}: {str(e)}")
            raise MarketDataError(f"Error fetching data for {symbol}: {str(e)}")

class MarketDataManager:
    """Manages market data fetching, caching, and updates."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache",
        config_path: Optional[Union[str, Path]] = None
    ):
        """Initialize the market data manager.
        
        Args:
            cache_dir: Directory for caching market data
            config_path: Path to configuration file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.source = YFinanceSource()
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config  # Store full config
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def get_market_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = '1d',
        use_cache: bool = True,
        add_indicators: bool = True
    ) -> pd.DataFrame:
        """Fetch market data for given symbols.
        
        Args:
            symbols: Single symbol or list of symbols to fetch
            start_date: Start date for data fetch
            end_date: End date for data fetch
            interval: Data interval (e.g., '1d', '1h', '1W')
            use_cache: Whether to use cached results
            add_indicators: Whether to add technical indicators
            
        Returns:
            DataFrame with market data for all symbols
            
        Raises:
            MarketDataError: If data validation fails or required columns are missing
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not symbols:
            raise MarketDataError("No symbols provided")
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now(pytz.UTC) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(pytz.UTC) - timedelta(days=1)  # Use UTC and subtract one day
        
        # Convert to UTC if not already
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)
        
        # Validate date range
        if end_date <= start_date:
            raise ValueError("end_date must be after start_date")
        
        # Validate future dates
        if end_date > datetime.now(pytz.UTC):
            raise ValueError("end_date cannot be in the future")
        
        data_frames = []
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                if use_cache and cache_key in self.data_cache:
                    df = self.data_cache[cache_key]
                    logger.info(f"Using cached data for {symbol}")
                else:
                    # Fetch data using yfinance
                    df = self.source.get_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df.empty:
                        raise MarketDataError(f"No data available for {symbol}")
                    
                    # Validate raw data immediately after fetching
                    try:
                        validate_market_data(
                            df,
                            min_rows=1,  # Allow any number of rows for initial validation
                            require_indicators=False  # Don't require indicators for raw data
                        )
                    except MarketDataError as e:
                        logger.error(f"Validation failed for {symbol}: {str(e)}")
                        raise MarketDataError(f"Invalid data for {symbol}: {str(e)}")
                    
                    # Add technical indicators if requested
                    if add_indicators:
                        df = self._add_technical_indicators(df)
                        # Validate again with indicators
                        try:
                            validate_market_data(
                                df,
                                min_rows=1,
                                require_indicators=True
                            )
                        except MarketDataError as e:
                            logger.error(f"Technical indicator validation failed for {symbol}: {str(e)}")
                            raise MarketDataError(f"Invalid technical indicators for {symbol}: {str(e)}")
                    
                    # Cache the data if valid
                    if use_cache:
                        self.data_cache[cache_key] = df
                        self.last_update[symbol] = datetime.now()
                
                # Ensure datetime index is properly set before resampling
                if not isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.rename(columns={'Date': 'datetime'})
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime')
                
                # Resample data if needed
                if interval != '1d':  # Only resample if not daily
                    df = resample_market_data(df, interval)
                
                # Add symbol column and set multi-index after resampling
                df = df.reset_index()
                
                # Check for datetime column name - it might be 'index', 'date', 'Date', or 'datetime'
                datetime_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'datetime', 'index', 'time', 'timestamp']:
                        datetime_col = col
                        break
                
                # If no datetime column found, the index itself might be the datetime
                if datetime_col is None:
                    df['datetime'] = df.index
                else:
                    # Rename to standardized 'datetime' column
                    df = df.rename(columns={datetime_col: 'datetime'})
                
                # Ensure datetime column is actually a datetime type
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['symbol'] = symbol
                
                # Now set the multi-index with the standardized column names
                df = df.set_index(['symbol', 'datetime'])
                df.index.names = ['symbol', 'datetime']  # Explicitly set index names
                
                data_frames.append(df)
                logger.info(f"Successfully fetched data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                if isinstance(e, MarketDataError):
                    raise
                raise MarketDataError(f"Error fetching data for {symbol}: {str(e)}")
        
        if not data_frames:
            raise MarketDataError("No data available for any symbols")
        
        # Concatenate all data frames
        result = pd.concat(data_frames, axis=0)
        
        return result
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits.
        
        Returns:
            bool: True if within limits, False otherwise
        """
        return self.source._check_rate_limit()
    
    def _update_rate_limit(self) -> None:
        """Update rate limit counters."""
        self.source.request_count += 1
        self.source.last_request_time = time.time()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Handle missing values first
        df = df.ffill().bfill()
        
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Ensure bb_upper > bb_lower
        mask = df['bb_upper'] <= df['bb_lower']
        if mask.any():
            df.loc[mask, 'bb_upper'] = df.loc[mask, 'bb_middle'] + (bb_std[mask] * 2)
            df.loc[mask, 'bb_lower'] = df.loc[mask, 'bb_middle'] - (bb_std[mask] * 2)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)  # Avoid division by zero
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.inf)  # Avoid division by zero
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        df['rate_of_change'] = df['close'].pct_change(periods=20)
        
        # Calculate ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def _cache_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Cache data for symbol.
        
        Args:
            symbol: Symbol to cache
            df: DataFrame to cache
        """
        self.data_cache[symbol] = df
        self.last_update[symbol] = datetime.now()
        
        # Save to disk
        cache_file = self.cache_dir / f"{symbol}.parquet"
        df.to_parquet(cache_file)
        
        # Save metadata
        metadata = {
            'last_update': self.last_update[symbol].isoformat(),
            'rows': len(df),
            'columns': df.columns.tolist()
        }
        with open(self.cache_dir / f"{symbol}_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def _needs_update(self, symbol: str) -> bool:
        """Check if data for symbol needs updating.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if update needed, False otherwise
        """
        if symbol not in self.last_update:
            return True
        
        update_interval = self.config.get('update_interval', '1d')
        if update_interval == '1d':
            return datetime.now() - self.last_update[symbol] > timedelta(days=1)
        elif update_interval == '1h':
            return datetime.now() - self.last_update[symbol] > timedelta(hours=1)
        else:
            return True
    
    def load_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for symbol from disk.
        
        Args:
            symbol: Symbol to load
            
        Returns:
            DataFrame if available, None otherwise
        """
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            self.data_cache[symbol] = df
            
            # Load metadata
            metadata_file = self.cache_dir / f"{symbol}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.last_update[symbol] = datetime.fromisoformat(
                    metadata['last_update']
                )
            
            return df
        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {e}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear data cache.
        
        Args:
            symbol: Optional symbol to clear, if None clears all
        """
        if symbol:
            if symbol in self.data_cache:
                del self.data_cache[symbol]
            if symbol in self.last_update:
                del self.last_update[symbol]
            
            # Remove cache files
            cache_file = self.cache_dir / f"{symbol}.parquet"
            metadata_file = self.cache_dir / f"{symbol}_metadata.json"
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        else:
            self.data_cache.clear()
            self.last_update.clear()
            
            # Remove all cache files
            for file in self.cache_dir.glob("*"):
                file.unlink()
        
        logger.info(f"Cleared cache for {symbol if symbol else 'all symbols'}")
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate market data.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol for error messages
            
        Returns:
            bool: True if data is valid, False otherwise
            
        Raises:
            MarketDataError: If validation fails
        """
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise MarketDataError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Validate data values
        if (df['volume'] < 0).any():
            raise MarketDataError(f"Invalid volume data for {symbol}")
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            raise MarketDataError(f"Invalid price data for {symbol}")
        
        # Validate technical indicators if they exist
        if all(col in df.columns for col in ['sma_20', 'sma_50', 'rsi_14', 'volume_ma']):
            if df[['sma_20', 'sma_50', 'volume_ma']].isnull().any().any():
                raise MarketDataError(f"Invalid technical indicators for {symbol}")
            if (df['rsi_14'] < 0).any() or (df['rsi_14'] > 100).any():
                raise MarketDataError(f"Invalid RSI values for {symbol}")
        
        return True

def _safe_lowercase_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [(str(level).lower() for level in col) for col in df.columns],
            names=df.columns.names
        )
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df

if __name__ == "__main__":
    # Example usage
    manager = MarketDataManager(config_path="config/strategy_config.yaml")
    
    # Get data for multiple symbols
    data = manager.get_market_data(['AAPL', 'MSFT'])
    
    for symbol, df in data.items():
        print(f"\nData for {symbol}:")
        print(df.tail())
        print(f"Last update: {manager.last_update[symbol]}") 