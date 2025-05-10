# File: data_input/market_feed.py

import logging
from typing import List, Optional, Union, Dict, Any
import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

# Constants
DEFAULT_INTERVALS = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "1h": timedelta(days=730),
    "1d": timedelta(days=3650),
    "1wk": timedelta(days=3650),
    "1mo": timedelta(days=3650)
}

class MarketDataError(Exception):
    """Custom exception for market data related errors."""
    pass

def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load YAML configuration with enhanced validation.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    yaml.YAMLError
        If file cannot be parsed as YAML.
    ValueError
        If configuration is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate configuration
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a dictionary")
    
    if "symbols" not in cfg:
        raise ValueError("Configuration must contain 'symbols' key")
    
    if not isinstance(cfg["symbols"], list):
        raise ValueError("'symbols' must be a list")
    
    if not all(isinstance(s, str) for s in cfg["symbols"]):
        raise ValueError("All symbols must be strings")

    logger.info(f"Loaded config from {config_path}")
    return cfg

def validate_date_range(start_date: Optional[str], end_date: Optional[str], interval: str) -> tuple:
    """
    Validate and adjust date range based on interval.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    interval : str
        Data frequency

    Returns
    -------
    tuple
        (start_date, end_date) as datetime objects
    """
    if interval not in DEFAULT_INTERVALS:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {list(DEFAULT_INTERVALS.keys())}")

    now = datetime.now(pytz.UTC)
    max_lookback = DEFAULT_INTERVALS[interval]

    if end_date is None:
        end_date = now
    else:
        end_date = pd.to_datetime(end_date, utc=True)

    if start_date is None:
        start_date = end_date - max_lookback
    else:
        start_date = pd.to_datetime(start_date, utc=True)

    if end_date < start_date:
        raise ValueError("end_date must be after start_date")

    if end_date > now:
        logger.warning("end_date is in the future, using current time instead")
        end_date = now

    if end_date - start_date > max_lookback:
        logger.warning(f"Date range exceeds maximum lookback period for {interval}. Adjusting start_date.")
        start_date = end_date - max_lookback

    return start_date, end_date

def fetch_single_symbol(symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
    """
    Fetch data for a single symbol with error handling.

    Parameters
    ----------
    symbol : str
        Stock symbol
    start_date : datetime
        Start date
    end_date : datetime
        End date
    interval : str
        Data frequency

    Returns
    -------
    pd.DataFrame
        DataFrame with market data
    """
    try:
        data = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Ensure we have a DataFrame with the expected columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns in data for {symbol}")
            return pd.DataFrame()
        
        # Reset index to make Date a column and rename it to datetime
        data = data.reset_index()
        if 'Date' not in data.columns:
            logger.error(f"Date column not found after reset_index for {symbol}")
            return pd.DataFrame()
            
        data.rename(columns={'Date': 'datetime'}, inplace=True)
        
        # Convert column names to lowercase and standardize format
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Ensure all required columns are present
        required_cols = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            logger.error(f"Missing columns after processing: {missing_cols}")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_market_data(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    config_path: Union[str, Path] = "config/strategy_config.yaml",
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Fetch market data for one or more symbols with enhanced error handling and parallel processing.

    Parameters
    ----------
    symbols : list of str, optional
        List of ticker symbols to fetch. If None, read from YAML config.
    start_date : str, optional
        Fetch data from this date (YYYY-MM-DD). Defaults to None (all available).
    end_date : str, optional
        Fetch data up to this date (YYYY-MM-DD). Defaults to None (today).
    interval : str
        Data frequency, e.g., "1d", "1h", "5m". Defaults to "1d".
    config_path : str or Path
        Path to YAML config containing default symbols.
    max_workers : int
        Maximum number of parallel workers for data fetching.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with MultiIndex [symbol, datetime] and columns:
        ['open','high','low','close','adj_close','volume'].

    Raises
    ------
    MarketDataError
        If data fetching fails or returns invalid data.
    """
    try:
        # Load and validate symbols
        if symbols is None:
            cfg = load_config(config_path)
            symbols = cfg.get("symbols", [])
            if not symbols:
                raise MarketDataError("No symbols specified in config or function arguments.")
            logger.info(f"Using symbols from config: {symbols}")
        else:
            logger.info(f"Using provided symbols: {symbols}")

        # Validate date range
        start_date, end_date = validate_date_range(start_date, end_date, interval)
        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date} at interval '{interval}'")

        # Fetch data in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_single_symbol, symbol, start_date, end_date, interval)
                for symbol in symbols
            ]
            results = [f.result() for f in futures]

        # Combine results
        if not any(not df.empty for df in results):
            raise MarketDataError("No data fetched for any symbol")

        result = pd.concat([df for df in results if not df.empty], ignore_index=True)
        
        # Final processing
        result = result[["symbol", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]
        result["datetime"] = pd.to_datetime(result["datetime"])
        result.set_index(["symbol", "datetime"], inplace=True)
        result.sort_index(inplace=True)

        # Data quality checks
        if result.isnull().any().any():
            logger.warning("Data contains missing values")
            
        if (result["volume"] < 0).any():
            logger.warning("Data contains negative volume values")
            result.loc[result["volume"] < 0, "volume"] = 0

        logger.info(f"Successfully fetched data for {len(symbols)} symbols")
        return result

    except Exception as e:
        logger.exception("Failed to fetch market data")
        raise MarketDataError(f"Market data fetch failed: {str(e)}")

if __name__ == "__main__":
    # --- Enhanced Usage Example ---
    try:
        df = get_market_data(
            symbols=None,                # will read from config/strategy_config.yaml
            start_date="2023-01-01",
            end_date="2023-12-31",
            interval="1d",
            config_path="config/strategy_config.yaml",
            max_workers=5
        )
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"Total symbols: {len(df.index.get_level_values('symbol').unique())}")
        print(f"Date range: {df.index.get_level_values('datetime').min()} to {df.index.get_level_values('datetime').max()}")
        print("\nSample data:")
        print(df.head())
        
        # Basic data quality checks
        print("\nData Quality Checks:")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Negative volumes: {(df['volume'] < 0).sum()}")
        
    except Exception as ex:
        logger.error(f"Error in main: {ex}")
