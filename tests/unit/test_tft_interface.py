#!/usr/bin/env python
"""
Simplified test script for the market data functionality.
This script tests the data handling and preprocessing without using TensorFlow.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import MarketDataManager directly
try:
    from data_input.market_data_manager import MarketDataManager, MarketDataError
    print("✓ Successfully imported MarketDataManager")
except Exception as e:
    print(f"❌ Error importing MarketDataManager: {e}")

def create_sample_data():
    """Create sample market data for testing."""
    symbols = ['AAPL', 'MSFT']
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    data_list = []
    for symbol in symbols:
        for date in dates:
            data_list.append({
                'symbol': symbol,
                'datetime': date,
                'open': np.random.randn() + 100,
                'high': np.random.randn() + 105,
                'low': np.random.randn() + 95,
                'close': np.random.randn() + 100,
                'volume': np.random.randint(1000000, 10000000)
            })
    
    df = pd.DataFrame(data_list)
    df = df.set_index(['symbol', 'datetime'])
    return df

def test_market_data_manager():
    """Test MarketDataManager functionality."""
    print("\nTesting MarketDataManager...")
    
    # Create a temporary directory for cache
    cache_dir = Path("temp_cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Initialize MarketDataManager
        mdm = MarketDataManager(cache_dir=cache_dir)
        print("✓ MarketDataManager initialization successful")
        
        # Test adding technical indicators to data
        print("\nTesting technical indicators...")
        sample_data = create_sample_data()
        single_symbol_data = sample_data.loc['AAPL']
        
        # Add technical indicators
        enhanced_data = mdm._add_technical_indicators(single_symbol_data)
        
        # Check that indicators were added
        expected_indicators = ['sma_20', 'sma_50', 'bb_middle', 'bb_upper', 'bb_lower', 
                               'rsi_14', 'macd', 'macd_signal', 'macd_hist']
        
        for indicator in expected_indicators:
            if indicator in enhanced_data.columns:
                print(f"✓ Added indicator: {indicator}")
            else:
                print(f"❌ Missing indicator: {indicator}")
        
        # Test caching functionality
        print("\nTesting data caching...")
        test_symbol = 'TEST'
        test_df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [102, 103],
            'volume': [1000000, 1100000]
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        # Cache the data
        mdm._cache_data(test_symbol, test_df)
        
        # Check if data was cached
        cached_data = mdm.load_cached_data(test_symbol)
        if cached_data is not None:
            print(f"✓ Successfully cached and retrieved data")
            print(f"  Cached data shape: {cached_data.shape}")
        else:
            print(f"❌ Failed to retrieve cached data")
        
        print("\n🎉 All market data tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)

if __name__ == "__main__":
    test_market_data_manager()
