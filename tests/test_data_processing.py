#!/usr/bin/env python
"""
Test script for data preprocessing functionality.
This script tests the data preprocessing functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import data processing modules
try:
    from prediction_engine.scaler_handler import ScalerHandler, ScalerHandlerError
    print("✓ Successfully imported ScalerHandler")
except Exception as e:
    print(f"❌ Error importing ScalerHandler: {e}")

def create_sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 105,
        'low': np.random.randn(len(dates)) + 95,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data

def test_scaler_handler():
    """Test ScalerHandler functionality."""
    print("\nTesting ScalerHandler...")
    
    try:
        # Create sample data
        sample_data = create_sample_data()
        print(f"✓ Created sample data with shape: {sample_data.shape}")
        
        # Create scaler handler
        scaler = ScalerHandler(scaler_type='minmax')
        print(f"✓ Created ScalerHandler with type: {scaler.scaler_type}")
        
        # Fit scaler
        print("\nFitting scaler to data...")
        scaler.fit(sample_data)
        
        # Check that scalers were created
        print("\nChecking fitted scalers:")
        for column, fitted_scaler in scaler.scalers.items():
            print(f"✓ Fitted scaler for column: {column}")
        
        # Transform data
        print("\nTransforming data...")
        transformed_data = scaler.transform(sample_data)
        
        # Check transformed data
        min_values = transformed_data.min().min()
        max_values = transformed_data.max().max()
        print(f"✓ Transformed data range: [{min_values:.4f}, {max_values:.4f}]")
        
        # Inverse transform
        print("\nInverse transforming data...")
        recovered_data = scaler.inverse_transform(transformed_data)
        
        # Check recovery accuracy
        mse = ((sample_data - recovered_data) ** 2).mean().mean()
        print(f"✓ Mean squared error after round-trip: {mse:.10f}")
        
        # Test saving and loading
        temp_dir = Path("temp_scalers")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        print("\nSaving scalers...")
        scaler.save(temp_dir)
        
        print("Loading scalers...")
        new_scaler = ScalerHandler()
        new_scaler.load(temp_dir)
        
        print("✓ Successfully saved and loaded scalers")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All data processing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_scaler_handler() 