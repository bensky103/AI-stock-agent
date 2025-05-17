#!/usr/bin/env python
"""
Simple test script to verify MarketDataManager import works correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_direct_import():
    """Test direct import of MarketDataManager."""
    try:
        from data_input.market_data_manager import MarketDataManager, MarketDataError
        print("✓ Direct import successful")
        return True
    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
        return False

def test_module_import():
    """Test importing MarketDataManager through the data_input module."""
    try:
        from data_input import MarketDataManager
        print("✓ Module import successful")
        return True
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False

def test_importlib():
    """Test importing MarketDataManager using importlib."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "market_data_manager", 
            str(Path(__file__).resolve().parent / "data_input" / "market_data_manager.py")
        )
        market_data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(market_data_module)
        MarketDataManager = market_data_module.MarketDataManager
        print("✓ Importlib approach successful")
        return True
    except Exception as e:
        print(f"❌ Importlib approach failed: {e}")
        return False

def test_file_existence():
    """Check if the market_data_manager.py file exists and contains the MarketDataManager class."""
    file_path = Path(__file__).resolve().parent / "data_input" / "market_data_manager.py"
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
        if "class MarketDataManager" in content:
            print(f"✓ File exists and contains MarketDataManager class")
            return True
        else:
            print(f"❌ File exists but does not contain MarketDataManager class")
            return False

def test_pycache():
    """Check if there's a __pycache__ directory that might contain outdated bytecode."""
    pycache_dir = Path(__file__).resolve().parent / "data_input" / "__pycache__"
    if pycache_dir.exists():
        print(f"✓ __pycache__ directory exists: {pycache_dir}")
        print("  You might want to delete it to force Python to recompile the modules")
        return True
    else:
        print(f"❌ __pycache__ directory does not exist")
        return False

if __name__ == "__main__":
    print("Testing MarketDataManager import...")
    print("\n1. Testing direct import...")
    test_direct_import()
    
    print("\n2. Testing module import...")
    test_module_import()
    
    print("\n3. Testing importlib approach...")
    test_importlib()
    
    print("\n4. Checking file existence...")
    test_file_existence()
    
    print("\n5. Checking __pycache__...")
    test_pycache()
    
    print("\nDone!") 