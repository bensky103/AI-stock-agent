"""TFT model training script for Google Colab environment.

This script handles the training of the Temporal Fusion Transformer model
for stock price prediction using the Kaggle tech companies dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, List, Optional, Tuple, Union
import yaml
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """Dataset class for stock data."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 100,
        prediction_horizons: List[int] = [5, 10, 20],  # 5-day, 10-day, 20-day predictions
        target_col: str = 'Close'
    ):
        """Initialize dataset.
        
        Args:
            data: DataFrame with stock data
            sequence_length: Length of input sequences
            prediction_horizons: List of prediction horizons in days
            target_col: Column to use as target
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.target_col = target_col
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict]:
        """Create sequences for training."""
        sequences = []
        
        for i in range(self.sequence_length, len(self.data) - max(self.prediction_horizons)):
            # Input sequence
            x = self.data.iloc[i-self.sequence_length:i]
            
            # Target values for different horizons
            targets = {}
            for horizon in self.prediction_horizons:
                target_idx = i + horizon
                if target_idx < len(self.data):
                    targets[f'target_{horizon}d'] = self.data.iloc[target_idx][self.target_col]
            
            if len(targets) == len(self.prediction_horizons):
                sequences.append({
                    'input': x,
                    'targets': targets,
                    'timestamp': self.data.index[i]
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        return {
            'input': torch.FloatTensor(seq['input'].values),
            'targets': torch.FloatTensor(list(seq['targets'].values())),
            'timestamp': seq['timestamp']
        }

def load_kaggle_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess Kaggle dataset.
    
    Args:
        data_path: Path to Kaggle dataset directory
        
    Returns:
        DataFrame with processed stock data
    """
    logger.info(f"Loading data from {data_path}")
    
    # List of target stocks
    target_stocks = [
        'TSLA', 'NVDA', 'MSFT', 'GOOG', 'META',
        'AAPL', 'PLTR', 'AMZN', 'CRM', 'ADBE',
        'INTC', 'AMD'
    ]
    
    # Load and combine all stock data
    dfs = []
    for stock in target_stocks:
        try:
            file_path = os.path.join(data_path, f"{stock}.csv")
            df = pd.read_csv(file_path)
            df['Symbol'] = stock
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load data for {stock}: {e}")
    
    if not dfs:
        raise ValueError("No stock data could be loaded")
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert date column
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Set multi-index
    combined_df.set_index(['Symbol', 'Date'], inplace=True)
    
    # Sort by date
    combined_df.sort_index(inplace=True)
    
    # Basic preprocessing
    combined_df = combined_df.astype({
        'Open': float,
        'High': float,
        'Low': float,
        'Close': float,
        'Volume': float
    })
    
    # Add basic features
    for symbol in target_stocks:
        symbol_data = combined_df.xs(symbol)
        
        # Calculate returns
        combined_df.loc[symbol, 'Returns'] = symbol_data['Close'].pct_change()
        
        # Calculate log returns
        combined_df.loc[symbol, 'LogReturns'] = np.log1p(symbol_data['Close'].pct_change())
        
        # Calculate volatility (20-day rolling std of returns)
        combined_df.loc[symbol, 'Volatility'] = symbol_data['Returns'].rolling(20).std()
    
    # Drop any remaining NaN values
    combined_df.dropna(inplace=True)
    
    logger.info(f"Loaded data shape: {combined_df.shape}")
    return combined_df

def prepare_training_data(
    data: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.15,
    sequence_length: int = 100,
    prediction_horizons: List[int] = [5, 10, 20],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data loaders for training.
    
    Args:
        data: DataFrame with stock data
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        sequence_length: Length of input sequences
        prediction_horizons: List of prediction horizons
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data by date
    dates = sorted(data.index.get_level_values('Date').unique())
    train_end = int(len(dates) * train_split)
    val_end = int(len(dates) * (train_split + val_split))
    
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    # Split data
    train_data = data[data.index.get_level_values('Date').isin(train_dates)]
    val_data = data[data.index.get_level_values('Date').isin(val_dates)]
    test_data = data[data.index.get_level_values('Date').isin(test_dates)]
    
    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=sequence_length,
        prediction_horizons=prediction_horizons
    )
    val_dataset = StockDataset(
        val_data,
        sequence_length=sequence_length,
        prediction_horizons=prediction_horizons
    )
    test_dataset = StockDataset(
        test_data,
        sequence_length=sequence_length,
        prediction_horizons=prediction_horizons
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Colab environment
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function."""
    # Set paths
    data_path = "/kaggle/input/top-tech-companies-stock-price"
    output_path = "/kaggle/working/model_output"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load and preprocess data
        data = load_kaggle_data(data_path)
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_training_data(
            data,
            sequence_length=100,
            prediction_horizons=[5, 10, 20],
            batch_size=32
        )
        
        # TODO: Initialize and train TFT model
        # This will be implemented in the next step
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 