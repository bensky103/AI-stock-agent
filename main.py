# File: main.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pytz

from data_input.market_feed import MarketFeed
from prediction_engine.sequence_preprocessor import SequencePreprocessor  
from prediction_engine.feature_engineering import FeatureEngineer
# Use predictor instead of enhanced_stock_predictor which doesn't exist
from prediction_engine.predictor import EnhancedStockPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config file."""
    import yaml
    
    try:
        with open('config/strategy_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return minimal default config
        return {
            'market_data': {
                'symbols': ['AAPL'],
                'intervals': ['1d']
            },
            'model': {
                'sequence_length': 10,
                'prediction_horizon': 5
            },
            'training': {
                'epochs': 10,
                'batch_size': 32
            },
            'output': {
                'predictions_path': 'saved_models/predictions.csv'
            }
        }

def get_market_data(config):
    """Get market data for symbols in config."""
    try:
        # Create MarketFeed instance
        market_feed = MarketFeed(config_path='config/strategy_config.yaml')
        
        # Set date range - using historical dates to avoid future date errors
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)
        start_date = end_date - timedelta(days=365)  # Last year of data
        
        # Fetch data for first symbol in config
        symbol = config['market_data']['symbols'][0]
        interval = config['market_data']['intervals'][0]
        
        market_data = market_feed.fetch_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            resample_interval=interval
        )
        
        logger.info(f"Fetched market data for {symbol}, shape: {market_data.shape}")
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def save_predictions(predictions, path):
    """Save predictions to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save predictions
        if isinstance(predictions, pd.DataFrame):
            predictions.to_csv(path)
        else:
            pd.DataFrame(predictions).to_csv(path)
            
        logger.info(f"Saved predictions to {path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")

def main():
    """Main entry point for the stock prediction system."""
    # 1. Load configuration
    config = load_config()
    logger.info("Loaded configuration")
    
    # 2. Get market data
    market_data = get_market_data(config)
    if market_data.empty:
        logger.error("No market data available. Exiting.")
        return
    
    # 3. Initialize and train predictor
    try:
        predictor = EnhancedStockPredictor(
            model_type='lstm',  # Using LSTM as default
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon']
        )
        
        # Simple feature split - assuming market_data has proper features
        features = market_data.drop('close', axis=1, errors='ignore')
        target = market_data['close'] if 'close' in market_data.columns else None
        
        if target is not None:
            # Train model
            logger.info("Training model...")
            predictor.train(features, target)
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = predictor.predict(features.iloc[-1:])
            
            # Save predictions
            save_predictions(predictions, config['output']['predictions_path'])
            logger.info("Prediction process complete")
        else:
            logger.error("Target column 'close' not found in market data")
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")

if __name__ == "__main__":
    logger.info("Starting stock prediction system")
    main()
    logger.info("Stock prediction system finished")

