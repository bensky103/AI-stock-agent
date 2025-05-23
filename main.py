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
    
    # 3. Initialize predictor
    try:
        logger.info("Initializing stock predictor")
        predictor = EnhancedStockPredictor(
            model_type='tft',  # Using TFT as it's the only supported model type
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon']
        )
        logger.info(f"Predictor initialized with sequence_length={config['model']['sequence_length']}, prediction_horizon={config['model']['prediction_horizon']}")
        
        # Get symbol from config
        symbol = config['market_data']['symbols'][0] if isinstance(config['market_data']['symbols'], list) else config['market_data']['symbols']
        logger.info(f"Using symbol: {symbol}")
        
        # Set date range for prediction
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)
        start_date = end_date - timedelta(days=60)  # Two months of data for prediction
        
        # Convert to string format
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        logger.info(f"Date range: {start_date_str} to {end_date_str}")
        
        # Skip training since it's not implemented and directly make predictions
        logger.info("Making predictions...")
        try:
            predictions = predictor.predict(
                symbols=[symbol],
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            # Save predictions
            save_predictions(predictions, config['output']['predictions_path'])
            logger.info("Prediction process complete")
            
        except Exception as predict_err:
            logger.error(f"Error in prediction call: {str(predict_err)}")
            import traceback
            logger.error(f"Prediction error traceback: {traceback.format_exc()}")
            raise
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Debug key attributes that might be causing issues
        if 'predictor' in locals():
            logger.error(f"Predictor attributes:")
            logger.error(f"  sequence_length: {getattr(predictor, 'sequence_length', 'Not found')}")
            logger.error(f"  feature_engineer: {getattr(predictor, 'feature_engineer', 'Not found')}")
            if hasattr(predictor, 'feature_engineer') and predictor.feature_engineer is not None:
                logger.error(f"  feature_engineer.sequence_length: {getattr(predictor.feature_engineer, 'sequence_length', 'Not found')}")
        else:
            logger.error("Predictor not initialized successfully")

if __name__ == "__main__":
    logger.info("Starting stock prediction system")
    main()
    logger.info("Stock prediction system finished")

