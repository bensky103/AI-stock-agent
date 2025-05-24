# File: main.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Set before any other imports, especially TensorFlow-related

import logging
# Import our centralized logging configuration first
import logging_config

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from data_input.market_feed import MarketFeed
from prediction_engine.sequence_preprocessor import SequencePreprocessor  
from prediction_engine.feature_engineering import FeatureEngineer
from prediction_engine.predictor import EnhancedStockPredictor

# Remove the old logging setup since we now use centralized logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/trading_system.log'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Ensure logs directory exists for file handler
os.makedirs('logs', exist_ok=True)
# Add file handler to the centralized logging
file_handler = logging.FileHandler('logs/trading_system.log')
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(file_handler)

def load_config():
    """Load configuration from config file."""
    import yaml
    
    # Ensure necessary directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
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

def save_predictions(predictions_data: dict, path: str, prediction_horizon: int):
    """Save predictions to file in the specified wide format."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        all_predictions_records = []

        for symbol, data in predictions_data.items():
            if "error" in data:
                logger.warning(f"Skipping {symbol} due to error: {data['error']}")
                # Optionally, write error to a separate log or include in CSV with NaNs
                # For now, we just skip rows with errors for the main prediction CSV
                record = {
                    'Prediction_Made_At_UTC': datetime.now(pytz.utc).isoformat(),
                    'Symbol': symbol,
                    'Last_Actual_Date': data.get('last_actual_date', 'N/A'),
                    'Error': data['error']
                }
                # Fill prediction columns with NaN if error
                for i in range(prediction_horizon):
                    record[f'Pred_T+{i+1}'] = np.nan
                all_predictions_records.append(record)
                continue

            # Ensure 'predictions' and 'last_actual_date' are present
            if 'predictions' not in data or 'last_actual_date' not in data or data['last_actual_date'] is None:
                logger.warning(f"Skipping {symbol} due to missing 'predictions' or 'last_actual_date'. Data: {data}")
                # Create a record indicating missing data
                record = {
                    'Prediction_Made_At_UTC': data.get('predicted_at', datetime.now(pytz.utc).isoformat()),
                    'Symbol': symbol,
                    'Last_Actual_Date': data.get('last_actual_date', 'N/A'),
                    'Error': 'Missing prediction data or last actual date'
                }
                for i in range(prediction_horizon):
                    record[f'Pred_T+{i+1}'] = np.nan
                all_predictions_records.append(record)
                continue
                
            predicted_at_utc_str = data['predicted_at']
            # Parse last_actual_date to datetime object to calculate future dates
            try:
                last_actual_dt = datetime.strptime(data['last_actual_date'], '%Y-%m-%d')
            except ValueError:
                logger.error(f"Could not parse last_actual_date '{data['last_actual_date']}' for {symbol}. Skipping this record.")
                # Create a record indicating parsing error
                record = {
                    'Prediction_Made_At_UTC': predicted_at_utc_str,
                    'Symbol': symbol,
                    'Last_Actual_Date': data['last_actual_date'],
                    'Error': f'Invalid last_actual_date format: {data['last_actual_date']}'
                }
                for i in range(prediction_horizon):
                    record[f'Pred_T+{i+1}'] = np.nan
                all_predictions_records.append(record)
                continue

            # One record per (symbol, prediction_made_at) instance
            record = {
                'Prediction_Made_At_UTC': predicted_at_utc_str,
                'Symbol': symbol,
                'Last_Actual_Date': data['last_actual_date'] # The last date for which actual data was available
            }
            
            # Add each prediction step as a new column T+1, T+2, ...
            # Also create column headers for the actual dates of these predictions
            for i, pred_value in enumerate(data['predictions']):
                future_date = last_actual_dt + timedelta(days=i + 1)
                # Column name includes the target date for clarity e.g., Pred_2024-06-15_TSLA
                # For simpler wide format, let's use T+1, T+2, and store the date in a separate column if needed
                # As per request: rows = dates (prediction_made_at), columns = stock prices (target_date_symbol)
                # This implies a very wide format if many symbols and many target dates.
                # Let's try a format: Prediction_Made_At, Symbol, Last_Actual_Date, TargetDate_T+1, Pred_T+1, TargetDate_T+2, Pred_T+2 ...
                # Or, as requested: Rows=dates (predicted_at), Columns=stock_prices (TargetDate_Symbol), Value=prediction.
                # This means one row per (prediction_made_at, symbol) and columns for each future step T+1, T+2...
                # The specific request "rows, we have dates, for collumns we have stock prices, at the intersection ... we have the predicted price"
                # is better achieved if one CSV per prediction run, or by having 'Prediction_Made_At' as a primary key / index.

                # For the requested format: Prediction_Made_At | TargetDate1_Symbol | TargetDate2_Symbol ...
                # This is tricky if multiple symbols are in `predictions_data`.
                # Let's create one row per prediction event (symbol + time)
                # and columns for each step of the horizon T+1, T+2, ... T+N.
                # The column headers will be like 'Pred_TSLA_2025-05-24', 'Pred_TSLA_2025-05-25', etc.
                # This makes more sense. The rows would be indexed by when the prediction was made.
                
                # Current proposal: each row is one symbol's prediction set at a point in time
                # Columns: Prediction_Made_At, Symbol, Last_Actual_Date, Pred_T+1_Date, Pred_T+1_Value, Pred_T+2_Date, Pred_T+2_Value, ...
                
                # Let's simplify based on the user's description:
                # Index: Prediction_Made_At (implicitly, or could be a column)
                # Columns: Date_T+1_Symbol, Date_T+2_Symbol, ...
                # Values: The predicted price.
                # This is best achieved by having one record (row) per prediction instance (symbol + predicted_at)
                # And the columns represent the future predictions for that specific symbol at that specific time.
                
                # We will have: Prediction_Made_At_UTC, Symbol, Last_Actual_Date, Pred_T+1_Price, Pred_T+1_Target_Date, ...
                record[f'Pred_T+{i+1}_Price'] = pred_value
                record[f'Pred_T+{i+1}_Target_Date'] = future_date.strftime('%Y-%m-%d')
            
            # Add uncertainty for each step if available (assuming uncertainties_array matches predictions array)
            if 'uncertainties' in data and isinstance(data['uncertainties'], list) and len(data['uncertainties']) == len(data['predictions']):
                for i, uncert_value in enumerate(data['uncertainties']):
                    record[f'Pred_T+{i+1}_Uncertainty'] = uncert_value
            else:
                 for i in range(len(data['predictions'])):
                    record[f'Pred_T+{i+1}_Uncertainty'] = np.nan # Fill with NaN if not available
            
            all_predictions_records.append(record)

        if not all_predictions_records:
            logger.info("No valid prediction records to save.")
            # Create an empty file with headers if path specified, or just log and return
            # For simplicity, just log if no data
            return

        # Create DataFrame from the list of records
        output_df = pd.DataFrame(all_predictions_records)
        
        # Define column order for better readability
        base_cols = ['Prediction_Made_At_UTC', 'Symbol', 'Last_Actual_Date', 'Error']
        ordered_cols = [col for col in base_cols if col in output_df.columns] # Start with base columns that exist
        
        # Dynamically add prediction and uncertainty columns in order T+1, T+2, ...
        for i in range(prediction_horizon):
            price_col = f'Pred_T+{i+1}_Price'
            date_col = f'Pred_T+{i+1}_Target_Date'
            uncert_col = f'Pred_T+{i+1}_Uncertainty'
            if price_col in output_df.columns:
                ordered_cols.append(price_col)
            if date_col in output_df.columns:
                ordered_cols.append(date_col)
            if uncert_col in output_df.columns:
                ordered_cols.append(uncert_col)
        
        # Ensure all df columns are in ordered_cols, add any missing (e.g. if error only run)
        for col in output_df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        output_df = output_df[ordered_cols]

        output_df.to_csv(path, index=False)
        logger.info(f"Saved predictions to {path} with new format")
        
    except Exception as e:
        logger.error(f"Error saving predictions with new format: {e}", exc_info=True)

def main():
    """Main entry point for the stock prediction system."""
    # 1. Load configuration
    config = load_config()
    logger.info("Loaded configuration")
    
    # 2. Get market data
    logger.info("Attempting to fetch initial market data for the system...")
    market_data = get_market_data(config)
    if market_data.empty:
        logger.error("No market data available to proceed. Exiting prediction system.")
        return
    logger.info(f"Successfully fetched initial market data. Shape: {market_data.shape}. This data will be used as a base for operations like scaler fitting if needed.")
    
    # 3. Initialize predictor
    try:
        # Access sequence_length and prediction_horizon directly from model config
        sequence_length = config['model']['sequence_length']
        prediction_horizon = config['model']['prediction_horizon']
        
        predictor = EnhancedStockPredictor(
            model_type='tft',  # Using TFT as the only supported model type
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Get symbol from config
        symbol = config['market_data']['symbols'][0] if isinstance(config['market_data']['symbols'], list) else config['market_data']['symbols']
        
        # Set date range for prediction
        end_date = datetime.now(pytz.UTC) - timedelta(days=1)
        start_date = end_date - timedelta(days=60)  # Two months of data for prediction
        
        # Convert to string format
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Skip training since it's not implemented and directly make predictions
        logger.info("Starting the prediction pipeline...")
        predictions = predictor.predict(
            symbols=[symbol],
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        # Save predictions with a default path if output config is missing
        if 'output' not in config:
            # Add the output section with default predictions path
            config['output'] = {'predictions_path': 'saved_models/predictions.csv'}
            logger.warning("Output configuration missing, using default path: saved_models/predictions.csv")
        
        save_predictions(predictions, config['output']['predictions_path'], prediction_horizon)
        logger.info(f"Prediction pipeline complete. All forecasts have been saved to: {config['output']['predictions_path']}")
    
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting stock prediction system")
    main()
    logger.info("Stock prediction system finished")

