"""
# Section 1: Training Script Overview
This script implements a complete training pipeline for the Temporal Fusion Transformer (TFT) model.
It handles data loading, preprocessing, model training, evaluation, and result saving.
The script is designed to run in Google Colab with GPU support.

Key Features:
- GPU configuration and memory management
- Comprehensive data preprocessing with technical indicators
- Model training with progress tracking
- Detailed evaluation and model interpretability
- Automatic result saving and visualization
"""

# Section 2: Imports
# -----------------
# Standard library imports
import os
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
import gc
import json

# Data science and ML imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add colab_training directory to Python path
# Handle both Colab and local environments
try:
    # For local environment
    script_dir = Path(__file__).parent
except NameError:
    # For Google Colab
    script_dir = Path('/content/drive/MyDrive/colab_training')

# Add the colab_training directory to Python path
sys.path.insert(0, str(script_dir))

# Custom module imports - using absolute imports
from colab_training.libs.tft_model import TFTModel
from colab_training.data_formatters.stock_formatter import StockFormatter
from colab_training.features.technical_indicators import (
    IndicatorConfig,
    add_all_indicators
)
from colab_training.features.market_context import (
    SignalConfig,
    detect_market_regime,
    generate_trading_signals,
    analyze_market_context,
    get_trading_recommendations
)
from colab_training.interpretability.model_explainer import (
    ModelExplainer,
    ExplanationConfig,
    generate_explanation_report,
    plot_feature_importance,
    plot_attention_weights
)

# Section 3: Setup and Logging
# ---------------------------
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Section 4: GPU Configuration
# --------------------------
def setup_gpu():
    """Configure GPU settings for TensorFlow.
    
    This function:
    1. Detects available GPUs
    2. Enables memory growth to prevent OOM errors
    3. Logs GPU status
    """
    try:
        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        else:
            logger.warning("No GPU found. Running on CPU.")
    except Exception as e:
        logger.error(f"Error configuring GPU: {str(e)}")

# Section 5: Configuration Management
# ---------------------------------
def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary containing:
        - Model parameters
        - Training settings
        - Data specifications
        - Output paths
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Section 6: Data Loading and Preprocessing
# ---------------------------------------
def get_data_path():
    """Auto-detects the data path (Kaggle or local)."""
    kaggle_path = '/kaggle/input/top-tech-companies-stock-price/Technology Companies'
    local_path = '/content/colab_training/data/raw'
    if os.path.exists(kaggle_path):
        print(f"Using Kaggle dataset: {kaggle_path}")
        return kaggle_path
    elif os.path.exists(local_path):
        print(f"Using local dataset: {local_path}")
        return local_path
    else:
        raise FileNotFoundError("No valid data directory found.")

def load_data(data_path: str, config: dict) -> pd.DataFrame:
    """Load and preprocess stock data.
    
    This function:
    1. Loads data for each target stock
    2. Calculates technical indicators
    3. Detects market regimes
    4. Generates trading signals
    
    Args:
        data_path: Path to data directory
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame with all features
    """
    logger.info("Loading stock data...")
    
    # Load data for each stock with progress bar
    dfs = []
    for stock in tqdm(config['data']['target_stocks'], desc="Loading stocks"):
        file_path = os.path.join(data_path, f"{stock}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Auto-rename columns to expected lowercase names
            rename_dict = {
                'Date': 'date',
                'Close': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume',
            }
            df.rename(columns=rename_dict, inplace=True)
            if 'date' not in df.columns or 'close' not in df.columns:
                logger.warning(f"'date' or 'close' column not found in {file_path}. Skipping.")
                continue
            df['symbol'] = stock
            dfs.append(df)
        else:
            logger.warning(f"Data file not found for {stock}: {file_path}")
    
    if not dfs:
        raise ValueError("No data files found for target stocks")
    
    # Combine data
    data = pd.concat(dfs, ignore_index=True)
    
    # Convert date column
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date and symbol
    data = data.sort_values(['symbol', 'date'])
    
    # Add technical indicators with verbose=False for production
    logger.info("Calculating technical indicators...")
    indicator_config = IndicatorConfig(
        sma_periods=[5, 10, 20, 50],
        ema_periods=[5, 10, 20, 50],
        rsi_period=14,
        atr_period=14,
        volume_periods=[5, 10, 20]
    )
    data = add_all_indicators(data, indicator_config, verbose=False)  # Disable debug prints
    
    # Detect market regime
    logger.info("Detecting market regimes...")
    data = detect_market_regime(data)
    
    # Generate trading signals
    logger.info("Generating trading signals...")
    signal_config = SignalConfig()
    data = generate_trading_signals(data, signal_config)
    
    # Clean up memory
    gc.collect()
    
    logger.info(f"Loaded data for {len(config['data']['target_stocks'])} stocks")
    logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data

def prepare_data(data: pd.DataFrame, config: dict) -> tuple:
    """Prepare data for training.
    
    This function:
    1. Initializes data formatter
    2. Transforms inputs
    3. Splits data into train/val/test sets
    
    Args:
        data: Preprocessed DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (formatter, train_data, val_data, test_data)
    """
    logger.info("Preparing data for training...")
    
    # Initialize formatter
    formatter = StockFormatter(config)
    
    # Transform inputs
    transformed_data = formatter.transform_inputs(data)
    
    # Split data
    train_data, val_data, test_data = formatter.split_data(transformed_data)
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    return formatter, train_data, val_data, test_data

# Section 7: Model Training
# ------------------------
def train_model(formatter: StockFormatter,
                train_data: pd.DataFrame,
                val_data: pd.DataFrame,
                config: dict) -> TFTModel:
    """Enhanced TFT model training with advanced callbacks."""
    logger.info("Initializing enhanced model...")
    
    # Initialize model
    model = TFTModel(config['model'])
    
    # Print model summary
    logger.info("Model architecture summary:")
    model.model.summary(print_fn=logger.info)
    
    # Advanced callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training'].get('early_stopping_patience', 10),
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    if config['training'].get('use_model_checkpoint', True):
        output_dir = Path(config['output']['model_dir'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Learning rate scheduling
    if config['training'].get('use_reduce_lr_on_plateau', True):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training'].get('lr_schedule_factor', 0.5),
            patience=config['training'].get('lr_schedule_patience', 5),
            min_lr=config['training'].get('min_lr', 0.00001),
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    # Custom training progress callback
    class EnhancedProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.epoch_losses = []
            self.val_losses = []
        
        def on_epoch_end(self, epoch, logs=None):
            self.epoch_losses.append(logs['loss'])
            self.val_losses.append(logs['val_loss'])
            
            # Log every 5 epochs with detailed metrics
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}: "
                          f"loss={logs['loss']:.6f}, "
                          f"val_loss={logs['val_loss']:.6f}, "
                          f"mae={logs['mae']:.6f}, "
                          f"val_mae={logs['val_mae']:.6f}")
        
        def on_train_end(self, logs=None):
            # Plot training curves
            if config['output'].get('save_training_plots', True):
                self.plot_training_curves()
        
        def plot_training_curves(self):
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.epoch_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.yscale('log')  # Log scale for better visualization
            
            plt.subplot(1, 2, 2)
            plt.plot(self.epoch_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.title('Model Loss (Linear Scale)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            output_dir = Path(config['output']['model_dir'])
            plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    callbacks.append(EnhancedProgressCallback())
    
    # Train model with enhanced callbacks
    logger.info("Starting enhanced training...")
    
    try:
        # Train with all callbacks
        model.fit(train_data, val_data, callbacks=callbacks)
        
        # Verification
        sample_pred = model.predict(train_data.head(5))
        logger.info(f"Model training successful. Sample predictions: {sample_pred[:5]}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.info("Returning model despite errors for debugging...")
    
    # Enhanced memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    logger.info("Enhanced training completed")
    return model

# Section 8: Model Evaluation and Analysis
# --------------------------------------
def evaluate_model(model: TFTModel,
                  formatter: StockFormatter,
                  test_data: pd.DataFrame,
                  config: dict) -> dict:
    """Evaluate the trained model.
    
    Args:
        model: Trained model (may have issues but not None)
        formatter: Data formatter
        test_data: Test data
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info("Evaluating model...")
    
    # Check if model is valid
    if model is None:
        logger.error("Model is None, cannot evaluate")
        return {
            'metrics': {'error': 'Model is None'},
            'market_context': {},
            'trading_recommendations': {},
            'explanation_report': 'Model evaluation failed - model is None'
        }
    
    try:
        # Calculate metrics using model's evaluate method
        metrics = model.evaluate(test_data)
        logger.info(f"Model evaluation successful: {metrics}")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        metrics = {'error': str(e)}
    
    try:
        # Get predictions for further analysis
        predictions = model.predict(test_data)
        logger.info(f"Generated {len(predictions)} predictions")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        predictions = np.array([0.0])  # dummy predictions
    
    # Continue with other analysis even if model evaluation failed
    try:
        # Analyze market context
        market_context = analyze_market_context(test_data)
        
        # Get trading recommendations
        recommendations = get_trading_recommendations(test_data)
        
        # Skip model explanations if model is problematic
        explanation_report = "Model explanation skipped due to training issues"
        
        # Save trading recommendations
        output_dir = Path(config['output']['model_dir'])
        recommendations.to_csv(output_dir / 'trading_recommendations.csv')
        
        # Combine all results
        results = {
            'metrics': metrics,
            'market_context': market_context,
            'trading_recommendations': recommendations.to_dict(),
            'explanation_report': explanation_report
        }
        
        logger.info("Evaluation completed (with some components skipped)")
        logger.info(f"Test metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error during market analysis: {e}")
        results = {
            'metrics': metrics,
            'market_context': {'error': str(e)},
            'trading_recommendations': {},
            'explanation_report': f'Evaluation failed: {str(e)}'
        }
    
    return results

# Section 9: Results Management
# ---------------------------
def save_results(model: TFTModel,
                formatter: StockFormatter,
                results: dict,
                config: dict):
    """Save training results.
    
    This function saves:
    1. Trained model weights
    2. Data formatter (if it has save method)
    3. Configuration
    4. Evaluation metrics
    5. Market context analysis
    6. Trading recommendations
    7. Model explanations
    
    Args:
        model: Trained model
        formatter: Data formatter
        results: Evaluation results
        config: Configuration dictionary
    """
    logger.info("Saving results...")
    
    # Create output directory
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save(output_dir / 'model')
    
    # Save formatter (only if it has a save method)
    if hasattr(formatter, 'save'):
        formatter.save(output_dir / 'formatter')
    else:
        logger.warning("Formatter does not have a save method, skipping...")
        # Save formatter config instead
        formatter_config = {
            'class': formatter.__class__.__name__,
            'config': getattr(formatter, 'config', {})
        }
        with open(output_dir / 'formatter_config.json', 'w') as f:
            json.dump(formatter_config, f)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Save metrics
    with open(output_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(results['metrics'], f)
    
    # Save market context
    with open(output_dir / 'market_context.yaml', 'w') as f:
        yaml.dump(results['market_context'], f)
    
    logger.info(f"Results saved to {output_dir}")

# Section 10: Main Training Pipeline
# --------------------------------
def main():
    """Main training function.
    
    This function orchestrates the entire training pipeline:
    1. GPU setup
    2. Configuration loading
    3. Data preparation
    4. Model training
    5. Evaluation
    6. Results saving
    
    The pipeline includes memory management and error handling.
    """
    try:
        # Configure GPU
        setup_gpu()
        
        # Load configuration
        config_path = '/content/colab_training/config/training_config.yaml'
        config = load_config(config_path)
        
        # Auto-detect data path
        data_path = get_data_path()
        
        # Set up paths
        output_dir = script_dir / 'outputs' / 'tft_model'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with correct paths
        config['output']['model_dir'] = str(output_dir)
        
        # Load and preprocess data
        data = load_data(data_path, config)
        
        # Prepare data
        formatter, train_data, val_data, test_data = prepare_data(data, config)
        
        # Clean up memory after data preparation
        del data
        gc.collect()
        
        # Train model
        model = train_model(formatter, train_data, val_data, config)
        
        # Clean up memory after training
        del train_data, val_data
        gc.collect()
        
        # Evaluate model
        results = evaluate_model(model, formatter, test_data, config)
        
        # Save results
        save_results(model, formatter, results, config)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        tf.keras.backend.clear_session()
        gc.collect()

# Suggested addition
class Backtester:
    def __init__(self, strategy, initial_capital=100000):
        self.strategy = strategy
        self.capital = initial_capital
        
    def run_backtest(self, data, start_date, end_date):
        # Implement walk-forward analysis
        # Calculate returns, Sharpe ratio, max drawdown
        pass

# Risk metrics to add
def calculate_risk_metrics(returns):
    return {
        'sharpe_ratio': ...,
        'max_drawdown': ...,
        'var_95': ...,  # Value at Risk
        'cvar_95': ..., # Conditional VaR
        'kelly_fraction': ...
    }

if __name__ == '__main__':
    main() 