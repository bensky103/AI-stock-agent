# File: main.py

import logging

from data_input.market_feed import get_market_data
from prediction_engine.sequence_preprocessor import SequencePreprocessor
from feature_engineering.feature_engineer import FeatureEngineer
from prediction_engine.enhanced_stock_predictor import EnhancedStockPredictor

def main():
    """Main entry point for the stock prediction system."""
    logging.basicConfig(level=logging.INFO)

    # 1. Load configuration
    config = load_config()
    
    # 2. Get market data
    market_data = get_market_data(config)
    
    # 3. Get sentiment data
    sentiment_data = get_sentiment_data(config)
    
    # 4. Prepare features and sequences for TFT model
    feature_engineer = FeatureEngineer(
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon'],
        use_feature_selection=True,
        use_pca=False,
        detect_regime=True
    )
    
    X, y, sentiment_features = feature_engineer.prepare_sequences(
        market_data,
        sentiment_data,
        fit=True
    )
    
    # 5. Initialize and train TFT predictor
    predictor = EnhancedStockPredictor(
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon'],
        device='cpu',
        model_type='tft',
        use_feature_selection=True,
        use_pca=False,
        detect_regime=True
    )
    
    # 6. Train model
    predictor.train(
        X,
        y,
        sentiment_features=sentiment_features,
        validation_split=0.2,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size']
    )
    
    # 7. Make predictions
    predictions = predictor.predict(
        X[-1:],  # Use last sequence for prediction
        sentiment_features=sentiment_features[-1:] if sentiment_features is not None else None
    )
    
    # 8. Save model and results
    predictor.save_model('AAPL')  # Save model for symbol
    save_predictions(predictions, config['output']['predictions_path'])

if __name__ == "__main__":
    main()

