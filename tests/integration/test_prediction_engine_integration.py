import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prediction_engine.sequence_preprocessor import SequencePreprocessor
from prediction_engine.feature_engineering import FeatureEngineer
from prediction_engine.scaler_handler import ScalerHandler
from prediction_engine.predictor import EnhancedStockPredictor
from pathlib import Path
import yaml
from prediction_engine.exceptions import StockPredictorError
import json

class TestPredictionEngineIntegration:
    @pytest.fixture
    def setup_prediction_components(self, test_config):
        """Set up prediction components for testing"""
        # Create model directory and config
        model_path = Path("colab_training/tft_model")
        config_path = model_path / "model_config.json"
        
        # Create model config if it doesn't exist
        if not config_path.exists():
            model_config = {
                "model_config": {
                    "hidden_layer_size": 128,
                    "attention_head_size": 8,
                    "dropout_rate": 0.1,
                    "max_gradient_norm": 0.01,
                    "learning_rate": 0.0005,
                    "num_encoder_steps": 30,
                    "num_steps": 5
                },
                "best_val_loss": float('inf'),
                "best_epoch": 0,
                "training_history": []
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(model_config, f)
        
        # Create necessary directories and files
        saved_models_dir = Path("saved_models")
        saved_models_dir.mkdir(exist_ok=True)
        
        # Create dummy model and config files
        model_path = saved_models_dir / "tft_model"
        model_path.mkdir(exist_ok=True)
        
        # Copy training config from colab_training
        colab_model_path = Path("colab_training/tft_model")
        if colab_model_path.exists():
            # Copy model config
            with open(colab_model_path / "model_config.json", 'r') as f:
                model_config = json.load(f)
            with open(model_path / "model_config.json", 'w') as f:
                json.dump(model_config, f)
            
            # Copy training config
            with open(colab_model_path / "config.yaml", 'r') as f:
                training_config = yaml.safe_load(f)
            with open(saved_models_dir / "training_config.json", 'w') as f:
                json.dump(training_config, f)
            
            # Copy formatter config if it exists
            formatter_config_path = colab_model_path / "formatter_config.json"
            if formatter_config_path.exists():
                with open(formatter_config_path, 'r') as f:
                    formatter_config = json.load(f)
                with open(model_path / "formatter_config.json", 'w') as f:
                    json.dump(formatter_config, f)
        else:
            # Create default model config if colab_training files don't exist
            model_config = {
                'model_config': {
                    'hidden_layer_size': 128,
                    'attention_head_size': 8,
                    'dropout_rate': 0.1,
                    'max_gradient_norm': 0.01,
                    'learning_rate': 0.0005,
                    'num_encoder_steps': 30,
                    'num_steps': 5
                },
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'training_history': []
            }
            with open(model_path / "model_config.json", 'w') as f:
                json.dump(model_config, f)
            
            # Create default training config
            training_config = {
                'model': {
                    'sequence_length': 20,
                    'prediction_horizon': 1,
                    'hidden_size': 64,
                    'num_heads': 4,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'max_epochs': 100,
                    'early_stopping_patience': 10
                }
            }
            with open(saved_models_dir / "training_config.json", 'w') as f:
                json.dump(training_config, f)
        
        # Create main config file
        config_path = saved_models_dir / "tft_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump({
                'sequence_length': 20,
                'prediction_horizon': 1,
                'hidden_size': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1
            }, f)
        
        # Initialize components with correct arguments
        feature_engineer = FeatureEngineer(
            sequence_length=20,
            prediction_horizon=1,
            normalize=True,
            use_feature_selection=True,
            n_features=20,
            use_pca=False,
            n_components=10,
            detect_regime=True
        )
        
        scaler_handler = ScalerHandler(
            model_type='tft',
            scaler_type='standard'
        )
        
        predictor = EnhancedStockPredictor(
            sequence_length=20,
            prediction_horizon=1,
            device='cpu',
            model_type='tft',
            use_feature_selection=True,
            use_pca=False,
            detect_regime=True
        )
        
        return {
            'config_path': config_path,
            'model_path': model_path,
            'feature_engineer': feature_engineer,
            'scaler_handler': scaler_handler,
            'predictor': predictor
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, 100),
            'High': np.random.normal(102, 2, 100),
            'Low': np.random.normal(98, 2, 100),
            'Close': np.random.normal(100, 2, 100),
            'Volume': np.random.normal(1000000, 200000, 100),
        }, index=dates)
        return data
    
    def test_feature_engineering_flow(self, setup_prediction_components, sample_market_data):
        """Test the complete feature engineering flow"""
        components = setup_prediction_components
        
        # 1. Generate features
        features = components['feature_engineer'].generate_features(sample_market_data)
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert len(features.columns) > len(sample_market_data.columns)
        
        # 2. Scale features
        scaler = components['scaler_handler'].fit_scaler(features)
        scaled_features = components['scaler_handler'].transform_data(features, scaler)
        assert isinstance(scaled_features, pd.DataFrame)
        assert not scaled_features.empty
        assert scaled_features.shape == features.shape
        
        # 3. Prepare sequence
        feature_cols = [col for col in scaled_features.columns if col != 'close']
        sequence = components['feature_engineer'].sequence_preprocessor.prepare_sequence(
            scaled_features,
            target_col='close',
            feature_cols=feature_cols
        )
        assert isinstance(sequence, dict)
        assert 'features' in sequence
        assert 'targets' in sequence
    
    def test_feature_engineering_functions(self, setup_prediction_components, sample_market_data):
        """Test individual feature engineering functions"""
        components = setup_prediction_components
        
        # Test technical indicators
        tech_indicators = components['feature_engineer'].calculate_technical_indicators(
            sample_market_data
        )
        assert isinstance(tech_indicators, pd.DataFrame)
        assert not tech_indicators.empty
        assert 'rsi_14' in tech_indicators.columns
        assert 'macd' in tech_indicators.columns
        assert 'sma_20' in tech_indicators.columns
        assert 'ema_12' in tech_indicators.columns
        assert 'bollinger_upper' in tech_indicators.columns
        assert 'bollinger_lower' in tech_indicators.columns
    
    def test_scaler_operations(self, setup_prediction_components, sample_market_data):
        """Test scaler operations"""
        components = setup_prediction_components
        
        # Test fit and transform
        features = components['feature_engineer'].generate_features(sample_market_data)
        scaler = components['scaler_handler'].fit_scaler(features)
        assert scaler is not None
        
        # Test transform
        scaled_data = components['scaler_handler'].transform_data(features, scaler)
        assert isinstance(scaled_data, pd.DataFrame)
        assert not scaled_data.empty
        assert scaled_data.shape == features.shape
        
        # Test inverse transform
        original_data = components['scaler_handler'].inverse_transform_data(
            scaled_data, scaler
        )
        assert isinstance(original_data, pd.DataFrame)
        assert not original_data.empty
        assert original_data.shape == features.shape
    
    def test_sequence_preprocessing(self, setup_prediction_components, sample_market_data):
        """Test sequence preprocessing operations"""
        components = setup_prediction_components
        
        # Generate and scale features
        features = components['feature_engineer'].generate_features(sample_market_data)
        scaler = components['scaler_handler'].fit_scaler(features)
        scaled_features = components['scaler_handler'].transform_data(features, scaler)
        
        # Create sequence
        sequence = components['feature_engineer'].sequence_preprocessor.prepare_sequence(
            scaled_features,
            target_col='close',
            feature_cols=[col for col in scaled_features.columns if col != 'close']
        )
        assert isinstance(sequence, dict)
        assert 'features' in sequence
        assert 'targets' in sequence
        assert sequence['features'].shape[1] == components['feature_engineer'].sequence_length
    
    def test_error_handling(self, setup_prediction_components):
        """Test error handling in prediction components"""
        components = setup_prediction_components
        
        # Test with empty data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            components['feature_engineer'].generate_features(empty_df)
        
        # Test with invalid data
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
        with pytest.raises(Exception):
            components['feature_engineer'].sequence_preprocessor.prepare_sequence(invalid_df)
        
        # Test with invalid sequence length
        valid_df = pd.DataFrame({
            'Close': np.random.normal(100, 2, 5),
            'Volume': np.random.normal(1000000, 200000, 5)
        })
        with pytest.raises(Exception):
            components['feature_engineer'].sequence_preprocessor.prepare_sequence(
                valid_df,
                target_col='close',
                feature_cols=[col for col in valid_df.columns if col != 'close']
            )
    
    def test_data_consistency(self, setup_prediction_components, sample_market_data):
        """Test data consistency across prediction components"""
        components = setup_prediction_components
        
        # Generate features
        features = components['feature_engineer'].generate_features(sample_market_data)
        
        # Scale features
        scaler = components['scaler_handler'].fit_scaler(features)
        scaled_features = components['scaler_handler'].transform_data(features, scaler)
        
        # Create sequence
        sequence = components['feature_engineer'].sequence_preprocessor.prepare_sequence(
            scaled_features,
            target_col='close',
            feature_cols=[col for col in scaled_features.columns if col != 'close']
        )
        
        assert isinstance(sequence, dict)
        assert 'features' in sequence
        assert 'targets' in sequence
        assert sequence['features'].shape[1] == components['feature_engineer'].sequence_length
        
        # Verify inverse transform
        original_features = components['scaler_handler'].inverse_transform_data(
            scaled_features, scaler
        )
        assert original_features.shape == features.shape
        assert np.allclose(original_features, features, rtol=1e-3, atol=1e-3, equal_nan=True)
    
    def test_model_loading(self, setup_prediction_components):
        """Test model loading functionality."""
        components = setup_prediction_components
        predictor = components['predictor']
        
        # Test loading model
        predictor.load_model(components['model_path'])
        assert predictor.model is not None
        
        # Test loading with invalid path
        with pytest.raises(StockPredictorError):
            predictor.load_model(Path("invalid_path")) 