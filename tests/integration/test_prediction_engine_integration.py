import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prediction_engine.sequence_preprocessor import SequencePreprocessor
from prediction_engine.feature_engineering import FeatureEngineer
from prediction_engine.scaler_handler import ScalerHandler
from prediction_engine.predictor import EnhancedStockPredictor

class TestPredictionEngineIntegration:
    @pytest.fixture
    def setup_prediction_components(self, test_config):
        """Setup prediction engine components"""
        preprocessor = SequencePreprocessor(sequence_length=20)
        feature_engineer = FeatureEngineer(config=test_config)
        scaler_handler = ScalerHandler(config=test_config)
        predictor = EnhancedStockPredictor(
            sequence_length=20,
            prediction_horizon=5,
            device='cpu'
        )
        
        return {
            'preprocessor': preprocessor,
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
        scaled_features = components['scaler_handler'].scale_features(features)
        assert isinstance(scaled_features, pd.DataFrame)
        assert not scaled_features.empty
        assert scaled_features.shape == features.shape
        
        # 3. Prepare sequence
        sequence = components['preprocessor'].prepare_sequence(scaled_features)
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
        assert 'RSI' in tech_indicators.columns
        assert 'MACD' in tech_indicators.columns
        
        # Test statistical features
        stat_features = components['feature_engineer'].calculate_statistical_features(
            sample_market_data
        )
        assert isinstance(stat_features, pd.DataFrame)
        assert not stat_features.empty
        assert 'volatility' in stat_features.columns
        assert 'returns' in stat_features.columns
    
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
        scaled_features = components['scaler_handler'].scale_features(features)
        
        # Test sequence creation
        sequence = components['preprocessor'].create_sequence(
            scaled_features,
            sequence_length=10,
            target_length=5
        )
        assert isinstance(sequence, dict)
        assert 'features' in sequence
        assert 'targets' in sequence
        assert len(sequence['features']) > 0
        assert len(sequence['targets']) > 0
        
        # Test sequence validation
        is_valid = components['preprocessor'].validate_sequence(sequence)
        assert is_valid
    
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
            components['preprocessor'].prepare_sequence(invalid_df)
        
        # Test with invalid sequence length
        valid_df = pd.DataFrame({
            'Close': np.random.normal(100, 2, 5),
            'Volume': np.random.normal(1000000, 200000, 5)
        })
        with pytest.raises(Exception):
            components['preprocessor'].create_sequence(
                valid_df,
                sequence_length=10,  # Longer than data
                target_length=5
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
        sequence = components['preprocessor'].create_sequence(
            scaled_features,
            sequence_length=10,
            target_length=5
        )
        
        # Verify data consistency
        assert len(sequence['features']) == len(sequence['targets'])
        assert not np.isnan(sequence['features']).any()
        assert not np.isnan(sequence['targets']).any()
        
        # Verify inverse transform
        original_features = components['scaler_handler'].inverse_transform_data(
            scaled_features, scaler
        )
        assert original_features.shape == features.shape
        assert np.allclose(original_features, features, rtol=1e-5) 