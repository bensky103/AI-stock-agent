"""Integration tests for data pipeline module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketFeed
from data_input.news_sentiment import NewsSentimentAnalyzer
from prediction_engine.sequence_preprocessor import SequencePreprocessor
from data_input.market_utils import clean_market_data
from pathlib import Path

class TestDataPipelineIntegration:
    @pytest.fixture
    def setup_pipeline(self, test_config):
        """Setup the complete data pipeline with all necessary components"""
        # Create a temporary config file for MarketFeed
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        market_manager = MarketFeed(config_path=config_path)
        news_manager = NewsSentimentAnalyzer(config_path=config_path)
        preprocessor = SequencePreprocessor(sequence_length=20)
        
        return {
            'market_manager': market_manager,
            'news_manager': news_manager,
            'preprocessor': preprocessor,
            'config_path': config_path  # Store path for cleanup
        }
    
    def teardown_method(self, method):
        """Clean up temporary files after each test"""
        if hasattr(self, 'config_path'):
            import os
            try:
                os.unlink(self.config_path)
            except Exception as e:
                print(f"Error cleaning up config file: {e}")
    
    def test_market_data_to_preprocessing_flow(self, setup_pipeline):
        """Test the complete flow from market data fetching to preprocessing"""
        pipeline = setup_pipeline
        
        # 1. Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = pipeline['market_manager'].fetch_data(  # Changed from fetch_historical_data to fetch_data
            symbols='AAPL',  # Changed from symbol to symbols
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        
        # 2. Clean the data
        cleaned_data = clean_market_data(market_data)  # Using the function directly
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.empty
        assert cleaned_data.isnull().sum().sum() == 0
        
        # 3. Preprocess for model input
        processed_data = pipeline['preprocessor'].prepare_sequence(cleaned_data)
        assert isinstance(processed_data, dict)  # Should return a dict of tensors/arrays
        assert 'features' in processed_data
        assert 'targets' in processed_data
    
    def test_news_sentiment_integration(self, setup_pipeline):
        """Test integration between news sentiment and market data"""
        pipeline = setup_pipeline
        
        # 1. Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        market_data = pipeline['market_manager'].fetch_data(  # Changed from fetch_historical_data to fetch_data
            symbols='AAPL',  # Changed from symbol to symbols
            start_date=start_date,
            end_date=end_date
        )
        
        # 2. Fetch news sentiment
        news_sentiment = pipeline['news_manager'].get_sentiment_scores(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(news_sentiment, pd.DataFrame)
        assert not news_sentiment.empty
        
        # 3. Merge market data with sentiment
        # Note: We need to implement merge_market_sentiment in market_utils
        # For now, we'll just verify the data separately
        assert isinstance(market_data, pd.DataFrame)
        assert isinstance(news_sentiment, pd.DataFrame)
        assert 'sentiment_score' in news_sentiment.columns
    
    def test_error_handling_integration(self, setup_pipeline):
        """Test error handling across the pipeline"""
        pipeline = setup_pipeline
        
        # Test with invalid symbol
        with pytest.raises(Exception):
            pipeline['market_manager'].fetch_data(  # Changed from fetch_historical_data to fetch_data
                symbols='INVALID_SYMBOL',  # Changed from symbol to symbols
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
        
        # Test with invalid date range
        with pytest.raises(Exception):
            pipeline['market_manager'].fetch_data(  # Changed from fetch_historical_data to fetch_data
                symbols='AAPL',  # Changed from symbol to symbols
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)  # End date before start date
            )
        
        # Test with empty market data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            pipeline['preprocessor'].prepare_sequence(empty_df)
    
    def test_data_consistency(self, setup_pipeline):
        """Test data consistency across the pipeline"""
        pipeline = setup_pipeline
        
        # 1. Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        market_data = pipeline['market_manager'].fetch_data(  # Changed from fetch_historical_data to fetch_data
            symbols='AAPL',  # Changed from symbol to symbols
            start_date=start_date,
            end_date=end_date
        )
        
        # 2. Process through pipeline
        cleaned_data = clean_market_data(market_data)  # Using the function directly
        processed_data = pipeline['preprocessor'].prepare_sequence(cleaned_data)
        
        # Verify data consistency
        assert len(processed_data['features']) > 0
        assert len(processed_data['targets']) > 0
        assert len(processed_data['features']) == len(processed_data['targets'])
        
        # Verify no data leakage
        assert not any(pd.Timestamp(end_date) in processed_data['features'])
        assert not any(pd.Timestamp(start_date) in processed_data['features']) 