"""Integration tests for data pipeline module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_input.market_feed import MarketFeed
from data_input.news_sentiment import NewsSentimentAnalyzer
from prediction_engine.sequence_preprocessor import SequencePreprocessor
from data_input.market_utils import clean_market_data
from pathlib import Path
import pytz

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
        
        market_manager = MarketFeed(config_path=config_path, default_interval='1W')  # Set weekly as default
        news_manager = NewsSentimentAnalyzer(config_path=config_path)
        preprocessor = SequencePreprocessor(sequence_length=8, data_frequency='1W')  # Use weekly data
        
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
        
        # 1. Fetch market data - use fixed historical dates instead of relative dates
        end_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)  # Fixed historical date with UTC timezone
        start_date = datetime(2022, 11, 1, tzinfo=pytz.UTC)  # ~6 months before end_date
        market_data = pipeline['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert len(market_data) >= 8  # Ensure we have enough weekly data points
        
        # 2. Clean the data
        cleaned_data = clean_market_data(market_data, remove_outliers=False)
        assert isinstance(cleaned_data, pd.DataFrame)
        assert not cleaned_data.empty
        assert cleaned_data.isnull().sum().sum() == 0
        
        # 3. Preprocess for model input
        processed_data = pipeline['preprocessor'].prepare_sequence(
            cleaned_data,
            target_col='close',  # Use 'close' as the target column
            feature_cols=['open', 'high', 'low', 'volume']  # Exclude 'close' from features
        )
        assert isinstance(processed_data, dict)
        assert 'features' in processed_data
        assert 'targets' in processed_data
        assert processed_data['data_frequency'] == '1W'  # Verify weekly data
    
    def test_news_sentiment_integration(self, setup_pipeline):
        """Test integration between news sentiment and market data"""
        pipeline = setup_pipeline
        
        # 1. Fetch market data - use fixed historical dates
        end_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)  # Fixed historical date with UTC timezone
        start_date = datetime(2022, 11, 1, tzinfo=pytz.UTC)  # ~6 months before end_date
        market_data = pipeline['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        # 2. Fetch news sentiment
        news_sentiment = pipeline['news_manager'].get_sentiment(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(news_sentiment, dict)
        assert 'AAPL' in news_sentiment
        assert isinstance(news_sentiment['AAPL'], pd.DataFrame)
        assert not news_sentiment['AAPL'].empty
        
        # 3. Verify weekly data
        assert len(market_data) >= 8  # Ensure enough weekly data points
        assert 'sentiment_score' in news_sentiment['AAPL'].columns
    
    def test_error_handling_integration(self, setup_pipeline):
        """Test error handling across the pipeline"""
        pipeline = setup_pipeline
        
        # Test with invalid symbol - use fixed historical dates
        with pytest.raises(Exception):
            # Use known historical dates
            end_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)
            start_date = datetime(2023, 4, 1, tzinfo=pytz.UTC)
            pipeline['market_manager'].fetch_data(
                symbols='INVALID_SYMBOL',
                start_date=start_date,
                end_date=end_date
            )
        
        # Test with invalid date range - use fixed historical dates
        with pytest.raises(Exception):
            # Use end date before start date
            end_date = datetime(2023, 4, 1, tzinfo=pytz.UTC)
            start_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)
            pipeline['market_manager'].fetch_data(
                symbols='AAPL',
                start_date=start_date,
                end_date=end_date  # End date before start date
            )
        
        # Test with empty market data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            pipeline['preprocessor'].prepare_sequence(empty_df)
    
    def test_data_consistency(self, setup_pipeline):
        """Test data consistency across the pipeline"""
        pipeline = setup_pipeline
        
        # 1. Get market data - use fixed historical dates
        end_date = datetime(2023, 5, 1, tzinfo=pytz.UTC)  # Fixed historical date with UTC timezone
        start_date = datetime(2022, 11, 1, tzinfo=pytz.UTC)  # ~6 months before end_date
        
        # Log the dates for debugging
        print(f"\nTest dates - Start: {start_date}, End: {end_date}")
        
        market_data = pipeline['market_manager'].fetch_data(
            symbols='AAPL',
            start_date=start_date,
            end_date=end_date,
            resample_interval='1W'  # Explicitly request weekly data
        )
        
        # Log the actual data range
        print(f"Market data range - Start: {market_data.index[0]}, End: {market_data.index[-1]}")
        
        # 2. Process through pipeline
        cleaned_data = clean_market_data(market_data, remove_outliers=False)
        processed_data = pipeline['preprocessor'].prepare_sequence(
            cleaned_data,
            target_col='close',  # Use 'close' as the target column
            feature_cols=['open', 'high', 'low', 'volume']  # Exclude 'close' from features
        )
        
        # Log the sequence timestamps for debugging
        print(f"Sequence timestamps range - Start: {processed_data['sequence_timestamps'][0]}, End: {processed_data['sequence_timestamps'][-1]}")
        
        # Verify data consistency
        assert len(processed_data['features']) > 0
        assert len(processed_data['targets']) > 0
        assert len(processed_data['features']) == len(processed_data['targets']) + 1  # Last sequence has no target
        assert processed_data['data_frequency'] == '1W'  # Verify weekly data
    
        # Verify no data leakage using the actual sequence timestamps
        sequence_timestamps = processed_data['sequence_timestamps']
        # Convert all timestamps to pandas Timestamp for consistent comparison
        end_timestamp = pd.Timestamp(end_date)
        # Extract just the timestamp part from the (symbol, timestamp) tuples
        sequence_timestamps = [pd.Timestamp(ts[1]) for ts in sequence_timestamps]
        
        # Log any problematic timestamps
        problematic_timestamps = [ts for ts in sequence_timestamps if ts >= end_timestamp]
        if problematic_timestamps:
            print(f"\nProblematic timestamps found: {problematic_timestamps}")
            print(f"End date used for comparison: {end_timestamp}")
        
        assert not any(ts >= end_timestamp for ts in sequence_timestamps) 