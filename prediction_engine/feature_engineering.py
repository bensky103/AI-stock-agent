"""Feature engineering module for stock prediction.

This module provides functionality for generating and transforming features
for TFT-based stock prediction models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator
from .sequence_preprocessor import SequencePreprocessor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

class MarketRegimeDetector:
    """Detect market regimes using technical indicators."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime using multiple indicators.
        
        Returns
        -------
        pd.Series
            Market regime labels: 1 (bullish), 0 (neutral), -1 (bearish)
        """
        # Calculate trend indicators
        sma_20 = SMAIndicator(close=df['close'], window=20).sma_indicator()
        sma_50 = SMAIndicator(close=df['close'], window=50).sma_indicator()
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        
        # Calculate momentum indicators
        rsi = RSIIndicator(close=df['close']).rsi()
        macd = MACD(close=df['close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        
        # Calculate volatility
        bb = BollingerBands(close=df['close'])
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Combine indicators for regime detection
        regime = pd.Series(0, index=df.index)  # Initialize as neutral
        
        # Trend-based rules
        regime[df['close'] > sma_20] += 1
        regime[df['close'] > sma_50] += 1
        regime[adx > 25] += 1  # Strong trend
        
        # Momentum-based rules
        regime[rsi > 60] += 1
        regime[rsi < 40] -= 1
        regime[macd_line > signal_line] += 1
        regime[macd_line < signal_line] -= 1
        
        # Volatility-based rules
        regime[bb_width > bb_width.rolling(window=self.window).mean()] -= 1  # High volatility
        
        # Normalize to [-1, 1]
        regime = regime / regime.abs().max()
        
        # Apply smoothing
        regime = regime.rolling(window=self.window).mean()
        
        return regime

class FeatureEngineer:
    """
    Enhanced feature engineering for stock prediction.
    
    This class handles:
    1. Technical indicators calculation
    2. Market regime detection
    3. Feature selection and dimensionality reduction
    4. Advanced data normalization
    5. Sequence preparation for TFT models
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        technical_indicators: Optional[List[str]] = None,
        normalize: bool = True,
        use_feature_selection: bool = True,
        n_features: int = 20,
        use_pca: bool = False,
        n_components: int = 10,
        detect_regime: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.use_pca = use_pca
        self.n_components = n_components
        self.detect_regime = detect_regime
        
        # Default technical indicators
        self.technical_indicators = technical_indicators or [
            'sma_20', 'sma_50', 'ema_20', 'ema_50',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr_14', 'vwap', 'mfi_14', 'stoch_k', 'stoch_d',
            'adx_14', 'cci_14', 'williams_r'
        ]
        
        # Initialize components
        self.sequence_preprocessor = SequencePreprocessor(sequence_length)
        self.market_regime_detector = MarketRegimeDetector() if detect_regime else None
        self.feature_selector = SelectKBest(f_regression, k=n_features) if use_feature_selection else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.scaler = RobustScaler() if normalize else None  # More robust to outliers
        
        # Store transformation parameters
        self.feature_means = None
        self.feature_stds = None
        self.selected_features = None
        self.pca_components = None
        
        logger.info(
            f"Initialized feature engineer with {len(self.technical_indicators)} "
            f"technical indicators"
        )
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Convert column names to lowercase for consistency
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        result = df.copy()
        
        # Trend indicators
        result['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        result['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        result['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        result['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        
        # Momentum indicators
        result['rsi_14'] = RSIIndicator(close=df['close']).rsi()
        macd = MACD(close=df['close'])
        result['macd'] = macd.macd()
        result['macd_signal'] = macd.macd_signal()
        result['macd_hist'] = macd.macd_diff()
        
        # Volatility indicators
        bb = BollingerBands(close=df['close'])
        result['bb_upper'] = bb.bollinger_hband()
        result['bb_middle'] = bb.bollinger_mavg()
        result['bb_lower'] = bb.bollinger_lband()
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # Volume indicators
        result['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).volume_weighted_average_price()
        result['mfi_14'] = MFIIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).money_flow_index()
        
        # Additional momentum indicators
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        result['stoch_k'] = stoch.stoch()
        result['stoch_d'] = stoch.stoch_signal()
        
        # Trend strength indicators
        result['adx_14'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        
        # Market regime
        if self.detect_regime:
            result['market_regime'] = self.market_regime_detector.detect_regime(df)
        
        return result
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """Normalize features using robust scaling."""
        if not self.normalize:
            return df, None, None
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            # Fit scaler
            self.scaler.fit(df[numeric_cols])
            self.feature_means = self.scaler.center_
            self.feature_stds = self.scaler.scale_
        
        # Transform data
        normalized = df.copy()
        normalized[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return normalized, self.feature_means, self.feature_stds
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Select most relevant features."""
        if not self.use_feature_selection:
            return X, None
        
        if fit:
            # Fit feature selector
            self.feature_selector.fit(X, y)
            self.selected_features = self.feature_selector.get_support()
        
        # Transform data
        if self.selected_features is not None:
            X_selected = X[:, self.selected_features]
        else:
            X_selected = X
        
        return X_selected, self.selected_features
    
    def apply_pca(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply PCA for dimensionality reduction."""
        if not self.use_pca:
            return X, None
        
        if fit:
            # Fit PCA
            self.pca.fit(X)
            self.pca_components = self.pca.components_
        
        # Transform data
        if self.pca_components is not None:
            X_pca = self.pca.transform(X)
        else:
            X_pca = X
        
        return X_pca, self.pca_components
    
    def prepare_sequences(
        self,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for TFT model input.
        
        Args:
            market_data: DataFrame containing market data
            sentiment_data: Optional DataFrame containing sentiment data
            fit: Whether to fit transformers on the data
            
        Returns:
            Tuple of (market_features, targets, sentiment_features)
            market_features: numpy array of shape (n_samples, sequence_length, n_features)
            targets: numpy array of shape (n_samples, prediction_horizon)
            sentiment_features: Optional numpy array of shape (n_samples, n_sentiment_features)
        """
        # Calculate technical indicators
        market_with_indicators = self.calculate_technical_indicators(market_data)
        
        # Normalize market data
        market_normalized, means, stds = self.normalize_features(
            market_with_indicators,
            fit=fit
        )
        
        # Prepare sequences
        sequences = self.sequence_preprocessor.prepare_sequence(
            market_normalized,
            target_col='close',
            feature_cols=[col for col in market_normalized.columns if col != 'close']
        )
        
        X = sequences['features']
        y = sequences['targets']
        
        # Apply feature selection if enabled
        if self.use_feature_selection:
            X, selected_features = self.select_features(X, y, fit=fit)
        
        # Apply PCA if enabled
        if self.use_pca:
            X, pca_components = self.apply_pca(X, fit=fit)
        
        # Prepare sentiment features if available
        sentiment_features = None
        if sentiment_data is not None and not sentiment_data.empty:
            # Align sentiment data with market data
            sentiment_aligned = sentiment_data.set_index(['symbol', 'datetime'])
            sentiment_aligned = sentiment_aligned.reindex(market_data.index)
            
            # Normalize sentiment features
            sentiment_normalized, _, _ = self.normalize_features(
                sentiment_aligned,
                fit=fit
            )
            
            # Get sentiment features for each sequence
            sentiment_features = []
            for i in range(len(X)):
                sentiment_features.append(
                    sentiment_normalized.iloc[i + self.sequence_length].values
                )
            sentiment_features = np.array(sentiment_features)
        
        return X, y, sentiment_features
    
    def prepare_prediction_data(
        self,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for making predictions.
        
        Args:
            market_data: DataFrame containing market data
            sentiment_data: Optional DataFrame containing sentiment data
            
        Returns:
            Tuple of (market_features, sentiment_features)
            market_features: numpy array of shape (n_samples, sequence_length, n_features)
            sentiment_features: Optional numpy array of shape (n_samples, n_sentiment_features)
        """
        # Ensure we have enough data
        if len(market_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} time steps of market data")
        
        # Calculate technical indicators
        market_with_indicators = self.calculate_technical_indicators(market_data)
        
        # Normalize market data
        market_normalized, _, _ = self.normalize_features(market_with_indicators)
        
        # Prepare sequences
        sequences = []
        for i in range(len(market_normalized) - self.sequence_length + 1):
            sequence = market_normalized.iloc[i:i + self.sequence_length].values
            sequences.append(sequence)
        
        market_features = np.array(sequences)
        
        # Prepare sentiment features if available
        sentiment_features = None
        if sentiment_data is not None and not sentiment_data.empty:
            # Get sentiment features for each sequence
            sentiment_sequences = []
            for i in range(len(sentiment_data) - self.sequence_length + 1):
                sentiment_sequence = sentiment_data.iloc[i:i + self.sequence_length].values
                sentiment_sequences.append(sentiment_sequence)
            
            sentiment_features = np.array(sentiment_sequences)
        
        return market_features, sentiment_features

    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction.
        
        This is a wrapper around prepare_prediction_data that returns only the 
        market features.
        
        Args:
            market_data: DataFrame containing market data
            
        Returns:
            numpy array of processed features
        """
        # Debug logging
        logger.setLevel(logging.DEBUG)
        logger.debug(f"prepare_features called with market_data shape: {market_data.shape}")
        logger.debug(f"self.sequence_length = {self.sequence_length}")
        
        # Ensure we have enough data
        if len(market_data) < self.sequence_length:
            logger.warning(f"Insufficient data: got {len(market_data)} rows, need at least {self.sequence_length}")
            raise ValueError(f"Need at least {self.sequence_length} time steps of market data")
        
        try:
            # Calculate technical indicators
            logger.debug("Calculating technical indicators")
            market_with_indicators = self.calculate_technical_indicators(market_data)
            logger.debug(f"Indicators calculated, result shape: {market_with_indicators.shape}")
            
            # Normalize market data
            logger.debug("Normalizing market data")
            market_normalized, _, _ = self.normalize_features(market_with_indicators)
            logger.debug(f"Data normalized, result shape: {market_normalized.shape}")
            
            # Prepare sequences
            logger.debug(f"Preparing sequences with length {self.sequence_length}")
            sequences = []
            for i in range(len(market_normalized) - self.sequence_length + 1):
                sequence = market_normalized.iloc[i:i + self.sequence_length].values
                sequences.append(sequence)
            
            result = np.array(sequences)
            logger.debug(f"Sequences prepared, result shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from market data.
        
        Args:
            df: DataFrame containing market data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            DataFrame with generated features
        """
        # Convert column names to lowercase for consistency
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate technical indicators
        features = self.calculate_technical_indicators(df)
        
        # Normalize features with fit=True to ensure scaler is fitted
        features, _, _ = self.normalize_features(features, fit=True)
        
        return features

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        sequence_length=10,
        use_feature_selection=True,
        use_pca=True,
        detect_regime=True
    )
    
    # Prepare sequences
    X, y, _ = engineer.prepare_sequences(data, fit=True)
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Selected features: {engineer.selected_features.sum() if engineer.selected_features is not None else 'None'}")
    print(f"PCA components: {engineer.pca_components.shape if engineer.pca_components is not None else 'None'}") 