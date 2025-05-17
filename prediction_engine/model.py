"""Enhanced LSTM model for stock price prediction."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

class AttentionLayer(nn.Module):
    """Attention layer for sequence processing."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class EnhancedLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,  # Reduced default hidden size
        num_layers: int = 1,    # Single layer by default
        dropout: float = 0.1,
        bidirectional: bool = False,  # Disable bidirectional by default
        attention: bool = False,      # Disable attention by default
        residual: bool = True,        # Keep residual connections
        uncertainty: bool = False     # Disable uncertainty by default
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        self.residual = residual
        self.uncertainty = uncertainty
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (only if enabled)
        if attention:
            self.attention_layer = nn.Linear(
                hidden_size * (2 if bidirectional else 1),
                hidden_size
            )
        
        # Dense layers (simplified)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Uncertainty estimation (only if enabled)
        if uncertainty:
            self.uncertainty_layer = nn.Linear(hidden_size, 1)
        
        logger.info(f"Initialized Enhanced LSTM model with {num_layers} layers, "
                   f"hidden size {hidden_size}, bidirectional={bidirectional}, "
                   f"attention={attention}, residual={residual}, uncertainty={uncertainty}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention if enabled
        if self.attention:
            attention_weights = torch.softmax(
                self.attention_layer(lstm_out),
                dim=1
            )
            lstm_out = lstm_out * attention_weights
        
        # Get final sequence output
        if self.bidirectional:
            # Concatenate forward and backward states
            final_out = torch.cat(
                (lstm_out[:, -1, :self.hidden_size],
                 lstm_out[:, 0, self.hidden_size:]),
                dim=1
            )
        else:
            final_out = lstm_out[:, -1, :]
        
        # Apply dense layers
        pred = self.dense_layers(final_out)
        
        # Estimate uncertainty if enabled
        uncertainty = None
        if self.uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_layer(final_out))
        
        return pred, uncertainty

def create_model(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    use_sentiment: bool = True,
    bidirectional: bool = True,
    use_attention: bool = True,
    use_residual: bool = True,
    estimate_uncertainty: bool = True
) -> EnhancedLSTM:
    """Create a new enhanced LSTM model instance.
    
    Args:
        input_size: Size of input features
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        use_sentiment: Whether to use sentiment features (not implemented yet)
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use attention mechanism
        use_residual: Whether to use residual connections
        estimate_uncertainty: Whether to estimate prediction uncertainty
        
    Returns:
        EnhancedLSTM: Configured LSTM model instance
    """
    return EnhancedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        attention=use_attention,
        residual=use_residual,
        uncertainty=estimate_uncertainty
    )

if __name__ == "__main__":
    # Example usage
    model = create_model(input_size=5)  # OHLCV
    print(model)
    
    # Test forward pass
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 5)
    sentiment = torch.randn(batch_size, 3)
    
    with torch.no_grad():
        mean_pred, uncertainty = model(x, sentiment)
        print(f"Input shape: {x.shape}")
        print(f"Mean prediction shape: {mean_pred.shape}")
        print(f"Uncertainty shape: {uncertainty.shape if uncertainty is not None else None}") 