"""Model interpretability and prediction explanations.

This module provides tools for explaining model predictions and analyzing
feature importance in the TFT model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExplanationConfig:
    """Configuration for model explanations."""
    # Feature importance
    n_top_features: int = 10
    
    # Attention analysis
    attention_threshold: float = 0.1
    
    # Decision path
    min_confidence: float = 0.7
    max_path_length: int = 5

class ModelExplainer:
    """Class for explaining TFT model predictions."""
    
    def __init__(self, model, config: Optional[ExplanationConfig] = None):
        """Initialize the explainer.
        
        Args:
            model: Trained TFT model
            config: Explanation configuration
        """
        self.model = model
        self.config = config or ExplanationConfig()
    
    def explain_prediction(self, 
                          inputs: Dict[str, np.ndarray],
                          prediction: np.ndarray) -> Dict:
        """Explain a single prediction.
        
        Args:
            inputs: Model inputs
            prediction: Model prediction
            
        Returns:
            Dictionary with explanation components
        """
        # Get attention weights
        attention_weights = self.model.get_attention_weights(inputs)
        
        # Get feature importance
        feature_importance = self._get_feature_importance(inputs, prediction)
        
        # Get decision path
        decision_path = self._get_decision_path(inputs, prediction)
        
        return {
            'attention_weights': attention_weights,
            'feature_importance': feature_importance,
            'decision_path': decision_path,
            'prediction_confidence': self._get_prediction_confidence(prediction)
        }
    
    def _get_feature_importance(self,
                              inputs: Dict[str, np.ndarray],
                              prediction: np.ndarray) -> Dict[str, float]:
        """Extract feature importance from inputs."""
        importance = {}
        
        # Handle both dictionary and tuple inputs
        if isinstance(inputs, tuple):
            # Convert tuple to dictionary format
            static_inputs, historical_inputs, future_inputs = inputs
            inputs_dict = {
                'static': static_inputs,
                'historical': historical_inputs, 
                'future': future_inputs
            }
        else:
            inputs_dict = inputs
        
        # Process each input type
        for input_type, input_data in inputs_dict.items():
            if input_type in ['static', 'historical', 'future']:
                # Get feature names for this input type
                feature_names = self._get_feature_names(input_type)
                
                # Calculate importance (using variance as a simple metric)
                if len(input_data.shape) > 2:
                    # For 3D inputs (historical/future), take mean across time
                    feature_variance = np.var(input_data, axis=(0, 1))
                else:
                    # For 2D inputs (static)
                    feature_variance = np.var(input_data, axis=0)
                
                # Ensure we have the correct number of features
                actual_features = len(feature_variance)
                expected_features = len(feature_names)
                
                if actual_features != expected_features:
                    logger.warning(f"Feature count mismatch for {input_type}: "
                                 f"expected {expected_features}, got {actual_features}")
                    # Adjust feature names to match actual features
                    if actual_features < expected_features:
                        feature_names = feature_names[:actual_features]
                    else:
                        # Pad with generic names if needed
                        feature_names.extend([f"feature_{i}" for i in range(expected_features, actual_features)])
                
                # Normalize importance
                total_variance = np.sum(feature_variance)
                if total_variance > 0:
                    normalized_importance = feature_variance / total_variance
                else:
                    normalized_importance = np.ones_like(feature_variance) / len(feature_variance)
                
                # Store with feature names
                for i, name in enumerate(feature_names[:len(normalized_importance)]):
                    importance[f"{input_type}_{name}"] = normalized_importance[i]
        
        return importance
    
    def _get_decision_path(self,
                          inputs: Dict[str, np.ndarray],
                          prediction: np.ndarray) -> List[Dict]:
        """Get the decision path for a prediction.
        
        Args:
            inputs: Model inputs
            prediction: Model prediction
            
        Returns:
            List of decision steps
        """
        decision_path = []
        
        # Get attention weights
        attention_weights = self.model.get_attention_weights(inputs)
        
        # Process each attention head
        for head_idx, head_weights in enumerate(attention_weights):
            # Get top attended features
            top_features = self._get_top_attended_features(
                head_weights, 
                threshold=self.config.attention_threshold
            )
            
            if top_features:
                decision_path.append({
                    'step': len(decision_path) + 1,
                    'attention_head': head_idx,
                    'features': top_features,
                    'confidence': self._get_step_confidence(head_weights)
                })
            
            # Limit path length
            if len(decision_path) >= self.config.max_path_length:
                break
        
        return decision_path
    
    def _get_prediction_confidence(self, prediction: np.ndarray) -> float:
        """Calculate prediction confidence.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate confidence based on prediction variance
        pred_std = np.std(prediction)
        pred_mean = np.mean(prediction)
        
        # Normalize confidence
        confidence = 1.0 / (1.0 + pred_std / (abs(pred_mean) + 1e-6))
        
        return float(confidence)
    
    def _get_step_confidence(self, attention_weights: np.ndarray) -> float:
        """Calculate confidence for a decision step.
        
        Args:
            attention_weights: Attention weights for the step
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate confidence based on attention weight distribution
        weight_entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-6)
        )
        max_entropy = np.log(len(attention_weights))
        
        # Normalize confidence
        confidence = 1.0 - (weight_entropy / max_entropy)
        
        return float(confidence)
    
    def _get_top_attended_features(self,
                                 attention_weights: np.ndarray,
                                 threshold: float) -> List[Tuple[str, float]]:
        """Get top attended features based on attention weights.
        
        Args:
            attention_weights: Attention weights
            threshold: Minimum attention weight threshold
            
        Returns:
            List of (feature_name, weight) tuples
        """
        # Get feature names
        feature_names = self._get_all_feature_names()
        
        # Get top features
        top_indices = np.where(attention_weights > threshold)[0]
        top_features = [
            (feature_names[i], float(attention_weights[i]))
            for i in top_indices
        ]
        
        # Sort by weight
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        return top_features
    
    def _get_feature_names(self, input_type: str) -> List[str]:
        """Get feature names for an input type.
        
        Args:
            input_type: Type of input ('static', 'historical', or 'future')
            
        Returns:
            List of feature names
        """
        # This should be implemented based on your model's feature structure
        if input_type == 'static':
            return ['market_cap', 'sector', 'industry']
        elif input_type == 'historical':
            return [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower',
                'volume_ratio', 'volatility'
            ]
        elif input_type == 'future':
            return ['day_of_week', 'month', 'is_holiday']
        else:
            return []
    
    def _get_all_feature_names(self) -> List[str]:
        """Get all feature names across input types.
        
        Returns:
            List of all feature names
        """
        all_features = []
        for input_type in ['static', 'historical', 'future']:
            all_features.extend(self._get_feature_names(input_type))
        return all_features

def generate_explanation_report(explanation: Dict,
                              save_path: Optional[str] = None) -> str:
    """Generate a human-readable explanation report.
    
    Args:
        explanation: Explanation dictionary from ModelExplainer
        save_path: Optional path to save the report
        
    Returns:
        Formatted explanation report
    """
    report = []
    
    # Prediction confidence
    confidence = explanation['prediction_confidence']
    report.append(f"Prediction Confidence: {confidence:.2%}")
    report.append("")
    
    # Feature importance
    report.append("Top Features by Importance:")
    feature_importance = explanation['feature_importance']
    top_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, importance in top_features:
        report.append(f"- {feature}: {importance:.2%}")
    report.append("")
    
    # Decision path
    report.append("Decision Path:")
    for step in explanation['decision_path']:
        report.append(f"Step {step['step']} (Confidence: {step['confidence']:.2%}):")
        for feature, weight in step['features']:
            report.append(f"  - {feature}: {weight:.2%}")
        report.append("")
    
    # Format report
    report_text = "\n".join(report)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text

def plot_feature_importance(explanation: Dict,
                          save_path: Optional[str] = None):
    """Plot feature importance.
    
    Args:
        explanation: Explanation dictionary from ModelExplainer
        save_path: Optional path to save the plot
    """
    # Get feature importance
    feature_importance = explanation['feature_importance']
    
    # Create DataFrame
    df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['feature', 'importance']
    )
    df = df.sort_values('importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_attention_weights(explanation: Dict,
                         save_path: Optional[str] = None):
    """Plot attention weights.
    
    Args:
        explanation: Explanation dictionary from ModelExplainer
        save_path: Optional path to save the plot
    """
    # Get attention weights
    attention_weights = explanation['attention_weights']
    
    # Create plot
    n_heads = len(attention_weights)
    fig, axes = plt.subplots(n_heads, 1, figsize=(10, 4*n_heads))
    
    for i, (head_weights, ax) in enumerate(zip(attention_weights, axes)):
        # Plot attention weights
        sns.heatmap(
            head_weights,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'}
        )
        ax.set_title(f'Attention Head {i+1}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Time Step')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.close() 