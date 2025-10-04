"""
Simplified ASL Model for Testing
A lightweight model that doesn't require PyTorch
"""
import numpy as np
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SimplifiedASLModel:
    """Simplified ASL model for testing without PyTorch dependency"""
    
    def __init__(self, feature_dim: int = 258, num_classes: int = 21):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model_type = "SimplifiedASL"
        
        # Default class names for ASL expressions
        self.class_names = [
            "AMAZING", "BAD", "BEAUTIFUL", "DEAF", "FINE", "GOOD", "GOODBYE",
            "HEARING", "HELLO", "HELP", "HOW_ARE_YOU", "LEARN", "LOVE", "NAME",
            "NICE_TO_MEET_YOU", "NO", "PLEASE", "SORRY", "THANK_YOU", "UNDERSTAND", "YES"
        ]
        
        # Initialize random weights for demo
        self.weights = np.random.randn(feature_dim, num_classes) * 0.1
        self.bias = np.random.randn(num_classes) * 0.1
        
        logger.info(f"Initialized SimplifiedASLModel with {num_classes} classes")
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict ASL expression for a single feature vector"""
        # Simple linear transformation + softmax
        if isinstance(features, list):
            features = np.array(features)
        
        # Ensure correct shape
        if features.shape[0] != self.feature_dim:
            # Pad or truncate to correct size
            padded_features = np.zeros(self.feature_dim)
            min_len = min(len(features), self.feature_dim)
            padded_features[:min_len] = features[:min_len]
            features = padded_features
        
        # Simple prediction logic
        scores = np.dot(features, self.weights) + self.bias
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))  # Stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(self.class_names):
            prob_dict[class_name] = float(probabilities[i])
        
        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": float(confidence),
            "predicted_index": int(predicted_idx),
            "probabilities": prob_dict
        }
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Dict]:
        """Predict ASL expressions for a batch of feature vectors"""
        results = []
        for features in features_batch:
            results.append(self.predict_single(features))
        return results
    
    def save_model(self, filepath: str, config: Optional[Dict] = None):
        """Save the model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist(),
            'model_config': {
                'feature_dim': self.feature_dim,
                'num_classes': self.num_classes,
                'model_type': self.model_type,
                'class_names': self.class_names
            }
        }
        
        if config:
            save_data['training_config'] = config
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Simplified model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SimplifiedASLModel':
        """Load a saved model"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}, creating default model")
            return cls()
        
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            model_config = save_data['model_config']
            model = cls(
                feature_dim=model_config.get('feature_dim', 258),
                num_classes=model_config.get('num_classes', 21)
            )
            
            # Load weights and bias
            model.weights = np.array(save_data['weights'])
            model.bias = np.array(save_data['bias'])
            
            # Load class names
            if 'class_names' in model_config:
                model.class_names = model_config['class_names']
            
            logger.info(f"Simplified model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Creating default model instead")
            return cls()
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            "model_type": self.model_type,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "weights_shape": list(self.weights.shape),
            "bias_shape": list(self.bias.shape)
        }

# Alias for compatibility
FeatureBasedASLModel = SimplifiedASLModel

def create_default_model() -> SimplifiedASLModel:
    """Create a default model for testing purposes"""
    model = SimplifiedASLModel(
        feature_dim=258,
        num_classes=21
    )
    
    logger.info("Created default simplified ASL model")
    return model