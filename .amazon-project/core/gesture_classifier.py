"""
Professional Gesture Classifier Adapter
Integrates with existing ASL models for real-time recognition

This module provides a clean interface to connect your existing
ASL models with the professional camera system.
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Any
from abc import ABC, abstractmethod
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class GestureClassifierInterface(ABC):
    """Abstract interface for gesture classifiers"""
    
    @abstractmethod
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from landmarks
        
        Args:
            landmarks: Hand landmarks as numpy array
            
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if classifier is ready for predictions"""
        pass

class SimpleGestureClassifier(GestureClassifierInterface):
    """
    Simple gesture classifier using basic hand geometry
    This is a demonstration classifier - replace with your actual model
    """
    
    def __init__(self):
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        logger.info("Simple gesture classifier initialized")
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Simple prediction based on hand geometry"""
        try:
            if landmarks is None or len(landmarks) == 0:
                return "Unknown", 0.0
            
            # Convert to numpy array if needed
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks)
            
            # Flatten if 3D
            if landmarks.ndim > 2:
                landmarks = landmarks.reshape(-1, landmarks.shape[-1])
            
            # Simple finger counting for demonstration
            fingers_up = self._count_fingers(landmarks)
            
            # Map finger count to letters (simplified)
            letter_map = {
                0: 'A',  # Fist
                1: 'D',  # One finger
                2: 'V',  # Two fingers
                3: 'W',  # Three fingers
                4: 'B',  # Four fingers
                5: 'B'   # Five fingers (open hand)
            }
            
            predicted_letter = letter_map.get(fingers_up, 'Unknown')
            confidence = 0.7 if predicted_letter != 'Unknown' else 0.0
            
            return predicted_letter, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0
    
    def _count_fingers(self, landmarks: np.ndarray) -> int:
        """Count extended fingers (simplified algorithm)"""
        try:
            if landmarks.shape[0] < 21:  # MediaPipe has 21 landmarks
                return 0
            
            # Finger tip and PIP joint indices for MediaPipe
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            finger_pips = [3, 6, 10, 14, 18]
            
            fingers_up = 0
            
            # Check each finger
            for tip, pip in zip(finger_tips, finger_pips):
                if tip < len(landmarks) and pip < len(landmarks):
                    # Simple check: is tip above PIP joint?
                    if landmarks[tip][1] < landmarks[pip][1]:  # Y coordinate (inverted)
                        fingers_up += 1
            
            return fingers_up
            
        except Exception as e:
            logger.error(f"Finger counting error: {e}")
            return 0
    
    def is_ready(self) -> bool:
        """Check if classifier is ready"""
        return True

class ExistingModelAdapter(GestureClassifierInterface):
    """
    Adapter for your existing ASL models
    Modify this class to integrate with your actual models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.ready = False
        
        try:
            self._load_existing_model()
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}")
            logger.info("Using fallback simple classifier")
            self.fallback_classifier = SimpleGestureClassifier()
    
    def _load_existing_model(self):
        """Load your existing ASL model"""
        try:
            # TODO: Replace this with your actual model loading code
            # Example integrations:
            
            # Option 1: Load from your signspeak_pro.py
            # sys.path.append(str(Path(__file__).parent.parent))
            # from signspeak_pro import YourASLModel
            # self.model = YourASLModel()
            
            # Option 2: Load from your streamlit_app functions
            # from streamlit_app import classify_sign
            # self.classify_function = classify_sign
            
            # Option 3: Load from Wave backend
            # sys.path.append(str(Path(__file__).parent.parent.parent / "Wave"))
            # from asl_model_simple import FeatureBasedASLModel
            # self.model = FeatureBasedASLModel()
            
            # For now, we'll use a placeholder
            logger.warning("Existing model loading not implemented - using simple classifier")
            self.fallback_classifier = SimpleGestureClassifier()
            self.ready = True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.fallback_classifier = SimpleGestureClassifier()
            self.ready = True
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict using your existing model"""
        try:
            if self.model is not None:
                # TODO: Implement your actual model prediction
                # Example:
                # result = self.model.predict(landmarks)
                # return result.letter, result.confidence
                pass
            
            # Fallback to simple classifier
            if hasattr(self, 'fallback_classifier'):
                return self.fallback_classifier.predict(landmarks)
            
            return "Unknown", 0.0
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0
    
    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self.ready

class LegacyStreamlitAdapter(GestureClassifierInterface):
    """
    Adapter to use functions from your existing streamlit_app.py
    """
    
    def __init__(self):
        self.classify_function = None
        self.ready = False
        
        try:
            self._import_legacy_functions()
        except Exception as e:
            logger.error(f"Failed to import legacy functions: {e}")
            self.fallback_classifier = SimpleGestureClassifier()
            self.ready = True
    
    def _import_legacy_functions(self):
        """Import classification functions from existing code"""
        try:
            # Try to import from Wave folder
            wave_path = Path(__file__).parent.parent.parent / "Wave"
            if wave_path.exists():
                sys.path.append(str(wave_path))
            
            # Look for your classification functions
            # Uncomment and modify based on your actual functions:
            
            # from streamlit_app import classify_sign
            # self.classify_function = classify_sign
            
            # Or from other modules
            # from asl_model_simple import predict_letter
            # self.predict_function = predict_letter
            
            logger.warning("Legacy function import not implemented - using simple classifier")
            self.fallback_classifier = SimpleGestureClassifier()
            self.ready = True
            
        except Exception as e:
            logger.error(f"Legacy import failed: {e}")
            self.fallback_classifier = SimpleGestureClassifier()
            self.ready = True
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict using legacy functions"""
        try:
            if self.classify_function:
                # Convert landmarks to format expected by your function
                # Modify this based on your function's expected input format
                result = self.classify_function(landmarks)
                
                # Convert result to standard format
                if isinstance(result, str):
                    return result, 0.8  # Default confidence
                elif isinstance(result, tuple):
                    return result[0], result[1]
                
            # Fallback
            if hasattr(self, 'fallback_classifier'):
                return self.fallback_classifier.predict(landmarks)
            
            return "Unknown", 0.0
            
        except Exception as e:
            logger.error(f"Legacy prediction error: {e}")
            return "Error", 0.0
    
    def is_ready(self) -> bool:
        """Check if adapter is ready"""
        return self.ready

# Factory function to create the best available classifier
def create_gesture_classifier(model_type: str = "auto") -> GestureClassifierInterface:
    """
    Create the best available gesture classifier
    
    Args:
        model_type: Type of classifier to create
                   - "auto": Automatically choose best available
                   - "simple": Use simple geometric classifier
                   - "existing": Use existing model adapter
                   - "legacy": Use legacy streamlit adapter
    
    Returns:
        GestureClassifierInterface: Ready-to-use classifier
    """
    
    if model_type == "simple":
        return SimpleGestureClassifier()
    
    elif model_type == "existing":
        return ExistingModelAdapter()
    
    elif model_type == "legacy":
        return LegacyStreamlitAdapter()
    
    elif model_type == "auto":
        # Try to find the best available classifier
        logger.info("Auto-selecting best available gesture classifier")
        
        # Try existing model first
        try:
            classifier = ExistingModelAdapter()
            if classifier.is_ready():
                logger.info("Using existing model adapter")
                return classifier
        except Exception:
            pass
        
        # Try legacy adapter
        try:
            classifier = LegacyStreamlitAdapter()
            if classifier.is_ready():
                logger.info("Using legacy streamlit adapter")
                return classifier
        except Exception:
            pass
        
        # Fall back to simple classifier
        logger.info("Using simple geometric classifier")
        return SimpleGestureClassifier()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Integration helper for your existing models
def integrate_your_model(prediction_function: Any) -> GestureClassifierInterface:
    """
    Helper function to integrate your existing prediction function
    
    Args:
        prediction_function: Your existing prediction function
        
    Returns:
        GestureClassifierInterface: Wrapped classifier
    """
    
    class CustomModelAdapter(GestureClassifierInterface):
        def __init__(self, pred_func):
            self.pred_func = pred_func
        
        def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
            try:
                result = self.pred_func(landmarks)
                if isinstance(result, str):
                    return result, 0.8
                elif isinstance(result, tuple):
                    return result[0], result[1]
                else:
                    return str(result), 0.7
            except Exception as e:
                logger.error(f"Custom prediction error: {e}")
                return "Error", 0.0
        
        def is_ready(self) -> bool:
            return True
    
    return CustomModelAdapter(prediction_function)