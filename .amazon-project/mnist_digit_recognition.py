"""
MNIST Digit Recognition System

This module provides a complete system for recognizing ASL digits (0-9) using 
machine learning models trained on MNIST sign language dataset.
"""

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MNISTDigitRecognizer:
    """
    A complete system for MNIST digit recognition using machine learning.
    
    Features:
    - Multiple ML algorithms (Random Forest, Logistic Regression, SVM)
    - Real-time hand tracking with MediaPipe
    - Model training, saving, and loading
    - Integration with camera input
    """
    
    def __init__(self):
        """Initialize the digit recognizer."""
        self.model = None
        self.model_type = None
        self.label_mapping = {}
        self.available_digits = []
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
    
    def load_data(self, train_csv: str, test_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data from CSV files.
        
        Args:
            train_csv (str): Path to training CSV file
            test_csv (str): Path to test CSV file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading MNIST data...")
        
        # Load datasets
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        # Separate features and labels
        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values
        
        # Normalize pixel values
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Store available digits
        self.available_digits = sorted(list(set(y_train)))
        
        logger.info(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        logger.info(f"Available digits: {self.available_digits}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_type: str = 'random_forest') -> None:
        """
        Train a machine learning model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            model_type (str): Type of model ('random_forest', 'logistic_regression', 'svm')
        """
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        self.model_type = model_type
        
        logger.info(f"✅ {model_type} model training completed!")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model trained yet!")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def save_model(self, filename: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filename (str): Name of the file to save the model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_path = os.path.join("models", filename)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'available_digits': self.available_digits
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filename (str): Name of the file containing the model
        """
        model_path = os.path.join("models", filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.available_digits = model_data['available_digits']
        
        logger.info(f"✅ Model loaded from {model_path}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Available digits: {self.available_digits}")
    
    def predict_digit_from_pixels(self, pixel_data: np.ndarray) -> Tuple[str, float]:
        """
        Predict digit from pixel data.
        
        Args:
            pixel_data (np.ndarray): Flattened pixel data (784 features)
            
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        # Ensure correct shape
        if pixel_data.shape != (784,):
            pixel_data = pixel_data.reshape(-1)[:784]
        
        # Normalize if needed
        if pixel_data.max() > 1.0:
            pixel_data = pixel_data / 255.0
        
        # Predict
        pixel_data = pixel_data.reshape(1, -1)
        prediction = self.model.predict(pixel_data)[0]
        
        # Get confidence if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(pixel_data)[0]
            confidence = probabilities.max()
        else:
            confidence = 1.0  # For models without probability estimates
        
        return str(prediction), confidence
    
    def predict_digit_from_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict digit from camera image (simplified version).
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        # This is a simplified implementation
        # In practice, you'd need to extract hand landmarks and convert to MNIST format
        
        # Convert to grayscale and resize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to MNIST dimensions
        resized = cv2.resize(gray, (28, 28))
        
        # Flatten and normalize
        pixel_data = resized.flatten() / 255.0
        
        return self.predict_digit_from_pixels(pixel_data)
    
    def draw_hand_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on the image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Image with hand landmarks drawn
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        # Draw landmarks
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return annotated_image
    
    def get_available_digits(self) -> List[str]:
        """Get list of available digits that can be recognized."""
        return [str(digit) for digit in self.available_digits]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "model_type": self.model_type,
            "available_digits": self.available_digits,
            "num_digits": len(self.available_digits)
        }

if __name__ == "__main__":
    # Simple test
    recognizer = MNISTDigitRecognizer()
    print("✅ MNIST Digit Recognizer initialized successfully!")
    print(f"Available digits: {recognizer.get_available_digits()}")
