"""
Real-time MNIST Digit Recognition Interface

This module provides a clean interface for real-time digit recognition that can be
easily integrated into Streamlit or other applications.
"""

import os
import cv2
import numpy as np
import streamlit as st
from mnist_digit_recognition import MNISTDigitRecognizer
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DigitRecognitionInterface:
    """
    A clean interface for digit recognition that can be used in Streamlit apps.
    """
    
    def __init__(self, model_path: str = "mnist_random_forest_model.pkl"):
        """
        Initialize the digit recognition interface.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.recognizer = MNISTDigitRecognizer()
        self.model_loaded = False
        self.model_path = model_path
        
        # Try to load model automatically
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.recognizer.load_model(self.model_path)
            self.model_loaded = True
            logger.info("MNIST digit recognition model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self.model_loaded = False
            return False
    
    def train_new_model(self, train_csv: str, test_csv: str, 
                       model_type: str = 'random_forest') -> bool:
        """
        Train a new model and save it.
        
        Args:
            train_csv (str): Path to training CSV
            test_csv (str): Path to test CSV
            model_type (str): Type of model to train
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.recognizer.load_data(train_csv, test_csv)
            
            # Train
            self.recognizer.train_model(X_train, y_train, model_type)
            
            # Evaluate
            eval_results = self.recognizer.evaluate_model(X_test, y_test)
            accuracy = eval_results['accuracy']
            
            # Save
            model_filename = f"mnist_{model_type}_model.pkl"
            self.recognizer.save_model(model_filename)
            
            self.model_loaded = True
            logger.info(f"Model trained successfully with accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_from_camera(self, frame: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Predict digit from camera frame.
        
        Args:
            frame (np.ndarray): Camera frame (BGR format)
            
        Returns:
            Tuple[Optional[str], float, np.ndarray]: (predicted_digit, confidence, annotated_frame)
        """
        if not self.model_loaded:
            return None, 0.0, frame
        
        # Draw hand landmarks on frame
        annotated_frame = self.recognizer.draw_hand_landmarks(frame)
        
        # Predict digit (Note: This is a simplified version)
        # For real implementation, you'd need to convert hand landmarks to MNIST format
        try:
            predicted_digit, confidence = self.recognizer.predict_digit_from_image(frame)
            return predicted_digit, confidence, annotated_frame
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0, annotated_frame
    
    def predict_from_pixels(self, pixel_data: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict digit from pixel data.
        
        Args:
            pixel_data (np.ndarray): Flattened pixel data (784 features)
            
        Returns:
            Tuple[Optional[str], float]: (predicted_digit, confidence)
        """
        if not self.model_loaded:
            return None, 0.0
        
        try:
            return self.recognizer.predict_digit_from_pixels(pixel_data)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0
    
    def get_available_digits(self) -> list:
        """Get list of available digits."""
        if self.model_loaded:
            return self.recognizer.get_available_digits()
        return []
    
    def is_ready(self) -> bool:
        """Check if the interface is ready for predictions."""
        return self.model_loaded


def create_streamlit_digit_interface():
    """
    Create a Streamlit interface for digit recognition.
    
    This function can be called from your main Streamlit app.
    """
    st.header("🔢 MNIST Digit Recognition")
    
    # Initialize interface
    if 'digit_interface' not in st.session_state:
        st.session_state.digit_interface = DigitRecognitionInterface()
    
    interface = st.session_state.digit_interface
    
    # Check if model is loaded
    if not interface.is_ready():
        st.warning("⚠️ Digit recognition model not loaded!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Try Loading Model"):
                if interface.load_model():
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not load model")
        
        with col2:
            if st.button("🤖 Train New Model"):
                train_csv = "../Wave/dataset/mnist_12_train.csv"
                test_csv = "../Wave/dataset/mnist_12_test.csv"
                
                with st.spinner("Training model... This may take a few minutes"):
                    if interface.train_new_model(train_csv, test_csv):
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Training failed")
        
        return
    
    # Model is ready
    st.success(f"✅ Model loaded! Available digits: {', '.join(interface.get_available_digits())}")
    
    # Mode selection
    mode = st.selectbox(
        "Select Recognition Mode:",
        ["📹 Real-time Camera", "📁 Upload Image", "🧪 Test Samples"]
    )
    
    if mode == "📹 Real-time Camera":
        create_camera_interface(interface)
    elif mode == "📁 Upload Image":
        create_upload_interface(interface)
    elif mode == "🧪 Test Samples":
        create_test_samples_interface(interface)


def create_camera_interface(interface: DigitRecognitionInterface):
    """Create camera-based digit recognition interface."""
    st.subheader("📹 Real-time Camera Recognition")
    
    # Camera input
    camera_input = st.camera_input("Show a digit sign to the camera")
    
    if camera_input is not None:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Make prediction
        predicted_digit, confidence, annotated_frame = interface.predict_from_camera(image)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            st.image(annotated_frame, caption="Hand Landmarks", use_container_width=True)
        
        if predicted_digit:
            st.success(f"🎯 Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.3f})")
        else:
            st.warning("⚠️ Could not detect digit")


def create_upload_interface(interface: DigitRecognitionInterface):
    """Create file upload interface."""
    st.subheader("📁 Upload Image for Recognition")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a digit sign"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Make prediction
        predicted_digit, confidence, annotated_frame = interface.predict_from_camera(image)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.image(annotated_frame, caption="Hand Detection", use_container_width=True)
        
        if predicted_digit:
            st.success(f"🎯 Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.3f})")
        else:
            st.warning("⚠️ Could not detect digit")


def create_test_samples_interface(interface: DigitRecognitionInterface):
    """Create test samples interface."""
    st.subheader("🧪 Test with MNIST Samples")
    
    # Load test data
    try:
        import pandas as pd
        test_csv = "../Wave/dataset/mnist_12_test.csv"
        
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            
            st.info(f"📊 Loaded {len(test_df)} test samples")
            
            # Random sample testing
            if st.button("🎲 Test Random Sample"):
                sample_idx = np.random.randint(0, len(test_df))
                sample_row = test_df.iloc[sample_idx]
                
                true_label = sample_row['label']
                pixel_data = sample_row.drop('label').values
                
                # Predict
                predicted_digit, confidence = interface.predict_from_pixels(pixel_data)
                
                # Display sample image
                image_28x28 = pixel_data.reshape(28, 28)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_28x28, caption=f"Sample #{sample_idx}", use_container_width=True)
                
                with col2:
                    st.metric("True Label", true_label)
                    st.metric("Predicted", predicted_digit)
                    st.metric("Confidence", f"{confidence:.3f}")
                    
                    if str(true_label) == str(predicted_digit):
                        st.success("✅ Correct Prediction!")
                    else:
                        st.error("❌ Incorrect Prediction")
            
            # Manual sample selection
            st.markdown("---")
            sample_num = st.number_input(
                "Select specific sample number:", 
                min_value=0, 
                max_value=len(test_df)-1, 
                value=0
            )
            
            if st.button("🔍 Test Selected Sample"):
                sample_row = test_df.iloc[sample_num]
                
                true_label = sample_row['label']
                pixel_data = sample_row.drop('label').values
                
                # Predict
                predicted_digit, confidence = interface.predict_from_pixels(pixel_data)
                
                # Display
                image_28x28 = pixel_data.reshape(28, 28)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_28x28, caption=f"Sample #{sample_num}", use_container_width=True)
                
                with col2:
                    st.metric("True Label", true_label)
                    st.metric("Predicted", predicted_digit)
                    st.metric("Confidence", f"{confidence:.3f}")
                    
                    if str(true_label) == str(predicted_digit):
                        st.success("✅ Correct Prediction!")
                    else:
                        st.error("❌ Incorrect Prediction")
        
        else:
            st.error("❌ Test data file not found")
            
    except Exception as e:
        st.error(f"❌ Error loading test data: {str(e)}")


if __name__ == "__main__":
    # This allows the file to be run as a standalone Streamlit app
    create_streamlit_digit_interface()
