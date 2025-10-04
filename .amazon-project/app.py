"""
Main user interface for SignSpeak application
"""
import streamlit as st
import cv2
import numpy as np
from utils.api_client import SignSpeakClient
from utils.video import VideoProcessor
import logging
from typing import Optional, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignSpeakUI:
    def __init__(self):
        # Initialize API client and video processor
        self.api_client = SignSpeakClient()
        self.video_processor = VideoProcessor()
        
        # Initialize session state
        if "frame_buffer" not in st.session_state:
            st.session_state.frame_buffer = []
        if "predictions" not in st.session_state:
            st.session_state.predictions = []
            
    def setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.title("SignSpeak Settings")
        
        # Model selection
        st.sidebar.subheader("Recognition Mode")
        mode = st.sidebar.radio(
            "Select recognition mode:",
            ["Expression", "Alphabet", "Sentence"]
        )
        
        # Camera selection
        st.sidebar.subheader("Camera Settings")
        camera_index = st.sidebar.selectbox(
            "Select camera:",
            options=[0, 1, 2],
            format_func=lambda x: f"Camera {x}"
        )
        
        # Recognition settings
        st.sidebar.subheader("Recognition Settings")
        confidence = st.sidebar.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        return {
            "mode": mode,
            "camera_index": camera_index,
            "confidence": confidence
        }
        
    def setup_main_area(self):
        """Setup main display area"""
        st.title("SignSpeak - Real-time Sign Language Recognition")
        
        # Create columns for video and predictions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Camera Feed")
            video_placeholder = st.empty()
            
        with col2:
            st.subheader("Predictions")
            prediction_placeholder = st.empty()
            
        return video_placeholder, prediction_placeholder
        
    def process_predictions(self, predictions: Optional[Dict[str, Any]]):
        """Process and display predictions"""
        if predictions:
            st.session_state.predictions.append(predictions)
            if len(st.session_state.predictions) > 10:
                st.session_state.predictions.pop(0)
                
    def run(self):
        """Main application loop"""
        # Setup UI components
        settings = self.setup_sidebar()
        video_placeholder, prediction_placeholder = self.setup_main_area()
        
        # Check API connection
        if not self.check_api_connection():
            return
            
        # Initialize video capture
        cap = self.video_processor.start_capture(settings["camera_index"])
        if cap is None:
            st.error("Could not open camera")
            return
            
        try:
            # Main loop
            for frame in self.video_processor.extract_frames(cap):
                # Process frame
                processed_frame, landmarks = self.video_processor.process_frame(frame)
                
                # Update video display
                video_placeholder.image(
                    processed_frame,
                    channels="BGR",
                    use_container_width=True
                )
                
                # Get predictions if hands detected
                if landmarks:
                    predictions = self.api_client.detect_expression(
                        self.video_processor.get_frame_buffer()
                    )
                    self.process_predictions(predictions)
                    
                # Update predictions display
                with prediction_placeholder:
                    for pred in st.session_state.predictions[-3:]:
                        st.write(pred)
                        
                time.sleep(0.1)  # Small delay to prevent overwhelming the UI
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            st.error("An error occurred. Please refresh the page.")
            
        finally:
            # Cleanup
            self.video_processor.release_resources(cap)
            
if __name__ == "__main__":
    app = SignSpeakUI()
    app.run()
