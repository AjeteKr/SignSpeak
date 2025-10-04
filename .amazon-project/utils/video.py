"""
Video processing utilities for real-time sign language detection
"""
import cv2
import numpy as np
from typing import List, Generator, Tuple, Optional
import time
import logging
from collections import deque
from pathlib import Path
import mediapipe as mp

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, frame_interval: float = 0.5, buffer_size: int = 30):
        """
        Initialize video processing utilities
        
        Args:
            frame_interval: Time between frame captures in seconds
            buffer_size: Max number of frames to store in buffer
        """
        self.frame_interval = frame_interval
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Configure drawing specs
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=1
        )
        
    def start_capture(self, source: int = 0) -> Optional[cv2.VideoCapture]:
        """
        Start video capture from camera
        
        Args:
            source: Camera index (default 0 for webcam)
            
        Returns:
            VideoCapture object if successful, None otherwise
        """
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error("Could not open video capture")
                return None
            return cap
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return None
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process video frame to detect hands and draw landmarks
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of:
            - Processed frame with landmarks drawn
            - List of hand landmarks if detected, None otherwise
        """
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get hand landmarks
            results = self.hands.process(frame_rgb)
            
            # Draw landmarks if hands detected
            if results.multi_hand_landmarks:
                landmarks_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append((landmark.x, landmark.y, landmark.z))
                    landmarks_list.append(landmarks)
                    
                    # Draw landmarks on frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec
                    )
                    
                return frame, landmarks_list
                
            return frame, None
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, None
            
    def extract_frames(self, cap: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video capture at specified interval
        
        Args:
            cap: VideoCapture object
            
        Yields:
            Video frames at specified intervals
        """
        last_capture = time.time()
        
        while True:
            current_time = time.time()
            
            # Check if enough time has passed
            if current_time - last_capture >= self.frame_interval:
                success, frame = cap.read()
                if not success:
                    break
                    
                self.frame_buffer.append(frame)
                last_capture = current_time
                
                yield frame
                
    def get_frame_buffer(self) -> List[np.ndarray]:
        """Get current frame buffer contents"""
        return list(self.frame_buffer)
        
    def release_resources(self, cap: Optional[cv2.VideoCapture] = None):
        """Release video capture and other resources"""
        if cap is not None:
            cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
