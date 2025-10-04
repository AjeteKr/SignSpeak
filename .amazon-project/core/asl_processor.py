"""
Professional Real-Time ASL Detection Processor
Enterprise-grade sign language recognition with smooth camera integration

This module provides production-ready ASL detection with:
- Real-time hand landmark extraction
- Optimized MediaPipe processing
- Smooth gesture recognition
- Professional error handling
- Memory-efficient operations
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import Optional, Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from collections import deque
import threading
from enum import Enum

from .camera_engine import ProfessionalCameraEngine

logger = logging.getLogger(__name__)

class DetectionState(Enum):
    """ASL detection states"""
    IDLE = "idle"
    DETECTING = "detecting"
    CONFIRMED = "confirmed"
    ERROR = "error"

@dataclass
class ASLDetectionResult:
    """ASL detection result with metadata"""
    letter: str
    confidence: float
    landmarks: Optional[List[List[float]]]
    timestamp: float
    detection_time_ms: float
    hand_detected: bool
    
@dataclass
class ASLProcessorConfig:
    """Configuration for ASL processor"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    confirmation_frames: int = 15  # 0.5 seconds at 30fps
    max_hands: int = 1
    smoothing_window: int = 5
    landmark_normalize: bool = True

class ProfessionalASLProcessor:
    """
    Professional ASL detection processor with real-time performance
    
    Features:
    - MediaPipe hand tracking
    - Gesture recognition with confirmation
    - Smooth detection with temporal filtering
    - Performance monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config: ASLProcessorConfig = None):
        self.config = config or ASLProcessorConfig()
        
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        
        # Detection state
        self.current_letter = None
        self.confirmation_count = 0
        self.detection_history = deque(maxlen=self.config.smoothing_window)
        self.last_detection_time = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.total_frames_processed = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Gesture classification (placeholder for your actual model)
        self.gesture_classifier = None
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe with optimized settings"""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                model_complexity=1  # Balance between accuracy and speed
            )
            logger.info("MediaPipe hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def set_gesture_classifier(self, classifier):
        """Set the gesture classification model"""
        self.gesture_classifier = classifier
        logger.info("Gesture classifier set")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ASLDetectionResult]:
        """
        Process a single frame for ASL detection
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple of (annotated_frame, detection_result)
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(rgb_frame)
                
                # Initialize result
                detection_result = ASLDetectionResult(
                    letter="",
                    confidence=0.0,
                    landmarks=None,
                    timestamp=time.time(),
                    detection_time_ms=0.0,
                    hand_detected=False
                )
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                if results.multi_hand_landmarks:
                    detection_result.hand_detected = True
                    
                    # Process first hand only
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract and normalize landmarks
                    landmarks = self._extract_landmarks(hand_landmarks)
                    detection_result.landmarks = landmarks
                    
                    # Classify gesture
                    if self.gesture_classifier and landmarks:
                        letter, confidence = self._classify_gesture(landmarks)
                        detection_result.letter = letter
                        detection_result.confidence = confidence
                        
                        # Apply temporal smoothing
                        smoothed_result = self._apply_temporal_smoothing(letter, confidence)
                        if smoothed_result:
                            detection_result.letter = smoothed_result['letter']
                            detection_result.confidence = smoothed_result['confidence']
                            
                        # Draw detection info
                        self._draw_detection_info(annotated_frame, detection_result)
                else:
                    # No hand detected - reset confirmation
                    self.confirmation_count = 0
                    self.current_letter = None
                    
                    # Draw "no hand" message
                    cv2.putText(
                        annotated_frame,
                        "Show your hand clearly",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                detection_result.detection_time_ms = processing_time
                self.processing_times.append(processing_time)
                self.total_frames_processed += 1
                
                return annotated_frame, detection_result
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return original frame with error indication
            error_frame = frame.copy()
            cv2.putText(
                error_frame,
                f"Processing Error: {str(e)[:50]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            
            detection_result = ASLDetectionResult(
                letter="ERROR",
                confidence=0.0,
                landmarks=None,
                timestamp=time.time(),
                detection_time_ms=(time.time() - start_time) * 1000,
                hand_detected=False
            )
            
            return error_frame, detection_result
    
    def _extract_landmarks(self, hand_landmarks) -> List[List[float]]:
        """Extract and normalize hand landmarks"""
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        if self.config.landmark_normalize:
            landmarks = self._normalize_landmarks(landmarks)
        
        return landmarks
    
    def _normalize_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """Normalize landmarks relative to wrist position"""
        if not landmarks:
            return landmarks
        
        # Convert to numpy for easier manipulation
        landmarks_array = np.array(landmarks)
        
        # Use wrist (landmark 0) as reference point
        wrist = landmarks_array[0]
        centered_landmarks = landmarks_array - wrist
        
        # Scale by maximum distance from wrist
        distances = np.linalg.norm(centered_landmarks, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized_landmarks = centered_landmarks / max_distance
        else:
            normalized_landmarks = centered_landmarks
        
        return normalized_landmarks.tolist()
    
    def _classify_gesture(self, landmarks: List[List[float]]) -> Tuple[str, float]:
        """
        Classify gesture from landmarks
        
        This is a placeholder - replace with your actual ASL classification model
        """
        if not self.gesture_classifier:
            return "Unknown", 0.0
        
        try:
            # Convert landmarks to format expected by your model
            features = np.array(landmarks).flatten()
            
            # Get prediction from your model
            prediction = self.gesture_classifier.predict(features)
            
            # Extract letter and confidence (adjust based on your model's output format)
            if hasattr(prediction, 'letter') and hasattr(prediction, 'confidence'):
                return prediction.letter, prediction.confidence
            else:
                # Fallback for different model output formats
                return str(prediction), 0.8  # Default confidence
            
        except Exception as e:
            logger.error(f"Gesture classification error: {e}")
            return "Error", 0.0
    
    def _apply_temporal_smoothing(self, letter: str, confidence: float) -> Optional[Dict[str, Any]]:
        """Apply temporal smoothing to reduce jitter"""
        if confidence < 0.5:  # Ignore low-confidence detections
            return None
        
        current_time = time.time()
        
        # Add to history
        self.detection_history.append({
            'letter': letter,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Check for consistent detection
        if letter == self.current_letter:
            self.confirmation_count += 1
        else:
            self.current_letter = letter
            self.confirmation_count = 1
        
        # Return confirmed detection
        if self.confirmation_count >= self.config.confirmation_frames:
            # Calculate average confidence over confirmation period
            recent_detections = [d for d in self.detection_history if d['letter'] == letter]
            avg_confidence = np.mean([d['confidence'] for d in recent_detections])
            
            # Reset confirmation count after successful detection
            self.confirmation_count = 0
            self.last_detection_time = current_time
            
            return {
                'letter': letter,
                'confidence': avg_confidence,
                'confirmed': True
            }
        
        return {
            'letter': letter,
            'confidence': confidence,
            'confirmed': False,
            'progress': self.confirmation_count / self.config.confirmation_frames
        }
    
    def _draw_detection_info(self, frame: np.ndarray, result: ASLDetectionResult):
        """Draw detection information on frame"""
        height, width = frame.shape[:2]
        
        # Draw detection text
        if result.letter and result.letter != "Unknown":
            cv2.putText(
                frame,
                f"Detecting: {result.letter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Confidence: {result.confidence:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Draw confirmation progress
        if self.confirmation_count > 0 and self.current_letter:
            progress = self.confirmation_count / self.config.confirmation_frames
            progress_text = f"Confirming: {self.current_letter} ({int(progress*100)}%)"
            
            cv2.putText(
                frame,
                progress_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            # Progress bar
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (10, 100), (210, 115), (0, 0, 0), 2)
            cv2.rectangle(frame, (10, 100), (10 + bar_width, 115), (0, 255, 0), -1)
        
        # Draw performance info
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times))
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            cv2.putText(
                frame,
                f"Processing: {avg_time:.1f}ms ({fps:.1f} FPS)",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        if not self.processing_times:
            return {}
        
        processing_times = list(self.processing_times)
        return {
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'processing_fps': 1000 / np.mean(processing_times),
            'total_frames_processed': self.total_frames_processed,
            'detection_history_size': len(self.detection_history),
            'current_confirmation_count': self.confirmation_count,
            'current_letter': self.current_letter
        }
    
    def reset_detection_state(self):
        """Reset detection state"""
        with self._lock:
            self.current_letter = None
            self.confirmation_count = 0
            self.detection_history.clear()
            logger.info("Detection state reset")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.hands:
                self.hands.close()
            logger.info("ASL processor cleaned up")
        except Exception as e:
            logger.error(f"Error during ASL processor cleanup: {e}")

# Integrated camera and ASL processing class
class RealTimeASLSystem:
    """
    Complete real-time ASL recognition system
    Combines professional camera and ASL processing
    """
    
    def __init__(self, 
                 camera_config=None, 
                 asl_config=None,
                 detection_callback: Callable[[ASLDetectionResult], None] = None):
        
        self.camera = ProfessionalCameraEngine(camera_config)
        self.asl_processor = ProfessionalASLProcessor(asl_config)
        self.detection_callback = detection_callback
        
        # Processing control
        self._processing = False
        self._process_thread = None
        
    def set_gesture_classifier(self, classifier):
        """Set the gesture classification model"""
        self.asl_processor.set_gesture_classifier(classifier)
    
    def start(self) -> bool:
        """Start the complete ASL recognition system"""
        logger.info("Starting real-time ASL system")
        
        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            return False
        
        # Start processing
        self._processing = True
        self._process_thread = threading.Thread(
            target=self._processing_loop,
            name="ASLProcessing",
            daemon=False
        )
        self._process_thread.start()
        
        logger.info("Real-time ASL system started successfully")
        return True
    
    def _processing_loop(self):
        """Main processing loop"""
        logger.info("ASL processing loop started")
        
        while self._processing and self.camera.is_running():
            try:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # Process for ASL detection
                    annotated_frame, result = self.asl_processor.process_frame(frame)
                    
                    # Store processed frame for display
                    self._current_display_frame = annotated_frame
                    
                    # Call detection callback if provided
                    if self.detection_callback and result.letter and result.confidence > 0.5:
                        try:
                            self.detection_callback(result)
                        except Exception as e:
                            logger.error(f"Detection callback error: {e}")
                
                else:
                    time.sleep(0.01)  # No frame available, wait briefly
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Error recovery delay
        
        logger.info("ASL processing loop ended")
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get the current display frame with annotations"""
        return getattr(self, '_current_display_frame', None)
    
    def stop(self):
        """Stop the ASL recognition system"""
        logger.info("Stopping real-time ASL system")
        
        # Stop processing
        self._processing = False
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
        
        # Stop camera
        self.camera.stop()
        
        # Cleanup ASL processor
        self.asl_processor.cleanup()
        
        logger.info("Real-time ASL system stopped")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'camera_metrics': self.camera.get_metrics(),
            'asl_metrics': self.asl_processor.get_performance_metrics(),
            'system_status': {
                'camera_running': self.camera.is_running(),
                'processing_active': self._processing,
                'overall_health': self.camera.is_running() and self._processing
            }
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()