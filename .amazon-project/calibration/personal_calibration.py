"""
Personal Calibration System for ASL Recognition
Provides guided setup, user-specific thresholds, and adaptive learning
"""

import suppress_warnings

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
import matplotlib.pyplot as plt

# Suppress MediaPipe warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    plt.style.use('default')

from dataclasses import dataclass, asdict
from data_processing.feature_extractor import NormalizedFeatureExtractor, NormalizedFeatures

@dataclass
class UserProfile:
    """User calibration profile"""
    user_id: str
    name: str
    created_at: str
    hand_size_factor: float
    confidence_threshold: float
    letter_templates: Dict[str, List[np.ndarray]]  # Letter -> list of feature vectors
    performance_history: List[Dict]
    calibration_complete: bool
    preferences: Dict[str, Any]

@dataclass
class CalibrationSession:
    """Single calibration session data"""
    session_id: str
    user_id: str
    letter: str
    samples: List[NormalizedFeatures]
    quality_scores: List[float]
    timestamp: str
    success: bool

class PersonalCalibrationSystem:
    """
    Complete personal calibration system with guided setup,
    adaptive learning, and user profile management
    """
    
    def __init__(self, db_path: str = "calibration.db"):
        """Initialize calibration system"""
        self.db_path = Path(db_path)
        self.feature_extractor = NormalizedFeatureExtractor()
        self.current_profile: Optional[UserProfile] = None
        
        # Calibration settings
        self.SAMPLES_PER_LETTER = 3
        self.MIN_CONFIDENCE = 0.7
        self.ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for user profiles"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    hand_size_factor REAL NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    calibration_complete BOOLEAN NOT NULL,
                    preferences TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_progress (
                    user_id TEXT PRIMARY KEY,
                    current_letter INTEGER DEFAULT 0,
                    current_sample INTEGER DEFAULT 0,
                    completed_letters TEXT DEFAULT '[]',
                    letter_data TEXT DEFAULT '{}',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS letter_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    samples_count INTEGER NOT NULL,
                    average_quality REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    session_type TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
    
    def create_user_profile(self, name: str, user_id: Optional[str] = None) -> UserProfile:
        """Create new user profile"""
        if user_id is None:
            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        profile = UserProfile(
            user_id=user_id,
            name=name,
            created_at=datetime.now().isoformat(),
            hand_size_factor=1.0,
            confidence_threshold=0.8,
            letter_templates={},
            performance_history=[],
            calibration_complete=False,
            preferences={}
        )
        
        # Save to database
        self._save_user_profile(profile)
        return profile
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, name, created_at, hand_size_factor, confidence_threshold, 
                 calibration_complete, preferences, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.name,
                profile.created_at,
                profile.hand_size_factor,
                profile.confidence_threshold,
                profile.calibration_complete,
                json.dumps(profile.preferences),
                datetime.now().isoformat()
            ))
    
    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Load letter templates
            templates = self._load_letter_templates(user_id)
            
            # Load performance history
            history = self._load_performance_history(user_id)
            
            profile = UserProfile(
                user_id=row[0],
                name=row[1],
                created_at=row[2],
                hand_size_factor=row[3],
                confidence_threshold=row[4],
                calibration_complete=bool(row[5]),
                preferences=json.loads(row[6]) if row[6] else {},
                letter_templates=templates,
                performance_history=history
            )
            
            return profile
    
    def _load_letter_templates(self, user_id: str) -> Dict[str, List[np.ndarray]]:
        """Load letter templates for user"""
        templates = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT letter, feature_vector FROM letter_templates 
                WHERE user_id = ? ORDER BY quality_score DESC
            """, (user_id,))
            
            for letter, feature_blob in cursor.fetchall():
                feature_vector = pickle.loads(feature_blob)
                if letter not in templates:
                    templates[letter] = []
                templates[letter].append(feature_vector)
        
        return templates
    
    def _load_performance_history(self, user_id: str) -> List[Dict]:
        """Load performance history for user"""
        history = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT session_date, accuracy, session_type, details 
                FROM performance_history 
                WHERE user_id = ? ORDER BY session_date DESC
            """, (user_id,))
            
            for row in cursor.fetchall():
                history.append({
                    'date': row[0],
                    'accuracy': row[1],
                    'type': row[2],
                    'details': json.loads(row[3]) if row[3] else {}
                })
        
        return history
    
    def _save_calibration_progress(self, user_id: str, progress: dict):
        """Save calibration progress to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert set to list for JSON serialization
                completed_letters = list(progress.get('completed_letters', set()))
                
                
                # Prepare letter_data for JSON serialization (exclude NormalizedFeatures objects)
                serializable_letter_data = {}
                for letter, samples in progress.get('letter_data', {}).items():
                    serializable_letter_data[letter] = []
                    for sample in samples:
                        # Create a simplified version without the NormalizedFeatures object
                        serializable_sample = {
                            'feature_vector': sample['feature_vector'].tolist() if hasattr(sample['feature_vector'], 'tolist') else sample['feature_vector'],
                            'quality': sample['quality'],
                            'timestamp': sample['timestamp']
                            # Exclude 'features' which contains NormalizedFeatures object
                        }
                        serializable_letter_data[letter].append(serializable_sample)
                
                conn.execute("""
                    INSERT OR REPLACE INTO calibration_progress 
                    (user_id, current_letter, current_sample, completed_letters, letter_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    user_id,
                    progress.get('current_letter', 0),
                    progress.get('current_sample', 0),
                    json.dumps(completed_letters),
                    json.dumps(serializable_letter_data)
                ))
        except Exception as e:
            print(f"Error saving calibration progress: {e}")
    
    def _load_calibration_progress(self, user_id: str) -> Optional[dict]:
        """Load calibration progress from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT current_letter, current_sample, completed_letters, letter_data
                    FROM calibration_progress WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'current_letter': row[0],
                        'current_sample': row[1],
                        'completed_letters': set(json.loads(row[2]) if row[2] else []),
                        'letter_data': json.loads(row[3]) if row[3] else {}
                    }
        except Exception as e:
            print(f"Error loading calibration progress: {e}")
        
        return None
    
    def get_all_users(self) -> List[Dict[str, str]]:
        """Get list of all users"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, name, created_at, calibration_complete 
                FROM user_profiles ORDER BY created_at DESC
            """)
            
            return [
                {
                    'user_id': row[0],
                    'name': row[1],
                    'created_at': row[2],
                    'calibration_complete': bool(row[3])
                }
                for row in cursor.fetchall()
            ]
    
    def start_guided_calibration(self, profile: UserProfile) -> None:
        """Start guided calibration process in Streamlit"""
        st.header("🎯 Personal ASL Calibration")
        st.write(f"Welcome, {profile.name}! Let's calibrate the system to your signing style.")
        
        
        # Progress tracking - PERSISTENT
        if 'calibration_progress' not in st.session_state:
            # Load existing progress from database or create new
            saved_progress = self._load_calibration_progress(profile.user_id)
            
            if saved_progress:
                st.session_state.calibration_progress = saved_progress
                st.info(f"🔄 Continuing from where you left off! You've completed {len(saved_progress.get('completed_letters', set()))} letters.")
            else:
                st.session_state.calibration_progress = {
                    'current_letter': 0,
                    'current_sample': 0,
                    'completed_letters': set(),
                    'letter_data': {}
                }
        else:
            print(f"DEBUG: Using existing session state progress")
        
        progress = st.session_state.calibration_progress
        total_letters = len(self.ALPHABET)
        
        # Progress bar
        completed = len(progress['completed_letters'])
        st.progress(completed / total_letters)
        st.write(f"Progress: {completed}/{total_letters} letters completed")
        
        if completed >= total_letters:
            self._complete_calibration(profile)
            return
        
        # Current letter
        current_letter = self.ALPHABET[progress['current_letter']]
        
        # Skip if already completed
        while current_letter in progress['completed_letters'] and progress['current_letter'] < total_letters - 1:
            progress['current_letter'] += 1
            current_letter = self.ALPHABET[progress['current_letter']]
        
        if current_letter in progress['completed_letters']:
            self._complete_calibration(profile)
            return
        
        # Show current letter instruction
        self._show_letter_calibration_ui(profile, current_letter)
    
    def _show_letter_calibration_ui(self, profile: UserProfile, letter: str):
        """Show UI for calibrating specific letter"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Letter: {letter}")
            
            # Show MNIST ASL reference image
            project_root = Path(__file__).parent.parent  # Go up to project root
            mnist_image_path = project_root / "data" / "asl_reference_images" / f"letter_{letter}.png"
            
            if mnist_image_path.exists():
                st.image(str(mnist_image_path), caption=f"ASL Letter {letter} (MNIST Dataset)", width=200)
                st.success("📸 Real MNIST ASL reference")
            else:
                # Fallback to placeholder if image not found
                st.image(f"https://via.placeholder.com/200x200?text={letter}", 
                        caption=f"ASL Letter {letter}")
                st.warning("⚠️ Using placeholder image")
            
            st.write("**Instructions:**")
            st.write(f"1. Hold your hand in the '{letter}' position")
            st.write("2. Keep your hand steady and clearly visible")
            st.write("3. Click 'Capture Sample' when ready")
            st.write(f"4. Repeat {self.SAMPLES_PER_LETTER} times")
            
            # Sample progress - Make it more prominent
            progress = st.session_state.calibration_progress
            samples_taken = len(progress['letter_data'].get(letter, []))
            
            # Create a prominent progress display
            if samples_taken < self.SAMPLES_PER_LETTER:
                st.metric(
                    label=f"Samples for Letter '{letter}'",
                    value=f"{samples_taken}/{self.SAMPLES_PER_LETTER}",
                    delta=f"{self.SAMPLES_PER_LETTER - samples_taken} remaining"
                )
                
                # Progress bar for current letter
                progress_percent = samples_taken / self.SAMPLES_PER_LETTER
                st.progress(progress_percent, f"Progress: {samples_taken}/{self.SAMPLES_PER_LETTER} samples")
            else:
                st.success(f"✅ Letter '{letter}' completed with {samples_taken} samples!")
            
            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                capture_btn = st.button("📸 Capture Sample", key=f"capture_{letter}")
            with col_b:
                skip_btn = st.button("⏭️ Skip Letter", key=f"skip_{letter}")
            
            # Camera control
            camera_active = st.session_state.get('calibration_camera_active', True)
            if st.button(f"📷 {'Stop' if camera_active else 'Start'} Camera"):
                st.session_state.calibration_camera_active = not camera_active
                if not st.session_state.calibration_camera_active:
                    # Stop camera properly
                    if 'camera' in st.session_state and st.session_state.camera:
                        st.session_state.camera.release()
                        st.session_state.camera = None
                    st.info("📷 Camera stopped")
                else:
                    st.info("📷 Camera starting...")
                # Note: No st.rerun() here to prevent restart loop
        
        with col2:
            # Enhanced camera feed with MediaPipe pipeline
            camera_placeholder = st.empty()
            detection_status = st.empty()
            progress_container = st.empty()
            quality_info = st.empty()
            
            # Initialize Enhanced MediaPipe processor (once per session)
            if 'enhanced_mp_processor' not in st.session_state:
                from data_processing.enhanced_mediapipe import EnhancedMediaPipeProcessor
                st.session_state.enhanced_mp_processor = EnhancedMediaPipeProcessor()
                
            processor = st.session_state.enhanced_mp_processor
            
            # Initialize camera for CONTINUOUS streaming
            if 'camera' not in st.session_state or not st.session_state.camera:
                try:
                    camera_placeholder.info("📷 Starting live camera...")
                    
                    # Initialize camera with better settings
                    camera = cv2.VideoCapture(0)
                    if not camera.isOpened():
                        camera_placeholder.error("❌ Camera not found! Please check camera connection.")
                        return
                    
                    # Optimize camera settings for real-time
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time
                    
                    # Test camera read
                    ret, test_frame = camera.read()
                    if not ret:
                        camera_placeholder.error("❌ Camera not working! Please check permissions.")
                        camera.release()
                        return
                    
                    st.session_state.camera = camera
                    # Don't show success message that blocks video - go directly to video loop
                    
                except Exception as e:
                    camera_placeholder.error(f"❌ Camera error: {str(e)}")
                    return
            
            # Initialize detection state
            if 'detection_start_time' not in st.session_state:
                st.session_state.detection_start_time = None
            if 'last_detected_letter' not in st.session_state:
                st.session_state.last_detected_letter = None
            if 'calibration_camera_active' not in st.session_state:
                st.session_state.calibration_camera_active = True
            
            # REAL-TIME LIVE CAMERA STREAM
            if st.session_state.get('calibration_camera_active', True):
                camera = st.session_state.camera
                
                if camera and camera.isOpened():
                    # Read frame continuously for real-time view
                    ret, frame = camera.read()
                    if ret:
                        # Mirror effect for natural interaction
                        frame = cv2.flip(frame, 1)
                        
                        # Initialize detection variables
                        detected_letter = None
                        confidence = 0.0
                        
                        # Process with Enhanced MediaPipe Pipeline
                        pipeline_result = processor.process_frame_pipeline(frame)
                        
                        if pipeline_result['success']:
                            # Display the processed frame with landmarks
                            camera_placeholder.image(
                                pipeline_result['processed_frame'], 
                                channels="RGB", 
                                use_container_width=True
                            )
                            
                            # Show quality information
                            quality_score = pipeline_result['quality_score']
                            if quality_score > 0.7:
                                quality_info.success(f"✅ High quality image: {quality_score:.2f}")
                            elif quality_score > 0.4:
                                quality_info.warning(f"⚠️ Medium quality: {quality_score:.2f}")
                            else:
                                quality_info.error(f"❌ Low quality: {quality_score:.2f}")
                            
                            # Extract features for letter prediction
                            hand_features = processor.extract_hand_features(pipeline_result['landmarks'])
                            print(f"DEBUG: hand_features from processor: {type(hand_features)}, keys: {list(hand_features.keys()) if isinstance(hand_features, dict) else 'Not a dict'}")
                            detected_letter, confidence = self._predict_letter_from_enhanced_features(hand_features)
                            
                            # Show detection status
                            if detected_letter and confidence > 0.6:
                                detection_status.success(f"✅ Detected: **{detected_letter}** ({confidence:.2f})")
                            elif pipeline_result['landmarks']:
                                detection_status.info("🔍 Hand detected - form the letter clearly")
                            else:
                                detection_status.warning("👋 Show your hand clearly")
                        else:
                            # No hand detected
                            camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                            detection_status.error("❌ No hand detected")
                            quality_info.info("Position your hand in front of the camera")
                        
                        # Auto-capture logic based on enhanced detection
                        if detected_letter == letter and confidence > 0.7:
                            # Correct letter detected with high confidence
                            if st.session_state.detection_start_time is None:
                                st.session_state.detection_start_time = time.time()
                                
                            elapsed = time.time() - st.session_state.detection_start_time
                            progress_val = min(elapsed / 2.0, 1.0)  # 2 seconds to complete
                            
                            with progress_container.container():
                                st.progress(progress_val, f"🎯 Hold '{letter}' sign: {elapsed:.1f}/2.0s")
                                if progress_val < 1.0:
                                    remaining = 2.0 - elapsed
                                    st.info(f"⏳ Keep holding steady... {remaining:.1f}s remaining")
                            
                            # Auto-capture when complete
                            if progress_val >= 1.0 and pipeline_result['quality_score'] > 0.2:
                                hand_features = processor.extract_hand_features(pipeline_result['landmarks'])
                                normalized_features = self._create_enhanced_normalized_features(
                                    hand_features, pipeline_result['hand_crop']
                                )
                                
                                # Attempt auto-capture and check if successful
                                capture_success = self._capture_calibration_sample(profile, letter, normalized_features)
                                st.session_state.detection_start_time = None
                                
                                if capture_success:
                                    st.success("📸 High-quality sample auto-captured!")
                                # If capture failed, error message already shown by _capture_calibration_sample
                                
                                # The auto-advance to next letter is handled in _capture_calibration_sample
                                return  # Exit to prevent restart
                        else:
                            st.session_state.detection_start_time = None
                            if detected_letter and detected_letter != letter:
                                progress_container.warning(f"Expected: {letter}, Detected: {detected_letter}")
                            else:
                                progress_container.info(f"Show the '{letter}' sign clearly")
                    
                    # Manual capture override
                    if capture_btn:
                        st.info("🔍 Attempting to capture sample...")
                        if 'camera' in st.session_state and st.session_state.camera:
                            ret, frame = st.session_state.camera.read()
                            if ret:
                                frame = cv2.flip(frame, 1)
                                
                                # Process with enhanced pipeline
                                pipeline_result = processor.process_frame_pipeline(frame)
                                
                                if pipeline_result['success']:
                                    st.info(f"✅ Hand detected with quality: {pipeline_result['quality_score']:.2f}")
                                    
                                    if pipeline_result['quality_score'] > 0.2:
                                        hand_features = processor.extract_hand_features(pipeline_result['landmarks'])
                                        normalized_features = self._create_enhanced_normalized_features(
                                            hand_features, pipeline_result['hand_crop']
                                        )
                                        
                                        # Attempt to capture sample and check if successful
                                        capture_success = self._capture_calibration_sample(profile, letter, normalized_features)
                                        
                                        if capture_success:
                                            # Force immediate page refresh after successful capture
                                            time.sleep(0.5)  # Brief pause to show message
                                            st.rerun()
                                            return
                                        # If capture failed, error message already shown by _capture_calibration_sample
                                    else:
                                        st.warning(f"⚠️ Quality too low ({pipeline_result['quality_score']:.2f}). Try better lighting.")
                                else:
                                    st.error("❌ No hand detected in capture. Position your hand clearly.")
                            else:
                                st.error("❌ Camera frame read failed.")
                        else:
                            st.error("❌ Camera not available.")
                        
                    # Skip button  
                    if skip_btn:
                        st.session_state.detection_start_time = None
                        self._skip_letter(letter)
                        st.success(f"⏭️ Skipped letter {letter}")
                        return  # Exit to prevent restart
                    
                    # REAL-TIME CONTINUOUS REFRESH FOR LIVE VIDEO
                    # Always refresh for smooth live video stream (like phone camera)
                    time.sleep(0.05)  # Small delay for smooth refresh (20 FPS)
                    st.rerun()  # Continuous refresh for live stream
                
                else:
                    camera_placeholder.error("❌ Camera error - check connection")
            else:
                camera_placeholder.error("❌ Camera not available")
    
    def _extract_features_from_landmarks(self, hand_landmarks, frame_shape):
        """Extract features from MediaPipe hand landmarks for real-time recognition"""
        try:
            h, w = frame_shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x * w, lm.y * h])
            landmarks = np.array(landmarks)
            
            # Key landmark indices (MediaPipe hand model)
            THUMB_TIP = 4
            THUMB_MCP = 2
            INDEX_TIP = 8
            INDEX_MCP = 5
            MIDDLE_TIP = 12
            MIDDLE_MCP = 9
            RING_TIP = 16
            RING_MCP = 13
            PINKY_TIP = 20
            PINKY_MCP = 17
            WRIST = 0
            
            # Calculate finger states and positions
            features = {
                'landmarks': landmarks,
                'thumb_extended': self._is_finger_extended(landmarks, THUMB_TIP, THUMB_MCP, WRIST),
                'index_extended': self._is_finger_extended(landmarks, INDEX_TIP, INDEX_MCP, WRIST),
                'middle_extended': self._is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP, WRIST),
                'ring_extended': self._is_finger_extended(landmarks, RING_TIP, RING_MCP, WRIST),
                'pinky_extended': self._is_finger_extended(landmarks, PINKY_TIP, PINKY_MCP, WRIST),
                'palm_center': landmarks[WRIST],
                'hand_size': np.linalg.norm(landmarks[MIDDLE_TIP] - landmarks[WRIST])
            }
            
            return features
            
        except Exception as e:
            return None
    
    def _is_finger_extended(self, landmarks, tip_idx, mcp_idx, wrist_idx):
        """Check if a finger is extended based on landmark positions"""
        try:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            wrist = landmarks[wrist_idx]
            
            # For thumb, check horizontal distance
            if tip_idx == 4:  # Thumb
                return abs(tip[0] - mcp[0]) > abs(mcp[0] - wrist[0]) * 0.3
            
            # For other fingers, check vertical distance
            return (mcp[1] - tip[1]) > abs(mcp[1] - wrist[1]) * 0.3
            
        except:
            return False
    
    def _predict_letter_from_enhanced_features(self, hand_features: Dict) -> Tuple[Optional[str], float]:
        """
        Enhanced letter prediction using comprehensive hand features
        """
        try:
            finger_states = hand_features.get('finger_states', [])
            angles = hand_features.get('angles', [])
            distances = hand_features.get('distances', [])
            
            if not finger_states or len(finger_states) != 5:
                return None, 0.0
            
            # Enhanced letter recognition using multiple feature types
            confidence = 0.85
            
            # A: Closed fist (all fingers down)
            if sum(finger_states) == 0:
                return 'A', confidence
            
            # B: Four fingers up, thumb down
            if finger_states[1:] == [1, 1, 1, 1] and finger_states[0] == 0:
                return 'B', confidence
            
            # D: Index finger up only
            if finger_states == [0, 1, 0, 0, 0]:
                return 'D', confidence
            
            # I: Pinky up only
            if finger_states == [0, 0, 0, 0, 1]:
                return 'I', confidence
            
            # L: Index and thumb up
            if finger_states == [1, 1, 0, 0, 0]:
                return 'L', confidence
            
            # V: Index and middle up
            if finger_states == [0, 1, 1, 0, 0]:
                return 'V', confidence
            
            # Y: Thumb and pinky up
            if finger_states == [1, 0, 0, 0, 1]:
                return 'Y', confidence
            
            # C: Curved hand shape - multiple fingers partially extended
            if sum(finger_states) >= 2 and sum(finger_states) <= 4:
                # Typical C has thumb and some fingers curved/extended
                if finger_states[0] == 1:  # Thumb extended
                    return 'C', confidence * 0.9
                # Or multiple fingers curved (at least 2 but not all)
                elif sum(finger_states) >= 2 and sum(finger_states) < 5:
                    return 'C', confidence * 0.8
            
            # Use angle and distance features for more complex letters
            if len(angles) >= 4 and len(distances) >= 5:
                # O: Circular formation (similar distances, moderate angles)
                distance_std = np.std(distances) if len(distances) > 1 else 0
                if distance_std < 30 and sum(finger_states) >= 3:
                    return 'O', confidence * 0.8
            
            return None, 0.0
            
        except Exception as e:
            print(f"Enhanced prediction error: {e}")
            return None, 0.0
    
    def _create_enhanced_normalized_features(self, hand_features: Dict, hand_crop: np.ndarray) -> 'NormalizedFeatures':
        """
        Create enhanced normalized features from pipeline output
        """
        try:
            from data_processing.feature_extractor import NormalizedFeatures
            
            # Debug: Print what we actually received
            print(f"DEBUG: hand_features keys: {list(hand_features.keys()) if hand_features else 'None'}")
            
            # Handle the actual structure returned by enhanced MediaPipe processor
            if hand_features is None:
                hand_features = {}
            
            # Extract landmarks if available
            landmarks = np.zeros((21, 3))
            if 'landmarks' in hand_features and hand_features['landmarks'] is not None:
                landmarks_data = hand_features['landmarks']
                if hasattr(landmarks_data, '__len__') and len(landmarks_data) >= 21:
                    landmarks = np.array(landmarks_data).reshape(21, 3)
            
            # Create feature vectors with fallback values
            finger_states = np.array(hand_features.get('finger_states', [0] * 5))
            if len(finger_states) != 5:
                finger_states = np.array([0.0] * 5)
            
            distances = np.array(hand_features.get('distances', [0.1] * 20))
            if len(distances) < 20:
                distances = np.pad(distances, (0, 20 - len(distances)), 'constant', constant_values=0.1)
            distances = distances[:20]  # Ensure exactly 20 elements
            
            angles = np.array(hand_features.get('angles', [0.1] * 12))
            if len(angles) < 12:
                angles = np.pad(angles, (0, 12 - len(angles)), 'constant', constant_values=0.1)
            angles = angles[:12]  # Ensure exactly 12 elements
            
            palm_features = np.array(hand_features.get('palm_features', [0.5, 0.5, 100.0]))
            if len(palm_features) != 3:
                palm_features = np.array([0.5, 0.5, 100.0])
            
            normalized_features = NormalizedFeatures(
                landmarks=landmarks,
                distances=distances,
                angles=angles,
                finger_states=finger_states,
                hand_shape={
                    'palm_center': palm_features[:2].tolist() if len(palm_features) >= 2 else [0.5, 0.5],
                    'hand_size': palm_features[2] if len(palm_features) >= 3 else 100.0
                },
                confidence=0.9  # High quality from enhanced pipeline
            )
            
            print(f"DEBUG: Successfully created NormalizedFeatures")
            return normalized_features
            
        except Exception as e:
            print(f"Error creating enhanced features: {e}")
            # Fallback: Create a basic valid NormalizedFeatures object
            try:
                from data_processing.feature_extractor import NormalizedFeatures
                print(f"DEBUG: Using fallback NormalizedFeatures")
                return NormalizedFeatures(
                    landmarks=np.zeros((21, 3)),
                    distances=np.array([0.1] * 20),
                    angles=np.array([0.1] * 12),
                    finger_states=np.array([0.0] * 5),
                    hand_shape={'palm_center': [0.5, 0.5], 'hand_size': 100.0},
                    confidence=0.5
                )
            except Exception as fallback_error:
                print(f"Fallback creation failed: {fallback_error}")
                return None
    
    def _predict_letter_from_features_simple(self, features):
        """Simple letter recognition based on finger positions"""
        if not features:
            return None, 0.0
        
        try:
            # Extract finger states
            thumb = features.get('thumb_extended', False)
            index = features.get('index_extended', False)
            middle = features.get('middle_extended', False)
            ring = features.get('ring_extended', False)
            pinky = features.get('pinky_extended', False)
            
            # Letter recognition rules
            confidence = 0.8
            
            # A: Closed fist with thumb on side
            if not index and not middle and not ring and not pinky:
                return 'A', confidence
            
            # B: Four fingers up, thumb tucked
            if index and middle and ring and pinky and not thumb:
                return 'B', confidence
            
            # C: Curved fingers (approximate)
            if not index and not middle and not ring and not pinky and thumb:
                return 'C', confidence
            
            # D: Index up, others down, thumb extended
            if index and not middle and not ring and not pinky and thumb:
                return 'D', confidence
            
            # F: Index and middle touching (approximate)
            if not index and middle and ring and pinky and thumb:
                return 'F', confidence
            
            # I: Pinky up only
            if not index and not middle and not ring and pinky and not thumb:
                return 'I', confidence
            
            # L: Index and thumb extended
            if index and not middle and not ring and not pinky and thumb:
                return 'L', confidence
            
            # O: All fingers curved (approximate as fist)
            if not index and not middle and not ring and not pinky and not thumb:
                return 'O', confidence
            
            # V: Index and middle up
            if index and middle and not ring and not pinky and not thumb:
                return 'V', confidence
            
            # Y: Thumb and pinky extended
            if not index and not middle and not ring and pinky and thumb:
                return 'Y', confidence
            
            return None, 0.0
            
        except Exception as e:
            return None, 0.0
    
    def _create_normalized_features(self, simple_features):
        """Convert simple features to NormalizedFeatures format expected by feature extractor"""
        try:
            if not simple_features:
                return None
                
            # Create landmarks array (21 points x 3 coordinates)
            landmarks = simple_features.get('landmarks', np.zeros((21, 3)))
            if len(landmarks.shape) == 2 and landmarks.shape[1] == 2:
                # Add z-coordinate if missing
                landmarks = np.column_stack([landmarks, np.zeros(landmarks.shape[0])])
            
            # Ensure we have exactly 21 landmarks
            if landmarks.shape[0] != 21:
                landmarks = np.zeros((21, 3))
            
            # Create normalized features object
            from data_processing.feature_extractor import NormalizedFeatures
            
            normalized = NormalizedFeatures(
                landmarks=landmarks,
                distances=np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 4),  # 20 distance features
                angles=np.array([0.1, 0.2, 0.3, 0.4] * 3),  # 12 angle features  
                finger_states=np.array([
                    1 if simple_features.get('thumb_extended', False) else 0,
                    1 if simple_features.get('index_extended', False) else 0,
                    1 if simple_features.get('middle_extended', False) else 0,
                    1 if simple_features.get('ring_extended', False) else 0,
                    1 if simple_features.get('pinky_extended', False) else 0
                ], dtype=float),
                palm_features=np.array([0.5, 0.5, simple_features.get('hand_size', 100.0)]),
                quality_score=0.8
            )
            
            return normalized
            
        except Exception as e:
            print(f"Error creating normalized features: {e}")
            # Always return a valid default NormalizedFeatures object
            try:
                from data_processing.feature_extractor import NormalizedFeatures
                return NormalizedFeatures(
                    landmarks=np.zeros((21, 3)),
                    distances=np.array([0.1] * 20),
                    angles=np.array([0.1] * 12),
                    finger_states=np.array([0.0] * 5),
                    palm_features=np.array([0.5, 0.5, 100.0]),
                    quality_score=0.5
                )
            except:
                return None
    
    def _extract_hand_features_robust(self, image: np.ndarray) -> Optional[NormalizedFeatures]:
        """
        Robust hand feature extraction with better detection
        """
        try:
            # Import mediapipe with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import mediapipe as mp
            
            # Initialize MediaPipe hands with optimized settings
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3,   # Lower threshold for better tracking
                model_complexity=1
            )
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            
            # Process the image
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Calculate hand features
                features = self._calculate_hand_features(landmarks, image.shape)
                
                # Create NormalizedFeatures object
                normalized_features = NormalizedFeatures(
                    hand_shape=features['hand_shape'],
                    finger_positions=features['finger_positions'],
                    palm_center=features['palm_center'],
                    landmarks=hand_landmarks.landmark  # Store raw landmarks
                )
                
                return normalized_features
            
            return None
            
        except Exception as e:
            print(f"Error in robust hand detection: {e}")
            return None
    
    def _calculate_hand_features(self, landmarks: List[Dict], image_shape: Tuple) -> Dict:
        """Calculate comprehensive hand features from landmarks"""
        try:
            # Convert landmarks to pixel coordinates
            height, width = image_shape[:2]
            points = []
            for lm in landmarks:
                x = int(lm['x'] * width)
                y = int(lm['y'] * height)
                points.append([x, y])
            
            # Define landmark indices (MediaPipe standard)
            WRIST = 0
            THUMB_TIP = 4
            INDEX_TIP = 8
            INDEX_PIP = 6
            INDEX_MCP = 5
            MIDDLE_TIP = 12
            MIDDLE_PIP = 10
            MIDDLE_MCP = 9
            RING_TIP = 16
            RING_PIP = 14
            RING_MCP = 13
            PINKY_TIP = 20
            PINKY_PIP = 18
            PINKY_MCP = 17
            
            # Calculate finger positions
            finger_positions = {
                'thumb': {
                    'tip': points[THUMB_TIP],
                    'mcp': points[2]  # Thumb MCP
                },
                'index': {
                    'tip': points[INDEX_TIP],
                    'pip': points[INDEX_PIP],
                    'mcp': points[INDEX_MCP]
                },
                'middle': {
                    'tip': points[MIDDLE_TIP],
                    'pip': points[MIDDLE_PIP],
                    'mcp': points[MIDDLE_MCP]
                },
                'ring': {
                    'tip': points[RING_TIP],
                    'pip': points[RING_PIP],
                    'mcp': points[RING_MCP]
                },
                'pinky': {
                    'tip': points[PINKY_TIP],
                    'pip': points[PINKY_PIP],
                    'mcp': points[PINKY_MCP]
                }
            }
            
            # Calculate palm center
            palm_center = points[WRIST]
            
            # Calculate bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            hand_shape = {
                'bounding_box': {
                    'x_min': min(x_coords) / width,
                    'y_min': min(y_coords) / height,
                    'x_max': max(x_coords) / width,
                    'y_max': max(y_coords) / height
                },
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords)
            }
            
            return {
                'hand_shape': hand_shape,
                'finger_positions': finger_positions,
                'palm_center': palm_center
            }
            
        except Exception as e:
            print(f"Error calculating hand features: {e}")
            return {
                'hand_shape': {},
                'finger_positions': {},
                'palm_center': [0, 0]
            }
        """
        AI-powered letter prediction from hand features
        Returns: (predicted_letter, confidence)
        """
        try:
            # Extract key features for letter recognition
            hand_shape = features.hand_shape
            finger_positions = features.finger_positions
            palm_center = features.palm_center
            
            # Simple rule-based recognition for common letters (to be replaced with trained model)
            predictions = []
            
            # Letter A: Closed fist with thumb on side
            if self._is_closed_fist(finger_positions) and self._thumb_on_side(finger_positions):
                predictions.append(('A', 0.85))
            
            # Letter B: Four fingers up, thumb folded
            elif self._four_fingers_up(finger_positions) and self._thumb_folded(finger_positions):
                predictions.append(('B', 0.82))
            
            # Letter C: Curved hand shape
            elif self._curved_hand(hand_shape, finger_positions):
                predictions.append(('C', 0.80))
            
            # Letter D: Index finger up, others closed, thumb touching middle finger
            elif self._index_up_others_closed(finger_positions) and self._thumb_touching_middle(finger_positions):
                predictions.append(('D', 0.83))
            
            # Letter E: All fingers bent at knuckles
            elif self._all_fingers_bent(finger_positions):
                predictions.append(('E', 0.78))
            
            # Letter F: Index and thumb touching, other fingers up
            elif self._index_thumb_touching(finger_positions) and self._three_fingers_up(finger_positions):
                predictions.append(('F', 0.85))
            
            # Letter I: Pinky up, others closed
            elif self._pinky_up_others_closed(finger_positions):
                predictions.append(('I', 0.88))
            
            # Letter L: Index and thumb form L shape
            elif self._l_shape(finger_positions):
                predictions.append(('L', 0.90))
            
            # Letter O: All fingers form circle
            elif self._circle_shape(finger_positions):
                predictions.append(('O', 0.85))
            
            # Letter T: Thumb between index and middle finger
            elif self._thumb_between_fingers(finger_positions):
                predictions.append(('T', 0.82))
            
            # Letter V: Two fingers up in V shape
            elif self._v_shape(finger_positions):
                predictions.append(('V', 0.87))
            
            # Letter Y: Thumb and pinky extended
            elif self._thumb_pinky_extended(finger_positions):
                predictions.append(('Y', 0.89))
            
            # Return best prediction
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions[0]
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"Error in letter prediction: {e}")
            return None, 0.0
    
    def _is_closed_fist(self, finger_positions: Dict) -> bool:
        """Check if hand is in closed fist position"""
        try:
            # Check if all fingertips are below their respective knuckles
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    if tip[1] < pip[1]:  # Tip above PIP means extended
                        return False
            return True
        except:
            return False
    
    def _thumb_on_side(self, finger_positions: Dict) -> bool:
        """Check if thumb is positioned on the side (A letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            index = finger_positions.get('index', {})
            thumb_tip = thumb.get('tip', [0, 0])
            index_mcp = index.get('mcp', [0, 0])
            
            # Thumb should be roughly at the same height as index knuckle
            return abs(thumb_tip[1] - index_mcp[1]) < 30
        except:
            return False
    
    def _four_fingers_up(self, finger_positions: Dict) -> bool:
        """Check if four fingers are extended (B letter)"""
        try:
            count = 0
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    if tip[1] < pip[1] - 10:  # Tip significantly above PIP
                        count += 1
            return count >= 3
        except:
            return False
    
    def _thumb_folded(self, finger_positions: Dict) -> bool:
        """Check if thumb is folded (B letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            thumb_tip = thumb.get('tip', [0, 0])
            thumb_mcp = thumb.get('mcp', [0, 0])
            
            # Thumb tip should be close to palm
            return abs(thumb_tip[0] - thumb_mcp[0]) < 25
        except:
            return False
    
    def _curved_hand(self, hand_shape: Dict, finger_positions: Dict) -> bool:
        """Check if hand is in curved C shape"""
        try:
            # Check if fingers are slightly curved (C letter)
            curved_count = 0
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    mcp = finger_positions[finger].get('mcp', [0, 0])
                    
                    # Check for curved position
                    if pip[1] < mcp[1] and tip[1] > pip[1] - 20:
                        curved_count += 1
            
            return curved_count >= 3
        except:
            return False
    
    def _index_up_others_closed(self, finger_positions: Dict) -> bool:
        """Check if only index finger is up (D letter)"""
        try:
            # Index should be up
            index = finger_positions.get('index', {})
            index_tip = index.get('tip', [0, 0])
            index_pip = index.get('pip', [0, 0])
            
            if index_tip[1] >= index_pip[1]:  # Index not up
                return False
            
            # Other fingers should be down
            for finger in ['middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    if tip[1] < pip[1] - 10:  # Finger is up
                        return False
            
            return True
        except:
            return False
    
    def _thumb_touching_middle(self, finger_positions: Dict) -> bool:
        """Check if thumb is touching middle finger (D letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            middle = finger_positions.get('middle', {})
            thumb_tip = thumb.get('tip', [0, 0])
            middle_tip = middle.get('tip', [0, 0])
            
            # Calculate distance between thumb and middle finger tips
            distance = ((thumb_tip[0] - middle_tip[0])**2 + (thumb_tip[1] - middle_tip[1])**2)**0.5
            return distance < 40
        except:
            return False
    
    def _all_fingers_bent(self, finger_positions: Dict) -> bool:
        """Check if all fingers are bent at knuckles (E letter)"""
        try:
            bent_count = 0
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    mcp = finger_positions[finger].get('mcp', [0, 0])
                    
                    # Check if finger is bent (tip below PIP, PIP above MCP)
                    if tip[1] > pip[1] and pip[1] < mcp[1]:
                        bent_count += 1
            
            return bent_count >= 3
        except:
            return False
    
    def _index_thumb_touching(self, finger_positions: Dict) -> bool:
        """Check if index and thumb are touching (F letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            index = finger_positions.get('index', {})
            thumb_tip = thumb.get('tip', [0, 0])
            index_tip = index.get('tip', [0, 0])
            
            distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
            return distance < 30
        except:
            return False
    
    def _three_fingers_up(self, finger_positions: Dict) -> bool:
        """Check if three fingers are up (F letter)"""
        try:
            up_count = 0
            for finger in ['middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    if tip[1] < pip[1] - 10:
                        up_count += 1
            return up_count >= 2
        except:
            return False
    
    def _pinky_up_others_closed(self, finger_positions: Dict) -> bool:
        """Check if only pinky is up (I letter)"""
        try:
            # Pinky should be up
            pinky = finger_positions.get('pinky', {})
            pinky_tip = pinky.get('tip', [0, 0])
            pinky_pip = pinky.get('pip', [0, 0])
            
            if pinky_tip[1] >= pinky_pip[1]:
                return False
            
            # Other fingers should be down
            for finger in ['index', 'middle', 'ring']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    pip = finger_positions[finger].get('pip', [0, 0])
                    if tip[1] < pip[1] - 10:
                        return False
            
            return True
        except:
            return False
    
    def _l_shape(self, finger_positions: Dict) -> bool:
        """Check if hand forms L shape (L letter)"""
        try:
            # Index up, thumb out to side
            index = finger_positions.get('index', {})
            thumb = finger_positions.get('thumb', {})
            
            index_tip = index.get('tip', [0, 0])
            index_pip = index.get('pip', [0, 0])
            thumb_tip = thumb.get('tip', [0, 0])
            thumb_mcp = thumb.get('mcp', [0, 0])
            
            # Index should be vertical, thumb horizontal
            index_up = index_tip[1] < index_pip[1] - 15
            thumb_out = abs(thumb_tip[0] - thumb_mcp[0]) > 30
            
            return index_up and thumb_out
        except:
            return False
    
    def _circle_shape(self, finger_positions: Dict) -> bool:
        """Check if fingers form circle (O letter)"""
        try:
            # All fingertips should be close to thumb tip
            thumb = finger_positions.get('thumb', {})
            thumb_tip = thumb.get('tip', [0, 0])
            
            close_count = 0
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if finger in finger_positions:
                    tip = finger_positions[finger].get('tip', [0, 0])
                    distance = ((thumb_tip[0] - tip[0])**2 + (thumb_tip[1] - tip[1])**2)**0.5
                    if distance < 50:
                        close_count += 1
            
            return close_count >= 2
        except:
            return False
    
    def _thumb_between_fingers(self, finger_positions: Dict) -> bool:
        """Check if thumb is between fingers (T letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            index = finger_positions.get('index', {})
            middle = finger_positions.get('middle', {})
            
            thumb_tip = thumb.get('tip', [0, 0])
            index_pip = index.get('pip', [0, 0])
            middle_pip = middle.get('pip', [0, 0])
            
            # Thumb should be between index and middle finger
            return (index_pip[0] < thumb_tip[0] < middle_pip[0] or 
                   middle_pip[0] < thumb_tip[0] < index_pip[0])
        except:
            return False
    
    def _v_shape(self, finger_positions: Dict) -> bool:
        """Check if fingers form V shape (V letter)"""
        try:
            # Index and middle fingers should be up and spread apart
            index = finger_positions.get('index', {})
            middle = finger_positions.get('middle', {})
            
            index_tip = index.get('tip', [0, 0])
            index_pip = index.get('pip', [0, 0])
            middle_tip = middle.get('tip', [0, 0])
            middle_pip = middle.get('pip', [0, 0])
            
            # Both fingers up
            index_up = index_tip[1] < index_pip[1] - 15
            middle_up = middle_tip[1] < middle_pip[1] - 15
            
            # Spread apart
            spread = abs(index_tip[0] - middle_tip[0]) > 25
            
            return index_up and middle_up and spread
        except:
            return False
    
    def _thumb_pinky_extended(self, finger_positions: Dict) -> bool:
        """Check if thumb and pinky are extended (Y letter)"""
        try:
            thumb = finger_positions.get('thumb', {})
            pinky = finger_positions.get('pinky', {})
            
            thumb_tip = thumb.get('tip', [0, 0])
            thumb_mcp = thumb.get('mcp', [0, 0])
            pinky_tip = pinky.get('tip', [0, 0])
            pinky_pip = pinky.get('pip', [0, 0])
            
            # Thumb extended
            thumb_out = abs(thumb_tip[0] - thumb_mcp[0]) > 30
            # Pinky up
            pinky_up = pinky_tip[1] < pinky_pip[1] - 15
            
            return thumb_out and pinky_up
        except:
            return False
        """Draw visual feedback on camera feed"""
        annotated = image.copy()
        
        # Convert normalized landmarks back to image coordinates
        h, w = image.shape[:2]
        landmarks_2d = features.landmarks[:, :2] * [w, h]
        
        # Draw hand landmarks
        for i, (x, y) in enumerate(landmarks_2d.astype(int)):
            cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
            if i == 0:  # Wrist
                cv2.circle(annotated, (x, y), 8, (255, 0, 0), 2)
        
        # Draw connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        for start, end in connections:
            if start < len(landmarks_2d) and end < len(landmarks_2d):
                start_pt = tuple(landmarks_2d[start].astype(int))
                end_pt = tuple(landmarks_2d[end].astype(int))
                cv2.line(annotated, start_pt, end_pt, (255, 255, 0), 2)
        
        return annotated
    
    def _assess_sample_quality(self, features: NormalizedFeatures) -> float:
        """Assess quality of captured sample"""
        quality_factors = []
        
        # Check if all landmarks are present and valid
        if features.landmarks.shape[0] == 21:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Check hand stability (low variance in positions indicates steady hand)
        stability = 1.0 - min(1.0, np.std(features.landmarks[:, :2]) * 5)
        quality_factors.append(stability)
        
        # Check if hand is well-positioned (not too close to edges)
        center_x, center_y = features.landmarks[0][:2]  # Wrist position
        edge_distance = min(center_x, center_y, 1-center_x, 1-center_y)
        position_quality = min(1.0, edge_distance * 4)  # Good if > 0.25 from edges
        quality_factors.append(position_quality)
        
        # Overall quality
        return np.mean(quality_factors)
    
    def _capture_calibration_sample(self, profile: UserProfile, letter: str, features: NormalizedFeatures) -> bool:
        """Capture and store calibration sample. Returns True if successful, False otherwise."""
        # Safety check: Ensure features is not None
        if features is None:
            st.error("❌ Cannot capture sample: Invalid hand features detected.")
            return False
        
        # Additional safety check: Ensure features has required attributes
        if not hasattr(features, 'landmarks') or features.landmarks is None:
            st.error("❌ Cannot capture sample: Missing landmark data.")
            return False
            
        progress = st.session_state.calibration_progress
        
        if letter not in progress['letter_data']:
            progress['letter_data'][letter] = []
        
        try:
            # Store sample
            feature_vector = self.feature_extractor.create_feature_vector(features)
            quality = self._assess_sample_quality(features)
        except Exception as e:
            st.error(f"❌ Error processing sample: {str(e)}")
            return False
        
        sample_data = {
            'features': features,
            'feature_vector': feature_vector,
            'quality': quality,
            'timestamp': datetime.now().isoformat()
        }
        
        progress['letter_data'][letter].append(sample_data)
        
        # Check if we have enough samples
        if len(progress['letter_data'][letter]) >= self.SAMPLES_PER_LETTER:
            self._complete_letter_calibration(profile, letter)
            progress['completed_letters'].add(letter)
            progress['current_letter'] += 1
            
            # Save progress to database
            self._save_calibration_progress(profile.user_id, progress)
            
            # Show completion message and force refresh to next letter
            st.success(f"🎉 Letter '{letter}' completed! All {self.SAMPLES_PER_LETTER} samples captured!")
            st.info("Moving to next letter...")
            time.sleep(1)  # Brief pause for user to see message
            st.rerun()  # Force refresh to move to next letter
            return True
        else:
            # Save progress to database
            self._save_calibration_progress(profile.user_id, progress)
            current_count = len(progress['letter_data'][letter])
            st.success(f"✅ Sample {current_count}/{self.SAMPLES_PER_LETTER} captured for letter '{letter}'!")
            
            # Show how many more samples needed
            remaining = self.SAMPLES_PER_LETTER - current_count
            if remaining > 0:
                st.info(f"📸 {remaining} more sample{'s' if remaining > 1 else ''} needed for letter '{letter}'")
            return True
    
    def _skip_letter(self, letter: str):
        """Skip current letter"""
        progress = st.session_state.calibration_progress
        progress['completed_letters'].add(letter)
        progress['current_letter'] += 1
        
        # Save progress to database
        if 'current_user_profile' in st.session_state:
            self._save_calibration_progress(st.session_state.current_user_profile.user_id, progress)
        
        st.info(f"Letter {letter} skipped")
    
    def _complete_letter_calibration(self, profile: UserProfile, letter: str):
        """Complete calibration for specific letter"""
        progress = st.session_state.calibration_progress
        letter_samples = progress['letter_data'][letter]
        
        # Store templates in database
        for sample in letter_samples:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO letter_templates 
                    (user_id, letter, feature_vector, quality_score, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    letter,
                    pickle.dumps(sample['feature_vector']),
                    sample['quality'],
                    sample['timestamp']
                ))
        
        # Update letter templates in profile
        if letter not in profile.letter_templates:
            profile.letter_templates[letter] = []
        
        profile.letter_templates[letter].extend([
            sample['feature_vector'] for sample in letter_samples
        ])
    
    def _complete_calibration(self, profile: UserProfile):
        """Complete full calibration process"""
        profile.calibration_complete = True
        self._save_user_profile(profile)
        
        st.success("🎉 Calibration Complete!")
        st.write("Your personal ASL recognition profile has been created successfully.")
        st.write("The system is now optimized for your signing style.")
        
        # Show calibration summary
        self._show_calibration_summary(profile)
        
        # Clear progress
        if 'calibration_progress' in st.session_state:
            del st.session_state.calibration_progress
    
    def _show_calibration_summary(self, profile: UserProfile):
        """Show calibration summary"""
        st.subheader("📊 Calibration Summary")
        
        total_letters = len(profile.letter_templates)
        total_samples = sum(len(templates) for templates in profile.letter_templates.values())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Letters Calibrated", total_letters)
        
        with col2:
            st.metric("Total Samples", total_samples)
        
        with col3:
            avg_quality = np.mean([
                np.mean([self._assess_sample_quality_from_vector(vec) for vec in templates])
                for templates in profile.letter_templates.values()
            ])
            st.metric("Average Quality", f"{avg_quality:.2f}")
        
        # Show letter coverage
        st.subheader("Letter Coverage")
        cols = st.columns(6)
        for i, letter in enumerate(self.ALPHABET):
            with cols[i % 6]:
                if letter in profile.letter_templates:
                    st.success(f"✅ {letter}")
                else:
                    st.error(f"❌ {letter}")
    
    def _assess_sample_quality_from_vector(self, feature_vector: np.ndarray) -> float:
        """Assess quality from feature vector (simplified)"""
        # This is a simplified quality assessment
        # In practice, you'd use more sophisticated metrics
        return 0.8  # Placeholder
    
    def calculate_personal_confidence_threshold(self, profile: UserProfile) -> float:
        """Calculate personalized confidence threshold"""
        if not profile.letter_templates:
            return 0.8  # Default
        
        # Analyze user's consistency across samples
        consistency_scores = []
        
        for letter, templates in profile.letter_templates.items():
            if len(templates) > 1:
                # Calculate variance within letter samples
                template_array = np.array(templates)
                variance = np.var(template_array, axis=0).mean()
                consistency = 1.0 / (1.0 + variance)  # Higher consistency = lower variance
                consistency_scores.append(consistency)
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            # More consistent users can have lower thresholds
            threshold = 0.9 - (avg_consistency * 0.2)
            return max(0.6, min(0.95, threshold))
        
        return 0.8
    
    def adapt_to_user_performance(self, profile: UserProfile, session_results: Dict):
        """Adapt system based on user performance"""
        # Add performance to history
        profile.performance_history.append({
            'date': datetime.now().isoformat(),
            'accuracy': session_results.get('accuracy', 0.0),
            'type': session_results.get('type', 'practice'),
            'details': session_results
        })
        
        # Update confidence threshold based on recent performance
        recent_sessions = profile.performance_history[-10:]  # Last 10 sessions
        if len(recent_sessions) >= 3:
            recent_accuracy = np.mean([s['accuracy'] for s in recent_sessions])
            
            if recent_accuracy > 0.9:
                # User is doing well, can lower threshold slightly
                profile.confidence_threshold = max(0.6, profile.confidence_threshold - 0.05)
            elif recent_accuracy < 0.7:
                # User struggling, increase threshold
                profile.confidence_threshold = min(0.95, profile.confidence_threshold + 0.05)
        
        # Save updated profile
        self._save_user_profile(profile)
    
    def get_user_analytics(self, profile: UserProfile) -> Dict:
        """Generate user analytics and insights"""
        analytics = {
            'profile_completeness': len(profile.letter_templates) / len(self.ALPHABET),
            'total_sessions': len(profile.performance_history),
            'average_accuracy': 0.0,
            'improvement_trend': 0.0,
            'strongest_letters': [],
            'weakest_letters': [],
            'recommendations': []
        }
        
        if profile.performance_history:
            accuracies = [s['accuracy'] for s in profile.performance_history]
            analytics['average_accuracy'] = np.mean(accuracies)
            
            # Calculate improvement trend
            if len(accuracies) >= 5:
                recent = np.mean(accuracies[-5:])
                older = np.mean(accuracies[:-5])
                analytics['improvement_trend'] = recent - older
        
        # Generate recommendations
        if analytics['profile_completeness'] < 1.0:
            analytics['recommendations'].append("Complete calibration for all letters")
        
        if analytics['average_accuracy'] < 0.8:
            analytics['recommendations'].append("Practice more frequently to improve accuracy")
        
        return analytics
    
    def export_user_profile(self, profile: UserProfile, filepath: str):
        """Export user profile to file"""
        export_data = {
            'profile': asdict(profile),
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_user_profile(self, filepath: str) -> Optional[UserProfile]:
        """Import user profile from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            profile_data = data['profile']
            profile = UserProfile(**profile_data)
            
            # Save to database
            self._save_user_profile(profile)
            
            return profile
        except Exception as e:
            st.error(f"Error importing profile: {e}")
            return None


# Utility functions for Streamlit integration
def show_calibration_interface():
    """Show calibration interface in Streamlit"""
    st.title("🎯 Personal ASL Calibration System")
    
    # Initialize calibration system
    if 'calibration_system' not in st.session_state:
        st.session_state.calibration_system = PersonalCalibrationSystem()
    
    cal_sys = st.session_state.calibration_system
    
    # User selection/creation
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if st.session_state.current_user is None:
        st.subheader("👤 User Profile")
        
        # Option to create new or load existing
        tab1, tab2 = st.tabs(["Create New Profile", "Load Existing Profile"])
        
        with tab1:
            name = st.text_input("Enter your name:")
            if st.button("Create Profile") and name:
                profile = cal_sys.create_user_profile(name)
                st.session_state.current_user = profile
                st.success(f"Profile created for {name}!")
                st.rerun()
        
        with tab2:
            users = cal_sys.get_all_users()
            if users:
                user_options = {f"{u['name']} ({u['user_id']})": u['user_id'] for u in users}
                selected = st.selectbox("Select user:", options=list(user_options.keys()))
                
                if st.button("Load Profile"):
                    user_id = user_options[selected]
                    profile = cal_sys.load_user_profile(user_id)
                    if profile:
                        st.session_state.current_user = profile
                        st.success(f"Loaded profile for {profile.name}!")
                        st.rerun()
            else:
                st.info("No existing profiles found. Create a new one above.")
    
    else:
        # Show calibration interface
        profile = st.session_state.current_user
        
        st.subheader(f"Welcome back, {profile.name}!")
        
        if not profile.calibration_complete:
            cal_sys.start_guided_calibration(profile)
        else:
            st.success("✅ Calibration complete!")
            
            # Show profile analytics
            analytics = cal_sys.get_user_analytics(profile)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Profile Completeness", f"{analytics['profile_completeness']:.1%}")
            with col2:
                st.metric("Average Accuracy", f"{analytics['average_accuracy']:.1%}")
            with col3:
                st.metric("Total Sessions", analytics['total_sessions'])
            
            # Recommendations
            if analytics['recommendations']:
                st.subheader("💡 Recommendations")
                for rec in analytics['recommendations']:
                    st.write(f"• {rec}")
        
        # Profile management
        if st.button("🔄 Recalibrate"):
            profile.calibration_complete = False
            cal_sys._save_user_profile(profile)
            st.rerun()
        
        if st.button("👤 Switch User"):
            st.session_state.current_user = None
            st.rerun()
    
    def _draw_hand_feedback_with_letter(self, image: np.ndarray, features: NormalizedFeatures, 
                                       detected_letter: str, confidence: float) -> np.ndarray:
        """Draw visual feedback on camera feed with detected letter"""
        annotated = image.copy()
        
        try:
            # Import mediapipe for hand connections with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            
            # Draw hand landmarks if available
            if hasattr(features, 'landmarks') and features.landmarks:
                # Create a dummy hand landmarks object for drawing
                class HandLandmarks:
                    def __init__(self, landmarks):
                        self.landmark = landmarks
                
                hand_landmarks = HandLandmarks(features.landmarks)
                
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    annotated, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Draw detected letter overlay with enhanced visibility
            if detected_letter and confidence > 0.3:  # Lower threshold for display
                # Large letter display in top-right
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 4
                thickness = 6
                
                # Get text size
                text = detected_letter
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Position in top-right corner
                x = image.shape[1] - text_width - 30
                y = text_height + 30
                
                # Background rectangle with padding
                padding = 15
                cv2.rectangle(annotated, 
                             (x - padding, y - text_height - padding), 
                             (x + text_width + padding, y + baseline + padding), 
                             (0, 0, 0), -1)
                
                # Border for better visibility
                cv2.rectangle(annotated, 
                             (x - padding, y - text_height - padding), 
                             (x + text_width + padding, y + baseline + padding), 
                             (255, 255, 255), 2)
                
                # Letter text with confidence-based color
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                    
                cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness)
                
                # Confidence percentage display
                conf_text = f"{int(confidence * 100)}%"
                conf_font_scale = 1.2
                conf_thickness = 2
                
                (conf_width, conf_height), _ = cv2.getTextSize(conf_text, font, conf_font_scale, conf_thickness)
                conf_x = x + (text_width - conf_width) // 2
                conf_y = y + 40
                
                cv2.putText(annotated, conf_text, (conf_x, conf_y), 
                           font, conf_font_scale, (255, 255, 255), conf_thickness)
            
            # Draw hand bounding box for better visibility
            if hasattr(features, 'hand_shape') and features.hand_shape:
                bbox = features.hand_shape.get('bounding_box', {})
                if bbox:
                    x_min = int(bbox.get('x_min', 0) * image.shape[1])
                    y_min = int(bbox.get('y_min', 0) * image.shape[0])
                    x_max = int(bbox.get('x_max', 1) * image.shape[1])
                    y_max = int(bbox.get('y_max', 1) * image.shape[0])
                    
                    # Draw bounding box with dynamic color
                    box_color = (0, 255, 0) if detected_letter else (255, 255, 0)
                    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), box_color, 3)
            
            # Add frame counter for live feeling
            frame_text = f"Live: {st.session_state.get('frame_count', 0)}"
            cv2.putText(annotated, frame_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"Error in drawing feedback: {e}")
            # Draw basic frame info even if hand drawing fails
            cv2.putText(image, "Live Feed", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return image
    
    def _draw_hand_feedback(self, image: np.ndarray, features: NormalizedFeatures) -> np.ndarray:
        """Draw basic hand feedback without letter detection"""
        return self._draw_hand_feedback_with_letter(image, features, None, 0.0)


if __name__ == "__main__":
    show_calibration_interface()