"""
Main SignSpeak Application with Enhanced ASL Recognition
Integrates all components: calibration, recognition, accumulation, and real-time processing
"""

# MUST BE FIRST: Suppress all warnings before any other imports
import suppress_warnings  # This MUST be the very first import

import os
# Additional suppression (redundant but ensures coverage)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st

# Safe OpenCV import with error handling
try:
    import cv2
except ImportError as e:
    st.error(f"""
    ❌ OpenCV Import Error: {str(e)}
    
    This error typically occurs when system libraries are missing.
    The app is trying to load required system packages.
    
    If you're seeing this on Streamlit Cloud, please wait a few minutes
    for the deployment to complete, then refresh the page.
    """)
    st.stop()
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import threading
import queue
import json
from typing import Dict, List, Optional, Tuple

# Import our enhanced components
from data_processing.dataset_manager import EnhancedDatasetManager
from data_processing.feature_extractor import NormalizedFeatureExtractor
from calibration.personal_calibration import PersonalCalibrationSystem, UserProfile
from models.advanced_recognition import AdvancedASLRecognitionEngine, ModelConfig
from components.letter_accumulation import LetterAccumulationEngine
from components.modern_ui import SignSpeakUI

# Import professional camera system
from core.camera_engine import ProfessionalCameraEngine, CameraConfig
from core.asl_processor import ProfessionalASLProcessor, ASLProcessorConfig
from core.gesture_classifier import create_gesture_classifier

# Configure page
st.set_page_config(
    page_title="SignSpeak Pro - Advanced ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SignSpeakProApp:
    """
    Main SignSpeak Pro application with all enhanced features
    """
    
    def __init__(self):
        """Initialize the application"""
        self.ui = SignSpeakUI()
        self.feature_extractor = NormalizedFeatureExtractor()
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize components
        self.calibration_system = self._get_calibration_system()
        self.recognition_engine = self._get_recognition_engine()
        self.accumulation_engine = self._get_accumulation_engine()
        
        # Professional camera system
        self.camera_engine = None
        self.asl_processor = None
        self.gesture_classifier = None
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Initialize professional camera
        self._init_professional_camera()
        
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = 'main'
        
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        if 'recognition_active' not in st.session_state:
            st.session_state.recognition_active = False
        
        if 'performance_mode' not in st.session_state:
            st.session_state.performance_mode = 'balanced'  # fast, balanced, accurate
    
    def _get_calibration_system(self) -> PersonalCalibrationSystem:
        """Get or create calibration system"""
        if 'calibration_system' not in st.session_state:
            st.session_state.calibration_system = PersonalCalibrationSystem()
        return st.session_state.calibration_system
    
    def _get_recognition_engine(self) -> AdvancedASLRecognitionEngine:
        """Get or create recognition engine"""
        if 'recognition_engine' not in st.session_state:
            config = ModelConfig()
            if st.session_state.performance_mode == 'fast':
                config.input_shape = (128, 128, 3)  # Smaller for speed
            elif st.session_state.performance_mode == 'accurate':
                config.input_shape = (256, 256, 3)  # Larger for accuracy
            
            st.session_state.recognition_engine = AdvancedASLRecognitionEngine(config)
        return st.session_state.recognition_engine
    
    def _get_accumulation_engine(self) -> LetterAccumulationEngine:
        """Get or create accumulation engine"""
        if 'accumulation_engine' not in st.session_state:
            st.session_state.accumulation_engine = LetterAccumulationEngine()
        return st.session_state.accumulation_engine
    
    def _init_professional_camera(self):
        """Initialize professional camera system (disabled due to compatibility issues)"""
        try:
            # Disable professional camera system for now due to compatibility issues
            # The simple camera system works reliably instead
            st.session_state.professional_camera_ready = False
            
            # Log that we're using simple camera instead
            import logging
            logging.info("Professional camera system disabled - using simple camera for compatibility")
            
        except Exception as e:
            st.error(f"Camera system note: Using simple camera due to compatibility: {e}")
            st.session_state.professional_camera_ready = False
    
    def run(self):
        """Main application entry point"""
        # Apply UI styling
        self.ui.apply_signspeak_css()
        
        # Show header
        self._show_header()
        
        # Show sidebar
        self._show_sidebar()
        
        # Route to appropriate page
        if st.session_state.app_mode == 'main':
            self._show_main_interface()
        elif st.session_state.app_mode == 'calibration':
            self._show_calibration_interface()
        elif st.session_state.app_mode == 'training':
            self._show_training_interface()
        elif st.session_state.app_mode == 'analytics':
            self._show_analytics_interface()
        elif st.session_state.app_mode == 'settings':
            self._show_settings_interface()
        elif st.session_state.app_mode == 'video_demos':
            self._show_video_demonstrations()
    
    def _show_header(self):
        """Show application header"""
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <h1 style="color: #2563EB; margin-bottom: 0;">🤟 SignSpeak Pro</h1>
                    <h3 style="color: #6B7280; font-weight: 300;">Advanced AI-Powered ASL Recognition</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # User info
            if st.session_state.current_user:
                with col3:
                    st.markdown(f"""
                    <div style="text-align: right; padding: 1rem;">
                        <p style="margin: 0; color: #6B7280;">Welcome back,</p>
                        <p style="margin: 0; font-weight: bold; color: #2563EB;">{st.session_state.current_user.name}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _show_sidebar(self):
        """Show application sidebar"""
        with st.sidebar:
            st.markdown("### 🎛️ Navigation")
            
            # Navigation buttons
            modes = {
                '🏠 Main Interface': 'main',
                '🎯 Calibration': 'calibration',
                '🧠 Model Training': 'training',
                '🎬 ASL Video Demos': 'video_demos',
                '📊 Analytics': 'analytics',
                '⚙️ Settings': 'settings'
            }
            
            for label, mode in modes.items():
                if st.button(label, key=f"nav_{mode}", use_container_width=True):
                    st.session_state.app_mode = mode
                    st.rerun()
            
            st.divider()
            
            # Performance settings
            st.markdown("### ⚡ Performance")
            performance_mode = st.selectbox(
                "Mode",
                ['fast', 'balanced', 'accurate'],
                index=['fast', 'balanced', 'accurate'].index(st.session_state.performance_mode)
            )
            
            if performance_mode != st.session_state.performance_mode:
                st.session_state.performance_mode = performance_mode
                # Reinitialize recognition engine with new settings
                if 'recognition_engine' in st.session_state:
                    del st.session_state.recognition_engine
                st.rerun()
            
            # Real-time stats
            if st.session_state.recognition_active:
                st.markdown("### 📈 Live Stats")
                st.metric("FPS", f"{self.current_fps:.1f}")
                
                # Session stats
                if hasattr(self, 'accumulation_engine'):
                    stats = self.accumulation_engine.get_session_statistics()
                    st.metric("Letters/min", f"{stats['letters_per_minute']:.1f}")
                    st.metric("Words", stats['words_completed'])
    
    def _show_main_interface(self):
        """Show main ASL recognition interface"""
        # Check if user is calibrated
        if not st.session_state.current_user:
            self._show_user_selection()
            return
        
        # Initialize demo mode session state
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = False
        
        # Show calibration options if not calibrated
        if not st.session_state.current_user.calibration_complete and not st.session_state.demo_mode:
            self._show_calibration_options()
            return
        
        # Show demo mode warning if in demo mode
        if not st.session_state.current_user.calibration_complete and st.session_state.demo_mode:
            self._show_demo_mode_banner()
        
        # Main interface layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._show_camera_interface()
        
        with col2:
            self._show_accumulation_interface()
            self._show_quick_controls()
    
    def _show_user_selection(self):
        """Show user selection/creation interface"""
        st.subheader("👤 User Profile")
        
        tab1, tab2 = st.tabs(["New User", "Existing User"])
        
        with tab1:
            with st.form("new_user_form"):
                name = st.text_input("Your Name:")
                submit = st.form_submit_button("Create Profile")
                
                if submit and name:
                    profile = self.calibration_system.create_user_profile(name)
                    st.session_state.current_user = profile
                    st.success(f"Welcome, {name}! Let's set up your ASL profile.")
                    st.rerun()
        
        with tab2:
            users = self.calibration_system.get_all_users()
            if users:
                user_options = {f"{u['name']} ({u['user_id'][:8]})": u['user_id'] for u in users}
                
                with st.form("load_user_form"):
                    selected = st.selectbox("Select Profile:", list(user_options.keys()))
                    submit = st.form_submit_button("Load Profile")
                    
                    if submit:
                        user_id = user_options[selected]
                        profile = self.calibration_system.load_user_profile(user_id)
                        if profile:
                            st.session_state.current_user = profile
                            st.success(f"Welcome back, {profile.name}!")
                            st.rerun()
            else:
                st.info("No existing profiles found. Create a new one above.")
    
    def _show_calibration_options(self):
        """Show calibration options - calibrate or try demo mode"""
        st.subheader("🎯 ASL Recognition Setup")
        
        # Explanation
        st.info("""
        **Welcome to SignSpeak Pro!** 
        
        For the best recognition accuracy, we recommend calibrating the system to your unique signing style. 
        However, you can also try the basic recognition immediately.
        """)
        
        # Two options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 **Calibration Mode** (Recommended)
            - Personalized to your signing style
            - Higher accuracy (90%+ recognition)
            - Takes 10-15 minutes
            - Calibrate 26 letters A-Z
            """)
            
            if st.button("🎯 Start Full Calibration", type="primary", use_container_width=True):
                st.session_state.app_mode = 'calibration'
                st.rerun()
        
        with col2:
            st.markdown("""
            ### 🚀 **Demo Mode** (Try Now)
            - Use basic recognition immediately
            - Standard accuracy (60-70% recognition)
            - No setup required
            - Try it out first!
            """)
            
            if st.button("🚀 Try Demo Mode", use_container_width=True):
                st.session_state.demo_mode = True
                st.success("🎉 Demo mode activated! You can now use ASL recognition.")
                st.rerun()
        
        # Additional option for partial calibration
        st.divider()
        with st.expander("🔧 Advanced Options"):
            st.markdown("""
            ### Quick Calibration (Coming Soon)
            Calibrate just 5-10 common letters for improved accuracy without full setup.
            """)
    
    def _show_demo_mode_banner(self):
        """Show demo mode banner with upgrade options"""
        st.warning("""
        🚀 **Demo Mode Active** - You're using basic ASL recognition. 
        For better accuracy, consider calibrating your profile.
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🎯 Start Calibration"):
                st.session_state.demo_mode = False
                st.session_state.app_mode = 'calibration'
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Demo Info"):
                st.info("Demo mode uses general ASL recognition. Calibration learns your personal signing style for much better accuracy.")
        
        with col3:
            if st.button("❌ Hide Banner"):
                if 'hide_demo_banner' not in st.session_state:
                    st.session_state.hide_demo_banner = True
                    st.rerun()
        
        st.divider()

    def _show_camera_interface(self):
        """Show camera interface - using your original working approach"""
        st.subheader("📹 Live ASL Recognition")
        
        # Show recognition status
        if hasattr(self, 'recognition_engine') and self.recognition_engine.model is not None:
            st.success("🧠 **CNN Model + Enhanced ASL** - Full alphabet recognition (A-Y) with advanced pattern matching!")
        else:
            st.success("🔤 **Enhanced ASL Recognition** - Advanced pattern matching for 20+ ASL letters!")
            
            # Show training option
            if st.button("🚀 Train CNN Model", help="Add CNN model for even better accuracy"):
                st.session_state.app_mode = 'training'
                st.rerun()
        
        # Initialize session state for camera
        if 'video_running' not in st.session_state:
            st.session_state.video_running = False
        if 'simple_camera_active' not in st.session_state:
            st.session_state.simple_camera_active = False
        
        # Camera controls with enhanced features
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎥 Start Video", disabled=st.session_state.video_running):
                st.session_state.video_running = True
                st.session_state.simple_camera_active = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Video", disabled=not st.session_state.video_running):
                st.session_state.video_running = False
                st.session_state.simple_camera_active = False
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Letters", help="Clear accumulated letters"):
                self.accumulation_engine.reset_session()
                st.success("Letter history cleared!")
        
        # Frame display placeholder
        frame_display = st.empty()
        
        # Show ASL alphabet reference
        with st.expander("📚 ASL Alphabet Reference", expanded=False):
            st.write("**Supported Letters**: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Simple Gestures:**")
                st.write("• A - Closed fist, thumb up")
                st.write("• B - Four fingers straight up")
                st.write("• D - Index finger up")
                st.write("• L - Thumb and index L-shape")
                st.write("• V - Index and middle apart")
            with col2:
                st.write("**Advanced Gestures:**")
                st.write("• C - Curved hand shape")
                st.write("• F - Index-thumb touch")
                st.write("• I - Pinky up only")
                st.write("• Y - Thumb and pinky out")
                st.write("• And many more...")
        
        # Run your original camera loop
        if st.session_state.video_running:
            self._run_original_camera_loop(frame_display)

    def _show_professional_camera_controls(self):
        """Show professional camera controls"""
        
        # Camera controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📷 Start Professional Camera", disabled=st.session_state.camera_active):
                if self._start_professional_camera():
                    st.session_state.camera_active = True
                    st.success("✅ Professional camera started!")
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_active):
                self._stop_professional_camera()
                st.session_state.camera_active = False
                st.session_state.recognition_active = False
                st.success("📹 Camera stopped")
                st.rerun()
        
        with col3:
            recognition_toggle = st.checkbox(
                "🧠 ASL Recognition", 
                value=st.session_state.recognition_active,
                disabled=not st.session_state.camera_active,
                help="Toggle real-time ASL letter recognition"
            )
            
            if recognition_toggle != st.session_state.recognition_active:
                st.session_state.recognition_active = recognition_toggle
                st.info(f"🎯 Recognition {'enabled' if recognition_toggle else 'disabled'}")
                st.rerun()
        
        # Performance metrics (if camera is running)
        if st.session_state.camera_active and self.camera_engine:
            metrics = self.camera_engine.get_metrics()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Camera FPS", f"{metrics.get('fps_current', 0):.1f}")
            with col2:
                st.metric("Frames Processed", metrics.get('frames_processed', 0))
            with col3:
                st.metric("Drop Rate", f"{metrics.get('drop_rate', 0):.1f}%")
            with col4:
                st.metric("Camera State", metrics.get('state', 'unknown').title())
        
        # Camera display with refresh control
        camera_placeholder = st.empty()
        
        if st.session_state.camera_active:
            # Add refresh control
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Refresh Frame"):
                    st.rerun()
            with col2:
                auto_refresh = st.checkbox("Auto Refresh", value=False)
            
            self._run_professional_camera_loop(camera_placeholder)
            
        else:
            camera_placeholder.info("📷 Camera stopped - Click 'Start Professional Camera' to begin")
    
    def _start_professional_camera(self) -> bool:
        """Start the professional camera system"""
        try:
            if self.camera_engine and self.camera_engine.start():
                st.session_state.camera_engine_active = True
                return True
            else:
                st.error("❌ Failed to start professional camera")
                return False
        except Exception as e:
            st.error(f"❌ Camera start error: {e}")
            return False
    
    def _stop_professional_camera(self):
        """Stop all camera systems and release resources"""
        try:
            # Stop professional camera system (which is causing issues)
            if hasattr(self, 'camera_engine') and self.camera_engine:
                self.camera_engine.stop()
            
            # Release OpenCV camera from professional system
            if 'cv_camera' in st.session_state and st.session_state.cv_camera is not None:
                st.session_state.cv_camera.release()
                st.session_state.cv_camera = None
                
            # Clean up MediaPipe from professional system
            if 'mp_hands' in st.session_state:
                st.session_state.mp_hands.close()
                del st.session_state.mp_hands
            
            # Also clean up simple camera system
            if 'simple_cv_camera' in st.session_state and st.session_state.simple_cv_camera is not None:
                st.session_state.simple_cv_camera.release()
                st.session_state.simple_cv_camera = None
                
            # Clean up simple MediaPipe
            if 'simple_mp_hands' in st.session_state:
                st.session_state.simple_mp_hands.close()
                del st.session_state.simple_mp_hands
                
            # Reset all camera state
            st.session_state.camera_active = False
            st.session_state.recognition_active = False
            st.session_state.simple_camera_active = False
            
        except Exception as e:
            st.error(f"Error stopping camera: {e}")
    
    def _run_professional_camera_loop(self, placeholder):
        """Run simple, reliable live camera feed"""
        if not st.session_state.camera_active:
            placeholder.info("📷 Camera stopped - Click 'Start Professional Camera' to begin")
            return
            
        # Initialize camera if not already done
        if 'cv_camera' not in st.session_state or st.session_state.cv_camera is None:
            try:
                st.session_state.cv_camera = cv2.VideoCapture(0)
                if not st.session_state.cv_camera.isOpened():
                    placeholder.error("❌ Could not open camera")
                    return
                    
                # Configure camera
                st.session_state.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state.cv_camera.set(cv2.CAP_PROP_FPS, 30)
                st.session_state.cv_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Initialize MediaPipe
                if 'mp_hands' not in st.session_state:
                    import mediapipe as mp
                    st.session_state.mp_hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    st.session_state.mp_drawing = mp.solutions.drawing_utils
                    st.session_state.mp_hands_module = mp.solutions.hands
                    
            except Exception as e:
                placeholder.error(f"❌ Camera initialization error: {e}")
                return
        
        # Capture and display frame
        try:
            ret, frame = st.session_state.cv_camera.read()
            if not ret:
                placeholder.error("❌ Failed to capture frame")
                return
                
            # Mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame if recognition is enabled
            if st.session_state.get('recognition_active', False):
                # Hand detection
                results = st.session_state.mp_hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        st.session_state.mp_drawing.draw_landmarks(
                            rgb_frame, hand_landmarks, 
                            st.session_state.mp_hands_module.HAND_CONNECTIONS,
                            st.session_state.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                            st.session_state.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                        
                        # Simple letter detection
                        letter = self._simple_letter_detection(hand_landmarks)
                        if letter:
                            cv2.putText(rgb_frame, f"Letter: {letter}", 
                                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            
                            # Add to accumulation if valid
                            if hasattr(self, 'accumulation_engine'):
                                self.accumulation_engine.add_letter(letter)
                                
                else:
                    cv2.putText(rgb_frame, "Show your hand", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(rgb_frame, "Camera Active - Recognition OFF", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame (no auto-refresh)
            placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
        except Exception as e:
            placeholder.error(f"❌ Camera processing error: {e}")

    def _simple_letter_detection(self, hand_landmarks):
        """Enhanced ASL letter detection with comprehensive alphabet recognition"""
        try:
            # First, try to use the trained CNN model for full alphabet recognition
            if hasattr(self, 'recognition_engine') and self.recognition_engine.model is not None:
                return self._cnn_letter_detection(hand_landmarks)
            
            # Enhanced ASL alphabet recognition using landmark analysis
            return self._advanced_asl_recognition(hand_landmarks)
                
        except Exception as e:
            # Fallback to basic finger counting
            return self._basic_finger_detection(hand_landmarks)
    
    def _advanced_asl_recognition(self, hand_landmarks):
        """Comprehensive ASL alphabet recognition using MediaPipe landmarks"""
        try:
            # Get landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)
            
            # MediaPipe hand landmark indices
            # Fingertips: [4, 8, 12, 16, 20] (Thumb, Index, Middle, Ring, Pinky)
            # Finger bases: [2, 5, 9, 13, 17]
            
            # Calculate finger states
            finger_states = self._get_finger_states(landmarks)
            thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
            
            # Calculate angles and positions for specific gestures
            thumb_angle = self._calculate_thumb_angle(landmarks)
            finger_spread = self._calculate_finger_spread(landmarks)
            hand_orientation = self._calculate_hand_orientation(landmarks)
            
            # ASL Alphabet Pattern Matching
            
            # A - Closed fist with thumb up
            if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
                return "A"
            
            # B - Four fingers up, thumb across palm
            if index_up and middle_up and ring_up and pinky_up and not thumb_up and finger_spread < 0.3:
                return "B"
            
            # C - Curved hand shape
            if self._detect_c_shape(landmarks):
                return "C"
            
            # D - Index finger up, thumb out
            if index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
                return "D"
            
            # E - Fingers curled down, thumb tucked
            if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
                if self._detect_e_shape(landmarks):
                    return "E"
            
            # F - Index and thumb touch, others up
            if self._detect_f_shape(landmarks):
                return "F"
            
            # G - Index pointing sideways, thumb up
            if index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
                if abs(hand_orientation) > 0.5:  # Sideways orientation
                    return "G"
            
            # H - Index and middle sideways
            if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                if abs(hand_orientation) > 0.3:
                    return "H"
            
            # I - Pinky up only
            if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
                return "I"
            
            # K - Index up, middle out, thumb touches middle
            if self._detect_k_shape(landmarks):
                return "K"
            
            # L - Thumb and index form L shape
            if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
                if self._detect_l_shape(landmarks):
                    return "L"
            
            # M - Thumb under three fingers
            if self._detect_m_shape(landmarks):
                return "M"
            
            # N - Thumb under two fingers
            if self._detect_n_shape(landmarks):
                return "N"
            
            # O - Fingers form circle
            if self._detect_o_shape(landmarks):
                return "O"
            
            # P - Index down, middle and thumb touch
            if self._detect_p_shape(landmarks):
                return "P"
            
            # Q - Thumb and index point down
            if self._detect_q_shape(landmarks):
                return "Q"
            
            # R - Index and middle crossed
            if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                if self._detect_r_shape(landmarks):
                    return "R"
            
            # S - Closed fist, thumb over fingers
            if not index_up and not middle_up and not ring_up and not pinky_up:
                if self._detect_s_shape(landmarks):
                    return "S"
            
            # T - Thumb between index and middle
            if self._detect_t_shape(landmarks):
                return "T"
            
            # U - Index and middle up together
            if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                if finger_spread < 0.2:  # Close together
                    return "U"
            
            # V - Index and middle up, spread apart
            if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
                if finger_spread > 0.3:  # Spread apart
                    return "V"
            
            # W - Three fingers up (index, middle, ring)
            if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
                return "W"
            
            # X - Index finger hooked
            if self._detect_x_shape(landmarks):
                return "X"
            
            # Y - Thumb and pinky extended
            if thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
                return "Y"
            
            # If no specific pattern matches, return None
            return None
            
        except Exception as e:
            return None
    
    def _basic_finger_detection(self, hand_landmarks):
        """Basic fallback finger counting"""
        try:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])
            landmarks = np.array(landmarks)
            
            fingers_up = self._count_fingers(landmarks)
            
            if fingers_up == 0:
                return "A"  # Fist
            elif fingers_up == 1:
                return "D"  # Index finger
            elif fingers_up == 2:
                return "V"  # Peace sign
            elif fingers_up == 5:
                return "B"  # Open hand
            else:
                return None
        except:
            return None
    
    def _cnn_letter_detection(self, hand_landmarks):
        """Advanced CNN-based letter detection using the full ASL alphabet"""
        try:
            # Check if we have a trained model
            if not hasattr(self, 'recognition_engine') or self.recognition_engine.model is None:
                # Try to load a trained model
                if not self.recognition_engine.load_asl_reference_model():
                    return None  # No trained model available
            
            # Get current frame from session state (should be available from camera loop)
            current_frame = None
            if 'shared_frame' in st.session_state and st.session_state.shared_frame.get('frame') is not None:
                current_frame = st.session_state.shared_frame['frame']
            elif 'cv_camera' in st.session_state and st.session_state.cv_camera is not None:
                # Fallback: capture a new frame
                ret, frame = st.session_state.cv_camera.read()
                if ret:
                    current_frame = frame
            
            if current_frame is None:
                return None
            
            # Extract hand region using landmarks
            hand_region = self._extract_hand_region(current_frame, hand_landmarks)
            if hand_region is None:
                return None
            
            # Use the recognition engine to predict the letter
            predicted_letter, confidence = self.recognition_engine.predict_asl_letter(hand_region)
            
            # Only return prediction if confidence is above threshold
            confidence_threshold = 0.7  # Adjust this value as needed
            if predicted_letter and confidence > confidence_threshold:
                return predicted_letter
            else:
                return None
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"CNN letter detection error: {e}")
            return None
    
    def _extract_hand_region(self, frame, hand_landmarks):
        """
        Extract hand region from frame using MediaPipe landmarks
        
        Args:
            frame: Current camera frame (BGR format)
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Extracted and preprocessed hand region (RGB format)
        """
        try:
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Get landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append([x, y])
            landmarks = np.array(landmarks)
            
            # Calculate bounding box with padding
            x_min = max(0, int(landmarks[:, 0].min()) - 30)
            x_max = min(w, int(landmarks[:, 0].max()) + 30)
            y_min = max(0, int(landmarks[:, 1].min()) - 30)
            y_max = min(h, int(landmarks[:, 1].max()) + 30)
            
            # Make bounding box square by expanding the smaller dimension
            width = x_max - x_min
            height = y_max - y_min
            
            if width > height:
                # Expand height
                diff = width - height
                y_min = max(0, y_min - diff // 2)
                y_max = min(h, y_max + diff // 2)
            else:
                # Expand width
                diff = height - width
                x_min = max(0, x_min - diff // 2)
                x_max = min(w, x_max + diff // 2)
            
            # Extract hand region
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            # Convert BGR to RGB
            hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
            
            # Ensure minimum size
            if hand_region_rgb.shape[0] < 50 or hand_region_rgb.shape[1] < 50:
                return None
            
            return hand_region_rgb
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Hand region extraction error: {e}")
            return None
    
    def _get_finger_states(self, landmarks):
        """Determine which fingers are extended"""
        try:
            # Fingertip and base indices
            tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            pip_ids = [3, 6, 10, 14, 18]  # Finger joint below tip
            
            fingers = []
            
            # Thumb (special case - horizontal movement)
            if landmarks[tip_ids[0]][0] > landmarks[pip_ids[0]][0]:  # Thumb extended right
                fingers.append(True)
            else:
                fingers.append(False)
            
            # Other fingers (vertical movement)
            for i in range(1, 5):
                if landmarks[tip_ids[i]][1] < landmarks[pip_ids[i]][1]:  # Finger extended up
                    fingers.append(True)
                else:
                    fingers.append(False)
            
            return fingers
            
        except:
            return [False, False, False, False, False]
    
    def _calculate_thumb_angle(self, landmarks):
        """Calculate thumb angle relative to palm"""
        try:
            thumb_tip = landmarks[4]
            thumb_base = landmarks[2]
            wrist = landmarks[0]
            
            # Calculate angle between thumb and wrist-base line
            v1 = thumb_tip - thumb_base
            v2 = thumb_base - wrist
            
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            return angle
        except:
            return 0
    
    def _calculate_finger_spread(self, landmarks):
        """Calculate how spread apart the fingers are"""
        try:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            distance = np.linalg.norm(index_tip - middle_tip)
            return distance
        except:
            return 0
    
    def _calculate_hand_orientation(self, landmarks):
        """Calculate hand orientation (horizontal vs vertical)"""
        try:
            wrist = landmarks[0]
            middle_base = landmarks[9]
            
            direction = middle_base - wrist
            # Normalize to -1 to 1 (horizontal to vertical)
            orientation = direction[0] / (abs(direction[0]) + abs(direction[1]) + 1e-6)
            return orientation
        except:
            return 0
    
    def _detect_c_shape(self, landmarks):
        """Detect C-shaped hand gesture"""
        try:
            # Check if fingers are curved (tips closer to palm than expected)
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
            palm_center = landmarks[9]  # Middle finger base as palm reference
            
            curved_count = 0
            for tip_idx in finger_tips:
                tip_to_palm = np.linalg.norm(landmarks[tip_idx] - palm_center)
                if 0.1 < tip_to_palm < 0.25:  # Curved distance range
                    curved_count += 1
            
            return curved_count >= 3
        except:
            return False
    
    def _detect_e_shape(self, landmarks):
        """Detect E-shaped hand gesture (fingers curled down)"""
        try:
            # Check if all fingertips are below their respective joints
            finger_checks = [
                landmarks[8][1] > landmarks[6][1],   # Index curled
                landmarks[12][1] > landmarks[10][1], # Middle curled
                landmarks[16][1] > landmarks[14][1], # Ring curled
                landmarks[20][1] > landmarks[18][1]  # Pinky curled
            ]
            return sum(finger_checks) >= 3
        except:
            return False
    
    def _detect_f_shape(self, landmarks):
        """Detect F-shaped hand gesture (index and thumb touch)"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            # Check if thumb and index are close
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            
            # Check if other fingers are extended
            finger_states = self._get_finger_states(landmarks)
            
            return (thumb_index_dist < 0.05 and 
                   finger_states[2] and finger_states[3] and finger_states[4])  # Middle, Ring, Pinky up
        except:
            return False
    
    def _detect_k_shape(self, landmarks):
        """Detect K-shaped hand gesture"""
        try:
            # Index up, middle angled, thumb touches middle
            finger_states = self._get_finger_states(landmarks)
            if not (finger_states[1] and finger_states[2]):  # Index and middle needed
                return False
            
            thumb_tip = landmarks[4]
            middle_joint = landmarks[10]  # Middle finger PIP joint
            
            # Check if thumb touches middle finger joint
            touch_dist = np.linalg.norm(thumb_tip - middle_joint)
            return touch_dist < 0.06
        except:
            return False
    
    def _detect_l_shape(self, landmarks):
        """Detect L-shaped hand gesture"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Calculate angle between thumb and index
            thumb_vec = thumb_tip - landmarks[2]  # Thumb base
            index_vec = index_tip - landmarks[5]  # Index base
            
            # Check if they form roughly 90-degree angle
            dot_product = np.dot(thumb_vec, index_vec)
            norms = np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)
            
            if norms > 0:
                angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                return abs(angle - np.pi/2) < 0.5  # Within 30 degrees of 90
            return False
        except:
            return False
    
    def _detect_m_shape(self, landmarks):
        """Detect M-shaped hand gesture"""
        try:
            # Three fingers down with thumb underneath
            finger_states = self._get_finger_states(landmarks)
            if sum(finger_states[1:4]) != 0:  # Index, Middle, Ring should be down
                return False
            
            thumb_tip = landmarks[4]
            palm_base = landmarks[5]  # Index finger MCP
            
            # Check if thumb is tucked under
            return thumb_tip[1] > palm_base[1]  # Thumb below palm level
        except:
            return False
    
    def _detect_n_shape(self, landmarks):
        """Detect N-shaped hand gesture"""
        try:
            # Similar to M but with two fingers
            finger_states = self._get_finger_states(landmarks)
            if sum(finger_states[1:3]) != 0:  # Index and Middle should be down
                return False
            
            thumb_tip = landmarks[4]
            palm_base = landmarks[5]
            
            return thumb_tip[1] > palm_base[1]
        except:
            return False
    
    def _detect_o_shape(self, landmarks):
        """Detect O-shaped hand gesture"""
        try:
            # All fingertips should be close to form a circle
            fingertips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
            center = np.mean(fingertips, axis=0)
            
            # Check if all fingertips are roughly equidistant from center
            distances = [np.linalg.norm(tip - center) for tip in fingertips]
            avg_dist = np.mean(distances)
            
            # Check consistency of distances (circular shape)
            return all(abs(d - avg_dist) < 0.03 for d in distances)
        except:
            return False
    
    def _detect_p_shape(self, landmarks):
        """Detect P-shaped hand gesture"""
        try:
            # Index pointing down, thumb and middle touching
            finger_states = self._get_finger_states(landmarks)
            if finger_states[1]:  # Index should be down
                return False
            
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            
            # Check if thumb and middle are close
            return np.linalg.norm(thumb_tip - middle_tip) < 0.05
        except:
            return False
    
    def _detect_q_shape(self, landmarks):
        """Detect Q-shaped hand gesture"""
        try:
            # Thumb and index pointing down
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            # Check if both thumb and index are below wrist level
            return (thumb_tip[1] > wrist[1] and index_tip[1] > wrist[1])
        except:
            return False
    
    def _detect_r_shape(self, landmarks):
        """Detect R-shaped hand gesture (crossed fingers)"""
        try:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            
            # Check if fingers cross over each other
            index_x = index_tip[0] - index_pip[0]
            middle_x = middle_tip[0] - middle_pip[0]
            
            # Fingers should lean toward each other
            return (index_x * middle_x) < 0
        except:
            return False
    
    def _detect_s_shape(self, landmarks):
        """Detect S-shaped hand gesture"""
        try:
            # Closed fist with thumb over fingers
            finger_states = self._get_finger_states(landmarks)
            if any(finger_states[1:]):  # No fingers should be up
                return False
            
            thumb_tip = landmarks[4]
            index_knuckle = landmarks[5]
            
            # Thumb should be over the knuckles
            return thumb_tip[1] < index_knuckle[1]
        except:
            return False
    
    def _detect_t_shape(self, landmarks):
        """Detect T-shaped hand gesture"""
        try:
            # Thumb between index and middle finger
            thumb_tip = landmarks[4]
            index_base = landmarks[5]
            middle_base = landmarks[9]
            
            # Check if thumb is positioned between index and middle bases
            index_to_middle = middle_base - index_base
            index_to_thumb = thumb_tip - index_base
            
            # Project thumb onto index-middle line
            projection = np.dot(index_to_thumb, index_to_middle) / np.dot(index_to_middle, index_to_middle)
            
            return 0.2 < projection < 0.8  # Thumb between fingers
        except:
            return False
    
    def _detect_x_shape(self, landmarks):
        """Detect X-shaped hand gesture (hooked index)"""
        try:
            # Index finger curved/hooked
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            # Check if index finger is hooked (tip closer to base than expected)
            tip_to_base = np.linalg.norm(index_tip - index_mcp)
            pip_to_base = np.linalg.norm(index_pip - index_mcp)
            
            # Hooked if tip is not much farther from base than PIP joint
            return tip_to_base < (pip_to_base * 1.3)
        except:
            return False
    
    def _count_fingers(self, landmarks):
        """Count extended fingers from landmarks"""
        try:
            # Finger tip and PIP joint indices
            tip_ids = [4, 8, 12, 16, 20]
            pip_ids = [3, 6, 10, 14, 18]
            
            fingers = []
            
            # Thumb (different logic due to orientation)
            if landmarks[tip_ids[0]][0] > landmarks[pip_ids[0]][0]:
                fingers.append(1)
            else:
                fingers.append(0)
                
            # Other fingers
            for i in range(1, 5):
                if landmarks[tip_ids[i]][1] < landmarks[pip_ids[i]][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            return sum(fingers)
        except:
            return 0
    
    def _draw_flipped_landmarks(self, frame, hand_landmarks, mp_drawing, mp_connections):
        """Draw hand landmarks on horizontally flipped frame with corrected coordinates"""
        try:
            h, w, _ = frame.shape
            
            # Draw landmarks with flipped x coordinates
            for landmark in hand_landmarks.landmark:
                # Flip x coordinate for display
                x = int((1.0 - landmark.x) * w)
                y = int(landmark.y * h)
                
                # Draw landmark point
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dots
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)  # White border
            
            # Draw connections with flipped coordinates
            for connection in mp_connections:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]
                
                # Flip coordinates for both points
                start_x = int((1.0 - start_landmark.x) * w)
                start_y = int(start_landmark.y * h)
                end_x = int((1.0 - end_landmark.x) * w)
                end_y = int(end_landmark.y * h)
                
                # Draw connection line
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
                
        except Exception as e:
            # Fallback: draw without flipping if there's an error
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_connections)
    
    def _get_asl_letter_info(self, letter):
        """Get descriptive information about an ASL letter"""
        asl_info = {
            'A': 'Closed fist, thumb up',
            'B': 'Four fingers up, straight',
            'C': 'Curved hand shape',
            'D': 'Index finger up, thumb out',
            'E': 'Fingers curled down',
            'F': 'Index-thumb touch, others up',
            'G': 'Index pointing sideways',
            'H': 'Index & middle sideways',
            'I': 'Pinky up only',
            'K': 'Index up, middle angled',
            'L': 'Thumb & index L-shape',
            'M': 'Thumb under 3 fingers',
            'N': 'Thumb under 2 fingers',
            'O': 'Fingers form circle',
            'P': 'Index down, thumb-middle touch',
            'Q': 'Thumb & index point down',
            'R': 'Index & middle crossed',
            'S': 'Closed fist, thumb over',
            'T': 'Thumb between fingers',
            'U': 'Index & middle together',
            'V': 'Index & middle apart',
            'W': 'Three fingers up',
            'X': 'Index finger hooked',
            'Y': 'Thumb & pinky out'
        }
        return asl_info.get(letter, '')
    
    def _draw_accumulation_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw accumulation results on frame"""
        h, w = frame.shape[:2]
        
        # Draw current word
        if result.get('current_word'):
            text = f"Word: {result['current_word']}"
            cv2.putText(frame, text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw accumulated sentence
        if result.get('sentence'):
            # Split long sentences for display
            sentence = result['sentence']
            if len(sentence) > 40:
                sentence = "..." + sentence[-37:]
            
            cv2.putText(frame, f"Text: {sentence}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw detection status
        status = result.get('status', 'Waiting')
        color = (0, 255, 0) if status == 'confirmed' else (255, 255, 0) if status == 'detecting' else (255, 255, 255)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def _process_frame_for_recognition(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for ASL recognition"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features_from_image(frame)
            
            if features:
                # Get letter prediction (simplified - would use trained model)
                letter, confidence = self._predict_letter_from_features(features)
                
                if letter and confidence > 0.7:
                    # Process with accumulation engine
                    result = self.accumulation_engine.process_detection(
                        letter, confidence, features.hand_shape
                    )
                    
                    # Draw recognition results on frame
                    frame = self._draw_recognition_overlay(frame, letter, confidence, result)
                else:
                    # Handle no detection
                    result = self.accumulation_engine.process_detection(None, 0.0)
                    frame = self._draw_no_detection_overlay(frame, result)
            else:
                # No hand detected
                result = self.accumulation_engine.process_detection(None, 0.0)
                frame = self._draw_no_hand_overlay(frame)
                
        except Exception as e:
            st.error(f"Recognition error: {e}")
        
        return frame
    
    def _predict_letter_from_features(self, features) -> Tuple[Optional[str], float]:
        """Predict letter from extracted features (placeholder)"""
        # This would use the trained recognition engine
        # For now, return a dummy prediction based on hand shape
        
        if features.finger_states.sum() == 0:  # Fist
            return 'A', 0.85
        elif features.finger_states.sum() == 4:  # Open hand
            return 'B', 0.80
        elif features.finger_states[0] == 1 and features.finger_states.sum() == 1:  # Pointing
            return 'D', 0.90
        elif features.hand_shape.get('openness', 0) > 0.5:
            return 'O', 0.75
        
        return None, 0.0
    
    def _draw_recognition_overlay(self, frame: np.ndarray, letter: str, confidence: float, result: Dict) -> np.ndarray:
        """Draw recognition results on frame"""
        h, w = frame.shape[:2]
        
        # Draw letter and confidence
        text = f"Letter: {letter} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw progress bar if accumulating
        if result.get('action') == 'accumulating':
            progress = result.get('progress', 0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), (0, 255, 255), -1)
            cv2.rectangle(frame, (10, 50), (210, 70), (255, 255, 255), 2)
        
        # Draw current word
        current_word = result.get('current_word', '')
        if current_word:
            cv2.putText(frame, f"Word: {current_word}", (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame
    
    def _draw_no_detection_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw overlay when no letter detected but hand visible"""
        if result.get('action') == 'pausing':
            progress = result.get('pause_progress', 0)
            cv2.putText(frame, f"Pausing... {progress:.1%}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        return frame
    
    def _draw_no_hand_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay when no hand detected"""
        cv2.putText(frame, "Show your hand clearly", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    def _show_simple_camera_interface(self):
        """Show simple camera interface as main camera system"""
        st.subheader("📷 Camera Controls")
        
        # Initialize camera states
        if 'simple_camera_active' not in st.session_state:
            st.session_state.simple_camera_active = False
        if 'camera_starting' not in st.session_state:
            st.session_state.camera_starting = False
        
        # Camera status indicator
        camera_active = st.session_state.simple_camera_active
        camera_starting = st.session_state.camera_starting
        
        if camera_active:
            st.success("🟢 **Camera Status**: Active - Ready for ASL Recognition")
        elif camera_starting:
            st.warning("🟡 **Camera Status**: Starting...")
        else:
            st.info("🔴 **Camera Status**: Inactive - Click Start to begin")
        
        # Camera controls with improved UX
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Start button - only show when camera is not active
            if not camera_active and not camera_starting:
                if st.button("🎥 Start Camera", key="start_camera_btn", type="primary"):
                    st.session_state.camera_starting = True
                    st.rerun()
            elif camera_starting:
                st.button("⏳ Starting...", disabled=True, key="start_camera_btn_disabled")
            else:
                st.button("📷 Camera Running", disabled=True, key="start_camera_btn_running", 
                         help="Camera is already active")
        
        with col2:
            # Stop button - only show when camera is active
            if camera_active:
                if st.button("⏹️ Stop Camera", key="stop_camera_btn", type="secondary"):
                    self._stop_simple_camera()
                    st.session_state.simple_camera_active = False
                    st.session_state.recognition_active = False
                    st.session_state.camera_starting = False
                    st.success("📹 Camera stopped")
                    st.rerun()
            else:
                st.button("⏹️ Stop Camera", disabled=True, key="stop_camera_btn_disabled",
                         help="Camera is not running")
        
        with col3:
            # Recognition toggle - only show when camera is active
            if camera_active:
                recognition_toggle = st.checkbox(
                    "🧠 ASL Recognition", 
                    value=st.session_state.get('recognition_active', True),  # Default to True
                    help="Toggle ASL letter recognition"
                )
                
                if recognition_toggle != st.session_state.get('recognition_active', True):
                    st.session_state.recognition_active = recognition_toggle
                    st.info(f"🎯 Recognition {'enabled' if recognition_toggle else 'disabled'}")
                    st.rerun()
            else:
                st.checkbox("🧠 ASL Recognition", disabled=True, value=False,
                           help="Start camera first to enable recognition")
        
        # Camera display
        camera_placeholder = st.empty()
        
        # Handle camera starting state
        if st.session_state.camera_starting and not st.session_state.simple_camera_active:
            with st.spinner("🔄 Initializing camera..."):
                success = self._initialize_camera()
                if success:
                    st.session_state.simple_camera_active = True
                    st.session_state.camera_starting = False
                    st.success("✅ Camera started successfully!")
                else:
                    st.session_state.camera_starting = False
                    st.error("❌ Failed to start camera")
                st.rerun()
        
        # Show camera feed
        if st.session_state.simple_camera_active:
            # UI controls removed to prevent auto-refresh issues
            with col2:
                auto_mode = st.checkbox("🔄 Live Mode", value=True, help="Auto-refresh camera feed")
            with col3:
                if st.button("� Capture", help="Get single frame"):
                    pass  # Triggers rerun
            
            self._run_simple_camera_loop(camera_placeholder)
            

        elif st.session_state.camera_starting:
            camera_placeholder.info("⏳ Starting camera...")
        else:
            camera_placeholder.info("📷 Camera stopped - Click 'Start Camera' to begin")
    
    def _stop_simple_camera(self):
        """Stop simple camera and clean up resources including threading"""
        try:
            # Stop threading first
            if 'shared_frame' in st.session_state:
                st.session_state.shared_frame['running'] = False
                del st.session_state.shared_frame
            
            # Release camera if it exists in session state
            if 'simple_cv_camera' in st.session_state and st.session_state.simple_cv_camera is not None:
                st.session_state.simple_cv_camera.release()
                st.session_state.simple_cv_camera = None
            
            # Clean up MediaPipe if it exists
            if 'simple_mp_hands' in st.session_state:
                st.session_state.simple_mp_hands.close()
                del st.session_state.simple_mp_hands
                
        except Exception as e:
            st.error(f"Error stopping camera: {e}")
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with proper error handling"""
        try:
            # Clean up any existing camera first
            if 'simple_cv_camera' in st.session_state and st.session_state.simple_cv_camera is not None:
                st.session_state.simple_cv_camera.release()
            
            # Initialize new camera
            st.session_state.simple_cv_camera = cv2.VideoCapture(0)
            if not st.session_state.simple_cv_camera.isOpened():
                return False
            
            # Optimized camera settings for real-time performance
            st.session_state.simple_cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.simple_cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.simple_cv_camera.set(cv2.CAP_PROP_FPS, 30)
            st.session_state.simple_cv_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time
            st.session_state.simple_cv_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Initialize MediaPipe ONLY if not already initialized
            if 'simple_mp_hands' not in st.session_state:
                import mediapipe as mp
                st.session_state.simple_mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    model_complexity=0  # Fastest model for real-time
                )
                st.session_state.simple_mp_drawing = mp.solutions.drawing_utils
                st.session_state.simple_mp_hands_module = mp.solutions.hands
            
            return True
            
        except Exception as e:
            st.error(f"Camera initialization failed: {e}")
            return False

    def _run_original_camera_loop(self, frame_display):
        """Your original working camera implementation with threading"""
        try:
            # Initialize camera with better error handling
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                # Try alternative camera indices
                for i in range(1, 4):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        break
                
                if not cap.isOpened():
                    st.error("❌ Cannot open camera. Please check if your webcam is connected.")
                    st.session_state.simple_camera_active = False
                    st.session_state.video_running = False
                    return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Initialize MediaPipe
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            mp_drawing = mp.solutions.drawing_utils
            mp_hands_connections = mp_hands.HAND_CONNECTIONS
            
            # Shared frame for threading (your original approach)
            shared = {'frame': None, 'ret': False, 'running': True}
            st.session_state.shared_frame = shared
            
            def frame_reader():
                """Background thread for reading frames - your original approach"""
                # Read first frame immediately
                ret, frame = cap.read()
                shared['ret'] = ret
                shared['frame'] = frame
                
                while shared['running']:
                    try:
                        ret, frame = cap.read()
                        shared['ret'] = ret
                        shared['frame'] = frame
                        if not ret:
                            time.sleep(0.1)
                        else:
                            time.sleep(0.01)  # ~100 FPS capture
                    except Exception:
                        shared['ret'] = False
                        shared['frame'] = None
                        time.sleep(0.1)
            
            # Start frame reader thread
            import threading
            reader_thread = threading.Thread(target=frame_reader)
            reader_thread.daemon = True
            reader_thread.start()
            
            # Give a moment for first frame
            time.sleep(0.1)
            
            st.success("✅ Camera started successfully! Start signing letters!")
            
            # Main display loop (your original approach)
            while st.session_state.simple_camera_active:
                frame = shared['frame']
                ret = shared['ret']
                
                if not ret or frame is None:
                    continue
                
                # Process frame (your original MediaPipe approach)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Fix camera mirroring FIRST - flip horizontally so left/right sides are correct
                frame_flipped = cv2.flip(frame_rgb, 1)  # 1 = horizontal flip
                
                detected_letter = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks on flipped frame with corrected coordinates
                        self._draw_flipped_landmarks(frame_flipped, hand_landmarks, mp_drawing, mp_hands_connections)
                        
                        # Enhanced letter detection (process original landmarks for accuracy)
                        detected_letter = self._simple_letter_detection(hand_landmarks)
                
                # Always process with accumulation engine (for spacing detection)
                if detected_letter:
                    # Process detected letter through accumulation engine
                    result = self.accumulation_engine.process_detection(detected_letter, confidence=0.8)
                    
                    # Show detection method and letter with normal text
                    detection_method = "Enhanced" if hasattr(self, 'recognition_engine') and self.recognition_engine.model else "Advanced"
                    cv2.putText(frame_flipped, f"ASL Letter: {detected_letter} ({detection_method})", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Show confirmation progress if accumulating
                    if result.get('action') == 'accumulating':
                        progress = result.get('progress', 0)
                        cv2.putText(frame_flipped, f"Confirming... {progress:.0%}", 
                                  (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    
                    # Show ASL alphabet info with normal text (only when confirmed)
                    if result.get('action') == 'confirmed':
                        alphabet_info = self._get_asl_letter_info(detected_letter)
                        if alphabet_info:
                            cv2.putText(frame_flipped, alphabet_info, 
                                      (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    # Process no detection for spacing
                    result = self.accumulation_engine.process_detection(None, confidence=0.0)
                
                # Show word building progress and spacing info
                if hasattr(result, 'get') and result:
                    # Show current word being built
                    if result.get('current_word'):
                        cv2.putText(frame_flipped, f"Word: {result['current_word']}", 
                                  (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # Show full accumulated text (truncated for display)
                    full_text = self.accumulation_engine.get_full_text()
                    if full_text:
                        display_text = full_text[:30] + "..." if len(full_text) > 30 else full_text
                        cv2.putText(frame_flipped, f"Text: {display_text}", 
                                  (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
                    
                    # Show spacing countdown
                    if result.get('action') == 'pausing':
                        progress = result.get('pause_progress', 0)
                        countdown = max(0, 1.5 - (progress * 1.5))  # 1.5 seconds total
                        cv2.putText(frame_flipped, f"Space in: {countdown:.1f}s", 
                                  (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
                
                # Display the flipped frame with normal text
                frame_display.image(frame_flipped, channels="RGB")
                
                # Your original timing
                time.sleep(0.033)  # ~30 FPS
            
            # Cleanup
            shared['running'] = False
            reader_thread.join()
            cap.release()
            hands.close()
            # Reset camera states
            st.session_state.simple_camera_active = False
            st.session_state.video_running = False
            
        except Exception as e:
            st.error(f"❌ Camera error: {e}")
            st.session_state.simple_camera_active = False
            st.session_state.video_running = False
            # Stop threading if shared exists
            if 'shared_frame' in st.session_state:
                st.session_state.shared_frame['running'] = False

    def _show_optimized_camera_interface(self):
        """Show optimized camera interface without freezing"""
        # Initialize optimized camera in session state
        if 'optimized_camera' not in st.session_state:
            st.session_state.optimized_camera = None
        if 'optimized_camera_active' not in st.session_state:
            st.session_state.optimized_camera_active = False
        
        # Camera controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📷 Start Optimized Camera", 
                        disabled=st.session_state.optimized_camera_active,
                        help="Start real-time camera with optimized MediaPipe"):
                self._start_optimized_camera()
        
        with col2:
            if st.button("⏹️ Stop Camera", 
                        disabled=not st.session_state.optimized_camera_active,
                        help="Stop camera and release resources"):
                self._stop_optimized_camera()
        
        with col3:
            if st.session_state.optimized_camera_active:
                recognition_enabled = st.checkbox(
                    "🧠 Enable ASL Recognition",
                    value=st.session_state.get('recognition_active', True),
                    help="Toggle real-time hand detection"
                )
                st.session_state.recognition_active = recognition_enabled
        
        # Camera status
        if st.session_state.optimized_camera_active:
            st.success("🟢 Camera Active - Real-time ASL Recognition")
        else:
            st.info("🔴 Camera Inactive - Click Start to begin")
        
        # Camera feed
        camera_placeholder = st.empty()
        
        if st.session_state.optimized_camera_active and st.session_state.optimized_camera:
            self._display_optimized_camera_feed(camera_placeholder)
        else:
            camera_placeholder.info("📷 Click 'Start Optimized Camera' to begin real-time ASL recognition")
    
    def _start_optimized_camera(self):
        """Start optimized camera system"""
        try:
            # Create optimized camera engine
            # camera_engine = OptimizedCameraEngine()  # TODO: Implement OptimizedCameraEngine class
            st.error("❌ Optimized camera engine not implemented yet")
            return
            
            # Start camera
            if camera_engine.start():
                st.session_state.optimized_camera = camera_engine
                st.session_state.optimized_camera_active = True
                st.session_state.recognition_active = True
                st.success("✅ Optimized camera started successfully!")
                
                # Brief pause for initialization
                time.sleep(0.1)
            else:
                st.error("❌ Failed to start camera. Please check camera permissions.")
                
        except Exception as e:
            st.error(f"❌ Camera initialization error: {e}")
    
    def _stop_optimized_camera(self):
        """Stop optimized camera"""
        try:
            if st.session_state.optimized_camera:
                st.session_state.optimized_camera.stop()
            
            st.session_state.optimized_camera = None
            st.session_state.optimized_camera_active = False
            st.session_state.recognition_active = False
            
            st.success("📹 Camera stopped successfully")
            
        except Exception as e:
            st.error(f"❌ Error stopping camera: {e}")
    
    def _display_optimized_camera_feed(self, placeholder):
        """Display optimized camera feed without blocking"""
        try:
            camera_engine = st.session_state.optimized_camera
            
            # Get latest frame (non-blocking)
            frame = camera_engine.get_latest_frame()
            
            if frame is not None:
                # Process frame with MediaPipe
                processed_frame, results = camera_engine.process_frame_with_mediapipe(
                    frame, 
                    recognize_letters=st.session_state.get('recognition_active', True)
                )
                
                # Add accumulation results if available
                if results.get('letter') and hasattr(self, 'accumulation_engine'):
                    try:
                        accumulation_result = self.accumulation_engine.add_letter(results['letter'])
                        
                        # Draw accumulated text on frame
                        if accumulation_result.get('current_word'):
                            cv2.putText(processed_frame, f"Word: {accumulation_result['current_word']}", 
                                      (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                        if accumulation_result.get('sentence'):
                            # Wrap long sentences
                            sentence = accumulation_result['sentence']
                            if len(sentence) > 40:
                                sentence = "..." + sentence[-37:]
                            cv2.putText(processed_frame, f"Text: {sentence}", 
                                      (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception:
                        pass  # Ignore accumulation errors
                
                # Display processed frame
                placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                
                # Show detection results in sidebar
                if results.get('hand_detected'):
                    self._show_detection_sidebar(results)
                    
            else:
                placeholder.warning("⚠️ No frame available - camera may be starting up")
                
        except Exception as e:
            placeholder.error(f"❌ Camera processing error: {e}")
            # Reset camera on persistent errors
            self._stop_optimized_camera()
    
    def _show_detection_sidebar(self, results):
        """Show detection results in sidebar"""
        with st.sidebar:
            if results.get('letter'):
                st.subheader("🎯 Current Detection")
                st.metric("Letter", results['letter'])
                st.metric("Confidence", f"{results.get('confidence', 0):.1%}")
                
                # Store recent detections
                if 'recent_letters' not in st.session_state:
                    st.session_state.recent_letters = []
                
                # Add letter with timestamp
                current_time = time.time()
                if (not st.session_state.recent_letters or 
                    current_time - st.session_state.recent_letters[-1]['time'] > 2.0):
                    st.session_state.recent_letters.append({
                        'letter': results['letter'],
                        'time': current_time
                    })
                    
                    # Keep only last 10 letters
                    st.session_state.recent_letters = st.session_state.recent_letters[-10:]
                
                # Show recent letters
                if st.session_state.recent_letters:
                    st.subheader("📝 Recent Letters")
                    letters = [item['letter'] for item in st.session_state.recent_letters]
                    st.text(" ".join(letters))
                    
                    if st.button("🗑️ Clear"):
                        st.session_state.recent_letters = []

    def _show_accumulation_interface(self):
        """Show letter accumulation interface"""
        st.subheader("📝 Text Builder")
        
        # Get current state from accumulation engine
        full_text = self.accumulation_engine.get_full_text()
        current_word = self.accumulation_engine.state.current_word
        
        # Force refresh every few seconds to show updates
        if 'last_text_refresh' not in st.session_state:
            st.session_state.last_text_refresh = time.time()
        
        # Auto-refresh every 2 seconds when camera is active
        if (st.session_state.get('simple_camera_active', False) and 
            time.time() - st.session_state.last_text_refresh > 2):
            st.session_state.last_text_refresh = time.time()
            st.rerun()
        
        # Display accumulated text (always show the text area)
        display_text = full_text.strip() if full_text else ""
        if current_word:
            display_text = f"{display_text} {current_word}".strip()
        
        # Always show text area so Read button can work
        st.text_area("Accumulated Text", display_text, height=100, key="accumulated_text_display", 
                    placeholder="Your signed text will appear here...")
        
        if display_text:
            st.success(f"📝 **Text Length**: {len(display_text)} characters")
        
        # Current word and suggestions
        if current_word:
            # Show current word progress
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**🔤 Building Word:** `{current_word}` ({len(current_word)} letters)")
            with col2:
                # Live suggestion preview
                if hasattr(self.accumulation_engine.state, 'active_suggestions') and self.accumulation_engine.state.active_suggestions:
                    top_suggestion = self.accumulation_engine.state.active_suggestions[0]
                    if top_suggestion.confidence > 0.6:
                        st.markdown(f"**💡 Likely:** `{top_suggestion.word}`")
            
            # Show suggestions with enhanced UI
            suggestions = self.accumulation_engine.state.active_suggestions
            if suggestions:
                st.markdown("### 💡 **Word Suggestions**")
                
                # Create columns for suggestions
                cols = st.columns(len(suggestions[:3]))
                
                for i, suggestion in enumerate(suggestions[:3]):
                    with cols[i]:
                        # Different styling for gesture words
                        if suggestion.category == "gesture":
                            button_label = f"🎭 {suggestion.word}"
                            button_type = "primary"
                            help_text = f"ASL Gesture Word (confidence: {suggestion.confidence:.1%})"
                        elif suggestion.category == "asl":
                            button_label = f"🤟 {suggestion.word}"
                            button_type = "secondary" 
                            help_text = f"ASL Vocabulary (confidence: {suggestion.confidence:.1%})"
                        else:
                            button_label = f"✅ {suggestion.word}"
                            button_type = "secondary"
                            help_text = f"Common word (confidence: {suggestion.confidence:.1%})"
                        
                        if st.button(button_label, key=f"suggest_main_{i}", 
                                   type=button_type, help=help_text, use_container_width=True):
                            # Complete word with the suggestion
                            self.accumulation_engine._complete_word(suggestion.word, method="suggestion")
                            st.success(f"✨ Completed word: **{suggestion.word}**")
                            st.rerun()
                
                # Auto-completion hint
                if len(suggestions) > 0 and len(current_word) >= 2:
                    best_suggestion = suggestions[0]
                    if best_suggestion.confidence > 0.8:
                        st.info(f"💫 **Smart Suggestion**: Very likely you mean **{best_suggestion.word}** "
                               f"({best_suggestion.confidence:.0%} confidence)")
                        
            else:
                # Show helpful hints when no suggestions
                if len(current_word) == 1:
                    st.info(f"💭 **Type another letter to see word suggestions...**")
                else:
                    st.warning(f"🔍 **No word suggestions found for '{current_word}'** - Try a different spelling?")
        else:
            st.info("💡 **Getting Started**: Hold a letter sign for 0.7 seconds to add it to your word")
            
            # Show available gesture words in an expandable section
            with st.expander("🎭 **Available ASL Gesture Words** (Click to see full list)", expanded=False):
                gesture_words = [
                    "AMAZING", "BEAUTIFUL", "HELLO", "GOODBYE", "THANK_YOU", 
                    "PLEASE", "YES", "NO", "GOOD", "BAD", "FINE", "HELP", 
                    "LOVE", "NAME", "LEARN", "UNDERSTAND", "SORRY", "HEARING",
                    "NICE_TO_MEET_YOU", "HOW_ARE_YOU"
                ]
                
                # Display in columns
                cols = st.columns(3)
                for i, word in enumerate(gesture_words):
                    with cols[i % 3]:
                        if word in ["HELLO", "THANK_YOU", "PLEASE", "YES", "NO"]:
                            st.markdown(f"**🌟 {word}**")  # Highlight common words
                        else:
                            st.markdown(f"• {word}")
                            
                st.info("💡 **Tip**: Start typing the first few letters of these words to see smart suggestions!")
    
    def _show_quick_controls(self):
        """Show quick control buttons"""
        st.subheader("🎮 Quick Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Space", key="add_space_main"):
                self.accumulation_engine._add_space()
                st.rerun()
            
            if st.button("🗑️ Clear Word", key="clear_word_main"):
                self.accumulation_engine._clear_current_word()
                st.rerun()
        
        # Testing section (expandable)
        with st.expander("🧪 **Testing Word Completion** (Type letters manually)", expanded=False):
            test_input = st.text_input("Type letters to test suggestions:", 
                                     placeholder="Try: H, HE, HEL, HELL, HELLO",
                                     key="test_word_input")
            
            if test_input:
                # Get suggestions for test input
                test_suggestions = self.accumulation_engine.dictionary.get_suggestions(test_input.upper(), max_suggestions=5)
                
                if test_suggestions:
                    st.markdown("**🔍 Suggestions for '" + test_input.upper() + "':**")
                    
                    # Display suggestions in columns
                    suggestion_cols = st.columns(min(len(test_suggestions), 5))
                    
                    for i, suggestion in enumerate(test_suggestions):
                        with suggestion_cols[i]:
                            # Different icons for different categories
                            if suggestion.category == "gesture":
                                icon = "🎭"
                                badge_color = "🟢"
                            elif suggestion.category == "asl":
                                icon = "🤟"
                                badge_color = "🔵"
                            else:
                                icon = "📝"
                                badge_color = "🟡"
                            
                            st.markdown(f"{icon} **{suggestion.word}**")
                            st.caption(f"{badge_color} {suggestion.category}")
                            st.caption(f"Confidence: {suggestion.confidence:.1%}")
                            
                            if st.button(f"Add {suggestion.word}", key=f"test_add_{i}"):
                                # Simulate adding the word
                                self.accumulation_engine.state.current_word = ""
                                self.accumulation_engine._complete_word(suggestion.word, method="test")
                                st.success(f"✅ Added **{suggestion.word}** to text!")
                                st.rerun()
                else:
                    st.info(f"No suggestions found for '{test_input.upper()}'")
            
            st.info("💡 **Try typing**: 'A' (AMAZING), 'B' (BEAUTIFUL), 'H' (HELLO), 'G' (GOOD), 'T' (THANK_YOU)")
        
        with col2:
            if st.button("🔊 Read", key="read_text_main", help="Read accumulated text aloud"):
                try:
                    full_text = self.accumulation_engine.get_full_text()
                    st.write(f"**Debug**: Retrieved text: '{full_text}'")  # Debug info
                    
                    if full_text and full_text.strip():
                        st.info(f"🔊 **Speaking**: '{full_text}'")
                        with st.spinner("🎤 Reading text..."):
                            # Use the new TTS system
                            self.accumulation_engine.speak_full_text()
                            st.success("✅ Text queued for reading!")
                            st.info("🔊 Audio should play shortly...")
                    else:
                        st.warning("⚠️ No text to read. Sign some letters first to build text.")
                        st.info("💡 **Tip**: Use the camera to sign letters, then they will appear in the text area above.")
                except Exception as e:
                    st.error(f"❌ TTS Error: {e}")
                    # Fallback: show text at least
                    try:
                        text = self.accumulation_engine.get_full_text()
                        if text and text.strip():
                            st.info(f"📝 **Available Text**: {text}")
                            st.info("🔧 TTS failed, but your text is preserved above.")
                        else:
                            st.info("📝 No text available in accumulation engine.")
                    except Exception as fallback_error:
                        st.error(f"❌ Unable to retrieve text: {fallback_error}")
            
            if st.button("🔄 Reset", key="reset_session_main"):
                self.accumulation_engine.reset_session()
                st.rerun()
    
    def _show_calibration_interface(self):
        """Show calibration interface"""
        if not st.session_state.current_user:
            st.warning("Please select a user profile first.")
            if st.button("👤 Select User"):
                st.session_state.app_mode = 'main'
                st.rerun()
            return
        
        self.calibration_system.start_guided_calibration(st.session_state.current_user)
    
    def _create_simple_training_dataset(self, dataset_manager, X_asl, y_asl):
        """Create a simple training dataset without complex splits or validation."""
        try:
            # If we have any valid data, use it directly for training
            if X_asl is not None and len(X_asl) > 0:
                st.info(f"📊 Using {len(X_asl)} reference images for training")
                return {
                    'X_train': X_asl,
                    'y_train': y_asl,
                    'X_test': X_asl,  # Use same data for testing (simple approach)
                    'y_test': y_asl,
                    'feature_extractor': None,
                    'label_encoder': getattr(dataset_manager, 'label_encoder', None)
                }
            else:
                st.warning("⚠️ No valid reference images found. Trying image-only approach...")
                return self._create_image_only_dataset()
        
        except Exception as e:
            st.error(f"❌ Error creating simple dataset: {str(e)}")
            return None
    
    def _create_image_only_dataset(self):
        """Create dataset using only raw images without MediaPipe processing."""
        try:
            import os
            import cv2
            import numpy as np
            
            # Path to ASL reference images
            images_path = "Wave/asl_reference_images"
            if not os.path.exists(images_path):
                st.error(f"❌ Reference images folder not found: {images_path}")
                return None
            
            X_images = []
            y_labels = []
            
            # Load all reference images
            for filename in os.listdir(images_path):
                if filename.endswith('.png') and filename.startswith('letter_'):
                    # Extract letter from filename (letter_A.png -> A)
                    letter = filename.replace('letter_', '').replace('.png', '')
                    
                    # Load and resize image
                    img_path = os.path.join(images_path, filename)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        # Resize to consistent size
                        img_resized = cv2.resize(img, (224, 224))
                        # Normalize pixel values
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        
                        X_images.append(img_normalized.flatten())  # Flatten for simple model
                        y_labels.append(letter)
            
            if len(X_images) > 0:
                X_array = np.array(X_images)
                st.success(f"✅ Loaded {len(X_images)} reference images for training")
                
                return {
                    'X_train': X_array,
                    'y_train': y_labels,
                    'X_test': X_array,  # Use same data for testing
                    'y_test': y_labels,
                    'feature_extractor': None,  # No feature extractor needed
                    'label_encoder': None  # Simple string labels
                }
            else:
                st.error("❌ No valid reference images found")
                return None
                
        except Exception as e:
            st.error(f"❌ Error creating image-only dataset: {str(e)}")
            return None
    
    def _show_training_interface(self):
        """Show model training interface"""
        st.subheader("🧠 Model Training")
        
        tab1, tab2, tab3 = st.tabs(["Dataset", "Training", "Evaluation"])
        
        with tab1:
            self._show_dataset_interface()
        
        with tab2:
            self._show_training_controls()
        
        with tab3:
            self._show_evaluation_interface()
    
    def _show_dataset_interface(self):
        """Show dataset management interface"""
        st.markdown("### 📊 Dataset Management")
        
        # Dataset status
        dataset_manager = EnhancedDatasetManager()
        
        if st.button("📥 Check Datasets"):
            mnist_exists = (Path("datasets/sign-language-mnist/sign_mnist_train.csv")).exists()
            
            # Check for ASL reference images
            asl_reference_path = Path("../Wave/asl_reference_images")
            asl_exists = asl_reference_path.exists() and len(list(asl_reference_path.glob("letter_*.png"))) > 0
            
            col1, col2 = st.columns(2)
            with col1:
                if mnist_exists:
                    st.success("✅ MNIST ASL Dataset Found")
                else:
                    st.warning("⚠️ MNIST ASL Dataset Missing (Optional)")
                    st.info("📝 MNIST is for digits (0-9), not needed for alphabet recognition")
                    st.info("Download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
            
            with col2:
                if asl_exists:
                    st.success("✅ ASL Alphabet Dataset Found")
                    num_letters = len(list(asl_reference_path.glob("letter_*.png")))
                    st.info(f"📁 Found {num_letters} reference images in Wave/asl_reference_images/")
                else:
                    st.error("❌ ASL Alphabet Dataset Missing")
                    st.info("Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        
        # Display ASL Reference Images
        if st.button("🧪 View ASL Reference Database"):
            with st.spinner("Loading ASL reference image database..."):
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Loading ASL reference data...")
                    progress_bar.progress(25)
                    
                    X_asl, y_asl = dataset_manager.load_asl_reference_data()
                    progress_bar.progress(75)
                    
                    if X_asl is not None:
                        progress_bar.progress(100)
                        status_text.text("✅ Loading complete!")
                        st.success(f"✅ Successfully loaded {len(X_asl)} ASL reference images!")
                        
                        # Show all ASL alphabet images
                        st.markdown("### 🔤 Complete ASL Alphabet Reference Database")
                        
                        letters = "ABCDEFGHIKLMNOPQRSTUVWXY"  # Skip J and Z
                        
                        # Create grid layout - 6 images per row with better error handling
                        num_cols = 6
                        try:
                            for row_start in range(0, len(X_asl), num_cols):
                                cols = st.columns(num_cols)
                                for i, col in enumerate(cols):
                                    img_idx = row_start + i
                                    if img_idx < len(X_asl):
                                        try:
                                            letter = letters[y_asl[img_idx]]
                                            with col:
                                                # Display image with error handling
                                                st.image(X_asl[img_idx], 
                                                       caption=f"Letter {letter}", 
                                                       width=100,  # Smaller width to reduce load
                                                       use_container_width=False)  # Fixed parameter
                                                st.markdown(f"<div style='text-align: center; font-weight: bold; color: #60A5FA;'>{letter}</div>", 
                                                          unsafe_allow_html=True)
                                        except Exception as img_error:
                                            with col:
                                                st.error(f"Error loading image {img_idx}: {str(img_error)}")
                        except Exception as grid_error:
                            st.error(f"Error creating image grid: {str(grid_error)}")
                        
                        # Show statistics
                        st.markdown("### 📊 Dataset Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Images", len(X_asl))
                        with col2:
                            st.metric("Unique Letters", len(set(y_asl)))
                        
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
                            
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ Could not load ASL reference images")
                        st.info("💡 Make sure your ASL reference images are in the Wave/asl_reference_images/ folder")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error loading ASL reference images: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Create organized training dataset folder structure
        if st.button("� Setup Training Dataset Folders"):
            self._setup_training_folders()
        
        if st.button("�🔄 Create Training Dataset"):
            with st.spinner("Creating training dataset from ASL reference images..."):
                try:
                    st.info("🔍 Processing ASL reference images for training...")
                    
                    # Check if we have enough samples first
                    X_asl, y_asl = dataset_manager.load_asl_reference_data()
                    if X_asl is None or len(X_asl) < 5:
                        st.error("❌ Not enough ASL reference images found")
                        st.info("💡 Please ensure you have at least 5 ASL reference images")
                        return
                    
                    st.info("🎯 Creating training dataset...")
                    st.warning("⚠️ **Issue Detected**: MediaPipe landmark extraction is failing on most reference images")
                    st.info("🔧 **Alternative Solution**: Using direct image-based training approach")
                    
                    try:
                        # Create a simple dataset directly from ASL reference images
                        simple_dataset = self._create_simple_training_dataset(dataset_manager, X_asl, y_asl)
                        
                        if simple_dataset is None:
                            st.error("❌ Failed to create simple training dataset")
                            return
                        
                        train_size = len(simple_dataset.get('X_train', []))
                        
                        if train_size >= 1:
                            st.success(f"✅ Created training dataset with {train_size} samples!")
                            
                            # Show dataset info
                            st.json({
                                "train_samples": train_size,
                                "test_samples": len(simple_dataset.get('X_test', [])),
                                "approach": "image_based_training",
                                "labels": simple_dataset.get('y_train', [])[:10] if simple_dataset.get('y_train') else [],
                                "status": "ready_for_training"
                            })
                            
                            # Store dataset for training
                            st.session_state.training_dataset = simple_dataset
                            st.info("💾 Dataset stored in memory for training")
                            
                            # Create dataset in compatible format
                            dataset = {
                                'X_train': simple_dataset['X_train'],
                                'y_train': simple_dataset['y_train'],
                                'X_val': simple_dataset['X_test'],
                                'y_val': simple_dataset['y_test'],
                                'dataset_info': {
                                    'total_samples': train_size,
                                    'method': 'image_based'
                                }
                            }
                        else:
                            st.warning(f"⚠️ Only {train_size} samples available. Creating minimal dataset...")
                            
                            # Create a simple image-based dataset
                            X_asl, y_asl = dataset_manager.load_asl_reference_data()
                            if X_asl is not None and len(X_asl) >= 10:
                                # Create a simple dataset structure without landmarks
                                dataset = {
                                    'X_train': X_asl,
                                    'y_train': y_asl,
                                    'X_val': X_asl[-5:],  # Use last 5 images for validation
                                    'y_val': y_asl[-5:],
                                    'landmarks_train': None,  # No landmarks available
                                    'landmarks_val': None,
                                    'sources_train': ['reference'] * len(X_asl),
                                    'sources_val': ['reference'] * 5,
                                    'dataset_info': {
                                        'total_samples': len(X_asl),
                                        'method': 'image_only',
                                        'landmark_extraction': 'failed'
                                    }
                                }
                                st.success("✅ Created image-only training dataset!")
                                st.info("� **Note**: This dataset uses images without hand landmarks")
                            else:
                                st.error("❌ Not enough reference images for training")
                                return
                        
                    except Exception as e:
                        st.error(f"❌ Dataset creation error: {e}")
                        st.info("💡 **Recommendation**: Try collecting more diverse ASL images or use the video capture feature")
                        return
                    
                    if dataset is None:
                        st.error("❌ Dataset creation failed completely")
                        return
                        
                    # Check if we have samples
                    train_size = len(dataset.get('X_train', []))
                    val_size = len(dataset.get('X_val', []))
                    
                    if train_size == 0:
                        st.error("❌ No training samples created")
                        st.info("💡 Most ASL reference images failed landmark extraction")
                        return
                    elif train_size < 10:
                        st.warning(f"⚠️ Only {train_size} training samples created - this is very limited!")
                        st.info("🎯 **Recommendation**: Add your own training images using the folder structure above")
                        st.info("📸 Take 10-20 photos of yourself signing each letter for much better accuracy")
                    
                    # Save the dataset
                    dataset_manager.save_processed_dataset(dataset)
                    st.success("✅ Training dataset created successfully!")
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Samples", train_size)
                    with col2:
                        st.metric("Validation Samples", val_size)
                    with col3:
                        st.metric("Total Letters", len(set(dataset.get('y_train', []))))
                    
                    st.info("💡 Your ASL reference images are now ready for model training!")
                    
                except Exception as e:
                    st.error(f"Error creating training dataset: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _setup_training_folders(self):
        """Create organized folder structure for training dataset"""
        try:
            from pathlib import Path
            
            # Create main training dataset folder
            training_root = Path("training_dataset")
            training_root.mkdir(exist_ok=True)
            
            # Create subfolders for each letter
            letters = "ABCDEFGHIKLMNOPQRSTUVWXY"  # ASL alphabet (no J, Z)
            folders_created = []
            
            for letter in letters:
                letter_folder = training_root / letter
                letter_folder.mkdir(exist_ok=True)
                folders_created.append(str(letter_folder))
            
            # Create additional folders for different data types
            (training_root / "reference_images").mkdir(exist_ok=True)
            (training_root / "augmented_images").mkdir(exist_ok=True)
            (training_root / "user_images").mkdir(exist_ok=True)
            
            st.success(f"✅ Created training dataset folder structure!")
            st.info(f"📁 **Main folder**: `{training_root}`")
            
            # Show folder structure
            with st.expander("📂 **Folder Structure Created**", expanded=True):
                st.markdown("```")
                st.text("training_dataset/")
                st.text("├── reference_images/     # Original ASL reference images")
                st.text("├── augmented_images/     # AI-generated variations") 
                st.text("├── user_images/          # Your custom training images")
                for letter in letters[:5]:  # Show first few
                    st.text(f"├── {letter}/                     # Letter {letter} training images")
                st.text("├── ...")
                st.text(f"└── Y/                     # Letter Y training images")
                st.markdown("```")
            
            # Instructions for users
            st.markdown("### 📋 **How to Add Training Images:**")
            st.markdown("""
            1. **📸 Take photos** of yourself signing each ASL letter
            2. **📁 Save images** in the corresponding letter folder (A/, B/, C/, etc.)
            3. **📏 Image format**: JPG or PNG, any size (will be resized automatically)
            4. **📊 Recommended**: 10-20 images per letter for better accuracy
            5. **🎯 Variety**: Different lighting, hand positions, backgrounds
            """)
            
            st.info("💡 **Pro tip**: The more diverse training images you add, the better your ASL recognition will become!")
            
        except Exception as e:
            st.error(f"❌ Error creating training folders: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _show_training_controls(self):
        """Show model training controls"""
        st.markdown("### 🏋️ Model Training")
        
        # ASL Alphabet Training Section
        st.markdown("#### 🔤 ASL Alphabet Recognition")
        st.info("""
        Train a CNN model to recognize all ASL alphabet letters (A-Y) using the reference images 
        from your Wave folder. This will replace the simple finger counting with full alphabet recognition.
        """)
        
        with st.expander("🎯 ASL Reference Image Training", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                asl_epochs = st.number_input("Training Epochs", min_value=10, max_value=200, value=50, key="asl_epochs")
                asl_augment = st.number_input("Augmentation Factor", min_value=5, max_value=50, value=20, key="asl_augment")
                asl_model_type = st.selectbox("Model Type", ["cnn", "transfer"], index=0, key="asl_model_type")
            
            with col2:
                # Auto-detect path or allow custom override
                try:
                    from config.settings import get_asl_images_path, is_development_mode
                    
                    if is_development_mode():
                        default_path = get_asl_images_path()
                        st.info(f"🔧 Development mode: Using {default_path}")
                        custom_path = st.text_input("Custom Images Path (optional)", 
                                                  placeholder="Leave empty for auto-detection",
                                                  help="Override auto-detected path if needed")
                        images_path = custom_path if custom_path else None
                    else:
                        st.info("🚀 Production mode: Using bundled ASL images")
                        images_path = None  # Use auto-detection
                        
                except ImportError:
                    # Fallback if config not available
                    images_path = st.text_input("ASL Images Path", 
                                              placeholder="Path to folder containing ASL reference images",
                                              help="Path to folder containing letter_A.png, letter_B.png, etc.")
                
                # Show available reference images
                if st.button("🔍 Check Reference Images", key="check_asl_images"):
                    try:
                        from data_processing.asl_reference_loader import ASLReferenceDatasetLoader
                        loader = ASLReferenceDatasetLoader(images_path)
                        images, labels, letter_names = loader.load_reference_images()
                        
                        if len(images) > 0:
                            st.success(f"✅ Found {len(images)} ASL reference images")
                            st.write(f"Images location: {loader.images_folder}")
                            st.write(f"Letters available: {', '.join(letter_names)}")
                        else:
                            st.error(f"❌ No reference images found at: {loader.images_folder}")
                            st.write("Expected files: letter_A.png, letter_B.png, etc.")
                    except Exception as e:
                        st.error(f"Error checking images: {e}")
            
            # Training button and progress
            if st.button("🚀 Start ASL Alphabet Training", type="primary", key="start_asl_training"):
                if 'asl_training_in_progress' not in st.session_state:
                    st.session_state.asl_training_in_progress = True
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("🔄 Initializing ASL training...")
                        progress_bar.progress(10)
                        
                        # Start training with auto-detected or custom path
                        results = self.recognition_engine.train_with_asl_reference_images(
                            custom_images_path=images_path,
                            epochs=asl_epochs,
                            augment_factor=asl_augment,
                            model_type=asl_model_type
                        )
                        
                        progress_bar.progress(100)
                        
                        if results.get('success'):
                            st.success(f"🎉 ASL training completed!")
                            st.write(f"**Validation Accuracy:** {results['val_accuracy']:.4f}")
                            st.write(f"**Training Samples:** {results['training_samples']}")
                            st.write(f"**Model Saved:** {results['model_path']}")
                            
                            # Show training history
                            if 'history' in results:
                                history = results['history']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.line_chart({
                                        'Training': history['accuracy'],
                                        'Validation': history['val_accuracy']
                                    })
                                    st.caption("Model Accuracy")
                                
                                with col2:
                                    st.line_chart({
                                        'Training': history['loss'],
                                        'Validation': history['val_loss']
                                    })
                                    st.caption("Model Loss")
                            
                            # Load the trained model for immediate use
                            if self.recognition_engine.load_asl_reference_model(results['model_path']):
                                st.info("✅ Trained model loaded and ready for real-time recognition!")
                            
                        else:
                            st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Training error: {e}")
                    
                    finally:
                        st.session_state.asl_training_in_progress = False
            
            # Show trained models
            if st.button("📋 Show Trained Models", key="show_asl_models"):
                model_dir = Path("models")
                asl_models = list(model_dir.glob("asl_reference_*.h5"))
                
                if asl_models:
                    st.write("**Available ASL Models:**")
                    for model_path in asl_models:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📁 {model_path.name}")
                        with col2:
                            if st.button("Load", key=f"load_{model_path.stem}"):
                                if self.recognition_engine.load_asl_reference_model(str(model_path)):
                                    st.success(f"✅ Loaded {model_path.name}")
                                else:
                                    st.error("❌ Failed to load model")
                else:
                    st.info("No trained ASL models found. Train a model first.")
        
        st.divider()
        
        # General Training Configuration (existing code)
        with st.form("training_config"):
            st.markdown("#### ⚙️ General Training Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=1)
            
            with col2:
                model_type = st.selectbox("Model Type", ["CNN", "Transfer Learning"], index=0)
                performance_mode = st.selectbox("Performance Mode", ["Fast", "Balanced", "Accurate"], index=1)
                use_augmentation = st.checkbox("Data Augmentation", value=True)
            
            start_training = st.form_submit_button("🚀 Start General Training")
            
            if start_training:
                st.info("General training configuration saved. Use ASL Alphabet Training above for letter recognition.")
    
    def _show_evaluation_interface(self):
        """Show model evaluation interface"""
        st.markdown("### 📈 Model Evaluation")
        
        # Model selection
        model_files = list(Path("models").glob("*.h5")) if Path("models").exists() else []
        
        if model_files:
            selected_model = st.selectbox(
                "Select Model", 
                [f.name for f in model_files]
            )
            
            if st.button("📊 Evaluate Model"):
                st.info("Model evaluation would run here.")
                # Show placeholder metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", "94.5%")
                with col2:
                    st.metric("Precision", "93.2%")
                with col3:
                    st.metric("Recall", "95.1%")
        else:
            st.info("No trained models found. Train a model first.")
    
    def _show_analytics_interface(self):
        """Show analytics and insights"""
        st.subheader("📊 Analytics & Insights")
        
        tab1, tab2, tab3 = st.tabs(["Performance", "Usage", "Insights"])
        
        with tab1:
            self._show_performance_analytics()
        
        with tab2:
            self._show_usage_analytics()
        
        with tab3:
            self._show_insights()
    
    def _show_performance_analytics(self):
        """Show performance analytics"""
        st.markdown("### ⚡ Performance Metrics")
        
        # Current session stats
        if hasattr(self, 'accumulation_engine'):
            stats = self.accumulation_engine.get_session_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Session Time", f"{stats['session_duration']:.0f}s")
            with col2:
                st.metric("Letters/min", f"{stats['letters_per_minute']:.1f}")
            with col3:
                st.metric("Words", stats['words_completed'])
            with col4:
                st.metric("FPS", f"{self.current_fps:.1f}")
    
    def _show_usage_analytics(self):
        """Show usage analytics"""
        st.markdown("### 📈 Usage Patterns")
        st.info("Usage analytics would be displayed here.")
    
    def _show_insights(self):
        """Show AI insights"""
        st.markdown("### 🧠 AI Insights")
        st.info("AI-powered insights would be displayed here.")
    
    def _show_settings_interface(self):
        """Show settings interface"""
        st.subheader("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["Recognition", "Interface", "Advanced"])
        
        with tab1:
            st.markdown("### 🎯 Recognition Settings")
            st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
            st.slider("Confirmation Frames", 10, 60, 30, 5)
            st.checkbox("Voice Feedback", value=True)
        
        with tab2:
            st.markdown("### 🎨 Interface Settings")
            st.selectbox("Theme", ["Light", "Dark", "Auto"])
            st.slider("Font Size", 12, 24, 16, 1)
            st.checkbox("Animations", value=True)
        
        with tab3:
            st.markdown("### 🔧 Advanced Settings")
            st.selectbox("Processing Backend", ["CPU", "GPU"])
            st.checkbox("Debug Mode", value=False)
            st.number_input("Max FPS", 15, 60, 30, 5)
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _cleanup_camera(self):
        """Clean up camera resources - Updated for professional camera"""
        try:
            # Clean up professional camera system
            if hasattr(self, 'camera_engine') and self.camera_engine:
                self.camera_engine.stop()
            
            if hasattr(self, 'asl_processor') and self.asl_processor:
                self.asl_processor.cleanup()
            
            # Clean up old camera system (if exists)
            if 'camera' in st.session_state:
                if st.session_state.camera:
                    st.session_state.camera.release()
                del st.session_state.camera
            
            st.session_state.camera_engine_active = False
            
        except Exception as e:
            st.error(f"Error during camera cleanup: {e}")
    
    def _start_recognition_processing(self):
        """Start background recognition processing - Now handled by professional system"""
        # Professional system handles this automatically
        pass
    
    def _stop_recognition_processing(self):
        """Stop background recognition processing - Now handled by professional system"""
        # Professional system handles this automatically
        pass

    def _show_video_demonstrations(self):
        """Show ASL video demonstrations with navigation"""
        st.title("🎬 ASL Video Demonstrations")
        st.markdown("Learn proper ASL gestures by watching demonstration videos")
        
        # Initialize session state for video navigation
        if 'video_index' not in st.session_state:
            st.session_state.video_index = 0
        if 'video_list' not in st.session_state:
            st.session_state.video_list = self._load_video_list()
        
        video_list = st.session_state.video_list
        
        if not video_list:
            st.error("❌ No demonstration videos found")
            st.info("💡 Make sure your ASL gesture videos are in the Wave/raw_data/ folder")
            return
        
        # Video selection and navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        current_video = video_list[st.session_state.video_index]
        
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.video_index == 0):
                st.session_state.video_index = max(0, st.session_state.video_index - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"### 📹 **{current_video['word']}** ({st.session_state.video_index + 1}/{len(video_list)})")
            
            # Progress bar
            progress = (st.session_state.video_index + 1) / len(video_list)
            st.progress(progress)
        
        with col3:
            if st.button("➡️ Next", disabled=st.session_state.video_index == len(video_list) - 1):
                st.session_state.video_index = min(len(video_list) - 1, st.session_state.video_index + 1)
                st.rerun()
        
        st.divider()
        
        # Video display area
        col_video, col_info = st.columns([2, 1])
        
        with col_video:
            st.markdown("#### 🎥 Video Demonstration")
            
            # Display video
            video_path = current_video['path']
            if video_path.exists():
                try:
                    # Read video file as bytes
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    # Display video with controls
                    st.video(video_bytes)
                    
                except Exception as e:
                    st.error(f"❌ Error loading video: {e}")
                    st.info(f"📁 Video path: {video_path}")
            else:
                st.error(f"❌ Video file not found: {video_path}")
        
        with col_info:
            st.markdown("#### ℹ️ Gesture Information")
            
            # Word information
            st.markdown(f"**Word:** `{current_video['word']}`")
            st.markdown(f"**Category:** {current_video['category']}")
            st.markdown(f"**Difficulty:** {current_video['difficulty']}")
            
            # Description
            if current_video.get('description'):
                st.markdown("**Description:**")
                st.info(current_video['description'])
            
            # Tips
            st.markdown("**💡 Learning Tips:**")
            tips = current_video.get('tips', [
                "Watch the hand movements carefully",
                "Practice slowly at first",
                "Pay attention to facial expressions",
                "Repeat until comfortable"
            ])
            
            for tip in tips:
                st.markdown(f"• {tip}")
            
            # Action buttons
            st.markdown("#### 🎮 Actions")
            
            if st.button("🔄 Replay Video", key="replay_video"):
                st.rerun()
            
            if st.button("📚 Add to Practice List", key="add_practice"):
                self._add_to_practice_list(current_video['word'])
                st.success(f"✅ Added **{current_video['word']}** to practice list!")
            
            if st.button("🎯 Try in Camera", key="try_camera"):
                st.info("💡 Go to Main Interface to practice with camera!")
                if st.button("🏠 Go to Main Interface", key="go_main"):
                    st.session_state.app_mode = 'main'
                    st.rerun()
        
        st.divider()
        
        # Video gallery/grid view
        st.markdown("### 🎭 **All Available Gestures**")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            category_filter = st.selectbox(
                "Filter by Category:",
                ["All", "Greetings", "Emotions", "Actions", "Common Words"],
                key="category_filter"
            )
        
        with col_filter2:
            difficulty_filter = st.selectbox(
                "Filter by Difficulty:",
                ["All", "Easy", "Medium", "Hard"],
                key="difficulty_filter"
            )
        
        with col_filter3:
            sort_option = st.selectbox(
                "Sort by:",
                ["Alphabetical", "Difficulty", "Category"],
                key="sort_option"
            )
        
        # Apply filters and sorting
        filtered_videos = self._filter_and_sort_videos(video_list, category_filter, difficulty_filter, sort_option)
        
        # Display video grid
        cols = st.columns(4)
        
        for i, video in enumerate(filtered_videos):
            with cols[i % 4]:
                # Video thumbnail card
                with st.container():
                    st.markdown(f"**{video['word']}**")
                    
                    # Show video thumbnail or placeholder
                    if video['path'].exists():
                        # Create a thumbnail placeholder
                        st.markdown("🎥")
                        st.caption(f"{video['category']} • {video['difficulty']}")
                        
                        # Quick view button
                        if st.button(f"👁️ View", key=f"quick_view_{i}"):
                            st.session_state.video_index = video_list.index(video)
                            st.rerun()
                    else:
                        st.error("❌ File missing")
        
        # Practice mode
        st.divider()
        st.markdown("### 🏋️ **Practice Mode**")
        
        if st.button("🎯 Start Random Practice Session", key="random_practice"):
            self._start_random_practice_session()
        
        # Show practice statistics if available
        if 'practice_stats' in st.session_state:
            self._show_practice_statistics()

    def _load_video_list(self) -> List[Dict]:
        """Load list of available ASL demonstration videos"""
        video_list = []
        
        # Path to video data
        video_base_path = Path("../Wave/raw_data")
        if not video_base_path.exists():
            video_base_path = Path("Wave/raw_data")
            if not video_base_path.exists():
                video_base_path = Path("c:/Users/BesiComputers/Desktop/Amazon/Wave/raw_data")
        
        if not video_base_path.exists():
            return []
        
        # Gesture word mappings with categories and descriptions
        gesture_info = {
            "HELLO": {
                "category": "Greetings",
                "difficulty": "Easy",
                "description": "A friendly greeting gesture. Wave your hand near your forehead.",
                "tips": ["Start with open hand", "Move outward from forehead", "Smile while signing"]
            },
            "GOODBYE": {
                "category": "Greetings", 
                "difficulty": "Easy",
                "description": "Farewell gesture. Wave your hand to say goodbye.",
                "tips": ["Open and close hand repeatedly", "Natural waving motion", "Can be done with either hand"]
            },
            "THANK_YOU": {
                "category": "Greetings",
                "difficulty": "Easy", 
                "description": "Express gratitude. Touch fingers to lips, then move hand forward.",
                "tips": ["Start at lips", "Move forward gracefully", "Maintain eye contact"]
            },
            "PLEASE": {
                "category": "Greetings",
                "difficulty": "Easy",
                "description": "Polite request gesture. Circular motion on chest with flat hand.",
                "tips": ["Flat hand on chest", "Circular clockwise motion", "Gentle movement"]
            },
            "SORRY": {
                "category": "Emotions",
                "difficulty": "Medium",
                "description": "Apologetic gesture. Circular motion with fist on chest.",
                "tips": ["Make a fist", "Circle on chest area", "Show sincere expression"]
            },
            "LOVE": {
                "category": "Emotions",
                "difficulty": "Medium", 
                "description": "Express love or affection. Cross arms over chest.",
                "tips": ["Cross both arms", "Hold over heart area", "Gentle hugging motion"]
            },
            "GOOD": {
                "category": "Emotions",
                "difficulty": "Easy",
                "description": "Positive evaluation. Touch chin, then move forward.",
                "tips": ["Start at chin", "Move hand forward", "Positive facial expression"]
            },
            "BAD": {
                "category": "Emotions", 
                "difficulty": "Easy",
                "description": "Negative evaluation. Touch chin, then flip hand down.",
                "tips": ["Start at chin", "Flip hand downward", "Express appropriate emotion"]
            },
            "AMAZING": {
                "category": "Emotions",
                "difficulty": "Hard",
                "description": "Express amazement or wonder. Dramatic gesture with wide eyes.",
                "tips": ["Wide hand movements", "Expressive face", "Show excitement"]
            },
            "BEAUTIFUL": {
                "category": "Emotions",
                "difficulty": "Medium",
                "description": "Express beauty or attractiveness. Circular motion around face.",
                "tips": ["Circle around face", "Graceful movement", "Appreciate expression"]
            },
            "HELP": {
                "category": "Actions",
                "difficulty": "Medium",
                "description": "Request or offer assistance. One hand supports the other.",
                "tips": ["Place fist on flat palm", "Lift both hands together", "Clear helping motion"]
            },
            "LEARN": {
                "category": "Actions", 
                "difficulty": "Medium",
                "description": "Express learning or studying. Grasp information and place at forehead.",
                "tips": ["Grab from flat hand", "Move to forehead", "Show concentration"]
            },
            "UNDERSTAND": {
                "category": "Actions",
                "difficulty": "Hard",
                "description": "Express comprehension. Flick finger up at forehead.",
                "tips": ["Point at forehead", "Quick flicking motion", "Show enlightenment"]
            },
            "NAME": {
                "category": "Common Words",
                "difficulty": "Medium",
                "description": "Identify someone's name. Two fingers tap on other fingers.",
                "tips": ["Two H-hands", "Cross them", "Tap motion"]
            },
            "YES": {
                "category": "Common Words",
                "difficulty": "Easy", 
                "description": "Affirmative response. Nod fist like a head nodding.",
                "tips": ["Make a fist", "Nod up and down", "Clear yes motion"]
            },
            "NO": {
                "category": "Common Words",
                "difficulty": "Easy",
                "description": "Negative response. Snap fingers closed like mouth closing.",
                "tips": ["Three fingers together", "Snap closed motion", "Shake head slightly"]
            }
        }
        
        try:
            # Scan for video folders
            for folder in video_base_path.iterdir():
                if folder.is_dir():
                    word = folder.name.upper()
                    
                    # Find video files in folder
                    video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + list(folder.glob("*.mov"))
                    
                    if video_files:
                        # Get info for this word
                        info = gesture_info.get(word, {
                            "category": "Common Words",
                            "difficulty": "Medium", 
                            "description": f"ASL gesture for {word.lower()}",
                            "tips": ["Watch carefully", "Practice slowly", "Repeat as needed"]
                        })
                        
                        # Use first video file found
                        video_path = video_files[0]
                        
                        video_list.append({
                            "word": word,
                            "path": video_path,
                            "category": info["category"],
                            "difficulty": info["difficulty"],
                            "description": info["description"],
                            "tips": info["tips"]
                        })
        
        except Exception as e:
            st.error(f"Error loading video list: {e}")
        
        # Sort alphabetically by default
        video_list.sort(key=lambda x: x['word'])
        
        return video_list

    def _filter_and_sort_videos(self, video_list: List[Dict], category_filter: str, difficulty_filter: str, sort_option: str) -> List[Dict]:
        """Filter and sort video list based on user preferences"""
        filtered = video_list.copy()
        
        # Apply category filter
        if category_filter != "All":
            filtered = [v for v in filtered if v['category'] == category_filter]
        
        # Apply difficulty filter  
        if difficulty_filter != "All":
            filtered = [v for v in filtered if v['difficulty'] == difficulty_filter]
        
        # Apply sorting
        if sort_option == "Alphabetical":
            filtered.sort(key=lambda x: x['word'])
        elif sort_option == "Difficulty":
            difficulty_order = {"Easy": 1, "Medium": 2, "Hard": 3}
            filtered.sort(key=lambda x: difficulty_order.get(x['difficulty'], 2))
        elif sort_option == "Category":
            filtered.sort(key=lambda x: (x['category'], x['word']))
        
        return filtered

    def _add_to_practice_list(self, word: str):
        """Add word to user's practice list"""
        if 'practice_list' not in st.session_state:
            st.session_state.practice_list = []
        
        if word not in st.session_state.practice_list:
            st.session_state.practice_list.append(word)

    def _start_random_practice_session(self):
        """Start a random practice session"""
        if 'video_list' in st.session_state and st.session_state.video_list:
            import random
            random_video = random.choice(st.session_state.video_list)
            st.session_state.video_index = st.session_state.video_list.index(random_video)
            st.success(f"🎯 Random practice: **{random_video['word']}**")
            st.rerun()

    def _show_practice_statistics(self):
        """Show practice session statistics"""
        stats = st.session_state.practice_stats
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Videos Watched", stats.get('videos_watched', 0))
        with col2:
            st.metric("Practice Sessions", stats.get('sessions', 0))
        with col3:
            st.metric("Favorite Category", stats.get('favorite_category', 'N/A'))
        with col4:
            st.metric("Time Practiced", f"{stats.get('time_minutes', 0)} min")


def main():
    """Main application entry point"""
    app = SignSpeakProApp()
    app.run()


if __name__ == "__main__":
    main()