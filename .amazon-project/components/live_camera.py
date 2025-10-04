"""
Live Camera ASL Recognition Interface
Beautiful real-time ASL recognition with letter accumulation
"""
import streamlit as st
import cv2
import numpy as np
import time
from components.asl_recognition import get_asl_recognizer
from components.modern_ui import SignSpeakUI
from word_dictionary import get_word_suggestions
import threading

class LiveCameraASL:
    """Live camera ASL recognition interface"""
    
    def __init__(self):
        self.recognizer = get_asl_recognizer()
        self.camera = None
        self.is_running = False
        
    def start_camera(self) -> bool:
        """Start camera capture with optimized settings"""
        try:
            # Fast camera initialization
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster startup on Windows
            if not self.camera.isOpened():
                # Fallback to default backend
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    st.error("❌ Could not open camera. Please check camera permissions.")
                    return False
            
            # Optimized camera properties for fast startup
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for low latency
            
            # Warm up camera with a quick frame read
            for _ in range(3):
                ret, _ = self.camera.read()
                if ret:
                    break
            
            self.is_running = True
            return True
            
        except Exception as e:
            st.error(f"❌ Camera initialization failed: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def capture_frame(self):
        """Capture and process single frame with error handling"""
        if not self.camera or not self.camera.isOpened() or not self.is_running:
            return None, None
        
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return None, None
        
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
        
            # Process frame through ASL recognizer (with error handling)
            try:
                processed_frame, detection_info = self.recognizer.process_frame(frame)
                return processed_frame, detection_info
            except Exception as e:
                # If ASL processing fails, return the raw frame
                return frame, {"error": str(e)}
                
        except Exception as e:
            return None, {"capture_error": str(e)}

def show_live_asl_recognition():
    """Show live ASL recognition interface"""
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    
    # Header
    SignSpeakUI.hero_section(
        title="Live ASL Recognition", 
        subtitle="Real-time sign language recognition with letter accumulation",
        icon="📹"
    )
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'live_camera' not in st.session_state:
        st.session_state.live_camera = LiveCameraASL()
    if 'frame_placeholder' not in st.session_state:
        st.session_state.frame_placeholder = None
    if 'last_frame_time' not in st.session_state:
        st.session_state.last_frame_time = 0
    if 'detection_info' not in st.session_state:
        st.session_state.detection_info = {}
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📹 Start Camera", type="primary", use_container_width=True):
            if st.session_state.live_camera.start_camera():
                st.session_state.camera_active = True
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.live_camera.stop_camera()
            st.session_state.camera_active = False
            # Clean up frame placeholder when stopping
            if st.session_state.frame_placeholder:
                st.session_state.frame_placeholder.empty()
                st.session_state.frame_placeholder = None
            st.success("✅ Camera stopped")
            st.rerun()
    
    with col3:
        if st.button("🔄 Add Space", use_container_width=True):
            st.session_state.live_camera.recognizer._add_space()
            st.success("✅ Space added")
    
    with col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.live_camera.recognizer.clear_all()
            st.success("✅ Text cleared")
    
    # Camera status
    if st.session_state.camera_active:
        
        # Video stream container
        video_container = st.container()
        with video_container:
            st.markdown("### 📹 Live Video Feed")
            
            # Create persistent frame placeholder
            if 'frame_placeholder' not in st.session_state:
                st.session_state.frame_placeholder = st.empty()
            
            # Process video frames more efficiently
            if st.session_state.camera_active:
                try:
                    frame, detection_info = st.session_state.live_camera.capture_frame()
                    if frame is not None:
                        # Convert BGR to RGB for Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Create frame placeholder if it doesn't exist
                        if st.session_state.frame_placeholder is None:
                            st.session_state.frame_placeholder = st.empty()
                        
                        # Use a more stable image display approach to prevent MediaFileStorageError
                        
                        # Create a unique key for this frame to prevent conflicts
                        frame_key = f"frame_{int(current_time * 1000) % 10000}"
                        
                        # Clear the placeholder less frequently to reduce memory pressure
                        if not hasattr(st.session_state, 'last_clear_time'):
                            st.session_state.last_clear_time = current_time
                        
                        # Clear and recreate placeholder every 10 seconds (increased from 5)
                        if current_time - st.session_state.last_clear_time > 10.0:
                            try:
                                if st.session_state.frame_placeholder:
                                    st.session_state.frame_placeholder.empty()
                                st.session_state.frame_placeholder = st.empty()
                                st.session_state.last_clear_time = current_time
                            except:
                                pass  # Ignore cleanup errors
                        
                        # Display frame with error handling
                        try:
                            # Reduce image quality slightly to reduce memory usage
                            height, width = frame_rgb.shape[:2]
                            if width > 640:  # Resize if too large
                                scale = 640.0 / width
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                            
                            # Display with simplified caption to reduce overhead
                            detection_caption = "🎯 Live ASL Recognition"
                            if detection_info.get('current_letter'):
                                detection_caption = f"🎯 Detecting: {detection_info['current_letter']}"
                            
                            if st.session_state.frame_placeholder:
                                st.session_state.frame_placeholder.image(
                                    frame_rgb, 
                                    channels="RGB", 
                                    use_container_width=True,
                                    caption=detection_caption
                                )
                        except Exception as img_error:
                            # If image display fails, recreate the placeholder
                            try:
                                st.session_state.frame_placeholder = st.empty()
                                st.session_state.frame_placeholder.warning("📷 Refreshing video feed...")
                            except:
                                pass
                        
                        st.session_state.detection_info = detection_info or {}
                    else:
                        if st.session_state.frame_placeholder:
                            st.session_state.frame_placeholder.warning("📷 Waiting for camera frame...")
                        
                except Exception as e:
                    if st.session_state.frame_placeholder:
                        st.session_state.frame_placeholder.error(f"❌ Camera error: {str(e)}")
                    else:
                        st.error(f"❌ Camera error: {str(e)}")
        
        # Auto-refresh with conservative frame rate to prevent MediaFileStorageError
        if st.session_state.camera_active:
            current_time = time.time()
            # Conservative 8 FPS (0.125 second intervals) to prevent memory issues
            if current_time - st.session_state.last_frame_time > 0.125:
                st.session_state.last_frame_time = current_time
                st.rerun()
            else:
                # Longer sleep to reduce memory pressure
                time.sleep(0.05)
                st.rerun()
    else:
        SignSpeakUI.status_card(
            title="Camera Status",
            status="warning",
            message="Camera is not active. Click 'Start Camera' to begin recognition",
            icon="⚠️"
        )
    
    # Always show recognition interface when camera is active
    if st.session_state.camera_active:
        st.markdown("---")
        # Show recognition results (always visible when camera is active)
        if st.session_state.detection_info:
            show_recognition_results(st.session_state.detection_info)
        else:
            # Show default recognition interface even without detection info
            show_recognition_results({
                'current_letter': None,
                'current_word': st.session_state.live_camera.recognizer.current_word if hasattr(st.session_state.live_camera.recognizer, 'current_word') else '',
                'accumulated_text': st.session_state.live_camera.recognizer.accumulated_text if hasattr(st.session_state.live_camera.recognizer, 'accumulated_text') else '',
                'suggestions': []
            })

def show_recognition_results(detection_info: dict):
    """Show recognition results and text accumulation"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Real-Time ASL Recognition")
        
        # Current detection (always show status)
        if detection_info.get('current_letter'):
            SignSpeakUI.status_card(
                title="Detected Sign",
                status="success",
                message=f"Letter: {detection_info['current_letter']} ({detection_info.get('confidence', 0)*100:.1f}% confidence)",
                icon="✋"
            )
            
            # Progress bar for confirmation
            if detection_info.get('progress', 0) > 0:
                SignSpeakUI.progress_bar(
                    progress=detection_info.get('progress', 0) * 100,
                    label="Confirmation Progress",
                    max_value=100.0
                )
        else:
            SignSpeakUI.status_card(
                title="Recognition Status",
                status="info",
                message="Show your hand to start ASL recognition",
                icon="👋"
            )
        
        # Current word being built (always show)
        current_word = detection_info.get('current_word', '')
        if current_word:
            st.markdown(f"**Building Word:** `{current_word}`")
        else:
            st.markdown("**Building Word:** _Ready to detect letters..._")
        
        # Accumulated text and voice controls (always show)
        accumulated_text = detection_info.get('accumulated_text', '')
        current_word = detection_info.get('current_word', '')
        
        st.markdown("**Text Output:**")
        display_text = accumulated_text + (" " + current_word if current_word else "")
        if not display_text.strip():
            display_text = "Start signing to see your text here..."
        
        st.text_area("Your ASL Text", value=display_text, height=100, disabled=True)
        
        # Voice controls (always visible)
        st.markdown("### 🔊 Voice Output Controls")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔊 Speak Current Word", use_container_width=True, disabled=not current_word):
                if current_word:
                    st.session_state.live_camera.recognizer.read_text(current_word)
                    st.success(f"🔊 Speaking: {current_word}")
        with col_b:
            if st.button("🔊 Speak All Text", use_container_width=True, disabled=not accumulated_text):
                if accumulated_text:
                    st.session_state.live_camera.recognizer.read_text(accumulated_text)
                    st.success(f"🔊 Speaking all text")
        with col_c:
            if st.button("✅ Complete Word", use_container_width=True, disabled=not current_word):
                if current_word:
                    st.session_state.live_camera.recognizer.finish_word(current_word)
                    st.success(f"✅ Added word: {current_word}")
                    st.rerun()
    
    with col2:
        st.markdown("### 💡 Word Suggestions")
        
        suggestions = detection_info.get('suggestions', [])
        current_word = detection_info.get('current_word', '')
        
        if suggestions and current_word and len(current_word) >= 2:
            st.markdown("💡 **Word Suggestions:**")
            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
                if st.button(f"📝 {suggestion}", key=f"suggest_{suggestion}_{i}", use_container_width=True):
                    st.session_state.live_camera.recognizer.finish_word(suggestion)
                    st.success(f"✅ Added word: {suggestion}")
                    st.rerun()
        else:
            st.markdown("💡 **Word Suggestions:**")
            if current_word:
                st.info(f"Type more letters to see suggestions for '{current_word}'")
            else:
                st.info("Start signing letters to see word suggestions")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("👋 Add 'HELLO'", use_container_width=True):
            st.session_state.live_camera.recognizer.finish_word("HELLO")
            st.success("✅ Added: HELLO")
        
        if st.button("🙏 Add 'THANK YOU'", use_container_width=True):
            st.session_state.live_camera.recognizer.finish_word("THANK")
            st.session_state.live_camera.recognizer.finish_word("YOU")
            st.success("✅ Added: THANK YOU")

def show_asl_instructions():
    """Show ASL recognition instructions"""
    st.markdown("### 📚 How to Use ASL Recognition")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        SignSpeakUI.feature_card(
            title="Letter Signs",
            description="Supported ASL letters and their hand positions",
            icon="✋",
            action_text="Learn More"
        )
        
        st.markdown("""
        **Supported Letters:**
        - **A** - Closed fist
        - **D** - Index finger pointing up
        - **O** - Thumb and index touching (OK sign)
        - **V** - Peace sign (two fingers up)
        - **Numbers** - 1, 3, 4, 5 fingers extended
        """)
    
    with col2:
        SignSpeakUI.feature_card(
            title="Special Gestures",
            description="Additional gestures for enhanced communication",
            icon="👍",
            action_text="Try Now"
        )
        
        st.markdown("""
        **Special Signs:**
        - **👍** - Thumbs up
        - **❤️** - I Love You sign
        - **SPACE** - Flat hand (all fingers extended)
        - **Pause** - Remove hand for 1.5 seconds
        """)
    
    with col3:
        SignSpeakUI.feature_card(
            title="Tips for Success",
            description="Best practices for accurate recognition",
            icon="💡",
            action_text="Follow Tips"
        )
        
        st.markdown("""
        **Recognition Tips:**
        - Hold each sign for 1+ seconds
        - Use good lighting
        - Keep hand clearly visible
        - Make clear, distinct shapes
        - Practice consistent signing
        """)

# Main interface function
def show_live_camera_interface():
    """Main live camera interface"""
    # Apply CSS
    SignSpeakUI.apply_signspeak_css()
    SignSpeakUI.accessibility_features()
    
    # Tab interface
    tab1, tab2 = st.tabs(["📹 Live Recognition", "📚 Instructions"])
    
    with tab1:
        show_live_asl_recognition()
    
    with tab2:
        show_asl_instructions()

if __name__ == "__main__":
    show_live_camera_interface()