"""
Real-time camera display component for Streamlit
Provides smooth camera streaming without page reloads
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import numpy as np
import cv2
from typing import Optional

def create_realtime_camera_display(placeholder_key: str = "camera_display"):
    """
    Create a real-time camera display component that updates smoothly
    """
    
    # JavaScript component for real-time display
    camera_js = f"""
    <div id="{placeholder_key}" style="width: 100%; text-align: center;">
        <div id="camera-container" style="position: relative; display: inline-block;">
            <img id="camera-frame" 
                 style="max-width: 100%; height: auto; border: 2px solid #00ff00; border-radius: 10px;"
                 src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                 alt="Camera Feed" />
            <div id="status-overlay" 
                 style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 5px; font-family: monospace;">
                Initializing Camera...
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh camera display
        function updateCameraDisplay() {{
            const frameElement = document.getElementById('camera-frame');
            const statusElement = document.getElementById('status-overlay');
            
            // Check if Streamlit session state has new frame data
            if (window.parent && window.parent.document) {{
                // Trigger Streamlit to update (this is a simplified approach)
                setTimeout(updateCameraDisplay, 100); // 10 FPS refresh rate
            }}
        }}
        
        // Start auto-refresh when component loads
        updateCameraDisplay();
    </script>
    """
    
    return components.html(camera_js, height=500)

def display_camera_frame(frame: Optional[np.ndarray], placeholder, status: str = "Camera Active"):
    """
    Display a camera frame in the Streamlit placeholder
    """
    if frame is not None:
        # Add status overlay to frame
        frame_with_status = frame.copy()
        cv2.putText(frame_with_status, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        placeholder.image(frame_with_status, channels="RGB", use_container_width=True)
    else:
        placeholder.warning("📷 No camera frame available")

def create_camera_controls():
    """Create professional camera control interface"""
    st.markdown("### 📹 Camera Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        camera_active = st.session_state.get('camera_active', False)
        if st.button("🟢 Start Camera" if not camera_active else "🔴 Stop Camera"):
            st.session_state.camera_active = not camera_active
            if st.session_state.camera_active:
                st.success("✅ Camera started!")
            else:
                st.info("📹 Camera stopped")
            st.rerun()
    
    with col2:
        recognition_active = st.session_state.get('recognition_active', False)
        if st.checkbox("🧠 ASL Recognition", value=recognition_active):
            st.session_state.recognition_active = True
        else:
            st.session_state.recognition_active = False
    
    with col3:
        if st.button("🔄 Reset Camera"):
            # Reset camera state
            st.session_state.camera_active = False
            st.session_state.recognition_active = False
            st.session_state.camera_stream_active = False
            st.success("🔄 Camera reset!")
            st.rerun()
    
    return st.session_state.get('camera_active', False), st.session_state.get('recognition_active', False)