"""
MediaPipe configuration and initialization
"""
import mediapipe as mp
import logging

# Configure logging to suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# MediaPipe configuration
mp_config = {
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False,
}

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands.Hands(**mp_config)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom drawing specs
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=2,
    circle_radius=2
)

hand_connection_drawing_spec = mp_drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=1
)
