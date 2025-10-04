"""
Constants and configuration values for SignSpeak UI
"""

# Camera settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# UI settings
SIDEBAR_WIDTH = 300
CONFIDENCE_THRESHOLD = 0.7
MAX_PREDICTIONS_DISPLAY = 3
FRAME_BUFFER_SIZE = 30

# Recognition modes
MODES = {
    "EXPRESSION": "Expression",
    "ALPHABET": "Alphabet",
    "SENTENCE": "Sentence"
}

# API endpoints
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": "/health",
    "expressions": "/expressions",
    "detect": {
        "expression": "/detect/expression",
        "alphabet": "/detect/alphabet",
        "sentence": "/detect/sentence"
    }
}

# Colors for visualization
COLORS = {
    "success": "#21c45d",
    "error": "#ff4b4b",
    "warning": "#faa825",
    "info": "#24c1e0"
}

# MediaPipe hand landmark indices
HAND_LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20
}
