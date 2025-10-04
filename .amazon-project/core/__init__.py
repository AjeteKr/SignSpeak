"""
Professional Camera and ASL Processing Core Module

This module provides enterprise-grade camera and ASL processing components:
- ProfessionalCameraEngine: Production-quality camera management
- ProfessionalASLProcessor: Real-time ASL detection and processing
- Gesture classification adapters for existing models

Usage:
    from core.camera_engine import create_camera_engine
    from core.asl_processor import RealTimeASLSystem
    from core.gesture_classifier import create_gesture_classifier
"""

__version__ = "1.0.0"
__author__ = "SignSpeak Professional Team"

# Import main components for easy access
from .camera_engine import ProfessionalCameraEngine, CameraConfig, create_camera_engine
from .asl_processor import ProfessionalASLProcessor, ASLProcessorConfig, RealTimeASLSystem
from .gesture_classifier import create_gesture_classifier, GestureClassifierInterface

__all__ = [
    "ProfessionalCameraEngine",
    "CameraConfig", 
    "create_camera_engine",
    "ProfessionalASLProcessor",
    "ASLProcessorConfig",
    "RealTimeASLSystem",
    "create_gesture_classifier",
    "GestureClassifierInterface"
]