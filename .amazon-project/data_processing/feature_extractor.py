"""
Normalized Feature Extraction for ASL Recognition
Implements wrist-centered, scale-invariant hand feature extraction
"""

# MUST BE FIRST: Suppress all warnings before any other imports
import suppress_warnings  # This MUST be the very first import

import os
# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import cv2
# Import MediaPipe with warning suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mediapipe as mp
    
from typing import Tuple, Dict, List, Optional, Union
import math
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class HandBounds:
    """Hand bounding box information"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float
    center_x: float
    center_y: float

@dataclass
class NormalizedFeatures:
    """Normalized hand features for scale-invariant recognition"""
    landmarks: np.ndarray  # 21x3 normalized coordinates
    distances: np.ndarray  # Distance features
    angles: np.ndarray     # Angle features
    finger_states: np.ndarray  # Finger extension states
    hand_shape: Dict       # Hand shape descriptors
    confidence: float      # Detection confidence

class NormalizedFeatureExtractor:
    """
    Extract normalized, scale-invariant features from hand landmarks
    All coordinates normalized to 0-1 scale with wrist as reference point
    """
    
    def __init__(self):
        """Initialize feature extractor with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe landmark indices
        self.LANDMARK_INDICES = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # Key landmarks for distance calculations
        self.KEY_DISTANCES = [
            (0, 4),   # Wrist to thumb tip
            (0, 8),   # Wrist to index tip
            (0, 12),  # Wrist to middle tip
            (0, 16),  # Wrist to ring tip
            (0, 20),  # Wrist to pinky tip
            (4, 8),   # Thumb to index
            (8, 12),  # Index to middle
            (12, 16), # Middle to ring
            (16, 20), # Ring to pinky
            (4, 20),  # Thumb to pinky span
        ]
        
        # Angle triplets for angle calculations
        self.KEY_ANGLES = [
            (2, 3, 4),   # Thumb angle
            (6, 7, 8),   # Index angle
            (10, 11, 12), # Middle angle
            (14, 15, 16), # Ring angle
            (18, 19, 20), # Pinky angle
            (0, 9, 13),   # Hand span angle
        ]
    
    def extract_hand_bounds(self, landmarks: np.ndarray, padding: float = 1.2) -> HandBounds:
        """
        Extract hand bounding box with padding
        
        Args:
            landmarks: Hand landmarks (21x3)
            padding: Padding factor around hand
            
        Returns:
            HandBounds object with bounding box information
        """
        # Get 2D coordinates
        coords_2d = landmarks[:, :2]
        
        # Calculate bounding box
        x_min, y_min = coords_2d.min(axis=0)
        x_max, y_max = coords_2d.max(axis=0)
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        
        pad_x = width * (padding - 1) / 2
        pad_y = height * (padding - 1) / 2
        
        # Ensure bounds stay within [0, 1]
        x_min_padded = max(0, x_min - pad_x)
        y_min_padded = max(0, y_min - pad_y)
        x_max_padded = min(1, x_max + pad_x)
        y_max_padded = min(1, y_max + pad_y)
        
        return HandBounds(
            x_min=x_min_padded,
            y_min=y_min_padded,
            x_max=x_max_padded,
            y_max=y_max_padded,
            width=x_max_padded - x_min_padded,
            height=y_max_padded - y_min_padded,
            center_x=(x_min_padded + x_max_padded) / 2,
            center_y=(y_min_padded + y_max_padded) / 2
        )
    
    def normalize_landmarks_to_wrist(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks with wrist as origin (0,0)
        Scale by hand size for scale invariance
        
        Args:
            landmarks: Raw landmarks (21x3)
            
        Returns:
            Normalized landmarks with wrist at origin
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0].copy()
        
        # Translate to make wrist the origin
        normalized = landmarks.copy()
        normalized[:, :2] -= wrist[:2]
        
        # Calculate hand scale (distance from wrist to middle finger tip)
        middle_tip = landmarks[12]
        hand_scale = np.linalg.norm(middle_tip[:2] - wrist[:2])
        
        # Avoid division by zero
        if hand_scale < 1e-6:
            hand_scale = 1.0
        
        # Scale by hand size for scale invariance
        normalized[:, :2] /= hand_scale
        
        # Z-coordinate normalization (relative depth)
        z_range = landmarks[:, 2].max() - landmarks[:, 2].min()
        if z_range > 1e-6:
            normalized[:, 2] = (landmarks[:, 2] - landmarks[:, 2].min()) / z_range
        
        return normalized
    
    def calculate_distances(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate key distances between landmarks
        
        Args:
            landmarks: Normalized landmarks (21x3)
            
        Returns:
            Distance features array
        """
        distances = []
        
        for i, j in self.KEY_DISTANCES:
            if i < len(landmarks) and j < len(landmarks):
                dist = np.linalg.norm(landmarks[i][:2] - landmarks[j][:2])
                distances.append(dist)
        
        return np.array(distances)
    
    def calculate_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate key angles between landmark triplets
        
        Args:
            landmarks: Normalized landmarks (21x3)
            
        Returns:
            Angle features array (in radians)
        """
        angles = []
        
        for i, j, k in self.KEY_ANGLES:
            if i < len(landmarks) and j < len(landmarks) and k < len(landmarks):
                # Vectors from j to i and j to k
                v1 = landmarks[i][:2] - landmarks[j][:2]
                v2 = landmarks[k][:2] - landmarks[j][:2]
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        return np.array(angles)
    
    def calculate_finger_states(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate finger extension states (extended/bent)
        
        Args:
            landmarks: Normalized landmarks (21x3)
            
        Returns:
            Binary array indicating finger states
        """
        finger_states = []
        
        # For each finger, check if tip is above/below key joint
        finger_checks = [
            (4, 3, 2),   # Thumb: tip vs IP vs MCP
            (8, 6, 5),   # Index: tip vs PIP vs MCP
            (12, 10, 9), # Middle: tip vs PIP vs MCP
            (16, 14, 13),# Ring: tip vs PIP vs MCP
            (20, 18, 17) # Pinky: tip vs PIP vs MCP
        ]
        
        for tip_idx, pip_idx, mcp_idx in finger_checks:
            if tip_idx < len(landmarks) and pip_idx < len(landmarks):
                # For most fingers, extended means tip is higher (lower y) than PIP
                if tip_idx == 4:  # Thumb special case
                    extended = landmarks[tip_idx][0] > landmarks[pip_idx][0]  # Thumb extends horizontally
                else:
                    extended = landmarks[tip_idx][1] < landmarks[pip_idx][1]  # Others extend vertically
                
                finger_states.append(1.0 if extended else 0.0)
        
        return np.array(finger_states)
    
    def calculate_hand_shape_descriptors(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate high-level hand shape descriptors
        
        Args:
            landmarks: Normalized landmarks (21x3)
            
        Returns:
            Dictionary of shape descriptors
        """
        descriptors = {}
        
        # Hand spread (distance between extreme points)
        thumb_tip = landmarks[4][:2]
        pinky_tip = landmarks[20][:2]
        hand_spread = np.linalg.norm(thumb_tip - pinky_tip)
        descriptors['hand_spread'] = hand_spread
        
        # Hand compactness (how clustered the fingertips are)
        fingertips = landmarks[[4, 8, 12, 16, 20]][:, :2]
        centroid = fingertips.mean(axis=0)
        compactness = np.mean([np.linalg.norm(tip - centroid) for tip in fingertips])
        descriptors['compactness'] = compactness
        
        # Hand orientation (angle of middle finger relative to wrist)
        wrist = landmarks[0][:2]
        middle_tip = landmarks[12][:2]
        orientation = math.atan2(middle_tip[1] - wrist[1], middle_tip[0] - wrist[0])
        descriptors['orientation'] = orientation
        
        # Finger curl (average curl of all fingers)
        finger_curls = []
        for i in range(5):  # 5 fingers
            base_idx = 1 + i * 4
            if base_idx + 3 < len(landmarks):
                # Distance from tip to base relative to extended length
                tip = landmarks[base_idx + 3][:2]
                base = landmarks[base_idx][:2]
                actual_length = np.linalg.norm(tip - base)
                
                # Estimate extended length (sum of bone segments)
                extended_length = sum([
                    np.linalg.norm(landmarks[base_idx + j + 1][:2] - landmarks[base_idx + j][:2])
                    for j in range(3)
                ])
                
                if extended_length > 1e-6:
                    curl = 1.0 - (actual_length / extended_length)
                    finger_curls.append(curl)
        
        descriptors['average_curl'] = np.mean(finger_curls) if finger_curls else 0.0
        
        # Hand openness (area of convex hull of fingertips)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(fingertips)
            descriptors['openness'] = hull.volume  # 2D area
        except:
            descriptors['openness'] = hand_spread  # Fallback
        
        return descriptors
    
    def extract_features_from_image(self, image: np.ndarray) -> Optional[NormalizedFeatures]:
        """
        Extract normalized features from image
        
        Args:
            image: Input RGB image
            
        Returns:
            NormalizedFeatures object or None if no hand detected
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        else:
            rgb_image = image
        
        # Process with MediaPipe
        results = self.hands.process((rgb_image * 255).astype(np.uint8) if rgb_image.dtype != np.uint8 else rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract raw landmarks
        raw_landmarks = []
        for landmark in hand_landmarks.landmark:
            raw_landmarks.append([landmark.x, landmark.y, landmark.z])
        raw_landmarks = np.array(raw_landmarks)
        
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks_to_wrist(raw_landmarks)
        
        # Calculate features
        distances = self.calculate_distances(normalized_landmarks)
        angles = self.calculate_angles(normalized_landmarks)
        finger_states = self.calculate_finger_states(normalized_landmarks)
        hand_shape = self.calculate_hand_shape_descriptors(normalized_landmarks)
        
        # Estimate confidence (simplified)
        confidence = 1.0  # Could be enhanced with actual confidence metrics
        
        return NormalizedFeatures(
            landmarks=normalized_landmarks,
            distances=distances,
            angles=angles,
            finger_states=finger_states,
            hand_shape=hand_shape,
            confidence=confidence
        )
    
    def extract_features_from_landmarks(self, landmarks: np.ndarray) -> NormalizedFeatures:
        """
        Extract features from pre-detected landmarks
        
        Args:
            landmarks: Raw landmarks (21x3)
            
        Returns:
            NormalizedFeatures object
        """
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks_to_wrist(landmarks)
        
        # Calculate features
        distances = self.calculate_distances(normalized_landmarks)
        angles = self.calculate_angles(normalized_landmarks)
        finger_states = self.calculate_finger_states(normalized_landmarks)
        hand_shape = self.calculate_hand_shape_descriptors(normalized_landmarks)
        
        return NormalizedFeatures(
            landmarks=normalized_landmarks,
            distances=distances,
            angles=angles,
            finger_states=finger_states,
            hand_shape=hand_shape,
            confidence=1.0
        )
    
    def create_feature_vector(self, features: NormalizedFeatures) -> np.ndarray:
        """
        Create a single feature vector from all extracted features
        
        Args:
            features: NormalizedFeatures object
            
        Returns:
            Concatenated feature vector
        """
        # Properly extract hand shape features
        hand_shape_features = []
        if isinstance(features.hand_shape, dict):
            # Extract palm center coordinates
            palm_center = features.hand_shape.get('palm_center', [0.5, 0.5])
            if isinstance(palm_center, (list, tuple, np.ndarray)):
                hand_shape_features.extend(palm_center[:2])  # Take first 2 elements
            else:
                hand_shape_features.extend([0.5, 0.5])
            
            # Extract hand size
            hand_size = features.hand_shape.get('hand_size', 100.0)
            if isinstance(hand_size, (int, float)):
                hand_shape_features.append(hand_size)
            else:
                hand_shape_features.append(100.0)
        else:
            # Fallback values
            hand_shape_features = [0.5, 0.5, 100.0]
        
        feature_components = [
            features.landmarks.flatten(),  # 21*3 = 63 features
            features.distances,            # Distance features
            features.angles,               # Angle features
            features.finger_states,        # Finger state features
            np.array(hand_shape_features)  # Shape descriptors (3 features)
        ]
        
        return np.concatenate(feature_components)
    
    def batch_extract_features(self, images: List[np.ndarray]) -> List[Optional[NormalizedFeatures]]:
        """
        Extract features from batch of images
        
        Args:
            images: List of RGB images
            
        Returns:
            List of NormalizedFeatures (None for images without hands)
        """
        features_list = []
        
        for image in images:
            features = self.extract_features_from_image(image)
            features_list.append(features)
        
        return features_list
    
    def save_feature_config(self, filepath: str):
        """Save feature extraction configuration"""
        config = {
            'landmark_indices': self.LANDMARK_INDICES,
            'key_distances': self.KEY_DISTANCES,
            'key_angles': self.KEY_ANGLES,
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


# Utility functions for testing
def test_feature_extraction():
    """Test the feature extraction system"""
    import matplotlib.pyplot as plt
    
    # Create test image (you would load a real image here)
    test_image = np.random.rand(224, 224, 3)
    
    # Initialize extractor
    extractor = NormalizedFeatureExtractor()
    
    # Extract features
    features = extractor.extract_features_from_image(test_image)
    
    if features:
        print("Features extracted successfully!")
        print(f"Landmarks shape: {features.landmarks.shape}")
        print(f"Distances: {len(features.distances)}")
        print(f"Angles: {len(features.angles)}")
        print(f"Finger states: {features.finger_states}")
        print(f"Hand shape: {features.hand_shape}")
        
        # Create feature vector
        feature_vector = extractor.create_feature_vector(features)
        print(f"Feature vector length: {len(feature_vector)}")
    else:
        print("No hand detected in test image")


if __name__ == "__main__":
    test_feature_extraction()