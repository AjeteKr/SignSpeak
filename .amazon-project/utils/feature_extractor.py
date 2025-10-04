"""
Feature extraction utilities for SignSpeak
"""
import numpy as np
import mediapipe as mp
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def extract_features(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Extract hand landmarks and compute features
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple containing:
            - Array of computed features
            - List of landmarks if detected, None otherwise
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Get hand landmarks
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if not results.multi_hand_landmarks:
            return np.zeros(63), None  # Return zero features if no hand detected
            
        # Get landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0]
        
        # Extract features
        features = []
        
        # 1. Joint angles
        features.extend(self._compute_joint_angles(landmarks))
        
        # 2. Distance ratios
        features.extend(self._compute_distance_ratios(landmarks))
        
        # 3. Hand orientation
        features.extend(self._compute_hand_orientation(landmarks))
        
        # 4. Finger states
        features.extend(self._compute_finger_states(landmarks))
        
        return np.array(features), landmarks.landmark
        
    def _compute_joint_angles(self, landmarks) -> List[float]:
        """Compute angles between finger joints"""
        angles = []
        
        # Finger joints indices
        finger_joints = [
            [1, 2, 3, 4],          # Thumb
            [5, 6, 7, 8],          # Index
            [9, 10, 11, 12],       # Middle
            [13, 14, 15, 16],      # Ring
            [17, 18, 19, 20]       # Pinky
        ]
        
        for finger in finger_joints:
            for i in range(len(finger)-2):
                p1 = landmarks.landmark[finger[i]]
                p2 = landmarks.landmark[finger[i+1]]
                p3 = landmarks.landmark[finger[i+2]]
                angle = self._calculate_angle(
                    [p1.x, p1.y],
                    [p2.x, p2.y],
                    [p3.x, p3.y]
                )
                angles.append(angle)
                
        return angles
        
    def _compute_distance_ratios(self, landmarks) -> List[float]:
        """Compute distance ratios between key points"""
        ratios = []
        
        # Get wrist point as reference
        wrist = landmarks.landmark[0]
        
        # Fingertip indices
        fingertips = [4, 8, 12, 16, 20]
        
        # Compute ratios between fingertips
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                tip1 = landmarks.landmark[fingertips[i]]
                tip2 = landmarks.landmark[fingertips[j]]
                
                # Distance between fingertips
                d_tips = self._euclidean_dist(
                    [tip1.x, tip1.y],
                    [tip2.x, tip2.y]
                )
                
                # Distance from wrist to each tip
                d1 = self._euclidean_dist(
                    [wrist.x, wrist.y],
                    [tip1.x, tip1.y]
                )
                d2 = self._euclidean_dist(
                    [wrist.x, wrist.y],
                    [tip2.x, tip2.y]
                )
                
                # Compute ratio
                ratio = d_tips / (d1 + d2 + 1e-6)  # Add small epsilon to avoid division by zero
                ratios.append(ratio)
                
        return ratios
        
    def _compute_hand_orientation(self, landmarks) -> List[float]:
        """Compute hand orientation features"""
        # Get key points
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]  # Middle finger MCP joint
        middle_tip = landmarks.landmark[12]  # Middle finger tip
        
        # Compute vectors
        v1 = [middle_mcp.x - wrist.x, middle_mcp.y - wrist.y]
        v2 = [middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y]
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0:
            v1 = [x/v1_norm for x in v1]
        if v2_norm > 0:
            v2 = [x/v2_norm for x in v2]
            
        return v1 + v2
        
    def _compute_finger_states(self, landmarks) -> List[float]:
        """Compute finger extension states"""
        states = []
        
        # Finger joint sets (MCP, PIP, DIP, TIP)
        fingers = [
            [2, 3, 4],      # Thumb
            [5, 6, 8],      # Index
            [9, 10, 12],    # Middle
            [13, 14, 16],   # Ring
            [17, 18, 20]    # Pinky
        ]
        
        for finger in fingers:
            # Get points
            mcp = landmarks.landmark[finger[0]]
            pip = landmarks.landmark[finger[1]]
            tip = landmarks.landmark[finger[2]]
            
            # Check if finger is extended
            extended = pip.y > mcp.y and tip.y < pip.y
            states.append(float(extended))
            
        return states
        
    @staticmethod
    def _calculate_angle(p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle between three points"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return angle
        
    @staticmethod
    def _euclidean_dist(p1: List[float], p2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
