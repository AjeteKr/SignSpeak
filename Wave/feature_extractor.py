"""
SignSpeak Feature Extractor
MediaPipe-based feature extraction for ASL recognition
"""
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoFeatures:
    """Container for extracted video features"""
    frames: List[np.ndarray]
    hand_landmarks: List[Optional[np.ndarray]]
    pose_landmarks: List[Optional[np.ndarray]]
    
    def to_array(self) -> np.ndarray:
        """Convert features to a single array format"""
        # Combine hand and pose features into a single feature vector
        # Feature dimension: 126 (hand) + 132 (pose) = 258
        combined_features = []
        
        for i in range(len(self.frames)):
            frame_features = np.zeros(258)  # 126 hand + 132 pose
            
            # Hand features (21 landmarks * 3 coords * 2 hands = 126)
            if self.hand_landmarks[i] is not None:
                hand_flat = self.hand_landmarks[i].flatten()
                frame_features[:len(hand_flat)] = hand_flat
            
            # Pose features (33 landmarks * 4 coords = 132) 
            if self.pose_landmarks[i] is not None:
                pose_flat = self.pose_landmarks[i].flatten()
                start_idx = 126
                frame_features[start_idx:start_idx + len(pose_flat)] = pose_flat
                
            combined_features.append(frame_features)
        
        # Return mean features across all frames
        if combined_features:
            return np.mean(combined_features, axis=0)
        else:
            return np.zeros(258)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "num_frames": len(self.frames),
            "has_hand_data": any(h is not None for h in self.hand_landmarks),
            "has_pose_data": any(p is not None for p in self.pose_landmarks),
            "feature_vector": self.to_array().tolist()
        }

class SignSpeakFeatureExtractor:
    """MediaPipe-based feature extractor for ASL recognition"""
    
    def __init__(self):
        """Initialize MediaPipe solutions"""
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("SignSpeak Feature Extractor initialized")
    
    def extract_frame_features(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features from a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_features = None
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = []
            for hand_landmark in hand_results.multi_hand_landmarks:
                for landmark in hand_landmark.landmark:
                    hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad or truncate to ensure consistent size (2 hands * 21 landmarks * 3 coords = 126)
            while len(hand_landmarks) < 126:
                hand_landmarks.append(0.0)
            hand_features = np.array(hand_landmarks[:126])
        
        # Extract pose landmarks
        pose_features = None
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            pose_landmarks = []
            for landmark in pose_results.pose_landmarks.landmark:
                pose_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Ensure consistent size (33 landmarks * 4 coords = 132)
            while len(pose_landmarks) < 132:
                pose_landmarks.append(0.0)
            pose_features = np.array(pose_landmarks[:132])
        
        return hand_features, pose_features
    
    def extract_from_video(self, video_path: str) -> VideoFeatures:
        """Extract features from a video file"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        hand_landmarks = []
        pose_landmarks = []
        
        frame_count = 0
        processed_frames = 0
        max_frames = 60  # Limit to 60 frames for faster processing
        frame_skip = 1  # Process every frame initially
        
        # Get total frame count to determine sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > max_frames:
            frame_skip = total_frames // max_frames  # Skip frames to get ~60 samples
        
        logger.info(f"Processing video: {total_frames} total frames, sampling every {frame_skip} frames")
        
        try:
            while cap.isOpened() and processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficient sampling
                if frame_count % frame_skip == 0:
                    # Resize frame for consistency
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
                    
                    # Extract features
                    hand_feat, pose_feat = self.extract_frame_features(frame)
                    hand_landmarks.append(hand_feat)
                    pose_landmarks.append(pose_feat)
                    
                    processed_frames += 1
                    
                    # Log progress every 10 frames
                    if processed_frames % 10 == 0:
                        logger.info(f"Processed {processed_frames}/{max_frames} frames")
                
                frame_count += 1
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        finally:
            cap.release()
        
        logger.info(f"Extracted features from {len(frames)} frames")
        return VideoFeatures(frames, hand_landmarks, pose_landmarks)
    
    def extract_from_frame_array(self, frames: List[np.ndarray]) -> VideoFeatures:
        """Extract features from an array of frames"""
        hand_landmarks = []
        pose_landmarks = []
        
        for frame in frames:
            hand_feat, pose_feat = self.extract_frame_features(frame)
            hand_landmarks.append(hand_feat)
            pose_landmarks.append(pose_feat)
        
        return VideoFeatures(frames, hand_landmarks, pose_landmarks)
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()