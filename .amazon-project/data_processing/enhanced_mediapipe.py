"""
Enhanced MediaPipe Processing Pipeline
Implements 5-step preprocessing for robust ASL recognition:
1. Hand detection with MediaPipe
2. Cropping with 1.2x padding  
3. Normalization to 224x224 pixels
4. Data augmentation (rotation ±15°, brightness ±20%)
5. Quality filtering to remove unclear images
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
import warnings
from contextlib import redirect_stderr
import io

# Suppress MediaPipe warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

class EnhancedMediaPipeProcessor:
    """
    Enhanced MediaPipe processor with 5-step preprocessing pipeline
    """
    
    def __init__(self):
        """Initialize MediaPipe hands with optimized settings"""
        # Suppress initialization warnings
        with redirect_stderr(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                
                # Initialize hands detector with optimal settings
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,  # Higher confidence for better quality
                    min_tracking_confidence=0.5,
                    model_complexity=1
                )
    
    def process_frame_pipeline(self, frame: np.ndarray) -> Dict:
        """
        Complete 5-step processing pipeline
        Returns: {
            'success': bool,
            'processed_frame': np.ndarray,
            'landmarks': list,
            'hand_crop': np.ndarray,
            'quality_score': float,
            'augmented_crops': list
        }
        """
        result = {
            'success': False,
            'processed_frame': frame.copy(),
            'landmarks': None,
            'hand_crop': None,
            'quality_score': 0.0,
            'augmented_crops': []
        }
        
        try:
            # Step 1: Hand Detection with MediaPipe
            detection_result = self._detect_hands(frame)
            if not detection_result['success']:
                return result
            
            # Step 2: Cropping with 1.2x padding
            crop_result = self._crop_hand_region(frame, detection_result['landmarks'])
            if crop_result is None:
                return result
            
            # Step 3: Normalization to 224x224 pixels
            normalized_crop = self._normalize_image(crop_result['crop'])
            
            # Step 4: Data augmentation (rotation ±15°, brightness ±20%)
            augmented_crops = self._apply_augmentation(normalized_crop)
            
            # Step 5: Quality filtering
            quality_score = self._assess_image_quality(normalized_crop, detection_result['landmarks'])
            
            # Update result
            result.update({
                'success': True,
                'processed_frame': detection_result['annotated_frame'],
                'landmarks': detection_result['landmarks'],
                'hand_crop': normalized_crop,
                'quality_score': quality_score,
                'augmented_crops': augmented_crops
            })
            
        except Exception as e:
            print(f"Pipeline error: {e}")
        
        return result
    
    def _detect_hands(self, frame: np.ndarray) -> Dict:
        """
        Step 1: Hand detection with MediaPipe
        """
        result = {'success': False, 'landmarks': None, 'annotated_frame': frame.copy()}
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe (suppress warnings)
            with redirect_stderr(io.StringIO()):
                results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks on frame
                annotated_frame = rgb_frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract landmark coordinates
                landmarks = []
                h, w = frame.shape[:2]
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': int(lm.x * w),
                        'y': int(lm.y * h),
                        'z': lm.z,
                        'visibility': getattr(lm, 'visibility', 1.0)
                    })
                
                result.update({
                    'success': True,
                    'landmarks': landmarks,
                    'annotated_frame': annotated_frame
                })
                
        except Exception as e:
            print(f"Hand detection error: {e}")
        
        return result
    
    def _crop_hand_region(self, frame: np.ndarray, landmarks: List[Dict]) -> Optional[Dict]:
        """
        Step 2: Cropping with 1.2x padding
        """
        try:
            if not landmarks:
                return None
            
            # Get bounding box of hand landmarks
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Calculate center and size
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            width = max_x - min_x
            height = max_y - min_y
            
            # Apply 1.2x padding
            padded_size = int(max(width, height) * 1.2)
            half_size = padded_size // 2
            
            # Calculate crop boundaries
            crop_x1 = max(0, center_x - half_size)
            crop_y1 = max(0, center_y - half_size)
            crop_x2 = min(frame.shape[1], center_x + half_size)
            crop_y2 = min(frame.shape[0], center_y + half_size)
            
            # Extract crop
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            return {
                'crop': crop,
                'bbox': (crop_x1, crop_y1, crop_x2, crop_y2),
                'center': (center_x, center_y),
                'size': padded_size
            }
            
        except Exception as e:
            print(f"Cropping error: {e}")
            return None
    
    def _normalize_image(self, image: np.ndarray, target_size: int = 224) -> np.ndarray:
        """
        Step 3: Normalization to 224x224 pixels
        """
        try:
            # Resize to target size
            normalized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize pixel values to [0, 1]
            normalized = normalized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Normalization error: {e}")
            return image
    
    def _apply_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Step 4: Data augmentation (rotation ±15°, brightness ±20%)
        """
        augmented = []
        
        try:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Original image
            augmented.append(image.copy())
            
            # Rotation augmentations (-15° to +15°)
            for angle in [-15, -7, 7, 15]:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                augmented.append(rotated)
            
            # Brightness augmentations (±20%)
            for brightness_factor in [0.8, 1.2]:
                brightened = np.clip(image * brightness_factor, 0, 1)
                augmented.append(brightened.astype(np.float32))
            
            # Combined augmentations (rotation + brightness)
            for angle in [-10, 10]:
                for brightness in [0.9, 1.1]:
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
                    combined = np.clip(rotated * brightness, 0, 1)
                    augmented.append(combined.astype(np.float32))
                    
        except Exception as e:
            print(f"Augmentation error: {e}")
            augmented = [image]  # Return original if augmentation fails
        
        return augmented
    
    def _assess_image_quality(self, image: np.ndarray, landmarks: List[Dict]) -> float:
        """
        Step 5: Quality filtering to remove unclear images
        """
        quality_score = 0.0
        
        try:
            # Convert to grayscale for quality assessment
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)
            
            # 1. Sharpness assessment using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)  # Normalize
            
            # 2. Contrast assessment
            contrast_score = gray.std() / 255.0
            
            # 3. Brightness assessment (prefer well-lit images)
            mean_brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2  # Peak at 0.5 brightness
            
            # 4. Hand completeness (all landmarks visible)
            visible_landmarks = sum(1 for lm in landmarks if lm.get('visibility', 1.0) > 0.5)
            completeness_score = visible_landmarks / len(landmarks)
            
            # 5. Hand size assessment (not too small, not too large)
            x_coords = [lm['x'] for lm in landmarks]
            y_coords = [lm['y'] for lm in landmarks]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_area = hand_width * hand_height
            
            # Ideal hand should occupy 10-60% of image
            image_area = image.shape[0] * image.shape[1]
            area_ratio = hand_area / image_area
            size_score = 1.0 if 0.1 <= area_ratio <= 0.6 else max(0.0, 1.0 - abs(area_ratio - 0.35) * 5)
            
            # Weighted combination
            quality_score = (
                sharpness_score * 0.25 +
                contrast_score * 0.2 +
                brightness_score * 0.15 +
                completeness_score * 0.3 +
                size_score * 0.1
            )
            
        except Exception as e:
            print(f"Quality assessment error: {e}")
        
        return min(1.0, max(0.0, quality_score))
    
    def extract_hand_features(self, landmarks: List[Dict]) -> Dict:
        """
        Extract comprehensive hand features from landmarks
        """
        features = {
            'finger_states': [],
            'angles': [],
            'distances': [],
            'palm_features': [],
            'geometric_features': []
        }
        
        try:
            # MediaPipe landmark indices
            WRIST = 0
            THUMB_TIP = 4
            INDEX_TIP = 8
            MIDDLE_TIP = 12
            RING_TIP = 16
            PINKY_TIP = 20
            
            # Extract finger states (extended vs folded)
            finger_tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            finger_pips = [3, 6, 10, 14, 18]  # PIP joints
            
            for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
                tip_y = landmarks[tip]['y']
                pip_y = landmarks[pip]['y']
                
                # For thumb, use x-coordinate (horizontal movement)
                if i == 0:  # Thumb
                    extended = abs(landmarks[tip]['x'] - landmarks[2]['x']) > 20
                else:
                    extended = tip_y < pip_y - 10  # Finger pointing up
                
                features['finger_states'].append(1.0 if extended else 0.0)
            
            # Calculate key distances
            wrist_pos = np.array([landmarks[WRIST]['x'], landmarks[WRIST]['y']])
            
            for tip_idx in finger_tips:
                tip_pos = np.array([landmarks[tip_idx]['x'], landmarks[tip_idx]['y']])
                distance = np.linalg.norm(tip_pos - wrist_pos)
                features['distances'].append(distance)
            
            # Calculate angles between fingers
            for i in range(len(finger_tips) - 1):
                vec1 = np.array([landmarks[finger_tips[i]]['x'], landmarks[finger_tips[i]]['y']]) - wrist_pos
                vec2 = np.array([landmarks[finger_tips[i+1]]['x'], landmarks[finger_tips[i+1]]['y']]) - wrist_pos
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features['angles'].append(angle)
            
            # Palm features
            palm_center = wrist_pos
            all_landmarks = np.array([[lm['x'], lm['y']] for lm in landmarks])
            
            # Hand bounding box
            min_coords = all_landmarks.min(axis=0)
            max_coords = all_landmarks.max(axis=0)
            hand_size = np.linalg.norm(max_coords - min_coords)
            
            features['palm_features'] = [
                palm_center[0] / 640,  # Normalized x
                palm_center[1] / 480,  # Normalized y  
                hand_size / 200        # Normalized size
            ]
            
            # Geometric features
            features['geometric_features'] = [
                (max_coords[0] - min_coords[0]) / (max_coords[1] - min_coords[1] + 1e-8),  # Aspect ratio
                hand_size,  # Hand span
                len([lm for lm in landmarks if lm.get('visibility', 1.0) > 0.8])  # Visible landmarks
            ]
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
        except Exception as e:
            print(f"Cleanup error: {e}")