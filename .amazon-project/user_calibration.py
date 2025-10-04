import os
import json
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

class UserProfile:
    def __init__(self, user_id, name=None):
        self.user_id = user_id
        self.name = name or user_id
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        # Calibration data for each letter
        self.letter_calibrations = {}
        # Personal thresholds
        self.confidence_threshold = 0.7
        self.hand_size_factor = 1.0
        self.position_offset = [0.0, 0.0]
        # Adaptive learning data
        self.letter_patterns = {}  # Store historical patterns for each letter
        self.success_rate = {}  # Track success rate per letter

    def add_calibration_data(self, letter, landmarks, features):
        """Add calibration data for a letter."""
        if letter not in self.letter_calibrations:
            self.letter_calibrations[letter] = {
                'landmarks': [],
                'features': [],
                'confidence_scores': []
            }
        
        self.letter_calibrations[letter]['landmarks'].append(landmarks)
        self.letter_calibrations[letter]['features'].append(features)
        self.last_updated = datetime.now().isoformat()

    def update_confidence_threshold(self, new_scores):
        """Adaptively update confidence threshold based on recent performance."""
        if new_scores:
            # Adjust threshold to be slightly below average successful detection
            self.confidence_threshold = np.mean(new_scores) * 0.9

    def update_hand_normalization(self, landmarks):
        """Update hand size and position normalization factors."""
        if landmarks:
            # Calculate hand size from landmarks
            points = np.array([[lm.x, lm.y] for lm in landmarks])
            current_size = np.max(np.ptp(points, axis=0))
            
            # Update size factor with moving average
            alpha = 0.3  # learning rate
            self.hand_size_factor = (alpha * current_size + 
                                   (1 - alpha) * self.hand_size_factor)
            
            # Update position offset
            wrist_pos = np.array([landmarks[0].x, landmarks[0].y])
            target_pos = np.array([0.5, 0.5])  # Center of frame
            current_offset = target_pos - wrist_pos
            self.position_offset = (alpha * current_offset + 
                                  (1 - alpha) * np.array(self.position_offset))

    def add_detection_result(self, letter, success, features):
        """Track detection results for adaptive learning."""
        if letter not in self.letter_patterns:
            self.letter_patterns[letter] = []
            self.success_rate[letter] = []
        
        # Store feature pattern
        self.letter_patterns[letter].append(features)
        if len(self.letter_patterns[letter]) > 100:  # Keep last 100 patterns
            self.letter_patterns[letter].pop(0)
        
        # Update success rate
        self.success_rate[letter].append(float(success))
        if len(self.success_rate[letter]) > 100:  # Keep last 100 results
            self.success_rate[letter].pop(0)
        
        self.last_updated = datetime.now().isoformat()

class UserCalibrationManager:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_profiles')
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.current_profile = None

    def create_profile(self, user_id, name=None):
        """Create a new user profile."""
        profile = UserProfile(user_id, name)
        self.save_profile(profile)
        return profile

    def save_profile(self, profile):
        """Save user profile to disk."""
        profile_path = self.base_path / f"{profile.user_id}.pkl"
        with open(profile_path, 'wb') as f:
            pickle.dump(profile, f)

    def load_profile(self, user_id):
        """Load user profile from disk."""
        profile_path = self.base_path / f"{user_id}.pkl"
        if profile_path.exists():
            with open(profile_path, 'rb') as f:
                return pickle.load(f)
        return None

    def list_profiles(self):
        """List all available user profiles."""
        profiles = []
        for profile_path in self.base_path.glob("*.pkl"):
            try:
                profile = self.load_profile(profile_path.stem)
                if profile:
                    profiles.append({
                        'user_id': profile.user_id,
                        'name': profile.name,
                        'created_at': profile.created_at,
                        'last_updated': profile.last_updated
                    })
            except Exception:
                continue
        return profiles

    def delete_profile(self, user_id):
        """Delete a user profile."""
        profile_path = self.base_path / f"{user_id}.pkl"
        if profile_path.exists():
            profile_path.unlink()
            return True
        return False

    def normalize_landmarks(self, landmarks, profile=None):
        """Apply user-specific normalization to landmarks."""
        if not profile:
            profile = self.current_profile
        if not profile or not landmarks:
            return landmarks

        # Apply hand size normalization
        normalized = []
        for lm in landmarks:
            x = lm.x * profile.hand_size_factor + profile.position_offset[0]
            y = lm.y * profile.hand_size_factor + profile.position_offset[1]
            z = lm.z * profile.hand_size_factor
            normalized.append(type(lm)(x=x, y=y, z=z))
        
        return normalized

    def get_confidence_threshold(self, letter=None):
        """Get confidence threshold, optionally letter-specific."""
        if not self.current_profile:
            return 0.7  # Default threshold
        
        if letter and letter in self.current_profile.success_rate:
            # Adjust threshold based on historical success rate
            success_rate = np.mean(self.current_profile.success_rate[letter])
            return min(self.current_profile.confidence_threshold * (1.0 + success_rate) / 2.0, 0.95)
        
        return self.current_profile.confidence_threshold
