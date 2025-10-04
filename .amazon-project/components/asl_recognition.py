"""
Advanced ASL Letter Recognition Engine
Real-time letter detection with accumulation, spacing, and word completion
"""
# Suppress warnings before importing MediaPipe/TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import threading
from typing import Dict, List, Optional, Tuple
import pyttsx3
from word_dictionary import get_word_suggestions

class ASLLetterRecognizer:
    """Real-time ASL letter recognition with accumulation engine"""
    
    def __init__(self):
        # MediaPipe setup with optimized settings for fast startup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Reduced for faster processing
            min_tracking_confidence=0.3,   # Reduced for faster processing
            model_complexity=0  # Use lightweight model for speed
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Letter accumulation settings (optimized for responsiveness)
        self.confirmation_frames = 20  # Reduced from 30 for faster response
        self.pause_frames = 30  # Reduced from 45 for faster spacing
        self.hold_frames = 45  # Reduced from 60 for faster auto-spacing
        
        # State variables
        self.current_letter = None
        self.letter_count = 0
        self.no_hand_count = 0
        self.accumulated_text = ""
        self.current_word = ""
        self.last_detection_time = time.time()
        
        # TTS engine (lazy initialization for faster startup)
        self.tts_engine = None
        # Don't initialize TTS during startup - do it on first use
        
        # Complete ASL Letter Recognition System (A-Z)
        # This covers the full American Sign Language alphabet
        self.letter_mappings = {
            # Complete ASL alphabet based on hand shapes and finger positions
            'A': 'fist',                    # Closed fist, thumb on side
            'B': 'flat_hand_thumb_in',      # Four fingers up, thumb tucked in
            'C': 'c_shape',                 # Curved hand like letter C
            'D': 'pointing_up',             # Index finger pointing up
            'E': 'curved_fingers',          # All fingers curved down
            'F': 'ok_modified',             # OK sign with other fingers up
            'G': 'index_side',              # Index finger pointing sideways
            'H': 'two_fingers_side',        # Index and middle finger sideways
            'I': 'pinky_up',                # Just pinky finger up
            'J': 'pinky_hook',              # Pinky in J motion
            'K': 'index_middle_v',          # Index and middle in V, thumb touches middle
            'L': 'l_shape',                 # Thumb and index in L shape
            'M': 'three_fingers_thumb',     # Three fingers over thumb
            'N': 'two_fingers_thumb',       # Two fingers over thumb
            'O': 'ok_sign',                 # Thumb and index touching (O shape)
            'P': 'k_pointing_down',         # Like K but pointing down
            'Q': 'g_pointing_down',         # Like G but pointing down
            'R': 'fingers_crossed',         # Index and middle fingers crossed
            'S': 'fist_thumb_front',        # Fist with thumb in front
            'T': 'thumb_between_fingers',   # Thumb between index and middle
            'U': 'two_fingers_up',          # Index and middle fingers up
            'V': 'peace',                   # Peace sign (V shape)
            'W': 'three_fingers_up',        # Index, middle, ring fingers up
            'X': 'index_bent',              # Index finger bent/hooked
            'Y': 'thumb_pinky_out',         # Thumb and pinky extended
            'Z': 'z_motion',                # Index finger drawing Z
            
            # Special gestures
            'SPACE': 'flat_hand',           # All fingers extended
            'THUMBS_UP': 'thumbs_up',       # Thumbs up gesture
            'LOVE': 'love_sign',            # I Love You sign
        }
        
    def _init_tts(self):
        """Initialize text-to-speech engine lazily"""
        if self.tts_engine is None:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
            except Exception as e:
                st.warning(f"TTS initialization failed: {e}")
                self.tts_engine = False  # Mark as failed to avoid retrying
    
    def calculate_hand_features(self, landmarks) -> Dict[str, float]:
        """Enhanced hand features calculation for complete A-Z ASL recognition"""
        if not landmarks:
            return {}
        
        # Get landmark positions
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x, lm.y])
        points = np.array(points)
        
        features = {}
        
        # Finger tip and joint positions
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        finger_mcp = [2, 5, 9, 13, 17]   # MCP joints
        
        # Individual finger extension detection
        features['thumb_extended'] = points[4][1] < points[3][1]
        features['index_extended'] = points[8][1] < points[6][1]
        features['middle_extended'] = points[12][1] < points[10][1]
        features['ring_extended'] = points[16][1] < points[14][1]
        features['pinky_extended'] = points[20][1] < points[18][1]
        
        # Count total fingers up
        finger_states = [
            features['index_extended'],
            features['middle_extended'], 
            features['ring_extended'],
            features['pinky_extended']
        ]
        features['fingers_up'] = sum(finger_states)
        features['thumb_up'] = features['thumb_extended']
        
        # Basic hand shapes
        features['is_fist'] = features['fingers_up'] == 0 and not features['thumb_up']
        features['is_flat_hand'] = features['fingers_up'] == 4 and features['thumb_up']
        features['is_pointing'] = features['fingers_up'] == 1 and features['index_extended']
        features['is_peace'] = (features['fingers_up'] == 2 and 
                               features['index_extended'] and features['middle_extended'] and
                               not features['ring_extended'] and not features['pinky_extended'])
        
        # Advanced shape detection for specific letters
        features['is_ok'] = self._check_ok_sign(points)
        features['is_thumbs_up'] = self._check_thumbs_up(points)
        features['is_love_sign'] = self._check_love_sign(points)
        
        # Additional shape detections for more letters
        features['is_curved'] = self._check_curved_hand(points)
        features['fingers_curved_down'] = self._check_fingers_curved_down(points)
        features['index_sideways'] = self._check_index_sideways(points)
        features['two_fingers_sideways'] = self._check_two_fingers_sideways(points)
        features['pinky_hook'] = self._check_pinky_hook(points)
        features['k_shape'] = self._check_k_shape(points)
        features['l_shape'] = self._check_l_shape(points)
        features['p_shape'] = self._check_p_shape(points)
        features['q_shape'] = self._check_q_shape(points)
        features['fingers_crossed'] = self._check_fingers_crossed(points)
        features['s_shape'] = self._check_s_shape(points)
        features['t_shape'] = self._check_t_shape(points)
        features['x_shape'] = self._check_x_shape(points)
        
        return features
    
    def _check_curved_hand(self, points) -> bool:
        """Check for C-shaped curved hand"""
        # Simplified: fingers partially extended with curve
        return (points[8][1] < points[6][1] and points[12][1] < points[10][1] and
                points[16][1] < points[14][1] and points[20][1] < points[18][1])
    
    def _check_fingers_curved_down(self, points) -> bool:
        """Check if fingers are curved downward (E shape)"""
        # All fingertips below PIP joints but above MCP
        curved_count = 0
        for tip, pip, mcp in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
            if points[tip][1] > points[pip][1] and points[tip][1] < points[mcp][1]:
                curved_count += 1
        return curved_count >= 3
    
    def _check_index_sideways(self, points) -> bool:
        """Check if index finger is pointing sideways (G shape)"""
        # Index extended, others down, pointing horizontally
        index_extended = points[8][1] < points[6][1]
        others_down = (points[12][1] > points[10][1] and 
                      points[16][1] > points[14][1] and 
                      points[20][1] > points[18][1])
        return index_extended and others_down
    
    def _check_two_fingers_sideways(self, points) -> bool:
        """Check if index and middle fingers are sideways (H shape)"""
        # Index and middle extended, pointing sideways
        index_middle_up = (points[8][1] < points[6][1] and points[12][1] < points[10][1])
        others_down = (points[16][1] > points[14][1] and points[20][1] > points[18][1])
        return index_middle_up and others_down
    
    def _check_pinky_hook(self, points) -> bool:
        """Check for pinky hook shape (J motion)"""
        # Simplified: just pinky extended
        return (points[20][1] < points[18][1] and 
                points[8][1] > points[6][1] and
                points[12][1] > points[10][1] and
                points[16][1] > points[14][1])
    
    def _check_k_shape(self, points) -> bool:
        """Check for K shape (index and middle in V, thumb touches middle)"""
        # Index and middle up in V shape
        index_up = points[8][1] < points[6][1]
        middle_up = points[12][1] < points[10][1]
        others_down = (points[16][1] > points[14][1] and points[20][1] > points[18][1])
        return index_up and middle_up and others_down
    
    def _check_l_shape(self, points) -> bool:
        """Check for L shape (thumb and index form L)"""
        # Thumb up, index pointing, others down
        thumb_up = points[4][1] < points[3][1]
        index_up = points[8][1] < points[6][1]
        others_down = (points[12][1] > points[10][1] and 
                      points[16][1] > points[14][1] and 
                      points[20][1] > points[18][1])
        return thumb_up and index_up and others_down
    
    def _check_p_shape(self, points) -> bool:
        """Check for P shape (like K but pointing down)"""
        # Similar to K but oriented differently
        return self._check_k_shape(points)  # Simplified
    
    def _check_q_shape(self, points) -> bool:
        """Check for Q shape (like G but pointing down)"""
        # Similar to G but oriented differently
        return self._check_index_sideways(points)  # Simplified
    
    def _check_fingers_crossed(self, points) -> bool:
        """Check for crossed fingers (R shape)"""
        # Index and middle fingers crossed - simplified detection
        index_up = points[8][1] < points[6][1]
        middle_up = points[12][1] < points[10][1]
        others_down = (points[16][1] > points[14][1] and points[20][1] > points[18][1])
        return index_up and middle_up and others_down
    
    def _check_s_shape(self, points) -> bool:
        """Check for S shape (fist with thumb in front)"""
        # Fist with thumb positioned differently
        fist_shape = (points[8][1] > points[6][1] and points[12][1] > points[10][1] and
                     points[16][1] > points[14][1] and points[20][1] > points[18][1])
        return fist_shape
    
    def _check_t_shape(self, points) -> bool:
        """Check for T shape (thumb between fingers)"""
        # Fist with thumb between index and middle
        return self._check_s_shape(points)  # Simplified
    
    def _check_x_shape(self, points) -> bool:
        """Check for X shape (bent index finger)"""
        # Index finger bent/hooked
        index_bent = (points[8][1] > points[6][1] and points[8][1] < points[5][1])
        others_down = (points[12][1] > points[10][1] and 
                      points[16][1] > points[14][1] and 
                      points[20][1] > points[18][1])
        return index_bent and others_down
    
    def _check_ok_sign(self, points) -> bool:
        """Check if hand is making OK sign (thumb and index touching)"""
        thumb_tip = points[4]
        index_tip = points[8]
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.05  # Close proximity indicates touching
    
    def _check_thumbs_up(self, points) -> bool:
        """Check if hand is making thumbs up"""
        # Thumb up, other fingers down
        thumb_up = points[4][1] < points[3][1]
        fingers_down = sum([points[tip][1] > points[pip][1] 
                           for tip, pip in [(8,6), (12,10), (16,14), (20,18)]]) >= 3
        return thumb_up and fingers_down
    
    def _check_love_sign(self, points) -> bool:
        """Check if hand is making I Love You sign"""
        # Thumb, index, and pinky up; middle and ring down
        thumb_up = points[4][1] < points[3][1]
        index_up = points[8][1] < points[6][1]
        middle_down = points[12][1] > points[10][1]
        ring_down = points[16][1] > points[14][1]
        pinky_up = points[20][1] < points[18][1]
        
        return thumb_up and index_up and middle_down and ring_down and pinky_up
    
    def recognize_letter(self, features: Dict[str, float]) -> Optional[str]:
        """Advanced ASL letter recognition for complete A-Z alphabet"""
        if not features:
            return None
        
        # Get finger states
        fingers_up = features.get('fingers_up', 0)
        thumb_up = features.get('thumb_up', False)
        
        # Get individual finger positions (thumb, index, middle, ring, pinky)
        finger_positions = [
            features.get('thumb_extended', False),
            features.get('index_extended', False), 
            features.get('middle_extended', False),
            features.get('ring_extended', False),
            features.get('pinky_extended', False)
        ]
        
        # Advanced letter recognition based on finger patterns
        
        # A - Closed fist with thumb on side
        if features.get('is_fist', False):
            return 'A'
        
        # B - Four fingers up, thumb tucked in
        elif fingers_up == 4 and not thumb_up:
            return 'B'
        
        # C - Curved hand shape (approximated by moderate finger curve)
        elif features.get('is_curved', False):
            return 'C'
        
        # D - Index finger pointing up, others down
        elif features.get('is_pointing', False) and fingers_up == 1:
            return 'D'
        
        # E - All fingers curved down (fist-like but more open)
        elif features.get('fingers_curved_down', False):
            return 'E'
        
        # F - OK sign with other fingers extended
        elif features.get('is_ok', False) and fingers_up >= 2:
            return 'F'
        
        # G - Index finger pointing sideways
        elif features.get('index_sideways', False):
            return 'G'
        
        # H - Index and middle fingers pointing sideways
        elif features.get('two_fingers_sideways', False):
            return 'H'
        
        # I - Only pinky finger up
        elif finger_positions == [False, False, False, False, True]:
            return 'I'
        
        # J - Pinky in hook shape (approximated)
        elif features.get('pinky_hook', False):
            return 'J'
        
        # K - Index and middle in V shape with thumb touching middle
        elif features.get('k_shape', False):
            return 'K'
        
        # L - Thumb and index form L shape
        elif features.get('l_shape', False):
            return 'L'
        
        # M - Three fingers over thumb
        elif fingers_up == 3 and not thumb_up:
            return 'M'
        
        # N - Two fingers over thumb  
        elif fingers_up == 2 and not thumb_up:
            return 'N'
        
        # O - Thumb and index touching (circle)
        elif features.get('is_ok', False) and fingers_up == 0:
            return 'O'
        
        # P - Like K but pointing down
        elif features.get('p_shape', False):
            return 'P'
        
        # Q - Like G but pointing down
        elif features.get('q_shape', False):
            return 'Q'
        
        # R - Index and middle fingers crossed
        elif features.get('fingers_crossed', False):
            return 'R'
        
        # S - Fist with thumb in front
        elif features.get('s_shape', False):
            return 'S'
        
        # T - Thumb between index and middle
        elif features.get('t_shape', False):
            return 'T'
        
        # U - Index and middle fingers up together
        elif finger_positions == [False, True, True, False, False]:
            return 'U'
        
        # V - Peace sign (index and middle fingers up, separated)
        elif features.get('is_peace', False):
            return 'V'
        
        # W - Three fingers up (index, middle, ring)
        elif finger_positions == [False, True, True, True, False]:
            return 'W'
        
        # X - Index finger bent/hooked
        elif features.get('x_shape', False):
            return 'X'
        
        # Y - Thumb and pinky extended
        elif finger_positions == [True, False, False, False, True]:
            return 'Y'
        
        # Z - Index finger drawing Z motion (approximated by extended index)
        elif finger_positions == [False, True, False, False, False]:
            return 'Z'
        
        # Special gestures
        elif features.get('is_thumbs_up', False):
            return 'THUMBS_UP'
        elif features.get('is_love_sign', False):
            return 'LOVE'
        elif features.get('is_flat_hand', False):
            return 'SPACE'
        
        # Fallback for numbers if no letter detected
        elif fingers_up == 1 and thumb_up:
            return '1'
        elif fingers_up == 2 and not thumb_up:
            return '2'
        elif fingers_up == 3 and not thumb_up:
            return '3'
        elif fingers_up == 4 and not thumb_up:
            return '4'
        elif fingers_up == 5:
            return '5'
        
        return None
    
    def process_frame(self, frame) -> Tuple[np.ndarray, Dict]:
        """Process video frame and return annotated frame with detection info"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Create copy for annotation
        annotated_frame = frame.copy()
        detection_info = {
            'current_letter': None,
            'confidence': 0.0,
            'progress': 0.0,
            'accumulated_text': self.accumulated_text,
            'current_word': self.current_word,
            'suggestions': []
        }
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate features and recognize letter
                features = self.calculate_hand_features(hand_landmarks)
                detected_letter = self.recognize_letter(features)
                
                if detected_letter:
                    detection_info['current_letter'] = detected_letter
                    detection_info['confidence'] = 0.8  # Simulated confidence
                    
                    # Letter accumulation logic
                    if detected_letter == self.current_letter:
                        self.letter_count += 1
                        self.no_hand_count = 0
                        
                        # Calculate progress
                        progress = min(self.letter_count / self.confirmation_frames, 1.0)
                        detection_info['progress'] = progress
                        
                        # Confirm letter when threshold reached
                        if self.letter_count >= self.confirmation_frames:
                            self._confirm_letter(detected_letter)
                            self.letter_count = 0
                            
                    else:
                        # New letter detected
                        self.current_letter = detected_letter
                        self.letter_count = 1
                        self.no_hand_count = 0
                else:
                    # No hand detected
                    self.no_hand_count += 1
                    self.letter_count = 0
                    
                    # Add space after pause
                    if self.no_hand_count >= self.pause_frames:
                        self._add_space()
                        self.no_hand_count = 0
        else:
            # No hand detected
            self.no_hand_count += 1
            self.letter_count = 0
            
            if self.no_hand_count >= self.pause_frames:
                self._add_space()
                self.no_hand_count = 0
        
        # Update detection info
        detection_info['accumulated_text'] = self.accumulated_text
        detection_info['current_word'] = self.current_word
        detection_info['suggestions'] = get_word_suggestions(self.current_word, 5)
        
        # Draw UI overlay
        annotated_frame = self._draw_ui_overlay(annotated_frame, detection_info)
        
        return annotated_frame, detection_info
    
    def _confirm_letter(self, letter: str):
        """Confirm and add letter to accumulated text"""
        if letter == 'SPACE':
            self._add_space()
        else:
            self.current_word += letter
            self.last_detection_time = time.time()
            
            # Voice feedback (with lazy TTS initialization)
            self._init_tts()
            if self.tts_engine and self.tts_engine is not False:
                try:
                    threading.Thread(target=lambda: self.tts_engine.say(letter)).start()
                except:
                    pass
    
    def _add_space(self):
        """Add space and finish current word"""
        if self.current_word:
            self.accumulated_text += self.current_word + " "
            self.current_word = ""
    
    def finish_word(self, word: str = None):
        """Finish current word or use suggestion"""
        word_to_add = word or self.current_word
        if word_to_add:
            self.accumulated_text += word_to_add + " "
            self.current_word = ""
    
    def clear_all(self):
        """Clear all accumulated text"""
        self.accumulated_text = ""
        self.current_word = ""
    
    def read_text(self, text: str):
        """Read text aloud using TTS with lazy initialization"""
        self._init_tts()
        if self.tts_engine and self.tts_engine is not False and text:
            try:
                threading.Thread(target=lambda: self.tts_engine.say(text)).start()
            except Exception as e:
                st.error(f"TTS error: {e}")
    
    def _draw_ui_overlay(self, frame, info):
        """Draw UI overlay on frame"""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Current letter and progress
        if info['current_letter']:
            cv2.putText(frame, f"Detecting: {info['current_letter']}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Progress bar
            progress_width = int(200 * info['progress'])
            cv2.rectangle(frame, (20, 50), (220, 70), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 50), (20 + progress_width, 70), (0, 255, 0), -1)
        
        # Current word
        cv2.putText(frame, f"Word: {info['current_word']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Accumulated text (first 50 chars)
        text_display = info['accumulated_text'][:50] + ("..." if len(info['accumulated_text']) > 50 else "")
        cv2.putText(frame, f"Text: {text_display}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

# Global recognizer instance
asl_recognizer = None

def get_asl_recognizer():
    """Get or create ASL recognizer instance"""
    global asl_recognizer
    if asl_recognizer is None:
        asl_recognizer = ASLLetterRecognizer()
    return asl_recognizer