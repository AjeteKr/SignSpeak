"""
Letter Accumulation Engine with Smart Spacing and Word Completion
Advanced system for building words from ASL letter recognition
"""

import numpy as np
import streamlit as st
import cv2
import time
import pyttsx3
import threading
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import re
from collections import defaultdict, deque
import difflib
import queue
import subprocess
import platform

class StreamlitTTS:
    """TTS system optimized for Streamlit environment"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.speech_queue = queue.Queue()
            self.worker_thread = None
            self.running = False
            self._initialized = True
    
    def speak(self, text: str):
        """Queue text for speaking"""
        if not text or not text.strip():
            return
        
        # Start worker thread if not running
        if not self.running:
            self._start_worker()
        
        # Add to queue
        self.speech_queue.put(text.strip())
    
    def _start_worker(self):
        """Start the TTS worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _speech_worker(self):
        """Worker thread that processes TTS queue"""
        while self.running:
            try:
                # Get text from queue with timeout
                text = self.speech_queue.get(timeout=1.0)
                
                # Try different TTS methods
                success = self._try_pyttsx3(text) or self._try_system_tts(text)
                
                if not success:
                    print(f"TTS: Failed to speak '{text}'")
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Worker Error: {e}")
    
    def _try_pyttsx3(self, text: str) -> bool:
        """Try using pyttsx3 for TTS"""
        try:
            # Create engine in worker thread
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Set event callbacks to avoid blocking
            def on_start(name):
                pass
            
            def on_word(name, location, length):
                pass
            
            def on_end(name, completed):
                pass
            
            engine.connect('started-utterance', on_start)
            engine.connect('started-word', on_word) 
            engine.connect('finished-utterance', on_end)
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
            print(f"TTS: Successfully spoke '{text}' using pyttsx3")
            return True
            
        except Exception as e:
            print(f"TTS pyttsx3 error: {e}")
            return False
    
    def _try_system_tts(self, text: str) -> bool:
        """Try using system TTS as fallback"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Use Windows SAPI
                import subprocess
                cmd = ['powershell', '-Command', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")']
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"TTS: Successfully spoke '{text}' using Windows SAPI")
                return True
                
            elif system == "darwin":  # macOS
                subprocess.run(['say', text], check=True)
                print(f"TTS: Successfully spoke '{text}' using macOS say")
                return True
                
            elif system == "linux":
                # Try espeak or festival
                try:
                    subprocess.run(['espeak', text], check=True)
                    print(f"TTS: Successfully spoke '{text}' using espeak")
                    return True
                except:
                    subprocess.run(['festival', '--tts'], input=text.encode(), check=True)
                    print(f"TTS: Successfully spoke '{text}' using festival")
                    return True
            
            return False
            
        except Exception as e:
            print(f"TTS system error: {e}")
            return False
    
    def stop(self):
        """Stop the TTS worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

@dataclass
class LetterDetection:
    """Single letter detection event"""
    letter: str
    confidence: float
    timestamp: datetime
    features: Dict
    frame_count: int

@dataclass
class WordSuggestion:
    """Word completion suggestion"""
    word: str
    confidence: float
    frequency: int
    category: str  # common, asl, technical, etc.

@dataclass
class AccumulationState:
    """Current state of letter accumulation"""
    current_word: str = ""
    completed_text: str = ""
    last_detection: Optional[LetterDetection] = None
    detection_buffer: List[LetterDetection] = field(default_factory=list)
    pause_start: Optional[datetime] = None
    confirmation_progress: float = 0.0
    active_suggestions: List[WordSuggestion] = field(default_factory=list)
    last_confirmed_frame: int = 0  # Track when last letter was confirmed
    cooldown_active: bool = False  # Track if in cooldown period

class WordDictionary:
    """Comprehensive word dictionary for suggestions"""
    
    def __init__(self):
        self.words = self._load_dictionary()
        self.frequency_data = self._load_frequency_data()
        self.asl_vocabulary = self._load_asl_vocabulary()
        self.user_dictionary = set()
    
    def _load_dictionary(self) -> Set[str]:
        """Load comprehensive English dictionary including ASL gesture vocabulary"""
        # Load gesture vocabulary from raw_data folder
        gesture_words = self._load_gesture_vocabulary()
        
        # Common English words
        common_words = {
            # Basic words
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR",
            "HAD", "BY", "WORD", "OIL", "SIT", "SET", "RUN", "EAT", "FAR", "SEA", "EYE", "OLD", "SEE",
            
            # ASL common words
            "HELLO", "GOODBYE", "THANK", "YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "LOVE", "NAME",
            "GOOD", "BAD", "HAPPY", "SAD", "ANGRY", "TIRED", "HUNGRY", "THIRSTY", "HOT", "COLD",
            
            # Family
            "MOTHER", "FATHER", "SISTER", "BROTHER", "FAMILY", "BABY", "CHILD", "GRANDMOTHER", "GRANDFATHER",
            
            # Time
            "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "MORNING", "AFTERNOON", "EVENING", "NIGHT",
            "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
            
            # Food
            "FOOD", "WATER", "MILK", "BREAD", "MEAT", "FRUIT", "VEGETABLE", "APPLE", "BANANA", "ORANGE",
            
            # Colors
            "RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "BROWN", "PINK", "PURPLE", "ORANGE",
            
            # Numbers (spelled out)
            "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
            "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY",
            
            # Common verbs
            "GO", "COME", "GET", "GIVE", "TAKE", "MAKE", "KNOW", "THINK", "SEE", "LOOK", "WANT", "LIKE",
            "WORK", "PLAY", "LEARN", "TEACH", "READ", "WRITE", "LISTEN", "SPEAK", "WALK", "RUN", "SIT", "STAND",
            
            # Common adjectives
            "BIG", "SMALL", "TALL", "SHORT", "FAST", "SLOW", "EASY", "HARD", "NICE", "GOOD", "BAD", "NEW", "OLD",
            
            # Education
            "SCHOOL", "TEACHER", "STUDENT", "BOOK", "PAPER", "PEN", "PENCIL", "COMPUTER", "LEARN", "STUDY",
            
            # Communication
            "SIGN", "LANGUAGE", "DEAF", "HEARING", "SPEAK", "VOICE", "HANDS", "FINGERS", "EYES", "FACE"
        }
        
        # Combine common words with gesture vocabulary
        all_words = common_words.union(gesture_words)
        return all_words
    
    def _load_gesture_vocabulary(self) -> Set[str]:
        """Load ASL gesture vocabulary from raw_data folder"""
        gesture_words = set()
        
        try:
            # Path to gesture data - adjust if needed
            gesture_path = Path("../Wave/raw_data")
            if not gesture_path.exists():
                # Try alternative paths
                gesture_path = Path("Wave/raw_data")
                if not gesture_path.exists():
                    gesture_path = Path("c:/Users/BesiComputers/Desktop/Amazon/Wave/raw_data")
            
            if gesture_path.exists():
                # Get all folder names (these are the gesture words)
                for folder in gesture_path.iterdir():
                    if folder.is_dir():
                        word = folder.name.upper()
                        gesture_words.add(word)
                
            else:
                # Fallback to known gesture words from your raw_data
                gesture_words = {
                    "AMAZING", "BAD", "BEAUTIFUL", "FINE", "GOOD", "GOODBYE", 
                    "HEARING", "HELLO", "HELP", "HOW_ARE_YOU", "LEARN", "LOVE", 
                    "NAME", "NICE_TO_MEET_YOU", "NO", "PLEASE", "SORRY", 
                    "THANK_YOU", "UNDERSTAND", "YES"
                }
                print(f"Using fallback gesture words: {len(gesture_words)} words")
                
        except Exception as e:
            print(f"Error loading gesture vocabulary: {e}")
            # Fallback gesture words
            gesture_words = {
                "AMAZING", "BAD", "BEAUTIFUL", "FINE", "GOOD", "GOODBYE", 
                "HEARING", "HELLO", "HELP", "HOW_ARE_YOU", "LEARN", "LOVE", 
                "NAME", "NICE_TO_MEET_YOU", "NO", "PLEASE", "SORRY", 
                "THANK_YOU", "UNDERSTAND", "YES"
            }
        
        return gesture_words
    
    def _load_frequency_data(self) -> Dict[str, int]:
        """Load word frequency data including gesture words"""
        # Base frequency data
        frequency_data = {
            "THE": 1000, "AND": 800, "FOR": 600, "ARE": 500, "BUT": 400,
            "HELLO": 300, "THANK": 250, "YOU": 900, "PLEASE": 200, "YES": 300, "NO": 250,
            "GOOD": 150, "BAD": 100, "HAPPY": 120, "LOVE": 180, "HELP": 140,
            
            # High frequency for gesture words (make them prioritized in suggestions)
            "AMAZING": 350, "BEAUTIFUL": 320, "HELLO": 400, "GOODBYE": 380,
            "THANK_YOU": 360, "PLEASE": 340, "SORRY": 300, "YES": 380, "NO": 370,
            "GOOD": 290, "BAD": 280, "FINE": 270, "HELP": 350, "LOVE": 330,
            "NAME": 310, "LEARN": 280, "UNDERSTAND": 250, "HEARING": 200,
            "NICE_TO_MEET_YOU": 220, "HOW_ARE_YOU": 240
        }
        
        return frequency_data
    
    def _load_asl_vocabulary(self) -> Set[str]:
        """Load ASL-specific vocabulary"""
        return {
            "ASL", "DEAF", "HEARING", "SIGN", "LANGUAGE", "INTERPRETER", "FINGERSPELL",
            "CHAT", "TALK", "CONVERSATION", "COMMUNICATE", "EXPRESSION", "GESTURE",
            "GALLAUDET", "DEAF-BLIND", "HARD-OF-HEARING", "COCHLEAR", "IMPLANT"
        }
    
    def add_user_word(self, word: str):
        """Add word to user dictionary"""
        self.user_dictionary.add(word.upper())
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 5) -> List[WordSuggestion]:
        """Get smart word suggestions for given prefix"""
        if len(prefix) < 1:  # Allow suggestions from first letter
            return []
        
        prefix_upper = prefix.upper()
        suggestions = []
        
        # Get gesture vocabulary (high priority)
        gesture_words = self._load_gesture_vocabulary()
        
        # Find matching words from all dictionaries
        all_words = self.words.union(self.asl_vocabulary).union(self.user_dictionary)
        
        for word in all_words:
            if word.startswith(prefix_upper):
                frequency = self.frequency_data.get(word, 50)
                
                # Determine category and boost gesture words
                if word in gesture_words:
                    category = "gesture"
                    frequency += 200  # Boost gesture words
                elif word in self.asl_vocabulary:
                    category = "asl"
                elif word in self.user_dictionary:
                    category = "user"
                    frequency += 100  # Boost user words
                else:
                    category = "common"
                
                # Calculate confidence based on match quality
                confidence = self._calculate_suggestion_confidence(prefix_upper, word, frequency)
                
                suggestions.append(WordSuggestion(
                    word=word,
                    confidence=confidence,
                    frequency=frequency,
                    category=category
                ))
        
        # Sort by category priority, then frequency and confidence
        category_priority = {"gesture": 4, "user": 3, "asl": 2, "common": 1}
        suggestions.sort(key=lambda x: (
            -category_priority.get(x.category, 0),
            -x.frequency, 
            -x.confidence
        ))
        
        return suggestions[:max_suggestions]
    
    def _calculate_suggestion_confidence(self, prefix: str, word: str, frequency: int) -> float:
        """Calculate suggestion confidence based on various factors"""
        # Base confidence from frequency
        base_confidence = min(1.0, frequency / 1000.0)
        
        # Boost for exact prefix match
        if word.startswith(prefix):
            match_ratio = len(prefix) / len(word)
            prefix_bonus = match_ratio * 0.3
        else:
            prefix_bonus = 0
        
        # Boost for common gesture words
        gesture_bonus = 0.2 if word in {"HELLO", "THANK_YOU", "PLEASE", "YES", "NO"} else 0
        
        return min(1.0, base_confidence + prefix_bonus + gesture_bonus)
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word is in any dictionary"""
        word_upper = word.upper()
        return (word_upper in self.words or 
                word_upper in self.asl_vocabulary or 
                word_upper in self.user_dictionary)

class LetterAccumulationEngine:
    """
    Advanced letter accumulation engine with smart spacing and word completion
    """
    
    def __init__(self, db_path: str = "accumulation.db"):
        """Initialize accumulation engine"""
        self.db_path = Path(db_path)
        self.dictionary = WordDictionary()
        self.tts = StreamlitTTS()  # Use new TTS system
        
        # Accumulation settings - Made more responsive
        self.CONFIRMATION_FRAMES = 20  # 0.7 seconds at 30 FPS (reduced from 30)
        self.CONFIRMATION_THRESHOLD = 0.75  # Only need 75% confirmation instead of 100%
        self.PAUSE_THRESHOLD = 45      # 1.5 seconds for space
        self.HOLD_THRESHOLD = 60       # 2 seconds for word completion
        self.MIN_CONFIDENCE = 0.7
        self.COOLDOWN_FRAMES = 20      # Cooldown period after adding a letter
        
        # State management
        self.state = AccumulationState()
        self.frame_count = 0
        self.last_letter_time = None
        self.gesture_detector = GestureDetector()
        
        # Performance tracking
        self.session_stats = {
            'letters_detected': 0,
            'words_completed': 0,
            'errors_corrected': 0,
            'session_start': datetime.now()
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for session tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accumulation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_letters INTEGER DEFAULT 0,
                    total_words INTEGER DEFAULT 0,
                    total_text TEXT DEFAULT '',
                    accuracy_score REAL DEFAULT 0.0,
                    user_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS letter_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    confirmed BOOLEAN DEFAULT FALSE,
                    features TEXT,
                    FOREIGN KEY (session_id) REFERENCES accumulation_sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS word_completions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    partial_word TEXT NOT NULL,
                    completed_word TEXT NOT NULL,
                    method TEXT NOT NULL,  -- manual, suggestion, auto
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES accumulation_sessions (session_id)
                )
            """)
    
    def process_detection(self, letter: str, confidence: float, features: Dict = None) -> Dict:
        """
        Process new letter detection and update accumulation state
        
        Args:
            letter: Detected letter
            confidence: Detection confidence
            features: Additional features from detection
            
        Returns:
            Dictionary with current state and actions
        """
        self.frame_count += 1
        current_time = datetime.now()
        
        # Create detection object
        detection = LetterDetection(
            letter=letter,
            confidence=confidence,
            timestamp=current_time,
            features=features or {},
            frame_count=self.frame_count
        )
        
        # Add to buffer
        self.state.detection_buffer.append(detection)
        
        # Keep buffer size manageable
        if len(self.state.detection_buffer) > 60:  # 2 seconds of buffer
            self.state.detection_buffer.pop(0)
        
        # Check for pause (no detection)
        if letter is None or confidence < self.MIN_CONFIDENCE:
            return self._handle_no_detection(current_time)
        
        # Reset pause tracking
        self.state.pause_start = None
        
        # Check for letter confirmation
        confirmation_result = self._check_letter_confirmation(detection)
        
        if confirmation_result['confirmed']:
            return self._confirm_letter(detection)
        
        # Update progress
        self.state.confirmation_progress = confirmation_result['progress']
        
        return {
            'action': 'accumulating',
            'current_letter': letter,
            'progress': self.state.confirmation_progress,
            'current_word': self.state.current_word,
            'suggestions': self.state.active_suggestions,
            'status': 'detecting'
        }
    
    def _handle_no_detection(self, current_time: datetime) -> Dict:
        """Handle case when no letter is detected"""
        # Start pause timer if not already started
        if self.state.pause_start is None:
            self.state.pause_start = current_time
        
        # Check if pause threshold reached
        pause_duration = current_time - self.state.pause_start
        
        if pause_duration.total_seconds() * 30 >= self.PAUSE_THRESHOLD:  # Convert to frames
            return self._add_space()
        
        return {
            'action': 'pausing',
            'pause_progress': pause_duration.total_seconds() * 30 / self.PAUSE_THRESHOLD,
            'current_word': self.state.current_word,
            'status': 'pausing'
        }
    
    def _check_letter_confirmation(self, detection: LetterDetection) -> Dict:
        """Check if letter should be confirmed based on consistency"""
        # Check if we're in cooldown period
        if self.state.cooldown_active and (self.frame_count - self.state.last_confirmed_frame) < self.COOLDOWN_FRAMES:
            return {
                'confirmed': False,
                'progress': 0.0,
                'consistent_detections': 0,
                'cooldown_remaining': self.COOLDOWN_FRAMES - (self.frame_count - self.state.last_confirmed_frame)
            }
        
        # Exit cooldown if enough frames have passed
        if self.state.cooldown_active and (self.frame_count - self.state.last_confirmed_frame) >= self.COOLDOWN_FRAMES:
            self.state.cooldown_active = False
        
        # Get recent detections of the same letter
        recent_same_letter = [
            d for d in self.state.detection_buffer[-self.CONFIRMATION_FRAMES:]
            if d.letter == detection.letter and d.confidence >= self.MIN_CONFIDENCE
        ]
        
        # Calculate confirmation progress
        progress = len(recent_same_letter) / self.CONFIRMATION_FRAMES
        
        # Check if confirmed - now only needs 75% instead of 100%
        confirmed = progress >= self.CONFIRMATION_THRESHOLD
        
        return {
            'confirmed': confirmed,
            'progress': progress,
            'consistent_detections': len(recent_same_letter)
        }
    
    def _confirm_letter(self, detection: LetterDetection) -> Dict:
        """Confirm and add letter to current word"""
        letter = detection.letter
        
        # Add to current word
        self.state.current_word += letter
        self.state.last_detection = detection
        self.last_letter_time = detection.timestamp
        
        # Start cooldown period to prevent rapid repeats
        self.state.last_confirmed_frame = self.frame_count
        self.state.cooldown_active = True
        
        # Update statistics
        self.session_stats['letters_detected'] += 1
        
        # Save to database
        self._save_letter_detection(detection, confirmed=True)
        
        # Generate word suggestions
        self.state.active_suggestions = self.dictionary.get_suggestions(self.state.current_word)
        
        # Text-to-speech for letter
        self._speak_letter(letter)
        
        # Check for automatic word completion
        auto_complete = self._check_auto_completion()
        
        result = {
            'action': 'letter_confirmed',
            'letter': letter,
            'current_word': self.state.current_word,
            'suggestions': self.state.active_suggestions,
            'auto_complete': auto_complete,
            'progress': 0.0,  # Reset progress
            'status': 'confirmed'
        }
        
        # Reset confirmation progress
        self.state.confirmation_progress = 0.0
        
        return result
    
    def _add_space(self) -> Dict:
        """Add space and complete current word"""
        if self.state.current_word:
            # Complete current word
            self._complete_word(self.state.current_word, method="auto_space")
        
        # Add space to completed text ONLY if it doesn't already end with space
        if self.state.completed_text:
            # Strip trailing spaces and add exactly one space
            self.state.completed_text = self.state.completed_text.rstrip() + ' '
        
        # Reset pause tracking
        self.state.pause_start = None
        
        return {
            'action': 'space_added',
            'completed_text': self.state.completed_text,
            'current_word': '',
            'suggestions': [],
            'status': 'spaced'
        }
    
    def _check_auto_completion(self) -> Optional[str]:
        """Check if current word should be auto-completed"""
        if len(self.state.current_word) < 3:
            return None
        
        # Check if current word exactly matches a high-frequency word
        for suggestion in self.state.active_suggestions:
            if (suggestion.word == self.state.current_word.upper() and 
                suggestion.frequency > 200):
                return suggestion.word
        
        # Check if only one suggestion with high confidence
        if (len(self.state.active_suggestions) == 1 and 
            self.state.active_suggestions[0].confidence > 0.8):
            return self.state.active_suggestions[0].word
        
        return None
    
    def complete_word_manually(self, word: str) -> Dict:
        """Manually complete current word with selected suggestion"""
        self._complete_word(word, method="manual")
        
        return {
            'action': 'word_completed',
            'completed_word': word,
            'completed_text': self.state.completed_text,
            'current_word': '',
            'suggestions': [],
            'status': 'completed'
        }
    
    def _complete_word(self, word: str, method: str = "manual"):
        """Complete current word and add to text"""
        if not word:
            return
        
        # Add to completed text with proper space handling
        if self.state.completed_text:
            # Only add space if text doesn't already end with space
            if not self.state.completed_text.endswith(' '):
                self.state.completed_text += ' '
        self.state.completed_text += word
        
        # Add to user dictionary if not already known
        if not self.dictionary.is_valid_word(word):
            self.dictionary.add_user_word(word)
        
        # Save to database
        self._save_word_completion(self.state.current_word, word, method)
        
        # Update statistics
        self.session_stats['words_completed'] += 1
        
        # Text-to-speech for word
        self._speak_word(word)
        
        # Reset current word and suggestions
        self.state.current_word = ''
        self.state.active_suggestions = []
    
    def handle_gesture_command(self, gesture: str) -> Dict:
        """Handle gesture-based commands"""
        if gesture == "space":
            return self._add_space()
        elif gesture == "delete":
            return self._delete_last_character()
        elif gesture == "complete_word":
            if self.state.active_suggestions:
                return self.complete_word_manually(self.state.active_suggestions[0].word)
        elif gesture == "clear_word":
            return self._clear_current_word()
        
        return {'action': 'unknown_gesture', 'gesture': gesture}
    
    def _delete_last_character(self) -> Dict:
        """Delete last character from current word"""
        if self.state.current_word:
            self.state.current_word = self.state.current_word[:-1]
            # Update suggestions
            self.state.active_suggestions = self.dictionary.get_suggestions(self.state.current_word)
        
        return {
            'action': 'character_deleted',
            'current_word': self.state.current_word,
            'suggestions': self.state.active_suggestions,
            'status': 'modified'
        }
    
    def _clear_current_word(self) -> Dict:
        """Clear current word"""
        self.state.current_word = ''
        self.state.active_suggestions = []
        
        return {
            'action': 'word_cleared',
            'current_word': '',
            'suggestions': [],
            'status': 'cleared'
        }
    
    def get_full_text(self) -> str:
        """Get complete accumulated text"""
        if self.state.current_word:
            return f"{self.state.completed_text} {self.state.current_word}".strip()
        return self.state.completed_text.strip()
    
    def _speak_letter(self, letter: str):
        """Speak letter using TTS"""
        self.tts.speak(letter)
    
    def _speak_word(self, word: str):
        """Speak word using TTS"""
        self.tts.speak(word)
    
    def speak_full_text(self):
        """Speak complete accumulated text"""
        text = self.get_full_text()
        if text.strip():
            print(f"TTS: Queuing text '{text}'")
            self.tts.speak(text)
        else:
            print("No text to speak")
    
    def _save_letter_detection(self, detection: LetterDetection, confirmed: bool = False):
        """Save letter detection to database"""
        # Implementation would save to database
        pass
    
    def _save_word_completion(self, partial: str, completed: str, method: str):
        """Save word completion to database"""
        # Implementation would save to database
        pass
    
    def get_session_statistics(self) -> Dict:
        """Get current session statistics"""
        duration = datetime.now() - self.session_stats['session_start']
        
        return {
            'session_duration': duration.total_seconds(),
            'letters_detected': self.session_stats['letters_detected'],
            'words_completed': self.session_stats['words_completed'],
            'letters_per_minute': self.session_stats['letters_detected'] / max(1, duration.total_seconds() / 60),
            'words_per_minute': self.session_stats['words_completed'] / max(1, duration.total_seconds() / 60),
            'current_word_length': len(self.state.current_word),
            'total_text_length': len(self.get_full_text()),
            'active_suggestions': len(self.state.active_suggestions)
        }
    
    def reset_session(self):
        """Reset current session"""
        self.state = AccumulationState()
        self.session_stats = {
            'letters_detected': 0,
            'words_completed': 0,
            'errors_corrected': 0,
            'session_start': datetime.now()
        }

class GestureDetector:
    """Detect special gestures for commands"""
    
    def __init__(self):
        self.gesture_buffer = deque(maxlen=30)  # 1 second buffer
    
    def detect_gesture(self, hand_features: Dict) -> Optional[str]:
        """Detect gesture from hand features"""
        # Add current features to buffer
        self.gesture_buffer.append(hand_features)
        
        # Check for specific gestures
        if self._is_flat_hand_horizontal():
            return "space"
        elif self._is_delete_gesture():
            return "delete"
        elif self._is_completion_gesture():
            return "complete_word"
        
        return None
    
    def _is_flat_hand_horizontal(self) -> bool:
        """Detect flat hand moving horizontally for space"""
        if len(self.gesture_buffer) < 10:
            return False
        
        recent = list(self.gesture_buffer)[-10:]
        flat_count = sum(1 for f in recent if f.get('is_flat_hand', False))
        
        return flat_count >= 8  # 80% of recent frames
    
    def _is_delete_gesture(self) -> bool:
        """Detect delete gesture (thumbs down or specific hand shape)"""
        if len(self.gesture_buffer) < 5:
            return False
        
        recent = list(self.gesture_buffer)[-5:]
        delete_count = sum(1 for f in recent if f.get('is_thumbs_down', False))
        
        return delete_count >= 3
    
    def _is_completion_gesture(self) -> bool:
        """Detect word completion gesture"""
        if len(self.gesture_buffer) < 5:
            return False
        
        recent = list(self.gesture_buffer)[-5:]
        complete_count = sum(1 for f in recent if f.get('is_ok_sign', False))
        
        return complete_count >= 3


# Streamlit interface functions
def show_accumulation_interface():
    """Show letter accumulation interface in Streamlit"""
    st.title("📝 ASL Letter Accumulation System")
    
    # Initialize engine
    if 'accumulation_engine' not in st.session_state:
        st.session_state.accumulation_engine = LetterAccumulationEngine()
    
    engine = st.session_state.accumulation_engine
    
    # Display current state
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Accumulated Text")
        
        # Show completed text
        completed_text = engine.state.completed_text
        if completed_text:
            st.markdown(f"**Completed:** {completed_text}")
        
        # Show current word being built
        current_word = engine.state.current_word
        if current_word:
            st.markdown(f"**Building:** {current_word}")
            
            # Show progress bar for current letter
            if engine.state.confirmation_progress > 0:
                st.progress(engine.state.confirmation_progress)
                st.caption(f"Confirming letter... {engine.state.confirmation_progress:.1%}")
        
        # Full text display
        full_text = engine.get_full_text()
        if full_text:
            st.text_area("Complete Text", full_text, height=100)
    
    with col2:
        st.subheader("💡 Suggestions")
        
        # Show word suggestions
        if engine.state.active_suggestions:
            for i, suggestion in enumerate(engine.state.active_suggestions):
                if st.button(f"✅ {suggestion.word}", key=f"suggest_{i}"):
                    result = engine.complete_word_manually(suggestion.word)
                    st.rerun()
        else:
            st.info("Type more letters for suggestions")
        
        # Manual controls
        st.subheader("🎮 Controls")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add Space"):
                engine._add_space()
                st.rerun()
        
        with col_b:
            if st.button("🗑️ Clear Word"):
                engine._clear_current_word()
                st.rerun()
        
        if st.button("🔊 Read All Text"):
            engine.speak_full_text()
        
        if st.button("🔄 Reset Session"):
            engine.reset_session()
            st.rerun()
    
    # Statistics
    with st.expander("📊 Session Statistics"):
        stats = engine.get_session_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Letters", stats['letters_detected'])
        with col2:
            st.metric("Words", stats['words_completed'])
        with col3:
            st.metric("WPM", f"{stats['words_per_minute']:.1f}")


if __name__ == "__main__":
    show_accumulation_interface()