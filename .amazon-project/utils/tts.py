"""
Text-to-Speech utility for SignSpeak
"""
import pyttsx3
import threading
import logging

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        """Initialize TTS engine"""
        self.engine = None
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the TTS engine safely"""
        try:
            self.engine = pyttsx3.init()
            # Configure properties
            self.engine.setProperty('rate', 150)    # Speaking rate
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
            
    def speak(self, text: str):
        """
        Speak text asynchronously
        
        Args:
            text: Text to speak
        """
        if not text or not self.engine:
            return
            
        def tts_worker():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                # Try to reinitialize engine
                self._init_engine()
                
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=tts_worker)
        thread.daemon = True
        thread.start()
        
    def stop(self):
        """Stop current TTS and cleanup"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
