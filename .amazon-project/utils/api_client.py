"""
SignSpeak API Client

Handles all communication with the Wave API server, including:
- ASL recognition and prediction requests
- Authentication and user management
- Feature processing and error handling
"""
import requests
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json
import time
import tempfile
import os
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

class SignSpeakClient:
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 120, max_retries: int = 3):
        """
        Initialize SignSpeak API client
        
        Args:
            api_url: Base URL for the Wave API server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.api_url = api_url
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def test_connection(self) -> bool:
        """Test connection to the API server"""
        try:
            response = self.session.get(
                f"{self.api_url}/health",
                timeout=5  # Shorter timeout for health checks
            )
            return response.status_code == 200
        except RequestException as e:
            logger.warning(f"API connection test failed: {e}")
            return False
            
    def get_health(self) -> Optional[Dict]:
        """Get API server health status"""
        return self._make_request("get", "/health", timeout=5)
        
    def get_model_info(self) -> Optional[Dict]:
        """Get information about loaded models"""
        return self._make_request("get", "/model/info")
        
    def get_expressions(self) -> List[str]:
        """Get list of supported ASL expressions"""
        response = self._make_request("get", "/expressions")
        return response.get("expressions", []) if response else []
            
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """
        Make an API request with proper error handling
        
        Args:
            method: HTTP method (get, post, etc)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
        
        Returns:
            JSON response or None if request failed
        """
        try:
            kwargs['timeout'] = kwargs.get('timeout', self.timeout)
            response = self.session.request(
                method,
                f"{self.api_url}/{endpoint.lstrip('/')}",
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
            
    def predict_video_file(self, video_file) -> Optional[Dict]:
        """
        Predict ASL expression from video file
        
        Args:
            video_file: File-like object or path to video file
            
        Returns:
            Prediction results or None if request failed
        """
        try:
            if hasattr(video_file, 'read'):
                # File-like object (e.g., UploadedFile from Streamlit)
                # Reset to beginning of file
                video_file.seek(0)
                files = {"file": (video_file.name, video_file, video_file.type)}
                
                response = self.session.post(
                    f"{self.api_url}/predict/video",
                    files=files,
                    timeout=180  # 3 minutes for video processing
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Prediction successful: {result}")
                return result
            else:
                # File path
                with open(video_file, 'rb') as f:
                    files = {"file": f}
                    response = self.session.post(
                        f"{self.api_url}/predict/video",
                        files=files,
                        timeout=180  # 3 minutes for video processing
                    )
                    response.raise_for_status()
                    result = response.json()
                    logger.info(f"Prediction successful: {result}")
                    return result
            
        except RequestException as e:
            logger.error(f"Error predicting video: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in predict_video_file: {e}")
            return None
    
    def extract_features_from_video(self, video_file) -> Optional[Dict]:
        """
        Extract features from video file without prediction
        
        Args:
            video_file: File-like object or path to video file
            
        Returns:
            Feature extraction results or None if request failed
        """
        try:
            if hasattr(video_file, 'read'):
                # File-like object (e.g., UploadedFile from Streamlit)
                video_file.seek(0)
                files = {"file": (video_file.name, video_file, video_file.type)}
                
                response = self.session.post(
                    f"{self.api_url}/extract/features",
                    files=files,
                    timeout=180  # 3 minutes for feature extraction
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Feature extraction successful: {result}")
                return result
            else:
                # File path
                with open(video_file, 'rb') as f:
                    files = {"file": f}
                    response = self.session.post(
                        f"{self.api_url}/extract/features",
                        files=files,
                        timeout=180  # 3 minutes for feature extraction
                    )
                    response.raise_for_status()
                    result = response.json()
                    logger.info(f"Feature extraction successful: {result}")
                    return result
            
        except RequestException as e:
            logger.error(f"Error extracting features: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in extract_features_from_video: {e}")
            return None
            logger.error(f"Error extracting features: {e}")
            return None
    
    def detect_expression_from_frames(self, frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Detect ASL expression from a sequence of video frames.
        Converts frames to video and sends to Wave API.

        Args:
            frames: List of numpy arrays containing video frames

        Returns:
            Dict containing detected expression and confidence, or None if error
        """
        try:
            if not frames:
                logger.error("No frames provided")
                return None

            # Create temporary video file from frames
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Convert frames to video using cv2
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            try:
                # Send video to API
                result = self.predict_video_file(temp_path)
                return result
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error detecting expression from frames: {e}")
            return None
    
    # Authentication methods
    def login(self, username: str, password: str) -> Optional[Dict]:
        """User login"""
        data = {"username": username, "password": password}
        return self._make_request("post", "/login", json=data)
    
    def register(self, username: str, password: str, name: str = None) -> Optional[Dict]:
        """User registration"""
        data = {"username": username, "password": password}
        if name:
            data["name"] = name
        return self._make_request("post", "/register", json=data)
    
    def get_user_profile(self, username: str) -> Optional[Dict]:
        """Get user profile information"""
        return self._make_request("get", f"/user/{username}")
    
    # Legacy methods for backward compatibility
    def predict_expression(self, features: np.ndarray, model: str = "expression") -> Optional[Dict]:
        """Legacy method - redirects to video prediction"""
        logger.warning("predict_expression is deprecated. Use predict_video_file instead.")
        return None
        
    def get_model_info_legacy(self, model: str) -> Optional[Dict]:
        """Legacy method - redirects to new model info endpoint"""
        return self.get_model_info()
        
    def get_supported_models(self) -> List[str]:
        """Legacy method - returns ASL expressions"""
        return self.get_expressions()
        
    def detect_expression(self, frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Legacy method - redirects to new frame detection"""
        return self.detect_expression_from_frames(frames)
