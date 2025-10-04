import os
import json
from datetime import datetime
from pathlib import Path

class UserAuthManager:
    def __init__(self):
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'user_data'
        self.base_path.mkdir(exist_ok=True)
        
    def register_user(self, username, password, name=None):
        """Register a new user"""
        user_file = self.base_path / f"{username}.json"
        if user_file.exists():
            return False, "Username already exists"
            
        user_data = {
            "username": username,
            "password": password,  # In production, use proper password hashing
            "name": name or username,
            "created_at": datetime.now().isoformat(),
            "calibration_data": {},
            "settings": {
                "hand_size_factor": 1.0,
                "position_offset": [0.0, 0.0],
                "confidence_threshold": 0.7
            }
        }
        
        with open(user_file, 'w') as f:
            json.dump(user_data, f, indent=2)
        return True, "User registered successfully"
    
    def login(self, username, password):
        """Login user"""
        user_file = self.base_path / f"{username}.json"
        if not user_file.exists():
            return False, "User not found"
            
        with open(user_file, 'r') as f:
            user_data = json.load(f)
            
        if user_data["password"] == password:  # In production, use proper password verification
            return True, user_data
        return False, "Invalid password"
    
    def save_calibration(self, username, letter, calibration_data):
        """Save calibration data for a user"""
        user_file = self.base_path / f"{username}.json"
        if not user_file.exists():
            return False, "User not found"
            
        with open(user_file, 'r') as f:
            user_data = json.load(f)
        
        user_data["calibration_data"][letter] = calibration_data
        user_data["last_updated"] = datetime.now().isoformat()
        
        with open(user_file, 'w') as f:
            json.dump(user_data, f, indent=2)
        return True, "Calibration saved"
    
    def get_user_settings(self, username):
        """Get user's calibration settings"""
        user_file = self.base_path / f"{username}.json"
        if not user_file.exists():
            return None
            
        with open(user_file, 'r') as f:
            user_data = json.load(f)
        return user_data.get("settings")
