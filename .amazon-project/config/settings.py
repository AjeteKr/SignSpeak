"""
Configuration settings for SignSpeak Pro
Handles paths and settings for both development and production environments
"""

import os
from pathlib import Path
from typing import Dict, Any

class SignSpeakConfig:
    """Configuration class for SignSpeak Pro application"""
    
    def __init__(self):
        # Base directories - ensure we get the main project directory
        config_dir = Path(__file__).parent  # config folder
        self.app_root = config_dir.parent   # main project folder (.amazon-project)
        self.project_root = self.app_root.parent if self.app_root.name == ".amazon-project" else self.app_root
        
        # Environment detection
        self.is_development = self._detect_development_mode()
        
        # Set up paths based on environment
        self._setup_paths()
        
        # Model configuration
        self.model_config = {
            'confidence_threshold': 0.7,
            'voting_window': 5,
            'max_fps': 30,
            'input_shape': (224, 224, 3),
            'num_classes': 24  # A-Y excluding J,Z
        }
    
    def _detect_development_mode(self) -> bool:
        """Detect if running in development or production mode"""
        # Check for development indicators
        dev_indicators = [
            os.path.exists(self.app_root / '.git'),  # Git repository
            'Desktop' in str(self.app_root),  # Desktop path
            'BesiComputers' in str(self.app_root),  # Development machine
            os.environ.get('STREAMLIT_ENV') == 'development'
        ]
        return any(dev_indicators)
    
    def _setup_paths(self):
        """Setup paths for different environments"""
        if self.is_development:
            # Development paths - try multiple locations
            possible_dev_paths = [
                self.project_root / "Wave" / "asl_reference_images",  # Original dev location
                self.app_root / "assets" / "asl_reference_images",   # Copied to assets
                self.app_root / ".." / "Wave" / "asl_reference_images"  # Relative to project
            ]
            
            self.asl_images_folder = None
            for path in possible_dev_paths:
                if path.exists():
                    self.asl_images_folder = path
                    break
            
            # Fallback to assets if nothing found
            if self.asl_images_folder is None:
                self.asl_images_folder = self.app_root / "assets" / "asl_reference_images"
                
            self.wave_folder = self.project_root / "Wave"
        else:
            # Production paths - ASL images should be included in the deployment
            self.asl_images_folder = self.app_root / "assets" / "asl_reference_images"
            self.wave_folder = self.app_root / "backend"  # Backend data in production
        
        # Common paths
        self.models_folder = self.app_root / "models"
        self.data_folder = self.app_root / "data"
        self.logs_folder = self.app_root / "logs"
        self.temp_folder = self.app_root / "temp"
        
        # Database paths
        if self.is_development:
            self.database_folder = self.app_root
        else:
            self.database_folder = self.app_root / "data"
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.models_folder,
            self.data_folder,
            self.logs_folder,
            self.temp_folder,
            self.database_folder
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_asl_images_path(self) -> Path:
        """Get the path to ASL reference images"""
        if self.asl_images_folder.exists():
            return self.asl_images_folder
        
        # Fallback: look for images in various locations
        possible_paths = [
            self.app_root / "assets" / "asl_reference_images",
            self.app_root / "asl_reference_images",
            self.app_root / "Wave" / "asl_reference_images",
            self.project_root / "Wave" / "asl_reference_images"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If no images found, return the preferred production path
        return self.app_root / "assets" / "asl_reference_images"
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a specific model"""
        return self.models_folder / model_name
    
    def get_database_path(self, db_name: str) -> Path:
        """Get path for a database file"""
        return self.database_folder / db_name
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'app_root': str(self.app_root),
            'asl_images_path': str(self.get_asl_images_path()),
            'models_folder': str(self.models_folder),
            'is_development': self.is_development,
            'model_config': self.model_config
        }

# Global configuration instance
config = SignSpeakConfig()

# For backward compatibility and easy access
def get_asl_images_path() -> str:
    """Get ASL images path as string"""
    return str(config.get_asl_images_path())

def get_models_folder() -> str:
    """Get models folder path as string"""
    return str(config.models_folder)

def is_development_mode() -> bool:
    """Check if running in development mode"""
    return config.is_development