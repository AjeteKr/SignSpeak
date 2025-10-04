"""
Configuration management for SignSpeak
"""
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
        
    def _load_config(self):
        """Load configuration files"""
        try:
            config_dir = Path(__file__).parent.parent / 'config'
            
            # Load settings
            with open(config_dir / 'settings.json') as f:
                self.settings = json.load(f)
                
            # Load model configurations
            with open(config_dir / 'models.json') as f:
                self.models = json.load(f)
                
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration: {e}")
            raise
            
    def get_setting(self, *keys, default=None):
        """
        Get a setting value using dot notation
        Example: config.get_setting('video', 'frame_interval', default=0.5)
        """
        value = self.settings
        for key in keys:
            try:
                value = value[key]
            except (KeyError, TypeError):
                return default
        return value
        
    def get_model_config(self, model_name):
        """Get configuration for a specific model"""
        return self.models.get(model_name)
