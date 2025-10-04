"""
ASL Reference Images Dataset Loader
Loads ASL alphabet reference images for training - deployment ready
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import configuration
try:
    from config.settings import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

class ASLReferenceDatasetLoader:
    """Load ASL reference images for model training - production ready"""
    
    def __init__(self, custom_path: str = None):
        """
        Initialize the ASL reference dataset loader
        
        Args:
            custom_path: Custom path to images folder (optional)
        """
        if custom_path:
            # Use custom path if provided
            self.images_folder = Path(custom_path)
        elif CONFIG_AVAILABLE:
            # Use configuration system
            self.images_folder = config.get_asl_images_path()
        else:
            # Fallback: look for images in relative paths
            self.images_folder = self._find_asl_images_folder()
        
        self.metadata_file = self.images_folder / "letter_metadata.json"
        
        # ASL alphabet letters (excluding J and Z which are motion-based)
        self.asl_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
        ]
        
        # Create letter to index mapping
        self.letter_to_index = {letter: idx for idx, letter in enumerate(self.asl_letters)}
        self.index_to_letter = {idx: letter for letter, idx in self.letter_to_index.items()}
        
        logger.info(f"Initialized ASL dataset loader")
        logger.info(f"Images folder: {self.images_folder}")
        logger.info(f"Available letters: {len(self.asl_letters)}")
    
    def _find_asl_images_folder(self) -> Path:
        """Find ASL images folder using relative paths"""
        current_dir = Path(__file__).parent
        
        # Possible relative locations
        possible_paths = [
            current_dir / ".." / "assets" / "asl_reference_images",  # Production
            current_dir / ".." / "asl_reference_images",  # Alternative
            current_dir / ".." / ".." / "Wave" / "asl_reference_images",  # Development
            current_dir / "asl_reference_images",  # Same directory
            Path("assets") / "asl_reference_images",  # Relative to working directory
            Path("Wave") / "asl_reference_images"  # Relative to working directory
        ]
        
        for path in possible_paths:
            abs_path = path.resolve()
            if abs_path.exists() and abs_path.is_dir():
                logger.info(f"Found ASL images at: {abs_path}")
                return abs_path
        
        # Default to production expected path
        default_path = current_dir / ".." / "assets" / "asl_reference_images"
        logger.warning(f"ASL images not found, using default: {default_path}")
        return default_path
    
    def load_metadata(self) -> Dict:
        """Load metadata about the ASL reference images"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} letters")
                return metadata
            else:
                logger.warning(f"Metadata file not found: {self.metadata_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def load_reference_images(self, target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all ASL reference images and prepare them for training
        
        Args:
            target_size: Target size for resizing images (width, height)
            
        Returns:
            Tuple of (images, labels, letter_names)
            - images: numpy array of shape (n_samples, height, width, channels)
            - labels: numpy array of integer labels
            - letter_names: list of letter names corresponding to labels
        """
        images = []
        labels = []
        letter_names = []
        
        metadata = self.load_metadata()
        
        for letter in self.asl_letters:
            # Construct image path
            image_filename = f"letter_{letter}.png"
            image_path = self.images_folder / image_filename
            
            if image_path.exists():
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Could not load image: {image_path}")
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    image = cv2.resize(image, target_size)
                    
                    # Normalize pixel values to [0, 1]
                    image = image.astype(np.float32) / 255.0
                    
                    # Add to arrays
                    images.append(image)
                    labels.append(self.letter_to_index[letter])
                    letter_names.append(letter)
                    
                    logger.debug(f"Loaded image for letter {letter}")
                    
                except Exception as e:
                    logger.error(f"Error loading image for letter {letter}: {e}")
                    continue
            else:
                logger.warning(f"Image not found for letter {letter}: {image_path}")
        
        # Convert to numpy arrays
        if images:
            images = np.array(images)
            labels = np.array(labels)
            logger.info(f"Successfully loaded {len(images)} reference images")
            logger.info(f"Images shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
        else:
            logger.error("No images were loaded!")
            images = np.array([])
            labels = np.array([])
        
        return images, labels, letter_names
    
    def create_augmented_dataset(self, images: np.ndarray, labels: np.ndarray, 
                                augment_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an augmented dataset by applying transformations to reference images
        
        Args:
            images: Original reference images
            labels: Corresponding labels
            augment_factor: Number of augmented versions per original image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        if len(images) == 0:
            return images, labels
        
        augmented_images = []
        augmented_labels = []
        
        # Keep original images
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        
        # Generate augmented versions
        for i, (image, label) in enumerate(zip(images, labels)):
            for _ in range(augment_factor):
                augmented_image = self._augment_image(image)
                augmented_images.append(augmented_image)
                augmented_labels.append(label)
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        logger.info(f"Created augmented dataset: {augmented_images.shape[0]} images")
        return augmented_images, augmented_labels
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to an image
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip((augmented - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # Random gaussian noise
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        
        return augmented
    
    def prepare_training_data(self, test_size: float = 0.2, augment_factor: int = 10, 
                            target_size: Tuple[int, int] = (224, 224)) -> Dict:
        """
        Prepare complete training dataset with train/validation split
        
        Args:
            test_size: Fraction of data to use for validation
            augment_factor: Number of augmented versions per reference image
            target_size: Target size for images
            
        Returns:
            Dictionary containing train and validation data
        """
        # Load reference images
        images, labels, letter_names = self.load_reference_images(target_size)
        
        if len(images) == 0:
            logger.error("No images loaded, cannot prepare training data")
            return {}
        
        # Create augmented dataset
        aug_images, aug_labels = self.create_augmented_dataset(images, labels, augment_factor)
        
        # Split into train and validation sets
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            aug_images, aug_labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=aug_labels
        )
        
        training_data = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'letter_names': letter_names,
            'num_classes': len(self.asl_letters),
            'class_names': self.asl_letters,
            'letter_to_index': self.letter_to_index,
            'index_to_letter': self.index_to_letter
        }
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        logger.info(f"  Classes: {len(self.asl_letters)}")
        
        return training_data
    
    def get_class_info(self) -> Dict:
        """Get information about ASL classes"""
        return {
            'num_classes': len(self.asl_letters),
            'class_names': self.asl_letters,
            'letter_to_index': self.letter_to_index,
            'index_to_letter': self.index_to_letter
        }


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing ASL Reference Dataset Loader...")
    
    # Test with auto-detection
    loader = ASLReferenceDatasetLoader()
    print(f"Using images folder: {loader.images_folder}")
    
    # Load reference images
    images, labels, letter_names = loader.load_reference_images()
    print(f"Loaded {len(images)} reference images")
    
    if len(images) > 0:
        print(f"Available letters: {letter_names}")
        
        # Prepare training data
        training_data = loader.prepare_training_data(augment_factor=5)
        if training_data:
            print(f"Training set: {training_data['X_train'].shape}")
            print(f"Validation set: {training_data['X_val'].shape}")
            print(f"Classes: {training_data['class_names']}")
    else:
        print("⚠️  No images found. Make sure ASL reference images are available.")
        print("Expected files: letter_A.png, letter_B.png, etc.")