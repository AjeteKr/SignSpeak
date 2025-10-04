"""
Enhanced Dataset Manager for ASL Recognition
Supports MNIST ASL + ASL Alphabet datasets with unified preprocessing pipeline
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import logging

# Optional imports with fallbacks
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not ALBUMENTATIONS_AVAILABLE:
    logger.warning("albumentations not available, using basic augmentation")

class EnhancedDatasetManager:
    """
    Comprehensive dataset management with MediaPipe integration
    Supports both MNIST ASL and ASL Alphabet datasets
    """
    
    def __init__(self, 
                 data_root: str = "datasets",
                 target_size: Tuple[int, int] = (224, 224),
                 hand_padding: float = 1.2):
        """
        Initialize dataset manager
        
        Args:
            data_root: Root directory for all datasets
            target_size: Target image size for preprocessing (224x224 for CNN)
            hand_padding: Padding factor around detected hand (1.2 = 20% padding)
        """
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.hand_padding = hand_padding
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Data paths
        self.mnist_path = self.data_root / "sign-language-mnist"
        self.asl_alphabet_path = self.data_root / "asl-alphabet"
        
        # ASL Reference Images (your existing dataset)
        # Try multiple possible paths to find the ASL reference images
        possible_paths = [
            Path("../Wave/asl_reference_images"),  # Original relative path
            Path("../../Wave/asl_reference_images"),  # In case we're deeper
            Path("c:/Users/BesiComputers/Desktop/Amazon/Wave/asl_reference_images"),  # Absolute path
            Path(__file__).parent.parent.parent / "Wave" / "asl_reference_images"  # Relative to this file
        ]
        
        self.asl_reference_path = None
        for path in possible_paths:
            if path.exists():
                self.asl_reference_path = path
                logger.info(f"Found ASL reference images at: {path}")
                break
        
        if self.asl_reference_path is None:
            # Default to first path for error messages
            self.asl_reference_path = possible_paths[0]
            logger.warning(f"ASL reference images not found at any expected location")
        
        self.processed_path = self.data_root / "processed"
        self.landmarks_path = self.data_root / "landmarks"
        
        # Create directories
        self._create_directories()
        
        # Initialize data structures
        self.label_encoder = LabelEncoder()
        self.class_mapping = {}
        self.dataset_stats = {}
        self._used_simple_loading = False
        
        # Augmentation pipeline
        if ALBUMENTATIONS_AVAILABLE:
            self.augmentation = A.Compose([
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.HorizontalFlip(p=0.5),  # Only for some letters
                A.Blur(blur_limit=3, p=0.1),
            ])
        else:
            self.augmentation = None
        
    def _create_directories(self):
        """Create necessary directories for data processing"""
        directories = [
            self.data_root,
            self.processed_path,
            self.landmarks_path,
            self.processed_path / "train",
            self.processed_path / "validation", 
            self.processed_path / "test",
            self.landmarks_path / "train",
            self.landmarks_path / "validation",
            self.landmarks_path / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created directory structure in {self.data_root}")
    
    def download_datasets(self):
        """
        Download and extract datasets
        Note: This assumes datasets are manually downloaded from Kaggle
        """
        # Check if datasets exist
        mnist_exists = (self.mnist_path / "sign_mnist_train.csv").exists()
        asl_exists = (self.asl_alphabet_path / "asl_alphabet_train").exists()
        asl_reference_exists = self.asl_reference_path.exists() and len(list(self.asl_reference_path.glob("letter_*.png"))) > 0
        
        if not mnist_exists:
            logger.warning(f"MNIST ASL dataset not found at {self.mnist_path}")
            logger.info("Please download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
            
        if not asl_exists and not asl_reference_exists:
            logger.warning(f"ASL Alphabet dataset not found at {self.asl_alphabet_path}")
            logger.info("Please download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        elif asl_reference_exists:
            num_letters = len(list(self.asl_reference_path.glob("letter_*.png")))
            logger.info(f"✅ Using ASL reference images: {num_letters} letters found in {self.asl_reference_path}")
            
        return mnist_exists and asl_exists
    
    def load_mnist_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load MNIST ASL dataset (CSV format)
        Returns: X_train, y_train, X_test, y_test
        """
        try:
            # Load training data
            train_df = pd.read_csv(self.mnist_path / "sign_mnist_train.csv")
            test_df = pd.read_csv(self.mnist_path / "sign_mnist_test.csv")
            
            # Extract features and labels
            X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
            y_train = train_df['label'].values
            
            X_test = test_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
            y_test = test_df['label'].values
            
            # Normalize pixel values
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            
            # Convert to RGB (repeat grayscale channel)
            X_train = np.repeat(X_train, 3, axis=-1)
            X_test = np.repeat(X_test, 3, axis=-1)
            
            # Resize to target size
            X_train_resized = np.array([cv2.resize(img, self.target_size) for img in X_train])
            X_test_resized = np.array([cv2.resize(img, self.target_size) for img in X_test])
            
            logger.info(f"Loaded MNIST data: Train {X_train_resized.shape}, Test {X_test_resized.shape}")
            return X_train_resized, y_train, X_test_resized, y_test
            
        except Exception as e:
            logger.error(f"Error loading MNIST data: {e}")
            return None, None, None, None
    
    def load_asl_alphabet_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ASL Alphabet dataset (image folders)
        Returns: X, y
        """
        try:
            images = []
            labels = []
            
            # Define alphabet mapping (A=0, B=1, ..., Z=25)
            alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            # Load from train directory
            train_path = self.asl_alphabet_path / "asl_alphabet_train"
            
            for letter in alphabet:
                letter_path = train_path / letter
                if letter_path.exists():
                    logger.info(f"Loading letter {letter} from {letter_path}")
                    
                    for img_file in letter_path.glob("*.jpg"):
                        try:
                            # Load and preprocess image
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                # Apply hand detection and cropping
                                processed_img = self._process_single_image(img)
                                
                                if processed_img is not None:
                                    images.append(processed_img)
                                    labels.append(alphabet.index(letter))
                                    
                        except Exception as e:
                            logger.warning(f"Error processing {img_file}: {e}")
                            continue
            
            if len(images) == 0:
                logger.info("No ASL Alphabet images found in Kaggle dataset structure")
                return None, None
                
            X = np.array(images)
            y = np.array(labels)
            
            logger.info(f"Loaded ASL Alphabet data: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading ASL Alphabet data: {e}")
            return None, None
    
    def load_asl_reference_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ASL Reference Images (your existing dataset)
        Returns: X, y
        """
        try:
            images = []
            labels = []
            
            logger.info(f"Trying to load ASL reference images from: {self.asl_reference_path}")
            
            if not self.asl_reference_path.exists():
                logger.error(f"ASL reference path not found: {self.asl_reference_path}")
                logger.info(f"Current working directory: {os.getcwd()}")
                return None, None
            
            # Get all letter images
            letter_files = list(self.asl_reference_path.glob("letter_*.png"))
            logger.info(f"Found {len(letter_files)} ASL reference image files")
            
            if len(letter_files) == 0:
                logger.error(f"No letter_*.png files found in {self.asl_reference_path}")
                # List what files are actually there
                all_files = list(self.asl_reference_path.iterdir())
                logger.info(f"Files in directory: {[f.name for f in all_files]}")
                return None, None
            
            processed_count = 0
            for img_file in letter_files:
                try:
                    # Extract letter from filename (e.g., "letter_A.png" -> "A")
                    letter = img_file.stem.split('_')[1]  # "letter_A" -> "A"
                    
                    # Skip J and Z as they're not in static ASL alphabet
                    if letter in ['J', 'Z']:
                        continue
                    
                    # Load and preprocess image
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Apply hand detection and cropping
                        processed_img = self._process_single_image(img)
                        
                        if processed_img is not None:
                            images.append(processed_img)
                            # Convert letter to index (A=0, B=1, etc., skip J)
                            label_idx = ord(letter) - ord('A')
                            if letter > 'J':  # Adjust for missing J
                                label_idx -= 1
                            labels.append(label_idx)
                            processed_count += 1
                            logger.info(f"Successfully processed reference image for letter {letter} (index {label_idx})")
                        else:
                            logger.warning(f"MediaPipe hand detection failed for letter {letter} - image will be skipped")
                    else:
                        logger.error(f"Could not load image file: {img_file}")
                            
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
            
            logger.info(f"Processing complete: {processed_count}/{len(letter_files)} images successfully processed")
            
            if len(images) == 0:
                logger.error("No valid ASL reference images loaded - likely MediaPipe hand detection failed on all images")
                logger.info("💡 This might be because the reference images don't have detectable hands or are in wrong format")
                return None, None
                
            X = np.array(images)
            y = np.array(labels)
            
            # Set flag to indicate regular loading was used
            self._used_simple_loading = False
            
            logger.info(f"✅ Successfully loaded ASL Reference data: {X.shape}, Letters: {len(set(labels))}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading ASL Reference data: {e}")
            return None, None
    
    def load_asl_reference_data_simple(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ASL Reference Images WITHOUT MediaPipe processing (for debugging)
        Just resizes images to target size without hand detection
        Returns: X, y
        """
        try:
            images = []
            labels = []
            
            logger.info(f"Loading ASL reference images (simple mode) from: {self.asl_reference_path}")
            
            if not self.asl_reference_path.exists():
                logger.error(f"ASL reference path not found: {self.asl_reference_path}")
                return None, None
            
            # Get all letter images
            letter_files = list(self.asl_reference_path.glob("letter_*.png"))
            logger.info(f"Found {len(letter_files)} ASL reference image files")
            
            if len(letter_files) == 0:
                logger.error(f"No letter_*.png files found in {self.asl_reference_path}")
                return None, None
            
            for img_file in letter_files:
                try:
                    # Extract letter from filename
                    letter = img_file.stem.split('_')[1]  # "letter_A" -> "A"
                    
                    # Skip J and Z
                    if letter in ['J', 'Z']:
                        continue
                    
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Simple resize without hand detection
                        img_resized = cv2.resize(img, self.target_size)
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        
                        images.append(img_normalized)
                        
                        # Convert letter to index
                        label_idx = ord(letter) - ord('A')
                        if letter > 'J':  # Adjust for missing J
                            label_idx -= 1
                        labels.append(label_idx)
                        
                        logger.info(f"Loaded (simple): {letter} -> index {label_idx}")
                    else:
                        logger.error(f"Could not load image: {img_file}")
                        
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
            
            if len(images) == 0:
                logger.error("No images loaded in simple mode")
                return None, None
            
            X = np.array(images)
            y = np.array(labels)
            
            # Set flag to indicate simple loading was used
            self._used_simple_loading = True
            
            logger.info(f"✅ Simple mode loaded: {X.shape}, Labels: {len(set(labels))}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in simple ASL loading: {e}")
            return None, None
    
    def _process_single_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Process single image with MediaPipe hand detection and cropping
        
        Args:
            image: Input RGB image
            
        Returns:
            Processed image normalized to target size, or None if no hand detected
        """
        try:
            # Detect hand landmarks
            results = self.hands.process(image)
            
            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract landmark coordinates
                h, w = image.shape[:2]
                landmarks = []
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                # Calculate bounding box with padding
                x_min, y_min = landmarks.min(axis=0)
                x_max, y_max = landmarks.max(axis=0)
                
                # Add padding
                box_w = x_max - x_min
                box_h = y_max - y_min
                pad_w = int(box_w * (self.hand_padding - 1) / 2)
                pad_h = int(box_h * (self.hand_padding - 1) / 2)
                
                # Ensure bounds are within image
                x_min = max(0, x_min - pad_w)
                y_min = max(0, y_min - pad_h)
                x_max = min(w, x_max + pad_w)
                y_max = min(h, y_max + pad_h)
                
                # Crop hand region
                hand_crop = image[y_min:y_max, x_min:x_max]
                
                # Resize to target size
                if hand_crop.size > 0:
                    processed = cv2.resize(hand_crop, self.target_size)
                    return processed.astype('float32') / 255.0
                    
            return None
            
        except Exception as e:
            logger.warning(f"Error processing image: {e}")
            return None
    
    def extract_hand_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized hand landmarks from image
        
        Args:
            image: Input RGB image
            
        Returns:
            Normalized landmarks array (21 points × 3 coordinates) or None
        """
        try:
            results = self.hands.process(image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks, dtype=np.float32)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting landmarks: {e}")
            return None
    
    def create_unified_dataset(self, 
                             use_mnist: bool = True, 
                             use_asl_alphabet: bool = True,
                             validation_split: float = 0.2) -> Dict:
        """
        Create unified dataset from all sources with comprehensive preprocessing
        
        Args:
            use_mnist: Include MNIST ASL data
            use_asl_alphabet: Include ASL Alphabet data
            validation_split: Fraction for validation set
            
        Returns:
            Dictionary with train/validation/test splits and metadata
        """
        logger.info("Creating unified dataset...")
        
        all_images = []
        all_labels = []
        all_landmarks = []
        dataset_sources = []
        
        # Load MNIST data
        if use_mnist:
            logger.info("Processing MNIST ASL data...")
            X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = self.load_mnist_data()
            
            if X_train_mnist is not None:
                # Extract landmarks for MNIST data
                for i, img in enumerate(X_train_mnist):
                    landmarks = self.extract_hand_landmarks((img * 255).astype(np.uint8))
                    if landmarks is not None:
                        all_images.append(img)
                        all_labels.append(y_train_mnist[i])
                        all_landmarks.append(landmarks)
                        dataset_sources.append('mnist_train')
                
                # Process test set
                for i, img in enumerate(X_test_mnist):
                    landmarks = self.extract_hand_landmarks((img * 255).astype(np.uint8))
                    if landmarks is not None:
                        all_images.append(img)
                        all_labels.append(y_test_mnist[i])
                        all_landmarks.append(landmarks)
                        dataset_sources.append('mnist_test')
        
        # Load ASL Alphabet data
        if use_asl_alphabet:
            logger.info("🔄 Processing ASL Alphabet data...")
            
            # Try regular ASL alphabet dataset first
            X_asl, y_asl = self.load_asl_alphabet_data()
            
            # If not available, use ASL reference images
            if X_asl is None:
                logger.info("📁 Regular ASL alphabet dataset not found, trying ASL reference images with MediaPipe...")
                X_asl, y_asl = self.load_asl_reference_data()
                
                # If MediaPipe fails, try simple loading as fallback
                if X_asl is None:
                    logger.info("⚡ MediaPipe processing failed, falling back to simple loading...")
                    X_asl, y_asl = self.load_asl_reference_data_simple()
                    
                    if X_asl is not None:
                        logger.info("✅ Simple loading successful - using images without hand landmarks")
            
            if X_asl is not None:
                logger.info(f"Processing {len(X_asl)} ASL alphabet images...")
                
                # Check if we used simple loading (which might not have good landmarks)
                used_simple_loading = hasattr(self, '_used_simple_loading') and self._used_simple_loading
                
                for i, img in enumerate(X_asl):
                    if used_simple_loading:
                        # For simple loading, create dummy landmarks or skip landmark extraction
                        # Use zeros as placeholder landmarks for now
                        dummy_landmarks = np.zeros(63)  # 21 landmarks × 3 coordinates
                        all_images.append(img)
                        all_labels.append(y_asl[i])
                        all_landmarks.append(dummy_landmarks)
                        dataset_sources.append('asl_reference_simple')
                        logger.info(f"Added ASL image {i} with dummy landmarks (simple loading)")
                    else:
                        # Try to extract real landmarks
                        landmarks = self.extract_hand_landmarks((img * 255).astype(np.uint8))
                        if landmarks is not None:
                            all_images.append(img)
                            all_labels.append(y_asl[i])
                            all_landmarks.append(landmarks)
                            dataset_sources.append('asl_reference')
                            logger.info(f"Added ASL image {i} with real landmarks")
                        else:
                            logger.warning(f"Could not extract landmarks from ASL image {i} - skipping")
            else:
                logger.warning("No ASL alphabet data available (neither Kaggle dataset nor reference images)")
        
        # Check if we have any samples
        if len(all_images) == 0:
            logger.error("❌ No samples loaded! Cannot create dataset.")
            logger.error("This usually means:")
            logger.error("1. ASL reference images not found")
            logger.error("2. MediaPipe hand detection failed on all images")
            logger.error("3. Image loading failed")
            return None
        
        # Convert to arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        landmarks = np.array(all_landmarks)
        sources = np.array(dataset_sources)
        
        logger.info(f"Total unified dataset: {X.shape} images, {landmarks.shape} landmark sets")
        
        # Check minimum samples for train/test split
        if len(X) < 5:
            logger.warning(f"Only {len(X)} samples - skipping train/test split and using all data as train set.")
            # All data goes to train, empty val/test
            dataset = {
                'train': {
                    'images': X,
                    'labels': y,
                    'landmarks': landmarks,
                    'sources': sources
                },
                'validation': {
                    'images': np.array([]),
                    'labels': np.array([]),
                    'landmarks': np.array([]),
                    'sources': np.array([])
                },
                'test': {
                    'images': np.array([]),
                    'labels': np.array([]),
                    'landmarks': np.array([]),
                    'sources': np.array([])
                },
                'metadata': {
                    'num_classes': len(np.unique(y)),
                    'image_shape': X.shape[1:],
                    'landmark_shape': landmarks.shape[1:] if len(landmarks.shape) > 1 else (),
                    'class_names': list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    'total_samples': len(X),
                    'train_samples': len(X),
                    'val_samples': 0,
                    'test_samples': 0
                }
            }
            self._save_dataset_stats(dataset)
            return dataset
        
        # Create train/validation/test splits
        # Use stratified split to maintain class balance
        X_temp, X_test, y_temp, y_test, landmarks_temp, landmarks_test, sources_temp, sources_test = \
            train_test_split(X, y, landmarks, sources, 
                           test_size=0.2, stratify=y, random_state=42)
        
        X_train, X_val, y_train, y_val, landmarks_train, landmarks_val, sources_train, sources_val = \
            train_test_split(X_temp, y_temp, landmarks_temp, sources_temp,
                           test_size=validation_split, stratify=y_temp, random_state=42)
        
        # Create dataset dictionary
        dataset = {
            'train': {
                'images': X_train,
                'labels': y_train,
                'landmarks': landmarks_train,
                'sources': sources_train
            },
            'validation': {
                'images': X_val,
                'labels': y_val,
                'landmarks': landmarks_val,
                'sources': sources_val
            },
            'test': {
                'images': X_test,
                'labels': y_test,
                'landmarks': landmarks_test,
                'sources': sources_test
            },
            'metadata': {
                'num_classes': len(np.unique(y)),
                'image_shape': X.shape[1:],
                'landmark_shape': landmarks.shape[1:],
                'class_names': list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
        }
        
        # Save dataset statistics
        self._save_dataset_stats(dataset)
        
        return dataset
    
    def _save_dataset_stats(self, dataset: Dict):
        """Save comprehensive dataset statistics"""
        stats = {
            'total_samples': dataset['metadata']['total_samples'],
            'class_distribution': {},
            'source_distribution': {},
            'quality_metrics': {}
        }
        
        # Calculate class distribution
        for split in ['train', 'validation', 'test']:
            labels = dataset[split]['labels']
            unique, counts = np.unique(labels, return_counts=True)
            stats['class_distribution'][split] = dict(zip(unique.tolist(), counts.tolist()))
            
            # Source distribution
            sources = dataset[split]['sources']
            unique_sources, source_counts = np.unique(sources, return_counts=True)
            stats['source_distribution'][split] = dict(zip(unique_sources.tolist(), source_counts.tolist()))
        
        # Save stats
        stats_file = self.processed_path / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Dataset statistics saved to {stats_file}")
        self.dataset_stats = stats
    
    def apply_augmentation(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to image and corresponding landmarks
        
        Args:
            image: Input image (0-1 normalized)
            landmarks: Hand landmarks
            
        Returns:
            Augmented image and adjusted landmarks
        """
        try:
            # Convert to uint8 for augmentation
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Apply augmentation if available
            if self.augmentation is not None:
                augmented = self.augmentation(image=img_uint8)
                aug_image = augmented['image'].astype(np.float32) / 255.0
            else:
                # Simple fallback augmentation using OpenCV
                aug_image = self._basic_augmentation(img_uint8).astype(np.float32) / 255.0
            
            # Note: For more advanced augmentation, we would need to transform landmarks too
            # For now, returning original landmarks (can be enhanced later)
            
            return aug_image, landmarks
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return image, landmarks
    
    def _basic_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Basic augmentation using OpenCV when albumentations is not available"""
        import random
        
        # Random brightness adjustment
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random horizontal flip (for some letters)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def save_processed_dataset(self, dataset: Dict, format: str = 'numpy'):
        """
        Save processed dataset in specified format
        
        Args:
            dataset: Processed dataset dictionary
            format: Save format ('numpy', 'pickle', 'hdf5')
        """
        logger.info(f"Saving dataset in {format} format...")
        
        try:
            if format == 'numpy':
                for split in ['train', 'validation', 'test']:
                    split_path = self.processed_path / split
                    
                    np.save(split_path / 'images.npy', dataset[split]['images'])
                    np.save(split_path / 'labels.npy', dataset[split]['labels'])
                    np.save(split_path / 'landmarks.npy', dataset[split]['landmarks'])
                    np.save(split_path / 'sources.npy', dataset[split]['sources'])
            
            elif format == 'pickle':
                with open(self.processed_path / 'dataset.pkl', 'wb') as f:
                    pickle.dump(dataset, f)
            
            # Save metadata
            with open(self.processed_path / 'metadata.json', 'w') as f:
                json.dump(dataset['metadata'], f, indent=2)
                
            logger.info(f"Dataset saved successfully to {self.processed_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def load_processed_dataset(self, format: str = 'numpy') -> Optional[Dict]:
        """
        Load previously processed dataset
        
        Args:
            format: Load format ('numpy', 'pickle')
            
        Returns:
            Dataset dictionary or None if not found
        """
        try:
            if format == 'numpy':
                dataset = {'train': {}, 'validation': {}, 'test': {}, 'metadata': {}}
                
                for split in ['train', 'validation', 'test']:
                    split_path = self.processed_path / split
                    
                    if (split_path / 'images.npy').exists():
                        dataset[split]['images'] = np.load(split_path / 'images.npy')
                        dataset[split]['labels'] = np.load(split_path / 'labels.npy')
                        dataset[split]['landmarks'] = np.load(split_path / 'landmarks.npy')
                        dataset[split]['sources'] = np.load(split_path / 'sources.npy')
                
                # Load metadata
                with open(self.processed_path / 'metadata.json', 'r') as f:
                    dataset['metadata'] = json.load(f)
                    
                return dataset
                
            elif format == 'pickle':
                with open(self.processed_path / 'dataset.pkl', 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.warning(f"Could not load processed dataset: {e}")
            return None
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary"""
        if hasattr(self, 'dataset_stats') and self.dataset_stats:
            return self.dataset_stats
        
        # Try to load from file
        stats_file = self.processed_path / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.dataset_stats = json.load(f)
                return self.dataset_stats
        
        return {"error": "No dataset statistics available"}


# Utility function for testing
def main():
    """Test the dataset manager"""
    dm = EnhancedDatasetManager()
    
    # Check if datasets are available
    if dm.download_datasets():
        logger.info("Datasets found, creating unified dataset...")
        
        # Create unified dataset
        dataset = dm.create_unified_dataset()
        
        # Save processed dataset
        dm.save_processed_dataset(dataset, format='numpy')
        
        # Print summary
        summary = dm.get_dataset_summary()
        logger.info(f"Dataset created successfully: {summary}")
    else:
        logger.info("Please download the required datasets first")


if __name__ == "__main__":
    main()