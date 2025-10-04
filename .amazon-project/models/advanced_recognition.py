"""
Advanced CNN Recognition Engine for ASL Letters
4-layer CNN with batch normalization, dropout, and 26-class output
Frame-based processing for real-time performance
"""

# MUST BE FIRST: Suppress all warnings before any other imports
import suppress_warnings  # This MUST be the very first import

import os
# Suppress TensorFlow and MediaPipe warnings BEFORE importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Create dummy classes for fallback (define outside try/except for type hints)
class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        return np.array([[0.04] * 26])  # Equal probability for all letters
    def fit(self, *args, **kwargs):
        return None
    def save(self, *args, **kwargs):
        pass
    def load_weights(self, *args, **kwargs):
        pass

# Optional TensorFlow imports with fallbacks
try:
    # Import TensorFlow with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tensorflow as tf
        
    # Import using tf.keras to avoid import resolution issues
    layers = tf.keras.layers
    models = tf.keras.models
    optimizers = tf.keras.optimizers
    callbacks = tf.keras.callbacks
    # Import applications using tf.keras
    MobileNetV2 = tf.keras.applications.MobileNetV2
    EfficientNetB0 = tf.keras.applications.EfficientNetB0
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import numpy as np
import cv2
from pathlib import Path
import json
import pickle
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt

# Optional visualization imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass
from datetime import datetime

# Import our ASL dataset loader
try:
    from data_processing.asl_reference_loader import ASLReferenceDatasetLoader
    ASL_LOADER_AVAILABLE = True
except ImportError:
    ASL_LOADER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not ASL_LOADER_AVAILABLE:
    logger.warning("ASL reference dataset loader not available")

@dataclass
class ModelConfig:
    """Configuration for CNN model"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 26
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    dropout_rate: float = 0.5
    l2_regularization: float = 0.001

@dataclass
class TrainingHistory:
    """Training history and metrics"""
    accuracy: List[float]
    val_accuracy: List[float]
    loss: List[float]
    val_loss: List[float]
    epoch_times: List[float]
    best_accuracy: float
    best_epoch: int
    total_training_time: float

class AdvancedASLRecognitionEngine:
    """
    Advanced CNN-based ASL recognition engine
    Supports both image and feature-based recognition
    """
    
    def __init__(self, config: ModelConfig = None):
        """Initialize recognition engine"""
        self.config = config or ModelConfig()
        self.model = None
        self.feature_model = None
        self.training_history = None
        self.class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Performance tracking
        self.frame_buffer = []
        self.prediction_buffer = []
        self.confidence_threshold = 0.8
        self.voting_window = 5
        
        # Setup paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize TensorFlow optimizations
        self._setup_tensorflow()
    
    def _setup_tensorflow(self):
        """Setup TensorFlow for optimal performance"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback recognition.")
            return
            
        # Enable mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.info("No GPU found, using CPU")
    
    def build_cnn_model(self) -> Union[tf.keras.Model, DummyModel]:
        """
        Build 4-layer CNN model with specified architecture
        32→64→128→256 filters with batch normalization and dropout
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using dummy model.")
            return DummyModel()
            
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.config.input_shape),
            
            # Layer 1: 32 filters
            layers.Conv2D(32, (3, 3), activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Layer 2: 64 filters
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Layer 3: 128 filters
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Layer 4: 256 filters
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.config.dropout_rate),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.config.dropout_rate),
            
            # Output layer: 26 classes for A-Z
            layers.Dense(self.config.num_classes, activation='softmax', dtype='float32')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model: str = 'mobilenetv2') -> Union[tf.keras.Model, DummyModel]:
        """
        Build transfer learning model for better performance
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using dummy model.")
            return DummyModel()
            
        if base_model == 'mobilenetv2':
            base = MobileNetV2(
                input_shape=self.config.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_model == 'efficientnet':
            base = EfficientNetB0(
                input_shape=self.config.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze base model initially
        base.trainable = False
        
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.config.num_classes, activation='softmax', dtype='float32')
        ])
        
        return model
    
    def build_feature_model(self, feature_dim: int = 85) -> Union[tf.keras.Model, DummyModel]:
        """
        Build model for hand landmark features (not images)
        For use with normalized feature vectors
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using dummy model for features.")
            return DummyModel()
            
        model = models.Sequential([
            layers.Input(shape=(feature_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: tf.keras.Model):
        """Compile model with optimized settings"""
        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def create_data_generators(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_val: np.ndarray = None,
                             y_val: np.ndarray = None) -> Tuple:
        """
        Create data generators with advanced augmentation
        """
        # Create validation split if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config.validation_split,
                stratify=y_train,
                random_state=42
            )
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Not good for ASL
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=1./255 if X_train.max() > 1 else 1.0
        )
        
        # Validation generator (no augmentation)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255 if X_val.max() > 1 else 1.0
        )
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_callbacks(self, model_name: str = "asl_model") -> List:
        """Create training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                filepath=self.model_dir / f"{model_name}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                self.model_dir / f"{model_name}_training.log"
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=self.model_dir / "tensorboard",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_with_asl_reference_images(self, 
                                      custom_images_path: str = None,
                                      epochs: int = 50,
                                      augment_factor: int = 20,
                                      model_type: str = 'cnn') -> Dict:
        """
        Train the model using ASL reference images - deployment ready
        
        Args:
            custom_images_path: Custom path to images folder (optional, auto-detected if None)
            epochs: Number of training epochs
            augment_factor: Number of augmented versions per reference image
            model_type: Type of model to train ('cnn' or 'transfer')
            
        Returns:
            Dictionary containing training results and model info
        """
        if not ASL_LOADER_AVAILABLE:
            logger.error("ASL reference dataset loader not available")
            return {'success': False, 'error': 'ASL loader not available'}
        
        try:
            # Initialize dataset loader with auto-detection
            loader = ASLReferenceDatasetLoader(custom_images_path)
            
            # Check if images folder exists
            if not loader.images_folder.exists():
                error_msg = f"ASL images folder not found: {loader.images_folder}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Prepare training data
            logger.info("Loading ASL reference images...")
            training_data = loader.prepare_training_data(
                test_size=0.2, 
                augment_factor=augment_factor,
                target_size=self.config.input_shape[:2]
            )
            
            if not training_data:
                logger.error("Failed to load training data")
                return {'success': False, 'error': 'Failed to load training data'}
            
            # Update config with correct number of classes
            self.config.num_classes = training_data['num_classes']
            self.class_names = training_data['class_names']
            
            logger.info(f"Training data loaded:")
            logger.info(f"  Training samples: {training_data['X_train'].shape[0]}")
            logger.info(f"  Validation samples: {training_data['X_val'].shape[0]}")
            logger.info(f"  Classes: {self.config.num_classes}")
            logger.info(f"  Class names: {self.class_names}")
            
            # Build and compile model
            if model_type == 'transfer':
                self.model = self.build_transfer_learning_model()
            else:
                self.model = self.build_cnn_model()
            
            self.model = self.compile_model(self.model)
            
            # Setup callbacks
            callbacks_list = self.create_callbacks(f"asl_reference_{model_type}")
            
            # Train model
            logger.info(f"Starting training for {epochs} epochs...")
            history = self.model.fit(
                training_data['X_train'], training_data['y_train'],
                validation_data=(training_data['X_val'], training_data['y_val']),
                epochs=epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Save model
            model_path = self.model_dir / f"asl_reference_{model_type}_final.h5"
            self.model.save(str(model_path))
            
            # Evaluate model
            val_loss, val_accuracy = self.model.evaluate(
                training_data['X_val'], training_data['y_val'], 
                verbose=0
            )
            
            training_results = {
                'success': True,
                'model_path': str(model_path),
                'val_accuracy': float(val_accuracy),
                'val_loss': float(val_loss),
                'num_classes': self.config.num_classes,
                'class_names': self.class_names,
                'training_samples': training_data['X_train'].shape[0],
                'validation_samples': training_data['X_val'].shape[0],
                'epochs': epochs,
                'history': {
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
                    'loss': [float(x) for x in history.history.get('loss', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])]
                }
            }
            
            logger.info(f"Training completed! Validation accuracy: {val_accuracy:.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error during ASL reference training: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_asl_reference_model(self, model_path: str = None) -> bool:
        """
        Load a trained ASL reference model
        
        Args:
            model_path: Path to the trained model file. If None, looks for default model.
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot load model.")
            return False
        
        try:
            if model_path is None:
                # Look for default ASL reference models
                possible_paths = [
                    self.model_dir / "asl_reference_cnn_final.h5",
                    self.model_dir / "asl_reference_transfer_final.h5",
                    self.model_dir / "asl_reference_cnn_best.h5",
                    self.model_dir / "asl_reference_transfer_best.h5"
                ]
                
                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break
                
                if model_path is None:
                    logger.error("No trained ASL reference model found")
                    return False
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            
            # Initialize ASL class names (24 letters A-Y, excluding J and Z)
            if not hasattr(self, 'class_names') or len(self.class_names) != 24:
                self.class_names = [
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
                ]
            
            logger.info(f"ASL reference model loaded from: {model_path}")
            logger.info(f"Model expects input shape: {self.model.input_shape}")
            logger.info(f"Model predicts {len(self.class_names)} classes: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading ASL reference model: {e}")
            return False
    
    def predict_asl_letter(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL letter from image using trained model
        
        Args:
            image: Input image (RGB, any size - will be resized)
            
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        if self.model is None:
            logger.warning("No model loaded. Loading default ASL reference model...")
            if not self.load_asl_reference_model():
                return None, 0.0
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image_for_prediction(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get best prediction
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Map to letter
            if predicted_class < len(self.class_names):
                predicted_letter = self.class_names[predicted_class]
            else:
                predicted_letter = "?"
            
            return predicted_letter, confidence
            
        except Exception as e:
            logger.error(f"Error predicting ASL letter: {e}")
            return None, 0.0
    
    def _preprocess_image_for_prediction(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ASL letter prediction
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # Get target size from model input shape or use default
        if self.model is not None:
            target_size = self.model.input_shape[1:3]  # (height, width)
        else:
            target_size = (224, 224)
        
        # Ensure image is in correct format
        if len(image.shape) == 4:
            image = image[0]  # Remove batch dimension if present
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            processed = image.copy()
        else:
            logger.warning("Unexpected image format, attempting to process anyway")
            processed = image
        
        # Resize to target size
        processed = cv2.resize(processed, target_size)
        
        # Normalize to [0, 1]
        if processed.max() > 1.0:
            processed = processed.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: np.ndarray = None,
                   y_val: np.ndarray = None,
                   model_type: str = 'cnn',
                   use_transfer_learning: bool = False) -> TrainingHistory:
        """
        Train the ASL recognition model
        """
        logger.info(f"Starting training with {len(X_train)} samples")
        
        # Build model
        if model_type == 'feature':
            self.feature_model = self.build_feature_model(X_train.shape[1])
            model = self.compile_model(self.feature_model)
        elif use_transfer_learning:
            self.model = self.build_transfer_learning_model()
            model = self.compile_model(self.model)
        else:
            self.model = self.build_cnn_model()
            model = self.compile_model(self.model)
        
        # Print model summary
        model.summary()
        
        # Create data generators or prepare data
        if model_type == 'feature':
            # For feature models, use simple validation split
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=self.config.validation_split,
                    stratify=y_train,
                    random_state=42
                )
            
            train_data = (X_train, y_train)
            val_data = (X_val, y_val)
            steps_per_epoch = None
            validation_steps = None
        else:
            # For image models, use data generators
            train_generator, val_generator = self.create_data_generators(
                X_train, y_train, X_val, y_val
            )
            train_data = train_generator
            val_data = val_generator
            steps_per_epoch = len(train_generator)
            validation_steps = len(val_generator)
        
        # Create callbacks
        callback_list = self.create_callbacks(
            f"asl_{model_type}_model"
        )
        
        # Train model
        start_time = datetime.now()
        
        history = model.fit(
            train_data,
            epochs=self.config.epochs,
            validation_data=val_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callback_list,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Create training history object
        self.training_history = TrainingHistory(
            accuracy=history.history['accuracy'],
            val_accuracy=history.history['val_accuracy'],
            loss=history.history['loss'],
            val_loss=history.history['val_loss'],
            epoch_times=[],  # Would need to track separately
            best_accuracy=max(history.history['val_accuracy']),
            best_epoch=np.argmax(history.history['val_accuracy']) + 1,
            total_training_time=training_time
        )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.training_history.best_accuracy:.4f}")
        
        return self.training_history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        model = self.model or self.feature_model
        if model is None:
            raise ValueError("No trained model available")
        
        # Make predictions
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        evaluation_results = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': {
                self.class_names[i]: per_class_accuracy[i] 
                for i in range(len(self.class_names))
            },
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'predicted_classes': y_pred
        }
        
        return evaluation_results
    
    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Process single frame for real-time recognition
        Returns: (predicted_letter, confidence, additional_info)
        """
        if self.model is None and self.feature_model is None:
            return None, 0.0, {}
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            if processed_frame is None:
                return None, 0.0, {'error': 'preprocessing_failed'}
            
            # Make prediction
            if self.model is not None:
                # Use image model
                prediction = self.model.predict(processed_frame[np.newaxis, ...], verbose=0)
            else:
                # Use feature model (would need feature extraction here)
                return None, 0.0, {'error': 'feature_extraction_needed'}
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Apply voting mechanism
            prediction_result = self._apply_voting_mechanism(predicted_class, confidence)
            
            letter = self.class_names[prediction_result['class']]
            
            return letter, prediction_result['confidence'], prediction_result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None, 0.0, {'error': str(e)}
    
    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess frame for model input"""
        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Resize to model input size
            resized = cv2.resize(frame_rgb, self.config.input_shape[:2])
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return None
    
    def _apply_voting_mechanism(self, predicted_class: int, confidence: float) -> Dict:
        """Apply multi-frame voting for stable predictions"""
        # Add to prediction buffer
        self.prediction_buffer.append({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # Keep only recent predictions
        if len(self.prediction_buffer) > self.voting_window:
            self.prediction_buffer.pop(0)
        
        # If not enough samples, return current prediction
        if len(self.prediction_buffer) < 3:
            return {
                'class': predicted_class,
                'confidence': confidence,
                'votes': 1,
                'stability': 0.0
            }
        
        # Count votes for each class
        class_votes = {}
        confidence_sum = {}
        
        for pred in self.prediction_buffer[-self.voting_window:]:
            cls = pred['class']
            conf = pred['confidence']
            
            if cls not in class_votes:
                class_votes[cls] = 0
                confidence_sum[cls] = 0
            
            class_votes[cls] += 1
            confidence_sum[cls] += conf
        
        # Find winning class
        winning_class = max(class_votes, key=class_votes.get)
        votes = class_votes[winning_class]
        avg_confidence = confidence_sum[winning_class] / votes
        
        # Calculate stability (how consistent recent predictions are)
        stability = votes / len(self.prediction_buffer)
        
        return {
            'class': winning_class,
            'confidence': avg_confidence,
            'votes': votes,
            'stability': stability
        }
    
    def save_model(self, filepath: str, include_config: bool = True):
        """Save trained model and configuration"""
        model = self.model or self.feature_model
        if model is None:
            raise ValueError("No trained model to save")
        
        # Save model
        model.save(filepath)
        
        if include_config:
            # Save configuration and metadata
            config_path = Path(filepath).parent / f"{Path(filepath).stem}_config.json"
            
            config_data = {
                'model_config': {
                    'input_shape': self.config.input_shape,
                    'num_classes': self.config.num_classes,
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'dropout_rate': self.config.dropout_rate
                },
                'class_names': self.class_names,
                'training_history': {
                    'best_accuracy': self.training_history.best_accuracy if self.training_history else None,
                    'best_epoch': self.training_history.best_epoch if self.training_history else None,
                    'total_training_time': self.training_history.total_training_time if self.training_history else None
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Try to load configuration
            config_path = Path(filepath).parent / f"{Path(filepath).stem}_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self.class_names = config_data.get('class_names', self.class_names)
                    logger.info("Configuration loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def plot_training_history(self):
        """Plot training history"""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(self.training_history.accuracy, label='Training')
        axes[0].plot(self.training_history.val_accuracy, label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.training_history.loss, label='Training')
        axes[1].plot(self.training_history.val_loss, label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        if SEABORN_AVAILABLE:
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
        else:
            # Fallback to matplotlib-only visualization
            plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add text annotations
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    plt.text(j, i, str(confusion_matrix[i, j]),
                            ha="center", va="center")
            
            plt.xticks(range(len(self.class_names)), self.class_names)
            plt.yticks(range(len(self.class_names)), self.class_names)
            
        plt.title('ASL Recognition Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        model = self.model or self.feature_model
        if model is None:
            return {'error': 'No model available'}
        
        if not TENSORFLOW_AVAILABLE or isinstance(model, DummyModel):
            return {
                'model_type': 'dummy_fallback',
                'tensorflow_available': False,
                'total_parameters': 0,
                'trainable_parameters': 0,
                'layers': 0,
                'model_size_mb': 0
            }
        
        summary = {
            'total_parameters': model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'model_size_mb': model.count_params() * 4 / (1024 * 1024),  # Rough estimate
        }
        
        if self.training_history:
            summary.update({
                'best_accuracy': self.training_history.best_accuracy,
                'training_time': self.training_history.total_training_time
            })
        
        return summary


# Utility functions for testing
def train_example_model():
    """Example training script"""
    # This would normally use real data
    logger.info("Creating example training data...")
    
    # Dummy data for demonstration
    X_train = np.random.rand(1000, 224, 224, 3)
    y_train = np.random.randint(0, 26, 1000)
    X_test = np.random.rand(200, 224, 224, 3)
    y_test = np.random.randint(0, 26, 200)
    
    # Initialize engine
    config = ModelConfig(epochs=5)  # Short training for demo
    engine = AdvancedASLRecognitionEngine(config)
    
    # Train model
    history = engine.train_model(X_train, y_train)
    
    # Evaluate
    results = engine.evaluate_model(X_test, y_test)
    print(f"Test accuracy: {results['overall_accuracy']:.4f}")
    
    # Save model
    engine.save_model("example_asl_model.h5")
    
    return engine


if __name__ == "__main__":
    train_example_model()