"""
MNIST ASL Dataset Helper Functions
Provides integration between live ASL detection and MNIST dataset
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os

# Label mapping for the 12 letters in your dataset
LABEL_MAPPING = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
    8: 'I', 10: 'K', 11: 'L', 12: 'M', 14: 'O', 21: 'V', 24: 'Y'
}

# Reverse mapping
LETTER_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class MNISTASLHelper:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            # Use absolute path to Wave repository
            current_dir = os.path.dirname(os.path.abspath(__file__))
            amazon_root = os.path.dirname(current_dir)  # Go up to Amazon folder
            dataset_path = os.path.join(amazon_root, "Wave", "dataset")
        
        self.dataset_path = dataset_path
        self.train_data = None
        self.test_data = None
        self.loaded = False
        
    def load_data(self):
        """Load MNIST ASL training and test data"""
        try:
            train_path = os.path.join(self.dataset_path, "mnist_12_train.csv")
            test_path = os.path.join(self.dataset_path, "mnist_12_test.csv")
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                self.train_data = pd.read_csv(train_path)
                self.test_data = pd.read_csv(test_path)
                self.loaded = True
                return True
            else:
                st.warning("MNIST CSV files not found in dataset folder")
                return False
        except Exception as e:
            st.error(f"Error loading MNIST data: {str(e)}")
            return False
    
    def get_letter_examples(self, letter, num_examples=5, from_test=True):
        """Get random examples of a specific letter from MNIST dataset"""
        if not self.loaded:
            if not self.load_data():
                return []
        
        if letter not in LETTER_TO_LABEL:
            return []
        
        label = LETTER_TO_LABEL[letter]
        data_source = self.test_data if from_test else self.train_data
        
        # Filter data for the specific letter
        letter_data = data_source[data_source['label'] == label]
        
        if len(letter_data) == 0:
            return []
        
        # Get random samples
        num_examples = min(num_examples, len(letter_data))
        samples = letter_data.sample(n=num_examples)
        
        images = []
        for _, row in samples.iterrows():
            # Extract pixel data (excluding label column)
            pixel_data = row.drop('label').values
            # Reshape to 28x28 image
            image = pixel_data.reshape(28, 28)
            images.append(image)
        
        return images
    
    def get_dataset_stats(self):
        """Get statistics about the MNIST dataset"""
        if not self.loaded:
            if not self.load_data():
                return {}
        
        stats = {}
        for dataset_name, data in [("Training", self.train_data), ("Test", self.test_data)]:
            if data is not None:
                stats[dataset_name] = {}
                for label, letter in LABEL_MAPPING.items():
                    count = len(data[data['label'] == label])
                    stats[dataset_name][letter] = count
        
        return stats
    
    def visualize_letter_grid(self, letter, num_examples=9):
        """Create a grid visualization of MNIST examples for a letter"""
        images = self.get_letter_examples(letter, num_examples)
        
        if not images:
            return None
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(images))))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        fig.suptitle(f'MNIST Examples: Letter "{letter}"', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(grid_size * grid_size):
            ax = axes[i]
            if i < len(images):
                ax.imshow(images[i], cmap='gray')
                ax.set_title(f'Example {i+1}')
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def get_letter_similarity_score(self, detected_letter, target_letter):
        """Calculate similarity score between detected and target letters"""
        if not self.loaded:
            return 0.0
        
        if detected_letter == target_letter:
            return 1.0
        
        # You could implement more sophisticated similarity based on
        # hand landmark patterns or visual similarity
        # For now, return a simple score
        similar_pairs = {
            ('V', 'U'): 0.8, ('U', 'V'): 0.8,
            ('B', 'F'): 0.7, ('F', 'B'): 0.7,
            ('A', 'S'): 0.6, ('S', 'A'): 0.6,
            ('M', 'N'): 0.7, ('N', 'M'): 0.7,
        }
        
        return similar_pairs.get((detected_letter, target_letter), 0.0)
    
    def generate_practice_sequence(self, difficulty='easy', length=5):
        """Generate a practice sequence of letters"""
        available_letters = list(LETTER_TO_LABEL.keys())
        
        if difficulty == 'easy':
            # Focus on distinct letters
            preferred = ['A', 'B', 'I', 'L', 'O', 'Y']
            sequence = np.random.choice([l for l in preferred if l in available_letters], 
                                      size=min(length, len(preferred)), replace=False)
        elif difficulty == 'medium':
            # Mix of easy and challenging
            sequence = np.random.choice(available_letters, size=length, replace=False)
        else:  # hard
            # Focus on similar letters
            challenging = ['V', 'U', 'B', 'C', 'A', 'E']
            sequence = np.random.choice([l for l in challenging if l in available_letters], 
                                      size=min(length, len(challenging)), replace=True)
        
        return list(sequence)
    
    def create_comparison_image(self, letter, hand_region=None):
        """Create side-by-side comparison of MNIST and live detection"""
        mnist_examples = self.get_letter_examples(letter, 1)
        
        if not mnist_examples:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # MNIST example
        axes[0].imshow(mnist_examples[0], cmap='gray')
        axes[0].set_title(f'MNIST: Letter "{letter}"')
        axes[0].axis('off')
        
        # Live detection (if hand region provided)
        if hand_region is not None:
            # Convert to grayscale and resize to match MNIST
            if len(hand_region.shape) == 3:
                hand_gray = cv2.cvtColor(hand_region, cv2.COLOR_RGB2GRAY)
            else:
                hand_gray = hand_region
            
            hand_resized = cv2.resize(hand_gray, (28, 28))
            axes[1].imshow(hand_resized, cmap='gray')
            axes[1].set_title(f'Your Sign: "{letter}"')
        else:
            axes[1].text(0.5, 0.5, 'No hand detected', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Your Sign')
        
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig

# Global instance for easy access
mnist_helper = MNISTASLHelper()
