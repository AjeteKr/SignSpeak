"""
ASL Alphabet Dataset Creator
Downloads and organizes ASL alphabet reference images
"""
import os
import requests
from pathlib import Path
import json

def create_asl_dataset():
    """Create ASL alphabet dataset with reference images"""
    
    # Create dataset directory
    dataset_dir = Path(__file__).parent / "data" / "asl_alphabet"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # ASL alphabet data with descriptions
    asl_letters = {
        'A': {'description': 'Closed fist with thumb on the side'},
        'B': {'description': 'Flat hand, fingers together, thumb across palm'},
        'C': {'description': 'Curved hand like holding a cup'},
        'D': {'description': 'Index finger up, other fingers touch thumb'},
        'E': {'description': 'All fingertips touch thumb'},
        'F': {'description': 'Index and thumb circle, other fingers up'},
        'G': {'description': 'Index finger and thumb parallel, pointing sideways'},
        'H': {'description': 'Index and middle finger together, pointing sideways'},
        'I': {'description': 'Pinky finger up, others down'},
        'J': {'description': 'Pinky finger traces a J in the air'},
        'K': {'description': 'Index up, middle finger touches thumb'},
        'L': {'description': 'L shape with index finger and thumb'},
        'M': {'description': 'Three fingers over thumb'},
        'N': {'description': 'Two fingers over thumb'},
        'O': {'description': 'All fingers curved like holding a ball'},
        'P': {'description': 'Like K but pointing down'},
        'Q': {'description': 'Like G but pointing down'},
        'R': {'description': 'Index and middle fingers crossed'},
        'S': {'description': 'Closed fist with thumb over fingers'},
        'T': {'description': 'Thumb between index and middle finger'},
        'U': {'description': 'Index and middle finger up together'},
        'V': {'description': 'Index and middle finger apart, V shape'},
        'W': {'description': 'Index, middle, and ring finger up'},
        'X': {'description': 'Index finger crooked, like a hook'},
        'Y': {'description': 'Thumb and pinky out, others down'},
        'Z': {'description': 'Index finger traces a Z in the air'}
    }
    
    # Create ASCII art representations for quick reference
    ascii_art = {
        'A': '''
    ████████
   ██      ██
  ██        ██
 ██    ██    ██
██      ██    ██
██      ██    ██
 ████████████
        ''',
        'B': '''
████████████
██          ██
██          ██
████████████
██          ██
██          ██
████████████
        ''',
        'C': '''
  ████████████
 ██          
██           
██           
██           
 ██          
  ████████████
        '''
    }
    
    # Save dataset info
    dataset_info = {
        'name': 'ASL Alphabet Reference',
        'description': 'American Sign Language alphabet reference images',
        'letters': asl_letters,
        'total_letters': 26,
        'created_date': '2025-09-16'
    }
    
    # Save dataset info as JSON
    with open(dataset_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create individual letter directories
    for letter, info in asl_letters.items():
        letter_dir = dataset_dir / letter
        letter_dir.mkdir(exist_ok=True)
        
        # Save letter info
        letter_info = {
            'letter': letter,
            'description': info['description'],
            'ascii_art': ascii_art.get(letter, ''),
            'examples': []
        }
        
        with open(letter_dir / f'{letter}_info.json', 'w') as f:
            json.dump(letter_info, f, indent=2)
        
        # Create placeholder image (you can replace these with actual images)
        create_placeholder_image(letter_dir / f'{letter}_reference.txt', letter, info['description'])
    
    print(f"✅ ASL Dataset created at: {dataset_dir}")
    print(f"📁 Contains {len(asl_letters)} letter directories")
    print("📝 Each letter has description and placeholder for images")
    
    return dataset_dir

def create_placeholder_image(file_path, letter, description):
    """Create a text placeholder for the letter image"""
    content = f"""
ASL Letter: {letter}
Description: {description}

Instructions:
1. Form your hand into the '{letter}' position
2. {description}
3. Hold steady for recognition
4. Make sure your hand is clearly visible

To add real images:
- Replace this file with {letter}_reference.jpg
- Add multiple examples: {letter}_example1.jpg, {letter}_example2.jpg, etc.
- Images should be 224x224 pixels for best results
"""
    
    with open(file_path, 'w') as f:
        f.write(content)

def get_letter_info(letter):
    """Get information about a specific ASL letter"""
    dataset_dir = Path(__file__).parent / "data" / "asl_alphabet"
    letter_file = dataset_dir / letter.upper() / f'{letter.upper()}_info.json'
    
    if letter_file.exists():
        with open(letter_file, 'r') as f:
            return json.load(f)
    return None

def list_available_letters():
    """List all available letters in the dataset"""
    dataset_dir = Path(__file__).parent / "data" / "asl_alphabet"
    if not dataset_dir.exists():
        return []
    
    return [d.name for d in dataset_dir.iterdir() if d.is_dir()]

if __name__ == "__main__":
    print("🤟 Creating ASL Alphabet Dataset...")
    create_asl_dataset()
    print("🎉 Dataset creation complete!")
    
    # Test the dataset
    print("\n📖 Available letters:", list_available_letters())
    
    # Show example letter info
    if info := get_letter_info('A'):
        print(f"\n📋 Letter A Info:")
        print(f"   Description: {info['description']}")