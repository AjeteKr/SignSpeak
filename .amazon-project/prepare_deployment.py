"""
Deployment Preparation Script
Prepares SignSpeak Pro for production deployment
"""

import shutil
import os
from pathlib import Path
import json

def prepare_for_deployment():
    """
    Prepare the application for deployment by:
    1. Copying ASL reference images to production location
    2. Creating production directory structure
    3. Generating deployment configuration
    """
    
    print("🚀 Preparing SignSpeak Pro for deployment...")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    # Define source and target paths
    dev_images_path = project_root / ".." / "Wave" / "asl_reference_images"
    prod_images_path = project_root / "assets" / "asl_reference_images"
    
    # Create production directory structure
    print("📁 Creating production directories...")
    directories = [
        project_root / "assets",
        project_root / "assets" / "asl_reference_images",
        project_root / "models",
        project_root / "data",
        project_root / "logs",
        project_root / "temp"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Copy ASL reference images to production location
    if dev_images_path.exists():
        print("📷 Copying ASL reference images...")
        
        # Copy all image files
        for image_file in dev_images_path.glob("*.png"):
            target_file = prod_images_path / image_file.name
            shutil.copy2(image_file, target_file)
            print(f"   ✅ {image_file.name}")
        
        # Copy metadata
        metadata_file = dev_images_path / "letter_metadata.json"
        if metadata_file.exists():
            target_metadata = prod_images_path / "letter_metadata.json"
            shutil.copy2(metadata_file, target_metadata)
            print(f"   ✅ letter_metadata.json")
        
        print(f"📷 ASL images copied to: {prod_images_path}")
    else:
        print(f"⚠️  Warning: Development images not found at {dev_images_path}")
        print("   You'll need to manually copy ASL reference images to assets/asl_reference_images/")
    
    # Create production requirements.txt
    create_production_requirements(project_root)
    
    # Create deployment configuration
    create_deployment_config(project_root)
    
    # Create docker files
    create_docker_files(project_root)
    
    print("✅ Deployment preparation complete!")
    print()
    print("📋 Next steps for deployment:")
    print("1. Upload the entire .amazon-project folder to your hosting platform")
    print("2. Install requirements: pip install -r requirements.txt")
    print("3. Run the app: streamlit run signspeak_pro.py")
    print("4. For Docker: docker build -t signspeak-pro . && docker run -p 8501:8501 signspeak-pro")

def create_production_requirements(project_root: Path):
    """Create production requirements.txt"""
    print("📦 Creating production requirements.txt...")
    
    requirements = [
        "streamlit>=1.28.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "pathlib2>=2.3.0",
        "pandas>=2.0.0"
    ]
    
    # Add TensorFlow for optional advanced features
    requirements.append("tensorflow>=2.13.0")
    
    requirements_file = project_root / "requirements_production.txt"
    with open(requirements_file, 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"   ✅ {requirements_file}")

def create_deployment_config(project_root: Path):
    """Create deployment configuration"""
    print("⚙️ Creating deployment configuration...")
    
    # Create .streamlit directory
    streamlit_dir = project_root / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml
    config_content = """
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#2563EB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1F2937"
"""
    
    config_file = streamlit_dir / "config.toml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"   ✅ {config_file}")

def create_docker_files(project_root: Path):
    """Create Docker files for containerized deployment"""
    print("🐳 Creating Docker files...")
    
    # Dockerfile
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "signspeak_pro.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
"""
    
    dockerfile = project_root / "Dockerfile"
    with open(dockerfile, 'w') as f:
        f.write(dockerfile_content)
    
    # .dockerignore
    dockerignore_content = """
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.git/
.gitignore
README.md
.env
.venv/
venv/
logs/
temp/
*.log
.DS_Store
"""
    
    dockerignore = project_root / ".dockerignore"
    with open(dockerignore, 'w') as f:
        f.write(dockerignore_content)
    
    print(f"   ✅ {dockerfile}")
    print(f"   ✅ {dockerignore}")

if __name__ == "__main__":
    prepare_for_deployment()