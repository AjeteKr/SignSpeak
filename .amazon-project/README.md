
# 🤟 SignSpeak Pro - Advanced ASL Recognition System

**Professional-grade American Sign Language recognition with real-time optimization, personal calibration, and comprehensive analytics.**

## 🌟 Overview

SignSpeak Pro is a complete overhaul of the original ASL recognition system, featuring:

- **Normalized Feature Extraction** - 0-1 pixel scaling for device independence
- **Personal Calibration System** - Guided setup with 3-5 samples per letter
- **4-Layer CNN Architecture** - 32→64→128→256 filters with batch normalization
- **Real-Time Optimization** - 30 FPS target with threading and performance monitoring
- **Letter Accumulation Engine** - Smart spacing and word completion
- **Comprehensive Database** - User profiles, session tracking, and performance analytics
- **Advanced Testing Framework** - Confusion matrix analysis and multi-frame voting

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- Webcam
- Windows, macOS, or Linux

### Installation

1. **Run the automated setup:**
   ```bash
   python setup_signspeak_pro.py
   ```

2. **Launch the application:**
   ```bash
   # Windows
   launch_signspeak_pro.bat
   
   # macOS/Linux
   ./launch_signspeak_pro.sh
   
   # Or manually
   python -m streamlit run signspeak_pro_complete.py
   ```

## 📁 Project Structure

```
SignSpeak-Pro/
├── 📱 signspeak_pro_complete.py     # Main application
├── 🔧 setup_signspeak_pro.py        # Setup script
├── 📄 README.md                     # This file
├── 📋 Wave/requirements.txt         # Dependencies
│
├── �️ data_processing/              # Enhanced data pipeline
│   ├── dataset_manager.py           # Unified MNIST+ASL integration
│   └── feature_extractor.py         # Normalized feature extraction
│
├── 🎯 calibration/                  # Personal calibration system
│   └── personal_calibration.py      # Guided user setup
│
├── 🧠 models/                       # Advanced recognition models
│   └── advanced_recognition.py      # 4-layer CNN implementation
│
├── 🔤 components/                   # Core components
│   └── letter_accumulation.py       # Smart letter accumulation
│
├── 🗄️ database/                     # Database management
│   └── asl_database.py              # Comprehensive data storage
│
├── ⚡ optimization/                 # Performance optimization
│   └── real_time_optimizer.py       # Threading and performance
│
├── 🧪 testing/                      # Testing framework
│   └── comprehensive_validator.py   # Advanced validation
│
└── 📊 Wave/                         # Original system (reference)
    ├── streamlit_app.py
    ├── api_server.py
    └── requirements.txt
```

## 🎯 Key Features

### 1. Normalized Feature Extraction
- Wrist-centered coordinate system with 0-1 scaling
- MediaPipe hand detection with 21-point landmarks
- Scale-invariant features for consistent recognition

### 2. Personal Calibration System
- Guided setup requiring 3-5 samples per letter
- SQLite database for user profiles and calibration data
- Adaptive confidence thresholds per user

### 3. Advanced CNN Architecture
- 4-layer design: 32→64→128→256 filters
- Batch normalization and dropout regularization
- 26-class softmax output for A-Z recognition

### 4. Real-Time Optimization
- Multi-threaded processing with 30 FPS target
- Frame buffering and smart frame dropping
- Performance monitoring and automatic optimization

### 5. Letter Accumulation Engine
- Smart spacing detection using pause/gesture/time-based logic
- Word completion with dictionary integration (200+ common words)
- Text-to-speech support for completed words

### 6. Comprehensive Database System
- User profiles with session tracking
- Performance metrics and analytics
- Data export/import capabilities

### 7. Advanced Testing Framework
- Confusion matrix analysis with error pattern detection
- Multi-frame voting for improved accuracy
- Real-world testing under various conditions

## 🎮 User Guide

### Getting Started

1. **Launch the application** using the setup instructions above
2. **Initialize the system** using the sidebar controls
3. **Create a user profile** with your name and details
4. **Run personal calibration** to optimize for your signing style
5. **Start a practice session** and begin signing letters!

### Navigation

The application has 5 main tabs:

- **🎥 Live Recognition** - Real-time ASL recognition
- **🎯 Calibration** - Personal calibration system
- **📊 Analytics** - Performance metrics and analytics
- **🧪 Testing** - System validation and testing
- **⚙️ Settings** - Configuration and data management

### Best Practices

1. **Lighting:** Use consistent, bright lighting for best results
2. **Background:** Plain backgrounds work better than complex ones
3. **Hand Position:** Keep hands centered in the camera frame
4. **Signing Style:** Be consistent with your signing speed and style
5. **Calibration:** Recalibrate if accuracy decreases over time

## 🔧 Technical Details

### System Requirements

- **Minimum:** Python 3.8, 4GB RAM, 2 CPU cores, webcam
- **Recommended:** Python 3.10+, 8GB+ RAM, 4+ CPU cores, good lighting

### Performance Specifications

- **Target FPS:** 30 frames per second
- **Latency:** <100ms total processing time
- **Accuracy:** >95% on calibrated users, >85% on uncalibrated
- **Memory Usage:** ~500MB-1GB depending on configuration

---

**SignSpeak Pro** - Bridging communication barriers through advanced AI-powered ASL recognition.

*Built with ❤️ for the deaf and hard-of-hearing community.*
```bash
python -m venv env
env\Scripts\activate
pip install opencv-python streamlit mediapipe tensorflow pyttsx3 matplotlib pandas
deactivate
```

### 🌿 Git workflow (Ajete’s branch)
```bash
git checkout -b ajete_dev
git add .
git commit -m "feat: created UI"
git push --set-upstream origin ajete_dev
git push
```

### ▶️ Run the Streamlit application
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
.AMAZON-PROJECT/
├── env/                  ← Virtual environment (do not track in Git)
├── architecture.md       ← Technical documentation
├── camera_test.py        ← OpenCV script for webcam testing
├── streamlit_app.py      ← Main Streamlit UI
├── tts_test.py           ← Simple Text-to-Speech test
├── tts_helper.py         ← Optional TTS helper functions
├── requirements.txt      ← List of dependencies
└── README.md             ← This file
```

## ✅ Week 1 Deliverables (Completed)

- Setup development environment
- Install core libraries
- Test webcam with OpenCV
- Build basic UI using Streamlit
- Integrate pyttsx3 for text-to-speech
- Document technical architecture
- Push code using Git branching strategy

## 📌 Notes

- If you're on macOS or Linux, replace `env\Scripts\activate` with `source env/bin/activate`.
- Make sure the `env/` folder is listed in your `.gitignore`.
