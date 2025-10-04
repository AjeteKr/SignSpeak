# SignSpeak - AI-Powered ASL Recognition

An AI-powered American Sign Language (ASL) recognition system that converts sign language gestures into text and speech output.

## Features

- Real-time ASL gesture recognition using computer vision
- Text-to-speech conversion for accessibility
- Machine learning-powered classification
- Automated data processing pipeline
- Batch processing of ASL training data
- Comprehensive logging and reporting
- Easy-to-use interface

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SignSpeak-AI-Powered-ASL-Recognition.git
cd SignSpeak-AI-Powered-ASL-Recognition
```

### 2. Create Virtual Environment

#### On Windows:
```bash
python -m venv venv
```

#### On macOS/Linux:
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

#### On Windows:
```bash
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install TTS (for Text-to-Speech functionality)

```bash
pip install TTS
```

## Usage

### Data Processing Pipeline

The project includes an automated data processing pipeline that handles data cleaning and preparation for ASL training data.

#### Basic Usage

```bash
# Process all files in the default ./data folder
python main.py

# Process files in a specific folder
python main.py --data-folder /path/to/your/asl/data

# Process a single file for testing
python main.py --single-file /path/to/image.jpg

# Enable detailed logging
python main.py --verbose

# Generate a comprehensive processing report
python main.py --generate-report

# Combine options
python main.py --data-folder ./training_data --verbose --generate-report
```

#### Supported File Types

The processing pipeline supports the following file formats:
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Videos**: `.mp4`, `.avi`, `.mov`

#### Data Folder Structure

Organize your ASL data in the following structure:
```
data/
├── sign_a/
│   ├── image1.jpg
│   ├── image2.png
│   └── video1.mp4
├── sign_b/
│   ├── image1.jpg
│   └── image2.jpg
└── sign_c/
    └── video1.mov
```

The script will automatically discover and process all supported files in subdirectories.

#### Processing Pipeline

1. **File Discovery**: Automatically scans folders and subfolders for supported files
2. **File Validation**: Checks files for accessibility and basic integrity
3. **Data Cleaning**: Removes duplicates and filters out corrupted files
4. **Data Preparation**: Resizes images, normalizes data formats for model training
5. **Logging**: Comprehensive logging of all processing steps
6. **Reporting**: Generates detailed success/failure reports

#### Output and Logs

The processing pipeline generates:
- **Console Output**: Real-time progress updates
- **Log File**: `signspeak_processing.log` with detailed processing information
- **Report File**: `signspeak_processing_report.txt` (when `--generate-report` is used)

### Advanced Options

#### Command Line Arguments

- `--data-folder`: Specify the path to your ASL data folder (default: `./data`)
- `--single-file`: Process only a specific file instead of entire folder
- `--generate-report`: Create a detailed processing report file
- `--verbose`: Enable debug-level logging for troubleshooting

#### Examples

```bash
# Process a large dataset with full reporting
python main.py --data-folder ./large_dataset --verbose --generate-report

# Test processing on a single file
python main.py --single-file ./test_data/sample.jpg --verbose

# Quick batch processing
python main.py --data-folder ./new_signs
```

### Running the Full Application

```bash
python main.py
```

## Project Structure

```
SignSpeak-AI-Powered-ASL-Recognition/
├── main.py                 # Main initialization script with data processing pipeline
├── data_cleaning.py        # Data cleaning functions
├── data_preparation.py     # Data preparation and normalization
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Default data folder (create this)
├── logs/                  # Generated log files
└── reports/               # Generated processing reports
```

## Deactivating Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Contributing

We welcome contributions to SignSpeak! Here are two ways to contribute:

### Option 1: Fork the Repository

1. **Fork the repository** by clicking the "Fork" button on the GitHub page
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/your-username/SignSpeak-AI-Powered-ASL-Recognition.git
   cd SignSpeak-AI-Powered-ASL-Recognition
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/SignSpeak-AI-Powered-ASL-Recognition.git
   ```
4. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

### Option 2: Create a New Branch (for collaborators)

If you have direct access to the repository:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/original-owner/SignSpeak-AI-Powered-ASL-Recognition.git
   cd SignSpeak-AI-Powered-ASL-Recognition
   ```

2. **Create and switch to a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

4. **Push the branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

### Branch Naming Conventions

- `feature/feature-name` - For new features
- `bugfix/bug-description` - For bug fixes
- `hotfix/urgent-fix` - For urgent fixes
- `docs/documentation-update` - For documentation updates

## Dependencies

The project uses the following main libraries:

- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand tracking and pose estimation
- **TensorFlow** - Machine learning framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning tools
- **TTS** - Text-to-speech synthesis

## Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure your virtual environment is activated and all dependencies are installed
2. **TTS installation issues**: Try installing TTS separately with `pip install TTS`
3. **Camera access problems**: Ensure your camera is not being used by another application
4. **File processing failures**: Check that your data files are not corrupted and are in supported formats
5. **Permission errors**: Ensure the script has read/write permissions for the data folder and log files

### Data Processing Issues

1. **No files found**: Ensure your data folder contains files with supported extensions
2. **Processing failures**: Check the log file `signspeak_processing.log` for detailed error information
3. **Memory issues with large datasets**: Process smaller batches or use the `--single-file` option for testing

### Getting Help

- Check the [Issues](https://github.com/yourusername/SignSpeak-AI-Powered-ASL-Recognition/issues) page
- Create a new issue if you encounter problems
- Make sure to include your OS, Python version, error messages, and log files
- Use the `--verbose` flag to get more detailed debugging information

## Performance Tips

- **Large Datasets**: Process data in smaller batches if you encounter memory issues
- **Network Storage**: For better performance, copy data to local storage before processing
- **Monitoring**: Use the `--verbose` flag to monitor processing progress and identify bottlenecks
- **Parallel Processing**: Future versions will include multi-threading support for faster processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Special thanks to contributors and testers
- ASL community for feedback and guidance