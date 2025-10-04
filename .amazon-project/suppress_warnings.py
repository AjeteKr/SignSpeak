"""
Comprehensive warning suppression for ASL Recognition App
This module must be imported FIRST to suppress all TensorFlow/MediaPipe warnings
"""

import os
import sys
import warnings
from contextlib import redirect_stderr
import io

# Suppress all TensorFlow warnings before ANY tensorflow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Force CPU (no CUDA warnings)

# Comprehensive suppression for MediaPipe C++ warnings
os.environ['GLOG_minloglevel'] = '3'  # Suppress all glog messages except fatal
os.environ['GLOG_stderrthreshold'] = '3'  # Only fatal errors to stderr  
os.environ['GLOG_v'] = '0'  # Disable verbose logging
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU processing warnings

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress specific TensorFlow and MediaPipe warnings that flood the terminal
warnings.filterwarnings('ignore', message='.*deprecat.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*XNNPACK.*')
warnings.filterwarnings('ignore', message='.*inference_feedback.*')
warnings.filterwarnings('ignore', message='.*Feedback manager.*')
warnings.filterwarnings('ignore', message='.*model with a single signature.*')
warnings.filterwarnings('ignore', message='.*protobuf.*')
warnings.filterwarnings('ignore', message='.*mediapipe.*')

# Suppress absl logging (used by TensorFlow)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Suppress TensorFlow logging
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass

# Redirect stderr to suppress C++ level warnings
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr

def suppress_tensorflow_warnings():
    """Apply all TensorFlow warning suppressions"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
    
def suppress_mediapipe_warnings():
    """Apply all MediaPipe warning suppressions"""
    warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Apply suppressions immediately when this module is imported
suppress_tensorflow_warnings()
suppress_mediapipe_warnings()