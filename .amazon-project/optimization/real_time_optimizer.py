"""
Real-time Performance Optimization System for ASL Recognition
Implements threading, frame buffering, and performance monitoring for 30 FPS target
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    fps: float = 0.0
    frame_processing_time: float = 0.0
    model_inference_time: float = 0.0
    feature_extraction_time: float = 0.0
    total_latency: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    queue_size: int = 0
    frames_dropped: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    target_fps: int = 30
    max_queue_size: int = 10
    num_worker_threads: int = 3
    frame_skip_threshold: float = 0.1  # Skip frames if processing takes longer
    memory_cleanup_interval: int = 100  # Frames between cleanup
    performance_log_interval: int = 30   # Seconds between performance logs
    enable_gpu_acceleration: bool = True
    use_frame_downsampling: bool = True
    downsample_factor: float = 0.8
    quality_adaptive: bool = True
    min_quality_threshold: float = 0.5

class FrameBuffer:
    """Thread-safe frame buffer with automatic cleanup"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.frames_added = 0
        self.frames_processed = 0
        self.frames_dropped = 0
    
    def add_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Add frame to buffer, returns False if dropped"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                # Drop oldest frame
                self.buffer.popleft()
                self.frames_dropped += 1
            
            frame_data = {
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self.frames_added,
                'metadata': metadata or {}
            }
            
            self.buffer.append(frame_data)
            self.frames_added += 1
            return True
    
    def get_frame(self) -> Optional[Dict[str, Any]]:
        """Get next frame from buffer"""
        with self.lock:
            if self.buffer:
                self.frames_processed += 1
                return self.buffer.popleft()
            return None
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get most recent frame, discarding others"""
        with self.lock:
            if not self.buffer:
                return None
            
            # Drop all but the latest frame
            latest = self.buffer[-1]
            dropped_count = len(self.buffer) - 1
            self.frames_dropped += dropped_count
            self.buffer.clear()
            self.buffer.append(latest)
            
            self.frames_processed += 1
            return self.buffer.popleft()
    
    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            self.buffer.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'size': len(self.buffer),
                'max_size': self.max_size,
                'frames_added': self.frames_added,
                'frames_processed': self.frames_processed,
                'frames_dropped': self.frames_dropped,
                'drop_rate': self.frames_dropped / max(self.frames_added, 1)
            }

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, log_interval: int = 30):
        self.log_interval = log_interval
        self.metrics_history = deque(maxlen=1000)
        self.last_log_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # System monitoring (if available)
        self.process = None
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except Exception:
                logger.warning("Could not initialize psutil process monitoring")
        
    def record_frame_time(self, frame_time: float):
        """Record frame processing time"""
        self.frame_times.append(frame_time)
    
    def record_processing_time(self, processing_time: float):
        """Record model processing time"""
        self.processing_times.append(processing_time)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        now = time.time()
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            recent_times = list(self.frame_times)[-10:]
            avg_frame_time = np.mean(recent_times)
            fps = 1.0 / max(avg_frame_time, 0.001)
        else:
            fps = 0.0
        
        # Calculate processing times
        frame_proc_time = np.mean(self.frame_times) if self.frame_times else 0.0
        model_proc_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        # System metrics
        cpu_percent = 0.0
        memory_mb = 0.0
        
        if self.process is not None:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
            except Exception:
                # Fallback if psutil fails
                pass
        
        # GPU usage (if available)
        gpu_usage = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except (ImportError, Exception):
            # GPUtil not available or no GPU detected
            pass
        
        metrics = PerformanceMetrics(
            fps=fps,
            frame_processing_time=frame_proc_time,
            model_inference_time=model_proc_time,
            total_latency=frame_proc_time + model_proc_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_mb,
            gpu_usage=gpu_usage,
            timestamp=now
        )
        
        self.metrics_history.append(metrics)
        
        # Log performance if interval elapsed
        if now - self.last_log_time > self.log_interval:
            self._log_performance(metrics)
            self.last_log_time = now
        
        return metrics
    
    def _log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        logger.info(f"Performance: FPS={metrics.fps:.1f}, "
                   f"Latency={metrics.total_latency*1000:.1f}ms, "
                   f"CPU={metrics.cpu_usage:.1f}%, "
                   f"Memory={metrics.memory_usage:.1f}MB")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary over recent history"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        return {
            'avg_fps': np.mean([m.fps for m in recent_metrics]),
            'min_fps': np.min([m.fps for m in recent_metrics]),
            'max_fps': np.max([m.fps for m in recent_metrics]),
            'avg_latency_ms': np.mean([m.total_latency * 1000 for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_mb': np.mean([m.memory_usage for m in recent_metrics]),
            'total_samples': len(recent_metrics)
        }

class AdaptiveQualityManager:
    """Manages adaptive quality based on performance"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_quality = 1.0
        self.performance_history = deque(maxlen=30)
        
    def update_quality(self, metrics: PerformanceMetrics) -> float:
        """Update quality based on performance metrics"""
        if not self.config.quality_adaptive:
            return 1.0
        
        self.performance_history.append(metrics)
        
        if len(self.performance_history) < 10:
            return self.current_quality
        
        # Calculate average FPS over recent frames
        avg_fps = np.mean([m.fps for m in list(self.performance_history)[-10:]])
        target_fps = self.config.target_fps
        
        # Adjust quality based on performance
        if avg_fps < target_fps * 0.8:  # Significantly below target
            self.current_quality = max(0.6, self.current_quality - 0.1)
        elif avg_fps < target_fps * 0.9:  # Slightly below target
            self.current_quality = max(0.8, self.current_quality - 0.05)
        elif avg_fps > target_fps * 1.1:  # Above target
            self.current_quality = min(1.0, self.current_quality + 0.05)
        
        return self.current_quality
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get current processing parameters based on quality"""
        return {
            'resize_factor': self.current_quality,
            'skip_frames': int((1.0 - self.current_quality) * 3),
            'model_precision': 'float16' if self.current_quality < 0.8 else 'float32',
            'enable_optimizations': self.current_quality < 0.9
        }

class RealTimeOptimizer:
    """Main real-time optimization system"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Core components
        self.frame_buffer = FrameBuffer(self.config.max_queue_size)
        self.performance_monitor = PerformanceMonitor(self.config.performance_log_interval)
        self.quality_manager = AdaptiveQualityManager(self.config)
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_worker_threads)
        self.processing_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        
        # State management
        self.is_running = False
        self.worker_threads = []
        self.frame_counter = 0
        self.last_cleanup = 0
        
        # Processing callbacks
        self.feature_extractor = None
        self.model_predictor = None
        self.result_callback = None
        
    def initialize(self, feature_extractor: Callable, model_predictor: Callable, 
                  result_callback: Callable = None):
        """Initialize with processing functions"""
        self.feature_extractor = feature_extractor
        self.model_predictor = model_predictor
        self.result_callback = result_callback
        
        logger.info("Real-time optimizer initialized")
    
    def start(self):
        """Start the optimization system"""
        if self.is_running:
            logger.warning("Optimizer already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.config.num_worker_threads):
            thread = threading.Thread(target=self._worker_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        # Start result processing thread
        result_thread = threading.Thread(target=self._result_thread)
        result_thread.daemon = True
        result_thread.start()
        
        logger.info(f"Started optimizer with {self.config.num_worker_threads} worker threads")
    
    def stop(self):
        """Stop the optimization system"""
        self.is_running = False
        
        # Clear queues
        self.frame_buffer.clear()
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Real-time optimizer stopped")
    
    def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Process a single frame (main entry point)"""
        if not self.is_running or not self.feature_extractor or not self.model_predictor:
            return False
        
        start_time = time.time()
        
        # Get current performance metrics
        metrics = self.performance_monitor.get_current_metrics()
        
        # Update quality settings
        current_quality = self.quality_manager.update_quality(metrics)
        processing_params = self.quality_manager.get_processing_params()
        
        # Apply frame preprocessing based on quality
        processed_frame = self._preprocess_frame(frame, processing_params)
        
        # Check if we should skip this frame
        if self._should_skip_frame(metrics):
            return False
        
        # Add frame to buffer
        frame_metadata = {
            'original_shape': frame.shape,
            'quality': current_quality,
            'processing_params': processing_params,
            'metrics': metrics,
            **(metadata or {})
        }
        
        success = self.frame_buffer.add_frame(processed_frame, frame_metadata)
        
        # Try to add to processing queue (non-blocking)
        try:
            frame_data = self.frame_buffer.get_latest_frame()
            if frame_data:
                self.processing_queue.put_nowait(frame_data)
        except queue.Full:
            # Queue is full, frame will be dropped
            pass
        
        # Record timing
        frame_time = time.time() - start_time
        self.performance_monitor.record_frame_time(frame_time)
        
        # Periodic cleanup
        self.frame_counter += 1
        if self.frame_counter - self.last_cleanup > self.config.memory_cleanup_interval:
            self._cleanup_memory()
            self.last_cleanup = self.frame_counter
        
        return success
    
    def _preprocess_frame(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Preprocess frame based on quality parameters"""
        processed = frame.copy()
        
        # Apply resize if needed
        resize_factor = params.get('resize_factor', 1.0)
        if resize_factor < 1.0 and self.config.use_frame_downsampling:
            height, width = processed.shape[:2]
            new_height = int(height * resize_factor)
            new_width = int(width * resize_factor)
            processed = cv2.resize(processed, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return processed
    
    def _should_skip_frame(self, metrics: PerformanceMetrics) -> bool:
        """Determine if frame should be skipped"""
        # Skip if FPS is too low
        if metrics.fps < self.config.target_fps * 0.5:
            return True
        
        # Skip if processing is taking too long
        if metrics.total_latency > self.config.frame_skip_threshold:
            return True
        
        # Skip if queue is getting full
        if self.processing_queue.qsize() > self.config.max_queue_size * 0.8:
            return True
        
        return False
    
    def _worker_thread(self, worker_id: int):
        """Worker thread for processing frames"""
        logger.info(f"Worker thread {worker_id} started")
        
        while self.is_running:
            try:
                # Get frame from queue
                frame_data = self.processing_queue.get(timeout=0.1)
                
                if frame_data is None:
                    continue
                
                start_time = time.time()
                
                # Extract features
                feature_start = time.time()
                features = self.feature_extractor(frame_data['frame'])
                feature_time = time.time() - feature_start
                
                # Model prediction
                model_start = time.time()
                prediction = self.model_predictor(features)
                model_time = time.time() - model_start
                
                total_time = time.time() - start_time
                
                # Create result
                result = {
                    'prediction': prediction,
                    'features': features,
                    'frame_data': frame_data,
                    'processing_time': total_time,
                    'feature_time': feature_time,
                    'model_time': model_time,
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
                # Add to result queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Result queue full, drop result
                    pass
                
                # Record processing time
                self.performance_monitor.record_processing_time(total_time)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread {worker_id}: {e}")
    
    def _result_thread(self):
        """Thread for handling results"""
        logger.info("Result processing thread started")
        
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=0.1)
                
                if result is None:
                    continue
                
                # Call result callback if provided
                if self.result_callback:
                    try:
                        self.result_callback(result)
                    except Exception as e:
                        logger.error(f"Error in result callback: {e}")
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in result thread: {e}")
    
    def _cleanup_memory(self):
        """Periodic memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear old performance metrics
            while len(self.performance_monitor.metrics_history) > 500:
                self.performance_monitor.metrics_history.popleft()
            
            # Clear old frame/processing times
            while len(self.performance_monitor.frame_times) > 50:
                self.performance_monitor.frame_times.popleft()
            
            while len(self.performance_monitor.processing_times) > 50:
                self.performance_monitor.processing_times.popleft()
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        current_metrics = self.performance_monitor.get_current_metrics()
        performance_summary = self.performance_monitor.get_performance_summary()
        buffer_stats = self.frame_buffer.get_stats()
        quality_params = self.quality_manager.get_processing_params()
        
        return {
            'current_metrics': {
                'fps': current_metrics.fps,
                'latency_ms': current_metrics.total_latency * 1000,
                'cpu_usage': current_metrics.cpu_usage,
                'memory_mb': current_metrics.memory_usage,
                'gpu_usage': current_metrics.gpu_usage
            },
            'performance_summary': performance_summary,
            'buffer_statistics': buffer_stats,
            'quality_settings': {
                'current_quality': self.quality_manager.current_quality,
                'processing_params': quality_params
            },
            'queue_status': {
                'processing_queue_size': self.processing_queue.qsize(),
                'result_queue_size': self.result_queue.qsize(),
                'max_queue_size': self.config.max_queue_size
            },
            'system_info': {
                'worker_threads': len(self.worker_threads),
                'target_fps': self.config.target_fps,
                'frames_processed': self.frame_counter
            }
        }


# Utility functions for easy integration
def create_optimized_pipeline(feature_extractor: Callable, model_predictor: Callable,
                            result_callback: Callable = None, 
                            target_fps: int = 30) -> RealTimeOptimizer:
    """Create and initialize optimized processing pipeline"""
    config = OptimizationConfig(target_fps=target_fps)
    optimizer = RealTimeOptimizer(config)
    optimizer.initialize(feature_extractor, model_predictor, result_callback)
    return optimizer

def optimize_for_hardware() -> OptimizationConfig:
    """Create optimized configuration based on current hardware"""
    # Detect system capabilities
    cpu_count = mp.cpu_count()
    
    # Get memory info if available
    memory_gb = 8.0  # Default assumption
    if PSUTIL_AVAILABLE:
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            logger.warning("Could not detect memory size, using default 8GB assumption")
    
    # Adjust settings based on hardware
    if memory_gb < 4:
        # Low memory system
        config = OptimizationConfig(
            target_fps=20,
            max_queue_size=5,
            num_worker_threads=max(1, cpu_count // 2),
            use_frame_downsampling=True,
            downsample_factor=0.6
        )
    elif memory_gb < 8:
        # Medium memory system
        config = OptimizationConfig(
            target_fps=25,
            max_queue_size=8,
            num_worker_threads=max(2, cpu_count // 2),
            use_frame_downsampling=True,
            downsample_factor=0.8
        )
    else:
        # High memory system
        config = OptimizationConfig(
            target_fps=30,
            max_queue_size=10,
            num_worker_threads=min(4, cpu_count),
            use_frame_downsampling=False
        )
    
    logger.info(f"Optimized for hardware: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    return config


if __name__ == "__main__":
    # Test the optimization system
    config = optimize_for_hardware()
    optimizer = RealTimeOptimizer(config)
    
    # Mock functions for testing
    def mock_feature_extractor(frame):
        time.sleep(0.01)  # Simulate processing
        return np.random.rand(100)
    
    def mock_model_predictor(features):
        time.sleep(0.02)  # Simulate inference
        return {'letter': 'A', 'confidence': 0.95}
    
    def mock_result_callback(result):
        print(f"Result: {result['prediction']['letter']} "
              f"({result['prediction']['confidence']:.2f})")
    
    optimizer.initialize(mock_feature_extractor, mock_model_predictor, mock_result_callback)
    optimizer.start()
    
    # Simulate frame processing
    try:
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            optimizer.process_frame(frame)
            time.sleep(1/30)  # 30 FPS
            
            if i % 30 == 0:
                report = optimizer.get_performance_report()
                print(f"Performance: FPS={report['current_metrics']['fps']:.1f}")
    
    finally:
        optimizer.stop()