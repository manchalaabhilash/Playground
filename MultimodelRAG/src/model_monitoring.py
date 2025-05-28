import os
import time
import logging
from typing import Dict, Any, Optional, List
import threading
import functools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if prometheus_client is available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed, metrics will not be collected")
    PROMETHEUS_AVAILABLE = False

# Check if torch is available for GPU metrics
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("torch not installed, GPU metrics will not be collected")
    TORCH_AVAILABLE = False

# Define metrics if Prometheus is available
if PROMETHEUS_AVAILABLE:
    # Model metrics
    model_inference_latency = Histogram(
        'model_inference_latency_seconds', 
        'Time taken for model inference',
        ['model_id', 'operation']
    )
    
    model_inference_count = Counter(
        'model_inference_total',
        'Total number of model inferences',
        ['model_id', 'operation']
    )
    
    model_tokens_count = Counter(
        'model_tokens_total',
        'Total number of tokens processed',
        ['model_id', 'direction']  # direction: input, output
    )
    
    model_tokens_per_second = Gauge(
        'model_tokens_per_second',
        'Tokens processed per second',
        ['model_id', 'direction']
    )
    
    model_errors_count = Counter(
        'model_errors_total',
        'Total number of model errors',
        ['model_id', 'error_type']
    )
    
    # GPU metrics
    if TORCH_AVAILABLE:
        gpu_memory_allocated = Gauge(
            'gpu_memory_allocated_mb',
            'GPU memory allocated in MB',
            ['device']
        )
        
        gpu_memory_reserved = Gauge(
            'gpu_memory_reserved_mb',
            'GPU memory reserved in MB',
            ['device']
        )

# Function to start Prometheus HTTP server
def start_prometheus_server(port: int = 9090) -> None:
    """
    Start Prometheus HTTP server
    
    Args:
        port: Port to listen on
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not installed, metrics server not started")
        return
    
    try:
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {str(e)}")

# Function to get GPU memory usage
def get_gpu_memory_usage() -> Dict[str, Dict[str, float]]:
    """
    Get GPU memory usage
    
    Returns:
        Dictionary with GPU memory usage
    """
    if not TORCH_AVAILABLE:
        return {}
    
    try:
        result = {}
        
        # Get number of GPUs
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            # Get memory usage
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)    # MB
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                gpu_memory_allocated.labels(device=f"cuda:{i}").set(allocated)
                gpu_memory_reserved.labels(device=f"cuda:{i}").set(reserved)
            
            # Add to result
            result[f"cuda:{i}"] = {
                "allocated_mb": allocated,
                "reserved_mb": reserved
            }
        
        return result
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {str(e)}")
        return {}

# Decorator for monitoring model operations
def model_metrics(model_id: str, operation: str):
    """
    Decorator for monitoring model operations
    
    Args:
        model_id: Model ID
        operation: Operation name
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            # Record start time
            start_time = time.time()
            
            try:
                # Call function
                result = func(*args, **kwargs)
                
                # Record latency
                latency = time.time() - start_time
                model_inference_latency.labels(model_id=model_id, operation=operation).observe(latency)
                
                # Increment inference count
                model_inference_count.labels(model_id=model_id, operation=operation).inc()
                
                # Record token counts if available in result
                if isinstance(result, dict) and 'input_tokens' in result:
                    model_tokens_count.labels(model_id=model_id, direction='input').inc(result['input_tokens'])
                
                if isinstance(result, dict) and 'output_tokens' in result:
                    model_tokens_count.labels(model_id=model_id, direction='output').inc(result['output_tokens'])
                    
                    # Calculate tokens per second if latency > 0
                    if latency > 0 and result['output_tokens'] > 0:
                        tokens_per_second = result['output_tokens'] / latency
                        model_tokens_per_second.labels(model_id=model_id, direction='output').set(tokens_per_second)
                
                return result
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                model_errors_count.labels(model_id=model_id, error_type=error_type).inc()
                
                # Re-raise exception
                raise
        
        return wrapper
    
    return decorator

# Start GPU monitoring thread
def start_gpu_monitoring(interval_seconds: int = 10) -> threading.Thread:
    """
    Start GPU monitoring thread
    
    Args:
        interval_seconds: Monitoring interval in seconds
    
    Returns:
        Monitoring thread
    """
    if not TORCH_AVAILABLE or not PROMETHEUS_AVAILABLE:
        return None
    
    def monitor_gpu():
        while True:
            try:
                get_gpu_memory_usage()
            except Exception as e:
                logger.error(f"Error in GPU monitoring thread: {str(e)}")
            
            time.sleep(interval_seconds)
    
    thread = threading.Thread(target=monitor_gpu, daemon=True)
    thread.start()
    logger.info(f"Started GPU monitoring thread with interval {interval_seconds}s")
    
    return thread

# Initialize monitoring if enabled
if os.environ.get("PROMETHEUS_ENABLED", "false").lower() in ("true", "1", "yes"):
    port = int(os.environ.get("PROMETHEUS_PORT", "9090"))
    start_prometheus_server(port)
    
    # Start GPU monitoring if enabled
    if os.environ.get("GPU_MONITORING_ENABLED", "true").lower() in ("true", "1", "yes"):
        interval = int(os.environ.get("GPU_MONITORING_INTERVAL", "10"))
        start_gpu_monitoring(interval)
