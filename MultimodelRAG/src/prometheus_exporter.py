import time
import threading
import logging
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed. Prometheus metrics will not be available.")
    PROMETHEUS_AVAILABLE = False

class PrometheusExporter:
    """
    Export model metrics to Prometheus
    """
    
    def __init__(self, port: int = 9090, update_interval: int = 15):
        """
        Initialize the Prometheus exporter
        
        Args:
            port: Port to expose metrics on
            update_interval: Interval in seconds to update metrics
        """
        self.port = port
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
            return
        
        # Create metrics
        self.latency = Histogram(
            'model_inference_latency_seconds',
            'Model inference latency in seconds',
            ['model_id']
        )
        
        self.tokens_per_second = Gauge(
            'model_tokens_per_second',
            'Model throughput in tokens per second',
            ['model_id']
        )
        
        self.memory_usage = Gauge(
            'model_memory_usage_mb',
            'Model memory usage in MB',
            ['model_id']
        )
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total number of model errors',
            ['model_id', 'error_type']
        )
        
        self.gpu_memory_allocated = Gauge(
            'gpu_memory_allocated_mb',
            'GPU memory allocated in MB',
            ['device']
        )
        
        self.gpu_memory_reserved = Gauge(
            'gpu_memory_reserved_mb',
            'GPU memory reserved in MB',
            ['device']
        )
        
        self.inference_count = Counter(
            'model_inference_total',
            'Total number of model inferences',
            ['model_id']
        )
        
        self.token_count = Counter(
            'model_tokens_total',
            'Total number of tokens processed',
            ['model_id', 'direction']  # direction: input or output
        )
    
    def start(self):
        """Start the Prometheus exporter"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Cannot start exporter.")
            return
        
        if self.running:
            logger.warning("Prometheus exporter already running")
            return
        
        # Start HTTP server
        try:
            start_http_server(self.port)
            logger.info(f"Started Prometheus metrics server on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
            return
        
        # Start update thread
        self.running = True
        self.thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        self.thread.start()
        logger.info("Started metrics update thread")
    
    def stop(self):
        """Stop the Prometheus exporter"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            self.thread = None
        
        logger.info("Stopped Prometheus exporter")
    
    def _update_metrics_loop(self):
        """Update metrics in a loop"""
        from src.model_monitoring import model_metrics, get_gpu_memory_usage
        
        while self.running:
            try:
                # Update GPU memory metrics
                gpu_memory = get_gpu_memory_usage()
                if "error" not in gpu_memory:
                    for gpu_id, memory_info in gpu_memory.items():
                        self.gpu_memory_allocated.labels(device=gpu_id).set(memory_info["allocated_mb"])
                        self.gpu_memory_reserved.labels(device=gpu_id).set(memory_info["reserved_mb"])
                
                # Update model metrics from model_metrics
                all_metrics = model_metrics.get_all_metrics(window_seconds=60)  # Last minute
                
                for model_id, metrics in all_metrics.items():
                    # Update latency histogram
                    if "latency" in metrics:
                        for _, latency_ms in model_metrics.metrics.get(model_id, {}).get("latency", []):
                            # Convert ms to seconds for Prometheus
                            self.latency.labels(model_id=model_id).observe(latency_ms / 1000.0)
                    
                    # Update tokens per second gauge
                    if "tokens_per_second" in metrics:
                        tps_stats = metrics["tokens_per_second"]
                        if "mean" in tps_stats:
                            self.tokens_per_second.labels(model_id=model_id).set(tps_stats["mean"])
                    
                    # Update memory usage gauge
                    if "memory_usage" in metrics:
                        memory_stats = metrics["memory_usage"]
                        if "mean" in memory_stats:
                            self.memory_usage.labels(model_id=model_id).set(memory_stats["mean"])
                    
                    # Update error counter
                    if "errors" in metrics:
                        error_stats = metrics["errors"]
                        if "by_type" in error_stats:
                            for error_type, count in error_stats["by_type"].items():
                                # Get current count from Prometheus
                                current_count = self.error_counter._value.get((model_id, error_type), 0)
                                # Only increment if needed
                                if count > current_count:
                                    self.error_counter.labels(model_id=model_id, error_type=error_type).inc(count - current_count)
            
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {str(e)}")
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def record_inference(self, model_id: str, input_tokens: int, output_tokens: int, latency_seconds: float):
        """
        Record a model inference
        
        Args:
            model_id: Identifier for the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_seconds: Inference latency in seconds
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Increment inference counter
        self.inference_count.labels(model_id=model_id).inc()
        
        # Add to latency histogram
        self.latency.labels(model_id=model_id).observe(latency_seconds)
        
        # Increment token counters
        self.token_count.labels(model_id=model_id, direction="input").inc(input_tokens)
        self.token_count.labels(model_id=model_id, direction="output").inc(output_tokens)
        
        # Calculate and update tokens per second
        if latency_seconds > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / latency_seconds
            self.tokens_per_second.labels(model_id=model_id).set(tokens_per_second)
    
    def record_error(self, model_id: str, error_type: str):
        """
        Record a model error
        
        Args:
            model_id: Identifier for the model
            error_type: Type of error
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.error_counter.labels(model_id=model_id, error_type=error_type).inc()
    
    def update_memory_usage(self, model_id: str, memory_mb: float):
        """
        Update memory usage
        
        Args:
            model_id: Identifier for the model
            memory_mb: Memory usage in MB
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.memory_usage.labels(model_id=model_id).set(memory_mb)

# Create a global instance with environment variable configuration
prometheus_exporter = PrometheusExporter(
    port=int(os.environ.get("PROMETHEUS_PORT", "9090")),
    update_interval=int(os.environ.get("PROMETHEUS_UPDATE_INTERVAL", "15"))
)

# Auto-start if enabled
if os.environ.get("PROMETHEUS_ENABLED", "false").lower() in ("true", "1", "yes"):
    prometheus_exporter.start()
