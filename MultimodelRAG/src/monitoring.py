"""
Monitoring and metrics collection for the MultimodalRAG system.
Provides performance tracking, usage statistics, and system health monitoring.
"""

import time
import threading
import logging
import os
import json
import psutil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

# Configure logging
logger = logging.getLogger('multimodal_rag.monitoring')

class PerformanceMetrics:
    """Collects and reports performance metrics for the MultimodalRAG system"""
    
    def __init__(self, metrics_dir: str = "logs/metrics"):
        """
        Initialize the performance metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.retrieval_metrics = defaultdict(list)
        self.llm_metrics = defaultdict(list)
        self.system_metrics = []
        
        # Initialize locks for thread safety
        self.metrics_lock = threading.Lock()
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def record_request(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
        """
        with self.metrics_lock:
            self.request_times[endpoint].append({
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
                "status_code": status_code
            })
    
    def record_error(self, component: str, error_type: str) -> None:
        """
        Record error occurrence.
        
        Args:
            component: System component where error occurred
            error_type: Type of error
        """
        error_key = f"{component}:{error_type}"
        with self.metrics_lock:
            self.error_counts[error_key] += 1
    
    def record_retrieval_metrics(self, 
                                query_type: str, 
                                num_results: int, 
                                latency_ms: float,
                                modality: str) -> None:
        """
        Record retrieval performance metrics.
        
        Args:
            query_type: Type of query
            num_results: Number of results retrieved
            latency_ms: Retrieval latency in milliseconds
            modality: Retrieval modality (text, image, multimodal)
        """
        with self.metrics_lock:
            self.retrieval_metrics[modality].append({
                "timestamp": datetime.now().isoformat(),
                "query_type": query_type,
                "num_results": num_results,
                "latency_ms": latency_ms
            })
    
    def record_llm_metrics(self, 
                          model: str, 
                          prompt_tokens: int, 
                          completion_tokens: int,
                          latency_ms: float) -> None:
        """
        Record LLM performance metrics.
        
        Args:
            model: LLM model name
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            latency_ms: LLM response latency in milliseconds
        """
        with self.metrics_lock:
            self.llm_metrics[model].append({
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms
            })
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system resource metrics.
        
        Returns:
            Dictionary of system metrics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "open_files": len(psutil.Process().open_files()),
            "threads": len(psutil.Process().threads())
        }
    
    def _background_monitoring(self) -> None:
        """Background thread for periodic system metrics collection"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                with self.metrics_lock:
                    self.system_metrics.append(metrics)
                
                # Save metrics periodically
                if len(self.system_metrics) >= 60:  # Save after collecting ~1 hour of data (at 1 min intervals)
                    self._save_metrics()
                
                # Sleep for 60 seconds
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in background monitoring: {str(e)}")
                time.sleep(60)  # Sleep and retry
    
    def _save_metrics(self) -> None:
        """Save collected metrics to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save request metrics
        with open(os.path.join(self.metrics_dir, f"request_metrics_{timestamp}.json"), 'w') as f:
            json.dump(dict(self.request_times), f)
        self.request_times.clear()
        
        # Save error counts
        with open(os.path.join(self.metrics_dir, f"error_metrics_{timestamp}.json"), 'w') as f:
            json.dump(dict(self.error_counts), f)
        self.error_counts.clear()
        
        # Save retrieval metrics
        with open(os.path.join(self.metrics_dir, f"retrieval_metrics_{timestamp}.json"), 'w') as f:
            json.dump(dict(self.retrieval_metrics), f)
        self.retrieval_metrics.clear()
        
        # Save LLM metrics
        with open(os.path.join(self.metrics_dir, f"llm_metrics_{timestamp}.json"), 'w') as f:
            json.dump(dict(self.llm_metrics), f)
        self.llm_metrics.clear()
        
        # Save system metrics
        with open(os.path.join(self.metrics_dir, f"system_metrics_{timestamp}.json"), 'w') as f:
            json.dump(self.system_metrics, f)
        self.system_metrics = []
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary of current metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        with self.metrics_lock:
            # Calculate request time statistics
            request_stats = {}
            for endpoint, times in self.request_times.items():
                if times:
                    durations = [t["duration_ms"] for t in times]
                    request_stats[endpoint] = {
                        "count": len(durations),
                        "avg_duration_ms": np.mean(durations),
                        "p50_ms": np.percentile(durations, 50),
                        "p95_ms": np.percentile(durations, 95),
                        "p99_ms": np.percentile(durations, 99)
                    }
            
            # Calculate retrieval statistics
            retrieval_stats = {}
            for modality, metrics in self.retrieval_metrics.items():
                if metrics:
                    latencies = [m["latency_ms"] for m in metrics]
                    retrieval_stats[modality] = {
                        "count": len(latencies),
                        "avg_latency_ms": np.mean(latencies),
                        "p95_latency_ms": np.percentile(latencies, 95)
                    }
            
            # Calculate LLM statistics
            llm_stats = {}
            for model, metrics in self.llm_metrics.items():
                if metrics:
                    latencies = [m["latency_ms"] for m in metrics]
                    prompt_tokens = [m["prompt_tokens"] for m in metrics]
                    completion_tokens = [m["completion_tokens"] for m in metrics]
                    
                    llm_stats[model] = {
                        "count": len(latencies),
                        "avg_latency_ms": np.mean(latencies),
                        "p95_latency_ms": np.percentile(latencies, 95),
                        "avg_prompt_tokens": np.mean(prompt_tokens),
                        "avg_completion_tokens": np.mean(completion_tokens)
                    }
            
            # Get latest system metrics
            latest_system = self.system_metrics[-1] if self.system_metrics else {}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.system_metrics[0]["timestamp"])).total_seconds() if self.system_metrics else 0,
                "request_stats": request_stats,
                "error_counts": dict(self.error_counts),
                "retrieval_stats": retrieval_stats,
                "llm_stats": llm_stats,
                "system_metrics": latest_system
            }
    
    def shutdown(self) -> None:
        """Shutdown monitoring and save final metrics"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self._save_metrics()


class RequestTracker:
    """Tracks API requests for rate limiting and usage monitoring"""
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        """
        Initialize the request tracker.
        
        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier (IP or API key)
        
        Returns:
            True if client is within rate limit, False otherwise
        """
        with self.lock:
            # Clean up old requests
            current_time = time.time()
            while (self.requests[client_id] and 
                   self.requests[client_id][0] < current_time - self.window_size):
                self.requests[client_id].popleft()
            
            # Check if client has exceeded rate limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            # Record new request
            self.requests[client_id].append(current_time)
            return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self.lock:
            current_time = time.time()
            stats = {}
            
            for client_id, requests in self.requests.items():
                # Clean up old requests
                while requests and requests[0] < current_time - self.window_size:
                    requests.popleft()
                
                # Calculate request rate
                stats[client_id] = {
                    "requests_in_window": len(requests),
                    "rate_limit_percent": (len(requests) / self.max_requests) * 100
                }
            
            return stats