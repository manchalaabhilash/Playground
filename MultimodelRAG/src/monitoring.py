"""
Monitoring utilities for the MultimodalRAG system.
Provides performance metrics, request tracking, and alerting.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# Configure logging
logger = logging.getLogger('multimodal_rag.monitoring')

class PerformanceMetrics:
    """Tracks performance metrics for the system"""
    
    def __init__(self):
        """Initialize performance metrics"""
        self.request_counts = defaultdict(int)
        self.request_durations = defaultdict(list)
        self.status_counts = defaultdict(int)
        self.retrieval_metrics = defaultdict(list)
        self.llm_metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.lock = threading.RLock()
        
        # System metrics
        self.system_metrics = {
            "start_time": datetime.now().isoformat(),
            "total_requests": 0,
            "total_errors": 0
        }
        
        # Try to initialize Prometheus metrics if available
        try:
            from prometheus_client import Counter, Histogram, Gauge
            
            # Request metrics
            self.prom_requests = Counter(
                'multimodal_rag_requests_total', 
                'Total requests', 
                ['endpoint', 'status']
            )
            
            self.prom_request_duration = Histogram(
                'multimodal_rag_request_duration_seconds',
                'Request duration in seconds',
                ['endpoint']
            )
            
            # Retrieval metrics
            self.prom_retrieval_latency = Histogram(
                'multimodal_rag_retrieval_latency_seconds',
                'Retrieval latency in seconds',
                ['query_type', 'modality']
            )
            
            self.prom_retrieval_count = Histogram(
                'multimodal_rag_retrieval_results',
                'Number of retrieval results',
                ['query_type', 'modality']
            )
            
            # LLM metrics
            self.prom_llm_latency = Histogram(
                'multimodal_rag_llm_latency_seconds',
                'LLM latency in seconds',
                ['model']
            )
            
            self.prom_llm_tokens = Histogram(
                'multimodal_rag_llm_tokens',
                'LLM token usage',
                ['model', 'type']  # type is 'prompt' or 'completion'
            )
            
            # Error metrics
            self.prom_errors = Counter(
                'multimodal_rag_errors_total',
                'Total errors',
                ['component', 'error_type']
            )
            
            # System metrics
            self.prom_system_memory = Gauge(
                'multimodal_rag_memory_usage_bytes',
                'Memory usage in bytes'
            )
            
            self.has_prometheus = True
        except ImportError:
            self.has_prometheus = False
    
    def record_request(self, endpoint: str, duration_ms: float, status: int) -> None:
        """
        Record a request.
        
        Args:
            endpoint: API endpoint
            duration_ms: Request duration in milliseconds
            status: HTTP status code
        """
        with self.lock:
            self.request_counts[endpoint] += 1
            self.request_durations[endpoint].append(duration_ms)
            self.status_counts[f"{endpoint}:{status}"] += 1
            
            # Update system metrics
            self.system_metrics["total_requests"] += 1
            
            # Record Prometheus metrics if available
            if self.has_prometheus:
                self.prom_requests.labels(endpoint=endpoint, status=str(status)).inc()
                self.prom_request_duration.labels(endpoint=endpoint).observe(duration_ms / 1000)
    
    def record_retrieval_metrics(self, query_type: str, num_results: int, 
                               latency_ms: float, modality: str) -> None:
        """
        Record retrieval metrics.
        
        Args:
            query_type: Type of query
            num_results: Number of results retrieved
            latency_ms: Retrieval latency in milliseconds
            modality: Retrieval modality (text, image)
        """
        with self.lock:
            key = f"{query_type}:{modality}"
            self.retrieval_metrics[key].append({
                "num_results": num_results,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat()
            })
            
            # Record Prometheus metrics if available
            if self.has_prometheus:
                self.prom_retrieval_latency.labels(
                    query_type=query_type, 
                    modality=modality
                ).observe(latency_ms / 1000)
                
                self.prom_retrieval_count.labels(
                    query_type=query_type,
                    modality=modality
                ).observe(num_results)
    
    def record_llm_metrics(self, model: str, prompt_tokens: int, 
                         completion_tokens: int, latency_ms: float) -> None:
        """
        Record LLM metrics.
        
        Args:
            model: LLM model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency_ms: LLM latency in milliseconds
        """
        with self.lock:
            self.llm_metrics[model].append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat()
            })
            
            # Record Prometheus metrics if available
            if self.has_prometheus:
                self.prom_llm_latency.labels(model=model).observe(latency_ms / 1000)
                self.prom_llm_tokens.labels(model=model, type="prompt").observe(prompt_tokens)
                self.prom_llm_tokens.labels(model=model, type="completion").observe(completion_tokens)
    
    def record_error(self, component: str, error_type: str) -> None:
        """
        Record an error.
        
        Args:
            component: System component
            error_type: Type of error
        """
        with self.lock:
            key = f"{component}:{error_type}"
            self.error_counts[key] += 1
            
            # Update system metrics
            self.system_metrics["total_errors"] += 1
            
            # Record Prometheus metrics if available
            if self.has_prometheus:
                self.prom_errors.labels(component=component, error_type=error_type).inc()
    
    def update_system_metrics(self) -> None:
        """Update system metrics"""
        with self.lock:
            # Update uptime
            start_time = datetime.fromisoformat(self.system_metrics["start_time"])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            self.system_metrics["uptime_seconds"] = uptime_seconds
            
            # Update memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                self.system_metrics["memory_usage_bytes"] = memory_info.rss
                
                # Record Prometheus metrics if available
                if self.has_prometheus:
                    self.prom_system_memory.set(memory_info.rss)
            except ImportError:
                pass
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        with self.lock:
            # Update system metrics
            self.update_system_metrics()
            
            # Calculate request stats
            request_stats = {}
            for endpoint, durations in self.request_durations.items():
                if durations:
                    request_stats[endpoint] = {
                        "count": self.request_counts[endpoint],
                        "avg_duration_ms": sum(durations) / len(durations),
                        "min_duration_ms": min(durations),
                        "max_duration_ms": max(durations),
                        "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else None
                    }
            
            # Calculate status code distribution
            status_distribution = {}
            for key, count in self.status_counts.items():
                endpoint, status = key.split(":")
                if endpoint not in status_distribution:
                    status_distribution[endpoint] = {}
                status_distribution[endpoint][status] = count
            
            return {
                "system_metrics": self.system_metrics,
                "request_stats": request_stats,
                "status_distribution": status_distribution,
                "error_counts": dict(self.error_counts)
            }
    
    def shutdown(self) -> None:
        """Shutdown metrics collection"""
        # Nothing to do for basic metrics
        pass

class RequestTracker:
    """Tracks API requests for rate limiting"""
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        """
        Initialize the request tracker.
        
        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(list)
        self.lock = threading.RLock()
    
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if within limit, False otherwise
        """
        with self.lock:
            # Clean up old requests
            self._cleanup(client_id)
            
            # Check if adding a new request would exceed the limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            # Add new request
            self.requests[client_id].append(time.time())
            return True
    
    def _cleanup(self, client_id: str) -> None:
        """
        Remove old requests outside the time window.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.requests:
            current_time = time.time()
            cutoff_time = current_time - self.window_size
            
            # Keep only requests within the time window
            self.requests[client_id] = [
                timestamp for timestamp in self.requests[client_id]
                if timestamp > cutoff_time
            ]
    
    def get_request_count(self, client_id: str) -> int:
        """
        Get number of requests for a client within the time window.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Number of requests
        """
        with self.lock:
            self._cleanup(client_id)
            return len(self.requests[client_id])

class PerformanceMonitor:
    """High-level monitoring interface for the MultimodalRAG system"""
    
    def __init__(self):
        """Initialize the performance monitor"""
        self.metrics = PerformanceMetrics()
        self.request_tracker = RequestTracker()
        self.alerts = AlertManager()
        
        # Initialize monitoring dashboard if available
        try:
            from prometheus_client import start_http_server
            self.has_prometheus = True
            start_http_server(8000)  # Start Prometheus metrics server
        except ImportError:
            self.has_prometheus = False
    
    def track_request(self, endpoint: str, client_id: str, start_time: float) -> Dict[str, Any]:
        """
        Track an API request.
        
        Args:
            endpoint: API endpoint
            client_id: Client identifier
            start_time: Request start time
        
        Returns:
            Dictionary with tracking information
        """
        # Check rate limit
        within_limit = self.request_tracker.check_rate_limit(client_id)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics (if request is processed)
        if within_limit:
            self.metrics.record_request(endpoint, duration_ms, 200)
        else:
            self.metrics.record_request(endpoint, duration_ms, 429)  # Too Many Requests
        
        return {
            "within_rate_limit": within_limit,
            "duration_ms": duration_ms
        }
    
    def track_retrieval(self, query_type: str, modality: str, start_time: float, results: List[Any]) -> None:
        """
        Track a retrieval operation.
        
        Args:
            query_type: Type of query
            modality: Retrieval modality
            start_time: Operation start time
            results: Retrieved results
        """
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_retrieval_metrics(
            query_type=query_type,
            num_results=len(results),
            latency_ms=duration_ms,
            modality=modality
        )
        
        # Check for slow retrievals and alert if necessary
        if duration_ms > 1000:  # Alert on retrievals taking more than 1 second
            self.alerts.add_alert(
                "slow_retrieval",
                f"Slow {modality} retrieval: {duration_ms:.2f}ms for {query_type}",
                level="warning"
            )
    
    def track_llm(self, model: str, prompt: str, response: str, start_time: float) -> None:
        """
        Track an LLM operation.
        
        Args:
            model: LLM model name
            prompt: Input prompt
            response: LLM response
            start_time: Operation start time
        """
        duration_ms = (time.time() - start_time) * 1000
        
        # Estimate token counts (rough approximation)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response.split())
        
        self.metrics.record_llm_metrics(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=duration_ms
        )
        
        # Check for slow LLM responses and alert if necessary
        if duration_ms > 5000:  # Alert on LLM calls taking more than 5 seconds
            self.alerts.add_alert(
                "slow_llm",
                f"Slow LLM response: {duration_ms:.2f}ms for {model}",
                level="warning"
            )
    
    def track_error(self, component: str, error_type: str, error_message: str) -> None:
        """
        Track an error.
        
        Args:
            component: System component
            error_type: Type of error
            error_message: Error message
        """
        self.metrics.record_error(component, error_type)
        
        # Add alert for critical errors
        self.alerts.add_alert(
            f"error_{component}",
            f"Error in {component}: {error_type} - {error_message}",
            level="error"
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dictionary with health status information
        """
        # Get summary metrics
        metrics_summary = self.metrics.get_summary_metrics()
        
        # Get active alerts
        active_alerts = self.alerts.get_active_alerts()
        
        # Determine overall health status
        if any(alert["level"] == "critical" for alert in active_alerts):
            health_status = "critical"
        elif any(alert["level"] == "error" for alert in active_alerts):
            health_status = "error"
        elif any(alert["level"] == "warning" for alert in active_alerts):
            health_status = "warning"
        else:
            health_status = "healthy"
        
        # Get system metrics
        system_metrics = metrics_summary.get("system_metrics", {})
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "active_alerts": active_alerts,
            "system_metrics": system_metrics,
            "request_stats": metrics_summary.get("request_stats", {}),
            "error_counts": metrics_summary.get("error_counts", {})
        }
    
    def shutdown(self) -> None:
        """Shutdown monitoring"""
        self.metrics.shutdown()


class AlertManager:
    """Manages alerts for the monitoring system"""
    
    def __init__(self, max_alerts: int = 100):
        """
        Initialize the alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to store
        """
        self.alerts = deque(maxlen=max_alerts)
        self.active_alerts = {}
        self.lock = threading.Lock()
    
    def add_alert(self, alert_id: str, message: str, level: str = "info", 
                 auto_resolve_seconds: Optional[int] = None) -> None:
        """
        Add a new alert.
        
        Args:
            alert_id: Unique alert identifier
            message: Alert message
            level: Alert level (info, warning, error, critical)
            auto_resolve_seconds: Seconds after which to auto-resolve the alert
        """
        with self.lock:
            timestamp = datetime.now()
            
            alert = {
                "id": alert_id,
                "message": message,
                "level": level,
                "timestamp": timestamp.isoformat(),
                "resolved": False,
                "resolved_timestamp": None
            }
            
            # Add to alerts history
            self.alerts.append(alert.copy())
            
            # Update active alerts
            self.active_alerts[alert_id] = alert
            
            # Set up auto-resolve if specified
            if auto_resolve_seconds:
                threading.Timer(
                    auto_resolve_seconds, 
                    self.resolve_alert, 
                    args=[alert_id]
                ).start()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
        
        Returns:
            True if alert was resolved, False if not found
        """
        with self.lock:
            if alert_id in self.active_alerts:
                # Mark as resolved
                self.active_alerts[alert_id]["resolved"] = True
                self.active_alerts[alert_id]["resolved_timestamp"] = datetime.now().isoformat()
                
                # Add resolved alert to history
                self.alerts.append(self.active_alerts[alert_id].copy())
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                return True
            return False
    
    def get_active_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active alerts, optionally filtered by level.
        
        Args:
            level: Alert level to filter by
        
        Returns:
            List of active alerts
        """
        with self.lock:
            if level:
                return [
                    alert for alert in self.active_alerts.values()
                    if alert["level"] == level
                ]
            return list(self.active_alerts.values())
    
    def get_alert_history(self, 
                         limit: int = 50, 
                         level: Optional[str] = None,
                         since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get alert history, optionally filtered.
        
        Args:
            limit: Maximum number of alerts to return
            level: Alert level to filter by
            since: Only return alerts after this time
        
        Returns:
            List of alerts
        """
        with self.lock:
            filtered_alerts = self.alerts
            
            if level:
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert["level"] == level
                ]
            
            if since:
                since_str = since.isoformat()
                filtered_alerts = [
                    alert for alert in filtered_alerts
                    if alert["timestamp"] >= since_str
                ]
            
            # Return most recent alerts first, up to limit
            return list(reversed(list(filtered_alerts)))[-limit:]
