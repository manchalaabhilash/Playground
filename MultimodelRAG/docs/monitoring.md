# Monitoring Guide

This document provides information about the monitoring capabilities of the MultimodalRAG system.

## Overview

The system includes comprehensive monitoring using Prometheus and Grafana:

- **Prometheus**: Collects and stores metrics from the system
- **Grafana**: Visualizes metrics in dashboards

## Metrics Collected

### Model Performance Metrics

- **Inference Latency**: Time taken for model inference (p50, p95, p99)
- **Tokens Per Second**: Generation speed
- **Memory Usage**: Memory used by models
- **GPU Memory**: Allocated and reserved GPU memory
- **Error Rates**: Count of errors by type
- **Inference Count**: Total number of inferences
- **Token Count**: Input and output tokens processed

### System Metrics

- **CPU Usage**: Overall and per-container CPU usage
- **Memory Usage**: Overall and per-container memory usage
- **Disk Space**: Available disk space
- **Network Traffic**: Network I/O

## Setup

### Using Docker Compose

1. Start the monitoring stack:

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

2. Start the application with monitoring enabled:

```bash
docker-compose up -d
```

3. Access Grafana at http://localhost:3000 with the default credentials:
   - Username: admin
   - Password: admin

4. Navigate to the dashboards section to view the pre-configured dashboards:
   - Model Performance Dashboard
   - System Performance Dashboard

## Prometheus Configuration

Prometheus is configured to scrape metrics from the following endpoints:

- API service: http://api:5000/metrics
- Model API service: http://model-api:5001/metrics
- Node exporter: http://node-exporter:9100/metrics

The scrape interval is set to 15 seconds by default.

## Available Dashboards

### Model Performance Dashboard

This dashboard provides insights into model performance:

- Inference latency (p50, p95, p99)
- Tokens per second
- GPU memory usage
- Model errors
- Total inferences and tokens
- Model usage distribution

### System Performance Dashboard

This dashboard provides insights into system performance:

- CPU usage
- Memory usage
- Container resource usage
- Disk space
- Network traffic

## Custom Metrics

### Model Metrics

The system exposes the following custom metrics:

- `model_inference_latency_seconds`: Histogram of inference latency
- `model_inference_total`: Counter of total inferences
- `model_tokens_total`: Counter of total tokens processed
- `model_tokens_per_second`: Gauge of tokens per second
- `model_errors_total`: Counter of errors by type
- `gpu_memory_allocated_mb`: Gauge of allocated GPU memory
- `gpu_memory_reserved_mb`: Gauge of reserved GPU memory

### API Metrics

The system exposes the following API metrics:

- `http_requests_total`: Counter of total HTTP requests
- `http_request_duration_seconds`: Histogram of request duration
- `http_request_size_bytes`: Histogram of request size
- `http_response_size_bytes`: Histogram of response size

## Alerting

Alerting is configured in Grafana for the following conditions:

- High inference latency (p95 > 5s)
- High error rate (> 5% of requests)
- Low GPU memory (< 10% available)
- High CPU usage (> 90% for 5 minutes)
- High memory usage (> 90% for 5 minutes)
- Low disk space (< 10% available)

## Troubleshooting

### Common Issues

1. **Metrics not showing up in Grafana**:
   - Check if Prometheus is running: `docker-compose ps`
   - Check if the services are exposing metrics: `curl http://localhost:5000/metrics`
   - Check Prometheus targets: http://localhost:9090/targets

2. **High latency alerts**:
   - Check GPU utilization
   - Check model cache configuration
   - Consider scaling the service or optimizing the models

3. **Memory usage alerts**:
   - Check for memory leaks
   - Adjust model cache size
   - Consider scaling the service

## Extending Monitoring

### Adding Custom Metrics

To add custom metrics to your code:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
custom_counter = Counter('custom_metric_total', 'Description of the metric', ['label1', 'label2'])
custom_histogram = Histogram('custom_duration_seconds', 'Description of the duration', ['label1'])
custom_gauge = Gauge('custom_value', 'Description of the value', ['label1'])

# Use metrics
custom_counter.labels(label1='value1', label2='value2').inc()
custom_histogram.labels(label1='value1').observe(duration)
custom_gauge.labels(label1='value1').set(value)
```

### Creating Custom Dashboards

1. Log in to Grafana
2. Click on "Create" > "Dashboard"
3. Add panels using the metrics from Prometheus
4. Save the dashboard

### Exporting Dashboards

To export a dashboard for version control:

1. Open the dashboard
2. Click on the settings icon (gear)
3. Click on "JSON Model"
4. Copy the JSON or use the "Save to file" button
5. Save the JSON file in the `monitoring/grafana/dashboards/` directory
