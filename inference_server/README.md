# Grounding DINO Inference Server

FastAPI-based inference server for bird detection using the Grounding DINO model.

## Overview

This server provides a REST API for bird detection in images. It uses the Grounding DINO (Grounding DINO with Transformer) model for zero-shot object detection with text prompts.

### Features

- **FastAPI REST API**: High-performance async API
- **GPU Acceleration**: CUDA support for fast inference
- **Configurable Detection**: Customizable thresholds and prompts
- **Health Monitoring**: Built-in health checks and statistics
- **Docker Support**: Containerized deployment with docker-compose
- **Metrics**: Inference statistics and performance monitoring

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Installing NVIDIA Container Toolkit

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Start the server
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Test the Server

```bash
# Health check
curl http://localhost:6061/health

# Test detection with an image
curl -X POST http://localhost:6061/detect \
  -F "image=@path/to/your/image.jpg" \
  -F "device_id=test-device"

# Check Prometheus metrics
curl http://localhost:6061/metrics
```

## API Endpoints

### 1. Root Endpoint

```bash
GET /
```

Returns basic server information.

**Response:**
```json
{
  "service": "Grounding DINO Inference Server",
  "version": "1.0.0",
  "status": "running"
}
```

### 2. Health Check

```bash
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2026-01-03T10:30:00.123456"
}
```

### 3. Detect Birds

```bash
POST /detect
```

Main detection endpoint. Accepts an image and returns bird detections.

**Parameters:**
- `image` (required): Image file (multipart/form-data)
- `timestamp` (optional): Capture timestamp
- `device_id` (optional): ID of the device that captured the image
- `text_prompt` (optional): Custom text prompt (default: "bird")
- `box_threshold` (optional): Box confidence threshold (default: 0.35)
- `text_threshold` (optional): Text matching threshold (default: 0.25)

**Example:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@bird.jpg" \
  -F "device_id=rpi-edge-001" \
  -F "timestamp=2026-01-03T10:30:00"
```

**Response:**
```json
{
  "num_detections": 2,
  "boxes": [
    [150.5, 200.3, 450.2, 550.8],
    [600.1, 100.5, 800.3, 350.7]
  ],
  "labels": ["bird", "bird"],
  "confidences": [0.87, 0.92],
  "inference_time_ms": 125.3,
  "image_size": {
    "width": 1024,
    "height": 1024
  },
  "metadata": {
    "device_id": "rpi-edge-001",
    "timestamp": "2026-01-03T10:30:00",
    "server_timestamp": "2026-01-03T10:30:01.234567"
  }
}
```

### 4. Get Statistics

```bash
GET /stats
```

Returns inference statistics.

**Response:**
```json
{
  "total_inferences": 1523,
  "total_time_seconds": 189.45,
  "average_time_ms": 124.3,
  "model_config": {
    "text_prompt": "bird",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "device": "cuda"
  }
}
```

### 5. Get Configuration

```bash
GET /config
```

Returns current server configuration.

**Response:**
```json
{
  "inference": {
    "text_prompt": "bird",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "max_image_size": 1333,
    "min_image_size": 800
  },
  "model": {
    "device": "cuda"
  }
}
```

### 6. Prometheus Metrics

```bash
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring and alerting.

**Example:**
```bash
curl http://localhost:6061/metrics
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `inference_http_requests_total` | Counter | Total HTTP requests by method, endpoint, and status |
| `inference_detections_total` | Counter | Total number of bird detections |
| `inference_detection_requests_total` | Counter | Total number of detection requests |
| `inference_duration_seconds` | Histogram | Time spent on model inference |
| `inference_request_duration_seconds` | Histogram | Total request processing time by endpoint |
| `inference_model_loaded` | Gauge | Whether the model is loaded (1) or not (0) |
| `inference_model_load_time_seconds` | Gauge | Time taken to load the model |
| `inference_errors_total` | Counter | Total number of errors by error type |
| `inference_image_size_bytes` | Histogram | Size of uploaded images in bytes |
| `inference_detection_confidence` | Histogram | Confidence scores distribution of detections |

**Example Response:**
```
# HELP inference_http_requests_total Total HTTP requests
# TYPE inference_http_requests_total counter
inference_http_requests_total{endpoint="/detect",method="POST",status="200"} 152.0
inference_http_requests_total{endpoint="/health",method="GET",status="200"} 45.0

# HELP inference_detections_total Total number of bird detections
# TYPE inference_detections_total counter
inference_detections_total 287.0

# HELP inference_duration_seconds Time spent on model inference
# TYPE inference_duration_seconds histogram
inference_duration_seconds_bucket{le="0.1"} 85.0
inference_duration_seconds_bucket{le="0.25"} 142.0
inference_duration_seconds_bucket{le="0.5"} 152.0
inference_duration_seconds_sum 18.45
inference_duration_seconds_count 152.0
```

## Configuration

Edit `config.yaml` to customize the server settings:

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000

# Model settings
model:
  device: "cuda"  # or "cpu"

# Inference settings
inference:
  text_prompt: "bird"
  box_threshold: 0.35
  text_threshold: 0.25
```

After editing the configuration, restart the container:

```bash
docker-compose restart
```

## Development

### Running Locally (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Grounding DINO
cd GroundingDINO
pip install -e . --no-build-isolation
cd ..

# Download model weights
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..

# Update config.yaml paths for local development
# Then run the server
python server.py
```

### Testing with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Detect birds
with open("bird_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"image": f},
        data={"device_id": "test-device"}
    )
print(response.json())
```

## Performance Optimization

### GPU Memory

The model requires approximately 4-6 GB of GPU memory. If you encounter out-of-memory errors, you can:

1. Reduce `max_image_size` in config.yaml
2. Use CPU inference (set `device: "cpu"` in config.yaml)

### Batch Processing

Currently, the server processes one image at a time. For high-throughput scenarios, consider:

1. Running multiple server instances
2. Using a load balancer (e.g., nginx)
3. Implementing batch processing (future enhancement)

## Monitoring

### Docker Logs

```bash
# View logs
docker-compose logs -f

# View logs for specific time range
docker-compose logs --since 1h
```

### Prometheus Integration

The server exposes Prometheus metrics at the `/metrics` endpoint on the main server port (6061). Configure Prometheus to scrape this endpoint:

**prometheus.yml example:**
```yaml
scrape_configs:
  - job_name: 'bird-detection-inference'
    static_configs:
      - targets: ['bird-detection-inference:6061']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Note:** All endpoints (API + metrics) are on the same port (6061) for simplicity. The `/metrics` endpoint is accessible to Prometheus on the internal Docker network.

**Key metrics to monitor:**
- `inference_duration_seconds`: Track inference performance
- `inference_detections_total`: Monitor detection counts
- `inference_errors_total`: Alert on errors
- `inference_model_loaded`: Ensure model is loaded
- `inference_request_duration_seconds`: Monitor end-to-end latency

### Testing the Server

Use the provided test script to verify all endpoints:

```bash
# Test all endpoints (including /metrics)
python test_server.py http://localhost:6061

# Test with an image
python test_server.py http://localhost:6061 path/to/bird.jpg
```

### Grafana Dashboards

Create dashboards to visualize:
1. **Request Rate**: `rate(inference_http_requests_total[5m])`
2. **Detection Rate**: `rate(inference_detections_total[5m])`
3. **Error Rate**: `rate(inference_errors_total[5m])`
4. **P95 Latency**: `histogram_quantile(0.95, inference_duration_seconds_bucket)`
5. **Average Confidence**: `histogram_quantile(0.5, inference_detection_confidence_bucket)`

## Troubleshooting

### GPU Not Detected

```bash
# Check if NVIDIA Docker runtime is available
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If this fails, ensure nvidia-container-toolkit is installed
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Model Loading Errors

Ensure the model weights are downloaded correctly:

```bash
docker-compose exec inference-server ls -lh /resources/GroundingDINO/weights/
```

### Port Already in Use

If port 8000 is already in use, edit `docker-compose.yml` to use a different port:

```yaml
ports:
  - "8080:8000"  # Maps host port 8080 to container port 8000
```

## Project Structure

```
grounding_dino_instance/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── server.py               # FastAPI server implementation
├── config.yaml             # Server configuration
├── requirements.txt        # Python dependencies
├── inference.py            # Original inference script (reference)
└── README.md               # This file
```

## Integration with Edge Devices

The Raspberry Pi edge capture agent sends images to this server using the `/detect` endpoint. Configure the RPI's `config.yaml`:

```yaml
server:
  url: "http://<server-ip>:6061/detect"
  timeout_sec: 10
  retry_attempts: 3
```

## License

See main project LICENSE.

## References

- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
