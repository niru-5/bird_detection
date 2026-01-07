# Bird Detection Project

A bird detection system using Raspberry Pi for edge capture and a remote inference server for detection.

## Components

- **Edge Capture Agent (Raspberry Pi)**: Captures images from camera during operating hours, applies motion filtering, and sends frames to inference server
- **Inference Server**: FastAPI-based server using Grounding DINO for bird detection

## Setup Guides

### Port management
rpi ports -> 6051, 6052
rpi node exports -> 6053
inference server ports -> 6061, 6062
postgres server ports -> 6071, 6072
minio ports -> 6076, 6077
lakfs ports -> 6078
prometheus ports -> 6081 
grafana ports -> 6091
pc node exported -> 6095


### Raspberry Pi Setup
See detailed setup instructions in [rpi/README.md](rpi/README.md)

Quick overview:
1. Set up SSH key authentication
2. Install system packages (Python, picamera2, OpenCV dependencies)
3. Create Python virtual environment and install dependencies
4. Configure the application
5. Run manually or as a systemd service

### Inference Server Setup
See detailed setup instructions in [grounding_dino_instance/README.md](grounding_dino_instance/README.md)

Quick overview:
1. Install Docker and NVIDIA Container Toolkit
2. Build the Docker image
3. Configure detection parameters in config.yaml
4. Run with docker-compose
5. Test the API endpoints

## Architecture

```
┌─────────────────┐                    ┌──────────────────┐
│  Raspberry Pi   │                    │ Inference Server │
│  Edge Agent     │   HTTP POST        │  (GPU Server)    │
│                 │─────/detect────────▶│                  │
│  - Camera       │   (JPEG image)     │  - Grounding DINO│
│  - Filtering    │                    │  - FastAPI       │
│  - Local Store  │◀────JSON────────────│  - CUDA 12.1     │
└─────────────────┘  (Detections)      └──────────────────┘
```

## Quick Start

1. **Set up the Inference Server** (on a machine with GPU):
   ```bash
   cd grounding_dino_instance
   docker-compose up -d
   ```

2. **Set up the Raspberry Pi Edge Agent**:
   ```bash
   # On the Raspberry Pi
   cd rpi
   source venv/bin/activate
   python src/main.py --config configs/image_capture.yaml
   ```

3. **Configure the connection**:
   Update `rpi/configs/image_capture.yaml` with your inference server URL:
   ```yaml
   server:
     url: "http://<server-ip>:8000/detect"
   ```
