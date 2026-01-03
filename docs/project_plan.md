# Bird Detection Project – Iteration 1 Milestones

This document outlines the major milestones for Iteration 1 of the Bird Detection Project.  
Each milestone defines goals, deliverables, tasks, skills to be learned, and tools involved.

---

## Milestone 1: Edge Image Capture & Pre-Filtering (Raspberry Pi)

### Goal
Build a robust, fault-tolerant image capture agent that runs continuously on a Raspberry Pi, efficiently selects candidate frames, and communicates with a remote inference server when available.

### Core Deliverables
- Continuous image capture at a fixed interval (e.g. every N seconds)
- Cropping via configurable ROI parameters
- Lightweight frame-difference filtering to avoid redundant frames
- Async communication with inference server
- Local disk persistence with bounded storage when server is unavailable
- System-managed background service

### Key Tasks
- Camera setup and capture loop
- ROI-based cropping (config-driven)
- Frame differencing / motion detection
- Multi-threaded or async pipeline:
  - capture → filter → send/store
- Server availability check + retry logic
- Disk management:
  - rolling window storage
  - LRU / time-based cleanup
- systemd service setup with restart policies and logs

### Skills & Concepts Learned
- Camera fundamentals (exposure, resolution, intrinsics intuition)
- OpenCV basics (frame differencing, blur/thresholding)
- Multithreading vs async on constrained devices
- Linux internals:
  - systemd services
  - journald logs
  - resource limits
- Designing resilient edge agents

### Tools & Tech
- Raspberry Pi OS
- libcamera / Picamera2
- Python, OpenCV
- threading / asyncio / multiprocessing
- systemd, journald
- YAML / TOML config files

---

## Milestone 2: Inference & Data Server (DINOv3-based)

### Goal
Create a production-style inference server that performs detection, persists results, and exposes observability metrics.

### Core Deliverables
- FastAPI-based inference server
- DINOv3 model served via REST API
- Detection filtering + confidence logic
- Persistent storage of detections
- Observability (metrics, traces, health checks)
- Containerized local deployment

### Key Tasks
- FastAPI app structure and routing
- Image upload + preprocessing
- DINOv3 inference pipeline
- Detection post-processing and validation
- Database schema:
  - images
  - detections
  - metadata (timestamps, versions)
- Metrics & tracing:
  - request latency
  - error rates
  - throughput
- API documentation & testing
- Docker + Docker Compose setup

### Skills & Concepts Learned
- API design & versioning
- Pydantic-based schema validation
- Model serving patterns
- Database modeling for ML systems
- Observability fundamentals (metrics, traces)
- Containerized development workflows

### Tools & Tech
- FastAPI, Uvicorn
- Pydantic
- PyTorch + DINOv3
- PostgreSQL
- OpenTelemetry
- Grafana + Prometheus
- Docker, Docker Compose
- pytest, HTTPX

---

## Milestone 3: Efficient Model Training, Distillation & Quantization

### Goal
Build a full ML lifecycle pipeline that turns collected detections into smaller, faster, and continuously improving models.

### Core Deliverables
- Dataset creation from stored detections
- Training pipeline for a compact detection model
- Knowledge distillation from DINOv3
- Quantized inference-ready model
- Model evaluation & comparison metrics
- Automated retraining pipeline

### Key Tasks
- Dataset extraction & cleaning
- Label validation and filtering
- Train a lightweight detector (YOLO / RT-DETR small, etc.)
- Pseudo-labeling using teacher model
- Distillation training loop
- Quantization (PTQ / QAT)
- Performance evaluation:
  - accuracy
  - false positives
  - inference latency
- Automated pipelines triggered by new data

### Skills & Concepts Learned
- Dataset engineering & hygiene
- Semi-supervised learning
- Knowledge distillation
- Model compression & quantization
- Drift detection & performance regression
- ML pipeline orchestration

### Tools & Tech
- PyTorch
- Ultralytics / MMDetection
- ONNX, TensorRT / TFLite
- Weights & Biases or MLflow
- Airflow or Dagster
- Docker
- Evaluation tooling (COCO metrics)

---

## Milestone 4: Web & Mobile UI for Monitoring & Visualization

### Goal
Build a developer- and user-facing UI to visualize detections, metrics, and system health.

### Core Deliverables
- Web dashboard for:
  - detection events
  - images
  - model & server metrics
- Responsive design (desktop + mobile)
- Secure API integration
- Typed end-to-end communication

### Key Tasks
- Frontend project setup
- Metrics visualization
- Image gallery with filters
- API integration using shared schemas
- Authentication (basic token/session)
- Testing and linting

### Skills & Concepts Learned
- Modern full-stack development
- Type-safe APIs
- State management & UI architecture
- Frontend testing strategies
- Design systems & component reuse

### Tools & Tech
- TypeScript
- Next.js / React
- Tailwind CSS
- shadcn/ui
- tRPC
- Prisma
- Turborepo
- Playwright / Vitest

---
