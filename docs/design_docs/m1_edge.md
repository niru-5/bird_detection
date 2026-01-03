# Design Doc: Milestone 1 – Edge Image Capture & Pre-Filtering Agent

**Project:** Bird Detection Project  
**Milestone:** 1 – Edge Image Capture & Pre-Filtering  
**Target Platform:** Raspberry Pi (Pi Camera)  
**Status:** Draft  
**Author:** <your-name>  
**Last Updated:** <date>

---

## 1. Problem Statement

We need a reliable, long-running edge agent that captures images from a Raspberry Pi–attached camera, filters redundant frames, and forwards candidate images to a remote inference server when available. The system must tolerate network failures, power interruptions, and limited storage, while remaining configurable and resource-efficient.

This agent is the foundation of the entire bird detection pipeline: bad data here will propagate downstream and degrade model performance.

---

## 2. Goals

- Capture images at a fixed, configurable interval during operating hours (9 AM - 5 PM)
- Reduce redundant or useless frames via lightweight filtering
- Support configurable ROI cropping
- Communicate asynchronously with a remote inference server
- Persist data locally when the server is unavailable
- Enforce bounded disk usage
- Run as a resilient system-managed background service
- Expose HTTP metrics endpoint for health monitoring

---

## 3. Non-Goals (Iteration 1)

- Species classification
- On-device deep learning inference
- Complex motion tracking or object detection
- UI or visualization on the Raspberry Pi
- High-FPS video capture

---

## 4. Constraints

- Limited CPU, RAM, and storage
- Intermittent or unreliable network connectivity
- Must survive reboots and crashes without manual intervention
- Must not block capture due to slow network or inference server
- Configuration must be editable without code changes

---

## 5. High-Level Architecture

```
+-------------------+
|    Pi Camera      |
+---------+---------+
          |
          v
+-------------------+
|  Capture Loop     |
| (interval-based)  |
+---------+---------+
          |
          v
+-------------------+
|   ROI Cropper     |
+---------+---------+
          |
          v
+-------------------+
|  Frame Filter     | (frame diff / motion check)
+---------+---------+
          |
          v
+-------------------+ +---------------------+
| Async Dispatcher  |------>| Inference Server |
|                   |       | (if available)   |
|                   |       +---------------------+
|                   |
|  +-----> Local Disk Store
+-------------------+
```

---

## 6. Component Breakdown

### 6.1 Capture Service
- Periodically captures frames from the Pi camera during operating hours
- Capture interval configurable (e.g. `capture_interval_sec`)
- Operating hours configurable (default: 9 AM - 5 PM)
- Sleeps or idles outside operating hours to conserve resources
- Emits timestamped raw frames

### 6.2 ROI Cropper
- Applies a rectangular crop based on config
- Coordinates defined in normalized or pixel space
- Allows masking irrelevant regions (e.g. sky, street)

### 6.3 Frame Difference Filter
- Compares current frame to previous accepted frame
- Uses simple grayscale + absolute difference
- Threshold-based decision to accept or discard frame
- Prevents flooding downstream systems

### 6.4 Async Dispatcher
- Non-blocking pipeline stage
- Attempts to send accepted frames to inference server
- Falls back to local storage on failure
- Retries with exponential backoff

### 6.5 Local Disk Persistence
- Stores images in a structured directory layout
- Enforces disk quota via:
  - rolling window
  - time-based deletion
- Maintains metadata (timestamp, reason stored)

### 6.6 Metrics HTTP Server
- Embedded lightweight HTTP server (Flask)
- Exposes `/health`, `/metrics`, and `/stats` endpoints
- Runs on configurable port (default: 8080)
- Non-blocking, runs in separate thread
- Provides real-time service health and statistics
- Prometheus-compatible metrics format

### 6.7 System Integration
- Runs as a `systemd` service
- Auto-restart on crash
- Logs to `journald`
- Resource limits applied (CPU, memory)

---

## 7. Data & File Layout

```
/var/lib/bird-capture/
├── images/
│ ├── 2026-01-01/
│ │ ├── 12-00-01.jpg
│ │ ├── 12-00-06.jpg
├── metadata/
│ └── events.jsonl
├── config.yaml
└── state.json
```

---

## 8. Configuration

Example `config.yaml`:
```yaml
capture_interval_sec: 5

operating_hours:
  enabled: true
  start_hour: 9      # 9 AM
  end_hour: 17       # 5 PM
  timezone: "local"

roi:
  x: 0.1
  y: 0.2
  width: 0.6
  height: 0.5

filter:
  frame_diff_threshold: 25
  min_changed_pixels: 500

storage:
  max_disk_mb: 3072 # 3 GB

server:
  url: "http://server:8000/detect"
  timeout_sec: 3

metrics:
  enabled: true
  port: 8080
  host: "0.0.0.0"
```

---

## 9. Failure Modes & Recovery

| Failure | Mitigation |
|---------|------------|
| Inference server down | Store locally, retry later |
| Network timeout | Non-blocking send + fallback |
| Disk full | Enforce deletion policy |
| Process crash | systemd auto-restart |
| Camera failure | Log error + retry capture |
| Power loss | Stateless restart |

---

## 10. Observability

- Structured logs (JSON-style)
- HTTP metrics server on port 8080:
  - `/health` - Service health check
  - `/metrics` - Prometheus-compatible metrics
  - `/stats` - JSON statistics summary
- Key metrics exposed:
  - frames captured
  - frames filtered
  - frames sent
  - frames stored locally
  - operating hours state (active/inactive)
  - disk usage
  - service uptime
- systemd health:
  - restart count
  - uptime
- External monitoring via Prometheus/Grafana

---

## 11. Security & Privacy

- Metrics HTTP server on port 8080 (local network only, read-only)
- Server communication via token-based auth (future)
- Configurable ROI to avoid capturing private areas
- Operating hours to prevent unwanted recording (e.g., night time)
- No permanent storage beyond retention window
- Metrics endpoints expose no sensitive data (aggregate stats only)

---

## 12. Testing Strategy

**Unit tests:**
- ROI cropping logic
- Frame differencing thresholds

**Integration tests:**
- Simulated server unavailable
- Disk limit enforcement

**Manual tests:**
- Reboot recovery
- Network unplugged scenarios

---

## 13. Open Questions

- Pixel vs normalized ROI coordinates?
- Best default thresholds for frame differencing?
- JPEG vs PNG tradeoffs?
- Async model: threads vs asyncio?
- Operating hours: Should they be timezone-aware or clock-based?
- Metrics server: Flask vs built-in http.server vs aiohttp?
- Should operating hours automatically adjust for daylight (sunrise/sunset)?

---

## 14. Success Criteria

- Runs continuously for 24+ hours without manual intervention
- No data loss when server is unavailable
- Disk usage remains bounded
- Captured frames meaningfully reduce redundancy
- Clean, readable logs suitable for debugging
- Camera only operates during configured hours (9 AM - 5 PM)
- Metrics endpoint responds reliably and accurately
- External monitoring tools successfully scrape health data
