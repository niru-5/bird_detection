# Bird Detection Project - Architecture

**Last Updated:** 2026-01-03
**Status:** Draft

---

## Overview

The Bird Detection Project is a distributed system for automated bird detection and monitoring using edge devices (Raspberry Pi) with remote inference capabilities. The architecture is designed to handle intermittent connectivity, limited edge resources, and provide a scalable foundation for data collection and analysis.

---

## System Components

### 1. Edge Capture Agent (Raspberry Pi)

The edge agent is a long-running service responsible for continuous image capture, preprocessing, and intelligent data transmission.

**Key Responsibilities:**
- Periodic image capture from Pi Camera (day-time only: 9 AM - 5 PM)
- Region of Interest (ROI) cropping
- Frame redundancy filtering
- Asynchronous communication with inference server
- Local data persistence during network outages
- Resilient operation with automatic recovery
- HTTP metrics endpoint for health monitoring

**Technology Stack:**
- Python 3
- picamera2 / libcamera
- OpenCV (for image processing)
- systemd (service management)
- YAML (configuration)

---

### 2. Metrics HTTP Server (Embedded)

**Purpose:** Lightweight HTTP server embedded within the edge agent for health monitoring.

**Key Responsibilities:**
- Expose real-time metrics via HTTP endpoint
- Provide service health status
- Report capture statistics and operational state
- Enable external monitoring tools to scrape metrics

**Endpoints:**
- `GET /health` - Service health check
- `GET /metrics` - Prometheus-compatible metrics
- `GET /stats` - JSON statistics summary

**Technology:** Flask (lightweight) or built-in http.server

---

### 3. Inference Server (Remote)

**Purpose:** Centralized bird detection and classification service.

**Key Responsibilities:**
- Receive images from edge devices
- Run detection models
- Store detection results
- (Future) Provide feedback to edge agents

**Status:** Milestone 2 (not yet implemented)

---

## Edge Agent Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Edge Capture Agent                       │
│                                                               │
│  ┌──────────────┐                                            │
│  │  Pi Camera   │                                            │
│  └──────┬───────┘                                            │
│         │                                                     │
│         v                                                     │
│  ┌──────────────────┐                                        │
│  │  Capture Loop    │  (configurable interval)               │
│  │  + Time Check    │  (9 AM - 5 PM only)                    │
│  └──────┬───────────┘                                        │
│         │                                                     │
│         v                                                     │
│  ┌──────────────────┐                                        │
│  │   ROI Cropper    │  (config-driven crop)                  │
│  └──────┬───────────┘                                        │
│         │                                                     │
│         v                                                     │
│  ┌──────────────────┐                                        │
│  │  Frame Diff      │  (motion/change detection)             │
│  │    Filter        │  threshold-based filtering             │
│  └──────┬───────────┘                                        │
│         │                                                     │
│         v                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │       Async Dispatcher               │                   │
│  │                                       │                   │
│  │  ┌─────────────┐    ┌──────────────┐ │                   │
│  │  │ Send Queue  │───>│ HTTP Client  │─┼──> Inference     │
│  │  └─────────────┘    └──────────────┘ │    Server         │
│  │         │                             │                   │
│  │         │ (on failure)                │                   │
│  │         v                             │                   │
│  │  ┌─────────────────────────────────┐ │                   │
│  │  │   Local Disk Persistence        │ │                   │
│  │  │  - Timestamped storage          │ │                   │
│  │  │  - Quota enforcement            │ │                   │
│  │  │  - Metadata tracking            │ │                   │
│  │  └─────────────────────────────────┘ │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Metrics HTTP Server (Port 8080)              │ │
│  │  Endpoints: /health, /metrics, /stats                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                         ↑                                     │
│                         │ (External monitoring tools)         │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Configuration & State                     │ │
│  │  - config.yaml (user-editable)                          │ │
│  │  - state.json (runtime state)                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Normal Operation (Server Available)

```
1. Check current time - if outside operating hours (9 AM - 5 PM), sleep
2. Camera captures frame every N seconds
3. Apply ROI crop (if configured)
4. Compare with previous frame using grayscale diff
5. If change exceeds threshold:
   a. Queue frame for transmission
   b. Async send to inference server
   c. Update metrics
   d. Log success
6. Else: discard frame, update metrics
```

### Degraded Mode (Server Unavailable)

```
1. Check current time - if outside operating hours (9 AM - 5 PM), sleep
2. Camera captures frame every N seconds
3. Apply ROI crop (if configured)
4. Compare with previous frame
5. If change exceeds threshold:
   a. Attempt send to server
   b. On timeout/failure: save to local disk
   c. Queue for retry (with backoff)
   d. Update metrics
6. Background process periodically retries queued images
```

---

## File System Layout

```
/var/lib/bird-capture/
├── images/                    # Captured frames
│   ├── 2026-01-01/
│   │   ├── 12-00-01.jpg
│   │   ├── 12-00-06.jpg
│   │   └── ...
│   ├── 2026-01-02/
│   └── ...
├── metadata/
│   └── events.jsonl           # Event log (captures, sends, failures)
├── config.yaml                # User configuration
└── state.json                 # Runtime state (last frame, queue, etc.)
```

---

## Configuration Architecture

### config.yaml Structure

```yaml
capture_interval_sec: 5

operating_hours:
  enabled: true
  start_hour: 9      # 9 AM
  end_hour: 17       # 5 PM (24-hour format)
  timezone: "local"  # Use system timezone

roi:
  enabled: true
  x: 0.1        # Normalized coordinates [0-1]
  y: 0.2
  width: 0.6
  height: 0.5

filter:
  enabled: true
  frame_diff_threshold: 25        # Grayscale diff threshold
  min_changed_pixels: 500         # Minimum pixels changed

storage:
  max_disk_mb: 3072               # 3 GB quota
  retention_days: 7               # Auto-delete after N days

server:
  url: "http://inference-server:8000/detect"
  timeout_sec: 3
  retry_attempts: 3
  retry_backoff_sec: 5

metrics:
  enabled: true
  port: 8080
  host: "0.0.0.0"    # Listen on all interfaces
```

### Design Principles

- **Human-editable:** YAML format, clear naming
- **Hot-reload capable:** (future) detect config changes without restart
- **Validation:** Schema validation on startup
- **Defaults:** Sensible defaults for all optional parameters

---

## Resilience & Fault Tolerance

### Service Management

- **systemd service:** Auto-start on boot, auto-restart on crash
- **Resource limits:** CPU and memory caps via systemd
- **Graceful shutdown:** Signal handling for clean termination
- **Stateless design:** Can restart without data loss

### Network Resilience

| Scenario | Behavior |
|----------|----------|
| Inference server down | Store images locally, retry with backoff |
| Network timeout | Non-blocking send, fallback to disk |
| Partial network failure | Queue management with exponential backoff |
| Server returns error | Log error, retry with backoff |

### Storage Resilience

| Scenario | Behavior |
|----------|----------|
| Disk approaching quota | Delete oldest images first |
| Disk full | Stop capture, alert via logs |
| Corrupted state file | Reinitialize with defaults |
| Missing directories | Auto-create on startup |

### Hardware Resilience

| Scenario | Behavior |
|----------|----------|
| Camera failure | Log error, retry capture next interval |
| Power loss | Clean restart via systemd, resume operation |
| Temperature throttling | Log warning, continue at reduced rate |

---

## Observability

### Logging Strategy

- **Structured logs:** JSON-formatted for machine parsing
- **Log levels:** DEBUG, INFO, WARNING, ERROR
- **Log destination:** systemd journald
- **Key events logged:**
  - Capture attempts and results
  - Filter decisions (accept/reject)
  - Send attempts and outcomes
  - Disk operations
  - Configuration changes
  - Operating hours state changes
  - Errors and exceptions

### Metrics HTTP Server

The edge agent exposes a lightweight HTTP server for real-time monitoring.

**Endpoints:**

1. **`GET /health`** - Service health check
   ```json
   {
     "status": "healthy|degraded|unhealthy",
     "uptime_seconds": 86400,
     "last_capture_time": "2026-01-03T14:30:00Z",
     "camera_status": "ok|error",
     "disk_usage_percent": 45.2,
     "current_operating_state": "active|inactive|sleeping"
   }
   ```

2. **`GET /metrics`** - Prometheus-compatible metrics
   ```
   # HELP frames_captured_total Total frames captured from camera
   # TYPE frames_captured_total counter
   frames_captured_total 1234

   # HELP frames_filtered_total Frames discarded by filter
   # TYPE frames_filtered_total counter
   frames_filtered_total 567

   # HELP frames_sent_total Frames successfully sent to server
   # TYPE frames_sent_total counter
   frames_sent_total 456

   # HELP frames_stored_locally_total Frames stored to local disk
   # TYPE frames_stored_locally_total counter
   frames_stored_locally_total 211

   # HELP send_failures_total Failed attempts to send to server
   # TYPE send_failures_total counter
   send_failures_total 15

   # HELP disk_usage_bytes Current disk usage in bytes
   # TYPE disk_usage_bytes gauge
   disk_usage_bytes 1572864000

   # HELP service_uptime_seconds Service uptime in seconds
   # TYPE service_uptime_seconds gauge
   service_uptime_seconds 86400

   # HELP operating_hours_active Is service in active hours (1=yes, 0=no)
   # TYPE operating_hours_active gauge
   operating_hours_active 1
   ```

3. **`GET /stats`** - Human-readable JSON summary
   ```json
   {
     "service": {
       "uptime_seconds": 86400,
       "status": "healthy",
       "version": "1.0.0"
     },
     "capture": {
       "frames_captured": 1234,
       "frames_filtered": 567,
       "frames_sent": 456,
       "frames_stored_locally": 211,
       "last_capture_time": "2026-01-03T14:30:00Z"
     },
     "operating_hours": {
       "enabled": true,
       "currently_active": true,
       "start_hour": 9,
       "end_hour": 17,
       "next_state_change": "2026-01-03T17:00:00Z"
     },
     "storage": {
       "disk_usage_mb": 1500,
       "disk_quota_mb": 3072,
       "usage_percent": 48.8
     },
     "network": {
       "send_failures": 15,
       "last_successful_send": "2026-01-03T14:25:00Z"
     }
   }
   ```

**Implementation:**
- Runs in separate thread/async task to avoid blocking capture
- Minimal overhead (<5 MB RAM, <1% CPU)
- Optional authentication via API token (future)

### Health Monitoring

- **HTTP health endpoint:** `curl http://pi-device:8080/health`
- **systemd status:** `systemctl status bird-capture`
- **Restart count:** Track crash frequency
- **Uptime:** Service availability
- **Log analysis:** Error rate trends
- **External monitoring:** Prometheus scraping, Grafana dashboards

---

## Security Considerations

### Current Implementation

- **Metrics port exposure:** Port 8080 for monitoring (HTTP only, local network)
- **Configurable ROI:** Avoid capturing sensitive areas
- **Local-only access:** Service runs as non-root user
- **Data retention:** Automatic cleanup of old images
- **Metrics security:** Read-only endpoints, no write operations exposed

### Security Considerations for Metrics Server

- **Network isolation:** Recommend binding to local network only
- **Firewall rules:** Restrict access to trusted monitoring systems
- **No sensitive data:** Metrics expose only aggregate statistics
- **Rate limiting:** (Future) Prevent metrics endpoint abuse

### Future Enhancements

- Token-based authentication for server communication
- TLS/HTTPS for encrypted transmission (metrics + inference)
- API token authentication for metrics endpoints
- Image hashing for integrity verification
- Access control for configuration files
- mTLS for metrics scraping

---

## Scalability & Future Considerations

### Horizontal Scaling

- **Multiple edge devices:** Each runs independently
- **Load distribution:** Server-side concern
- **Device identification:** (future) unique device IDs

### Vertical Scaling

- **CPU/Memory:** Optimized for Raspberry Pi constraints
- **Storage:** Bounded by quota configuration
- **Camera resolution:** Configurable (affects processing load)

### Future Milestones

- **M2:** Inference server implementation
- **M3:** Model training pipeline
- **M4:** Feedback loop (edge model updates)
- **M5:** Multi-device coordination and analytics

---

## Technology Choices & Rationale

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3 | Raspberry Pi support, rich ecosystem |
| Camera API | picamera2 | Official support for Pi Camera |
| Image Processing | OpenCV | Efficient, well-documented |
| Configuration | YAML | Human-readable, standard format |
| Service Management | systemd | Native to Raspberry Pi OS |
| Logging | Python logging + journald | Standard, queryable logs |
| Async I/O | asyncio | Non-blocking network operations |
| Storage Format | JPEG | Good compression, widely supported |
| Metrics Server | Flask | Lightweight, minimal dependencies, easy integration |
| Time Handling | Python datetime | Built-in, reliable timezone support |

---

## Open Architectural Decisions

### Current Decisions Needed

1. **ROI Coordinates:** Pixel-based vs normalized (0-1)?
   - Recommendation: Normalized for resolution independence

2. **Frame Format:** JPEG vs PNG?
   - Recommendation: JPEG for smaller file sizes

3. **Async Model:** Threading vs asyncio?
   - Recommendation: asyncio for better resource control

4. **Retry Strategy:** Fixed backoff vs exponential?
   - Recommendation: Exponential backoff with jitter

### Future Architectural Considerations

- **On-device inference:** Add optional lightweight model
- **Multi-camera support:** Single agent, multiple cameras
- **Edge coordination:** Device-to-device communication
- **Compression:** Video encoding vs individual frames
- **Power management:** Sleep modes for battery operation

---

## Dependencies

### System Dependencies

- Raspberry Pi OS (Debian-based)
- Python 3.9+
- libcamera
- systemd

### Python Dependencies

- picamera2
- opencv-python
- PyYAML
- requests (or httpx for async)
- pillow
- flask (for metrics HTTP server)
- prometheus_client (optional, for Prometheus metrics formatting)

---

## Testing Architecture

### Unit Testing

- ROI cropping logic
- Frame difference calculation
- Configuration validation
- Storage quota enforcement
- Operating hours time checking logic
- Metrics calculation and formatting

### Integration Testing

- End-to-end capture pipeline
- Server unavailable scenarios
- Disk limit enforcement
- Configuration reload
- Metrics HTTP endpoints
- Time-based capture activation/deactivation

### System Testing

- 24+ hour continuous operation (across operating hours transitions)
- Network disconnect/reconnect
- Power cycle recovery
- Resource usage profiling
- Metrics server load testing
- Operating hours edge cases (midnight, DST transitions)

---

## Success Metrics

- **Reliability:** 99%+ uptime over 7 days
- **Data integrity:** Zero frame loss when properly stored
- **Resource efficiency:** <20% CPU, <200MB RAM (including metrics server)
- **Storage compliance:** Never exceed configured quota
- **Responsiveness:** Continue capture during network issues
- **Operating hours compliance:** Camera only active 9 AM - 5 PM (±1 minute accuracy)
- **Metrics availability:** HTTP metrics endpoint responds within 100ms
- **Monitoring integration:** Prometheus successfully scrapes metrics every 15s
