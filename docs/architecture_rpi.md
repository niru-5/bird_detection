# Architecture for RPI Code (main.py)

**File:** `rpi/src/main.py`
**Purpose:** Edge capture agent for bird detection on Raspberry Pi
**Last Updated:** 2026-01-03

---

## Threading Architecture

The application uses a **multi-threaded architecture** with two main threads:

### Thread 1: Main Thread (Capture Loop)
**Owner:** `EdgeCaptureAgent` class

**Responsibilities:**
- Image capture from Pi Camera
- Frame filtering
- Server communication (with retries)
- Local storage management
- Operating hours enforcement

**Classes/Components in Main Thread:**

1. **`EdgeCaptureAgent`**
   - Orchestrates the entire capture pipeline
   - Runs the main capture loop
   - Coordinates all components

2. **`CaptureService`**
   - Manages Raspberry Pi Camera (picamera2)
   - Captures frames at configured intervals
   - Applies ROI cropping
   - Owned by EdgeCaptureAgent

3. **`StorageManager`**
   - Handles local disk storage
   - Enforces disk quota
   - Deletes old images based on retention policy
   - Tracks disk usage (in-memory counter)
   - Owned by EdgeCaptureAgent

4. **`filter_frame()` function**
   - Frame difference filtering (placeholder)
   - Runs synchronously in main loop

5. **`send_frame_to_server()` function**
   - HTTP POST to inference server
   - Retry logic with exponential backoff
   - Runs synchronously (blocking)

**Data Flow in Main Thread:**
```
Camera → Capture → ROI Crop → Filter → Send to Server
                                    ↓ (if fails)
                               Store Locally
```

---

### Thread 2: Metrics HTTP Server (Daemon Thread)
**Owner:** Flask application

**Responsibilities:**
- Expose HTTP endpoints for monitoring
- Serve health checks, metrics, and statistics
- Respond to Prometheus scraping requests

**Components:**

1. **Flask App**
   - Runs in background daemon thread
   - Started by `start_metrics_server()`
   - Non-blocking HTTP server

2. **HTTP Endpoints:**
   - `GET /health` - Service health status
   - `GET /metrics` - Prometheus-compatible metrics
   - `GET /stats` - JSON statistics summary

**Thread Characteristics:**
- Daemon thread (exits when main thread exits)
- Started with `threading.Thread(target=run_server, daemon=True)`
- Non-blocking (doesn't interfere with capture loop)

---

## Shared Resources (Thread-Safe)

### 1. **`MetricsCollector` (Global `metrics` instance)**

**Purpose:** Thread-safe metrics aggregation and reporting

**Thread Safety Mechanism:**
- Uses `threading.Lock()` for all read/write operations
- All mutation methods use `with self._lock:` context manager
- `get_snapshot()` returns atomic copy of all metrics

**Accessed By:**
- **Main Thread:** Writes metrics (increments counters, updates state)
- **Metrics Thread:** Reads metrics (via `get_snapshot()`)

**Shared Data:**
```python
# Counters (written by main, read by metrics server)
- frames_captured_total
- frames_filtered_total
- frames_sent_total
- frames_stored_locally_total
- send_failures_total

# State (written by main, read by metrics server)
- camera_status
- disk_usage_mb
- current_operating_state
- last_capture_time
- last_successful_send
```

**Thread-Safe Methods:**
- `increment_captured()` - Main thread writes
- `increment_filtered()` - Main thread writes
- `increment_sent()` - Main thread writes
- `increment_stored_locally()` - Main thread writes
- `increment_send_failures()` - Main thread writes
- `set_camera_status()` - Main thread writes
- `set_operating_state()` - Main thread writes
- `update_disk_usage()` - Main thread writes
- `get_snapshot()` - Metrics thread reads (atomic copy)

---

### 2. **`Config` (Read-Only, Shared)**

**Thread Safety Mechanism:**
- Immutable after initialization
- No locks needed (read-only)

**Accessed By:**
- **Main Thread:** Reads configuration for capture settings
- **Metrics Thread:** Reads configuration for display in `/stats`

**Shared Data:**
- All configuration parameters (intervals, thresholds, paths, etc.)

---

## Component Ownership & Lifecycle

### Main Thread Components (Not Shared)

These components are **owned exclusively** by the main thread and are **NOT** shared:

1. **`CaptureService`**
   - Camera operations (picamera2)
   - Frame capture
   - ROI processing
   - **Not thread-safe** - camera can only be accessed from one thread

2. **`StorageManager`**
   - Disk I/O operations
   - File deletion
   - Disk usage calculation
   - **Not thread-safe** - disk operations are sequential

3. **Frame buffers**
   - `previous_accepted_frame` in EdgeCaptureAgent
   - Captured frames (NumPy arrays)
   - **Not shared** - processed and discarded in main thread

---

## Concurrency Model

### Design Pattern: Producer-Consumer (Decoupled)

**Producer (Main Thread):**
- Produces metrics data by calling `metrics.increment_*()` methods
- Writes to shared `MetricsCollector` instance

**Consumer (Metrics Thread):**
- Consumes metrics data by calling `metrics.get_snapshot()`
- Reads from shared `MetricsCollector` instance
- Serves data via HTTP endpoints

### Key Design Decisions

1. **No Message Queues:**
   - Metrics are aggregated in-place (counters)
   - No need for queue-based communication
   - Simple lock-based synchronization sufficient

2. **Daemon Thread for Metrics:**
   - Metrics server automatically stops when main thread exits
   - No explicit shutdown coordination needed
   - Flask server runs in non-blocking mode

3. **Synchronous Server Communication:**
   - HTTP POST to inference server is **blocking**
   - Runs in main thread (not async)
   - Retry logic with exponential backoff
   - Falls back to local storage on failure

4. **No Async/Await:**
   - Could use `asyncio` for non-blocking server calls
   - Current design uses threading + synchronous calls
   - Simpler for Raspberry Pi environment

---

## Thread Communication Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Thread                              │
│                                                               │
│  EdgeCaptureAgent                                            │
│    │                                                          │
│    ├─> CaptureService ──> Camera                            │
│    │                                                          │
│    ├─> filter_frame()                                        │
│    │                                                          │
│    ├─> send_frame_to_server() ──> Inference Server          │
│    │                                                          │
│    └─> StorageManager ──> Disk                              │
│                                                               │
│  All components write to ──┐                                │
│                             │                                 │
└─────────────────────────────┼─────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  MetricsCollector    │ ◄── Thread-Safe (Lock)
                   │  (Global Instance)   │
                   └──────────────────────┘
                              ▲
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                             │                                 │
│  Metrics Thread (Daemon)    │                                 │
│                             │                                 │
│  Flask App                  │                                 │
│    │                        │                                 │
│    ├─> /health ─────────────┘                                │
│    │                                                          │
│    ├─> /metrics ────────────┘                                │
│    │                                                          │
│    └─> /stats ──────────────┘                                │
│                                                               │
│  Reads from MetricsCollector (atomic snapshot)               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Potential Concurrency Issues & Mitigations

### 1. ✅ **Race Conditions on Metrics**
**Mitigation:** All `MetricsCollector` methods use locks

### 2. ✅ **Stale Metrics Data**
**Mitigation:** `get_snapshot()` returns atomic copy at point in time

### 3. ✅ **Camera Access from Multiple Threads**
**Mitigation:** Camera only accessed from main thread (not shared)

### 4. ✅ **Disk Quota Enforcement Race**
**Mitigation:** Only main thread writes to disk (no concurrent writes)

### 5. ⚠️ **Flask Server Blocking Main Thread**
**Status:** Not an issue - runs in separate daemon thread

### 6. ⚠️ **Metrics Server Crash**
**Status:** Daemon thread, won't crash main thread; main thread continues

---

## Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                        Process Memory                        │
│                                                               │
│  Heap (Shared):                                              │
│    • MetricsCollector instance (global 'metrics')            │
│    • Config instance (read-only)                             │
│    • Flask app instance                                      │
│                                                               │
│  Main Thread Stack:                                          │
│    • EdgeCaptureAgent instance                               │
│    • CaptureService instance                                 │
│    • StorageManager instance                                 │
│    • Frame buffers (NumPy arrays)                            │
│                                                               │
│  Metrics Thread Stack:                                       │
│    • Flask request handlers                                  │
│    • HTTP connection state                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Main Thread Bottlenecks

1. **Camera Capture:** ~100-200ms per frame (hardware limited)
2. **Server HTTP POST:** 3s timeout + retries (blocking)
3. **Disk I/O:** File writes (~10-50ms per image)
4. **Quota Enforcement:** Directory scanning (only when quota exceeded)

### Metrics Thread Characteristics

- **Low CPU:** ~1% average
- **Low Memory:** ~5-10 MB
- **Response Time:** <100ms per HTTP request
- **Blocking Risk:** Minimal (only reads metrics, no I/O)

### Lock Contention

**Expected:** Very low
- Main thread writes metrics ~1 time per capture interval (every 5 seconds)
- Metrics thread reads ~1 time per scrape (every 15 seconds)
- Lock hold time: <1ms per operation
- No contention expected

---

## Testing Strategy

### Unit Tests (No Threading)

- `MetricsCollector` methods (increment, set, get_snapshot)
- `filter_frame()` logic
- `CaptureService.capture_frame()` (mocked camera)
- `StorageManager` disk operations (temp directory)

### Integration Tests (Single Thread)

- End-to-end capture pipeline
- Metrics updates during capture
- Disk quota enforcement

### Concurrency Tests (Multi-Threaded)

1. **Metrics Thread Safety**
   - Main thread: Rapid metrics updates (100/sec)
   - Metrics thread: Rapid snapshot reads (100/sec)
   - Verify: No lost updates, no stale reads

2. **Daemon Thread Cleanup**
   - Start metrics server
   - Stop main thread (SIGTERM)
   - Verify: Metrics thread exits automatically

3. **Long-Running Stability**
   - Run for 24+ hours
   - Monitor: Memory leaks, CPU usage, lock contention
   - Verify: Stable operation, no deadlocks

### Load Tests

- **Metrics Endpoint:** 100 req/sec to `/metrics`
- **Main Thread:** Capture at max rate (limited by camera)
- **Verify:** No impact on capture loop, metrics remain accurate

---

## Future Enhancements

### Potential Threading Improvements

1. **Async Server Communication**
   - Replace blocking `requests.post()` with `aiohttp`
   - Benefits: Non-blocking main loop, faster recovery from server failures
   - Trade-off: Complexity increases

2. **Separate Disk I/O Thread**
   - Offload `StorageManager.store_frame()` to background thread
   - Benefits: Main loop not blocked by disk writes
   - Trade-off: Need queue + thread coordination

3. **Concurrent Image Processing**
   - Process multiple frames in parallel (e.g., filtering)
   - Benefits: Higher throughput
   - Trade-off: Raspberry Pi has limited CPU cores

### Not Recommended

- ❌ **Multi-threaded Camera Access:** picamera2 not thread-safe
- ❌ **Async Metrics Collection:** Overhead not worth it for low-frequency updates
- ❌ **Thread Pool for Filtering:** Filtering is lightweight (current placeholder)

---

## Summary

| Aspect | Design Choice | Rationale |
|--------|---------------|-----------|
| **Thread Model** | 2 threads (main + metrics daemon) | Simple, sufficient for workload |
| **Shared State** | Single `MetricsCollector` with locks | Minimal shared state, easy to reason about |
| **Server Calls** | Synchronous (blocking) | Simpler than async, retry logic sufficient |
| **Disk I/O** | Main thread (synchronous) | Sequential writes avoid race conditions |
| **Metrics Server** | Daemon thread | Automatically cleaned up, non-blocking |
| **Lock Strategy** | Coarse-grained locks on MetricsCollector | Low contention, no performance issues |

**Overall:** Simple, robust, easy to test and debug. Suitable for Raspberry Pi edge deployment.
