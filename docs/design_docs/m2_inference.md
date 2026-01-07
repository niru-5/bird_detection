# Design Doc: Milestone 2 – Inference & Data Server

**Project:** Bird Detection Project  
**Milestone:** 2 – Inference & Data Server  
**Target Platform:** Docker Compose (Local/Cloud)  
**Status:** Draft  
**Author:** [Your Name]  
**Last Updated:** 2026-01-04

---

## 1. Problem Statement

We need a production-grade inference server that:
- Receives images from edge devices (M1)
- Performs bird detection using Grounding DINO
- Stores detection results with full data lineage
- Manages image storage using LakeFS for versioning
- Persists structured metadata in PostgreSQL
- Exposes observability metrics for monitoring
- Scales horizontally to handle multiple edge devices

This server is the central intelligence hub of the bird detection pipeline. Poor design here will create bottlenecks, data quality issues, and operational headaches downstream.

---

## 2. Goals

### Functional Goals
- Accept image uploads from authenticated edge devices
- Run Grounding DINO inference with configurable prompts
- Filter detections based on confidence thresholds
- Store images in LakeFS with proper versioning
- Persist detection metadata in PostgreSQL with queryable schema
- Support batch processing for backfill scenarios from edge devices
- Provide database connectivity for dataset creation and experiments

### Non-Functional Goals
- **Performance:** <500ms p95 inference latency for 1024x1024 images
- **Reliability:** 99.9% uptime with graceful degradation
- **Observability:** Full request tracing, metrics, and structured logging
- **Scalability:** Handle 10+ concurrent edge devices
- **Maintainability:** Clean separation of concerns, typed interfaces
- **Data Integrity:** ACID guarantees for metadata, immutable image storage

---

## 3. Non-Goals (Iteration 1)

- Real-time streaming inference
- Multi-model ensemble predictions
- Online learning or model updates
- Species-level classification (bird/no-bird only)
- Complex user authentication (basic token auth is sufficient)
- Federated learning coordination
- Server-side image preprocessing (handled by edge devices)
- RESTful query API (direct database access for experiments/datasets)
- Web UI (covered in M4)

---

## 4. Constraints

### Technical Constraints
- Must run on modest hardware (16GB RAM, 4-8 CPU cores, GPU optional)
- Model size limited by GPU VRAM (Grounding DINO Base ~1.5GB)
- Storage: Maximum 100GB for images (managed by cleanup policies)
- Network: Assume unreliable connectivity from edge devices
- Latency: Edge devices expect responses within 5-10 seconds
- Preprocessing: Edge devices send preprocessed images, no server-side preprocessing

### Operational Constraints
- Must be deployable via Docker Compose for development
- Production deployment to cloud (future) should be straightforward
- Database migrations must be reversible
- No manual data cleanup or maintenance required

---

## 5. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Inference Server                            │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  FastAPI Application                        │ │
│  │                                                              │ │
│  │  ┌──────────────────────┐  ┌──────────────────────────┐   │ │
│  │  │   Upload Endpoint    │  │   Health/Metrics         │   │ │
│  │  │   /api/v1/detect     │  │   Endpoints              │   │ │
│  │  └──────────┬───────────┘  └──────────────────────────┘   │ │
│  │             │                                               │ │
│  └─────────────┼───────────────────────────────────────────────┘ │
│                │                                                  │
│                v                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Service Layer (Business Logic)              │   │
│  │                                                           │   │
│  │  ┌─────────────────────────┐  ┌───────────────────────┐ │   │
│  │  │  Detection Service      │  │   Metrics Service     │ │   │
│  │  │  - Inference            │  │                       │ │   │
│  │  │  - Post-processing      │  │                       │ │   │
│  │  │  - Validation           │  │                       │ │   │
│  │  └────────┬────────────────┘  └───────┬───────────────┘ │   │
│  └───────────┼────────────────────────────┼──────────────────┘   │
│              │                            │                      │
│              v                            v                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Data Access Layer                       │  │
│  │                                                            │  │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │  │
│  │  │ LakeFS Client   │  │   PostgreSQL Client (SQLAlchemy)│ │  │
│  │  │ - Image storage │  │   - Metadata storage          │  │  │
│  │  │ - Versioning    │  │   - Transactional operations  │  │  │
│  │  └────────┬────────┘  └──────────┬───────────────────┘  │  │
│  └───────────┼────────────────────────┼──────────────────────┘  │
└──────────────┼────────────────────────┼─────────────────────────┘
               │                        │
               v                        v
    ┌─────────────────┐      ┌──────────────────┐
    │     LakeFS      │      │   PostgreSQL     │
    │   (S3-backed)   │      │   (Relational    │
    │                 │      │    Metadata)     │
    └─────────────────┘      └──────────────────┘
               │                        │
               v                        │
    ┌─────────────────┐                │
    │  MinIO / S3     │                │
    │ (Object Store)  │                │
    └─────────────────┘                │
                                        │
                                        v
                              ┌──────────────────┐
                              │  Database        │
                              │  Clients         │
                              │  (Jupyter, CLI,  │
                              │   Scripts)       │
                              └──────────────────┘
```

---

## 6. Component Breakdown

### 6.1 FastAPI Application

**Responsibility:** HTTP request handling, routing, validation

**Key Endpoints:**

```python
POST   /api/v1/detect          # Upload image for inference
POST   /api/v1/detect/batch    # Batch upload for backfill
GET    /api/v1/health          # Health check
GET    /api/v1/metrics         # Prometheus metrics
```

**Request/Response Schemas (Pydantic):**

```python
# Upload Request
class DetectionRequest(BaseModel):
    image: UploadFile
    device_id: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# Batch Upload Request
class BatchDetectionRequest(BaseModel):
    images: List[UploadFile]
    device_id: str
    metadata: Optional[Dict[str, Any]] = None

# Detection Response
class Detection(BaseModel):
    id: UUID
    image_id: UUID
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    confidence: float
    label: str
    created_at: datetime

class DetectionResponse(BaseModel):
    image_id: UUID
    lakefs_path: str
    detections: List[Detection]
    inference_time_ms: float
    model_version: str

class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total_processed: int
    total_failed: int
    failed_images: List[str]
```

**Authentication:**
- Simple token-based authentication via `X-API-Token` header
- Each edge device has a unique token
- Tokens mapped to device IDs for tracking

**Middleware:**
- Request ID injection for tracing
- Token authentication
- Error handling and logging
- Request/response timing

---

### 6.2 Detection Service

**Responsibility:** Image validation, model inference, post-processing, result validation

**Key Operations:**

```python
class DetectionService:
    def __init__(self, model_config: ModelConfig):
        self.model = self._load_grounding_dino()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def validate_image(self, file: UploadFile) -> bool:
        """Validate image format, size, dimensions - edge already preprocessed"""
        # Basic validation only:
        # - Check file size (<10MB)
        # - Verify MIME type (image/jpeg, image/png)
        # - Quick dimension check (already preprocessed by edge)
        
    async def detect(
        self, 
        image_bytes: bytes,
        text_prompt: str = "bird"
    ) -> List[Detection]:
        """Run inference and return filtered detections"""
        # Load image from bytes
        # Run Grounding DINO inference
        # Post-process and filter
        
    def _postprocess_detections(
        self, 
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: List[str],
        confidence_threshold: float = 0.3
    ) -> List[Detection]:
        """Filter and format detections"""
```

**Grounding DINO Configuration:**

```yaml
model:
  name: "grounding-dino-base"
  checkpoint: "/models/groundingdino_swint_ogc.pth"
  config_file: "/models/GroundingDINO_SwinT_OGC.py"
  
inference:
  text_prompt: "bird"
  box_threshold: 0.3
  text_threshold: 0.25
  device: "cuda"  # or "cpu"
  
postprocessing:
  nms_threshold: 0.5
  max_detections: 100
  min_box_area: 100  # pixels
```

**Performance Optimizations:**
- Model warmup on startup
- Batch processing support for backfill
- TorchScript compilation (optional)
- Mixed precision inference (FP16)

**Validation Rules:**
- File size: <10MB
- Formats: JPEG, PNG
- Minimum dimensions: 256x256 (edge already preprocesses to target size)
- No server-side resizing or normalization (edge responsibility)

---

### 6.3 Data Access Layer

#### 6.3.1 LakeFS Client

**Responsibility:** Versioned image storage with Git-like semantics

**Key Operations:**

```python
class LakeFSClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = lakefs.Client(endpoint, access_key, secret_key)
        self.repository = "bird-detection"
        
    async def upload_image(
        self, 
        branch: str,
        path: str,
        data: bytes,
        metadata: Dict[str, str]
    ) -> str:
        """Upload image and return LakeFS path"""
        
    async def commit_batch(
        self, 
        branch: str,
        message: str,
        paths: List[str]
    ) -> str:
        """Commit multiple images atomically"""
        
    async def create_branch(self, branch_name: str, source: str = "main") -> None:
        """Create new branch for experimentation"""
```

**LakeFS Repository Structure:**

```
bird-detection/
├── main/                          # Production branch
│   ├── images/                    # Preprocessed images from edge
│   │   ├── device-001/
│   │   │   ├── 2026-01-04/
│   │   │   │   ├── 12-00-01_a3f5.jpg
│   │   │   │   └── metadata.json
│   │   ├── device-002/
│   │   └── ...
│   └── with-detections/           # Images with bboxes drawn (optional)
│       └── ...
├── experiment-yolo/               # Experimental branch
└── training-v2/                   # Training data branch
```

**Notes:**
- No `raw/` folder - edge devices send preprocessed images
- `images/` contains preprocessed images ready for inference
- `with-detections/` used for visualization/debugging (optional)
- May add `raw/` later if needed for reprocessing experiments

**Branch Strategy:**
- **`main`**: Production data with detections
- **`experiment-*`**: Testing new models/prompts
- **`training-*`**: Curated datasets for model training
- **Tags**: Version milestones (e.g., `dataset-v1.0`, `model-checkpoint-20260104`)

---

#### 6.3.2 PostgreSQL Schema

**Responsibility:** Structured metadata storage with ACID guarantees

**Schema Design:**

```sql
-- Core Tables

CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    location JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    INDEX idx_device_id (device_id)
);

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    lakefs_path VARCHAR(512) NOT NULL,
    lakefs_commit_id VARCHAR(64),
    original_filename VARCHAR(255),
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Image properties
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format VARCHAR(10) NOT NULL,
    size_bytes INTEGER NOT NULL,
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Flags
    has_detections BOOLEAN DEFAULT FALSE,
    is_flagged BOOLEAN DEFAULT FALSE,
    
    -- Additional metadata
    metadata JSONB,
    
    INDEX idx_device_captured (device_id, captured_at DESC),
    INDEX idx_lakefs_path (lakefs_path),
    INDEX idx_processing_status (processing_status),
    INDEX idx_has_detections (has_detections)
);

CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    
    -- Bounding box (normalized coordinates [0, 1])
    bbox_x1 REAL NOT NULL CHECK (bbox_x1 >= 0 AND bbox_x1 <= 1),
    bbox_y1 REAL NOT NULL CHECK (bbox_y1 >= 0 AND bbox_y1 <= 1),
    bbox_x2 REAL NOT NULL CHECK (bbox_x2 >= 0 AND bbox_x2 <= 1),
    bbox_y2 REAL NOT NULL CHECK (bbox_y2 >= 0 AND bbox_y2 <= 1),
    
    -- Detection properties
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    label VARCHAR(100) NOT NULL,
    
    -- Model information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Additional properties
    metadata JSONB,
    
    INDEX idx_image_id (image_id),
    INDEX idx_confidence (confidence DESC),
    INDEX idx_label (label),
    INDEX idx_model (model_name, model_version)
);

CREATE TABLE inference_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    
    -- Session metadata
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    -- Model information
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    text_prompt TEXT,
    
    -- Configuration
    config JSONB,
    
    -- Results summary
    num_detections INTEGER DEFAULT 0,
    max_confidence REAL,
    
    -- Error tracking
    status VARCHAR(50) DEFAULT 'success',  -- success, failed, timeout
    error_message TEXT,
    
    INDEX idx_image_id (image_id),
    INDEX idx_started_at (started_at DESC),
    INDEX idx_status (status)
);

-- Metrics and Monitoring

CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    labels JSONB,
    
    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_metric_name (metric_name)
);

-- Data Quality

CREATE TABLE data_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    
    -- Annotation type
    annotation_type VARCHAR(50) NOT NULL,  -- false_positive, missed_detection, incorrect_bbox, etc.
    
    -- Corrected data
    corrected_bbox JSONB,
    corrected_label VARCHAR(100),
    
    -- Annotator info
    annotator VARCHAR(100),
    annotated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    notes TEXT,
    
    INDEX idx_image_id (image_id),
    INDEX idx_annotation_type (annotation_type)
);
```

**Design Principles:**
- **Normalization:** Separate tables for devices, images, detections
- **Referential Integrity:** Foreign keys with CASCADE deletes
- **Indexing Strategy:** Indexes on common query patterns (device_id + timestamp, has_detections, confidence)
- **JSONB Fields:** Flexible metadata storage for evolving requirements
- **Time-series Ready:** Partitioning by time (future optimization)
- **Data Quality:** Dedicated annotations table for feedback loop

**Query Patterns:**

```sql
-- Get recent detections for a device
SELECT d.*, i.lakefs_path, i.captured_at
FROM detections d
JOIN images i ON d.image_id = i.id
WHERE i.device_id = 'device-001'
  AND i.captured_at > NOW() - INTERVAL '7 days'
  AND d.confidence > 0.5
ORDER BY i.captured_at DESC
LIMIT 100;

-- Get images with high-confidence detections
SELECT i.*, COUNT(d.id) as detection_count, MAX(d.confidence) as max_confidence
FROM images i
JOIN detections d ON d.image_id = i.id
WHERE d.confidence > 0.7
GROUP BY i.id
HAVING COUNT(d.id) >= 2
ORDER BY i.captured_at DESC;

-- Daily detection summary
SELECT 
    DATE(i.captured_at) as date,
    COUNT(DISTINCT i.id) as images_with_birds,
    COUNT(d.id) as total_detections,
    AVG(d.confidence) as avg_confidence
FROM images i
JOIN detections d ON d.image_id = i.id
WHERE i.captured_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(i.captured_at)
ORDER BY date DESC;
```

---

#### 6.3.3 Database Access for Experiments

**Direct Database Connectivity:**

The inference server does not provide RESTful query endpoints. Instead, data scientists and developers can connect directly to PostgreSQL for:

- **Dataset Creation:** Query and export images/detections for training
- **Experiments:** Join with LakeFS paths to retrieve images
- **Analytics:** Run ad-hoc queries for model evaluation
- **Debugging:** Investigate false positives/negatives

**Access Methods:**

```python
# 1. Jupyter Notebook with SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@postgres:5432/bird_detection')
df = pd.read_sql("SELECT * FROM detections WHERE confidence > 0.7", engine)

# 2. CLI with psql
psql -h postgres -U user -d bird_detection
\x  # Expanded display
SELECT * FROM images WHERE has_detections = true LIMIT 10;

# 3. Python scripts for dataset export
import psycopg2
conn = psycopg2.connect("dbname=bird_detection user=user password=pass host=postgres")
cur = conn.cursor()
cur.execute("SELECT lakefs_path FROM images WHERE has_detections = true")
paths = cur.fetchall()
# Download from LakeFS using paths
```

**Security:**
- Read-only user credentials for experiments
- Write access only for inference server
- Network isolation (database not exposed publicly)

---

### 6.4 Observability

#### 6.4.1 Metrics (Prometheus)

**Key Metrics:**

```python
# Request metrics
http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request latency')

# Inference metrics
inference_requests_total = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
inference_duration_seconds = Histogram('inference_duration_seconds', 'Inference latency', ['model'])
detections_per_image = Histogram('detections_per_image', 'Number of detections per image')

# Storage metrics
images_stored_total = Counter('images_stored_total', 'Total images stored', ['device_id'])
storage_bytes_total = Gauge('storage_bytes_total', 'Total storage used')

# Model metrics
model_load_time_seconds = Gauge('model_load_time_seconds', 'Model load time')
gpu_memory_bytes = Gauge('gpu_memory_bytes', 'GPU memory used')
```

#### 6.4.2 Tracing (OpenTelemetry)

**Trace Spans:**
- `http.request` → `service.detect` → `model.inference` → `storage.save`
- Attributes: device_id, image_id, model_version, status

#### 6.4.3 Logging (Structured JSON)

```json
{
  "timestamp": "2026-01-04T12:00:01Z",
  "level": "INFO",
  "logger": "detection_service",
  "request_id": "req-a3f5b2c1",
  "device_id": "device-001",
  "image_id": "img-d4e6f7g8",
  "message": "Inference completed",
  "inference_time_ms": 234,
  "num_detections": 2,
  "max_confidence": 0.87
}
```

---

## 7. Data Flow

### 7.1 Normal Detection Flow

```
1. Edge device POSTs preprocessed image to /api/v1/detect with auth token
2. FastAPI validates request (auth token, file size, format)
3. DetectionService validates image (size, format - no preprocessing needed)
4. DetectionService runs Grounding DINO inference on preprocessed image
5. Parallel operations (within database transaction):
   a. LakeFS: Store preprocessed image with metadata
   b. PostgreSQL: Insert image record
6. For each detection above confidence threshold:
   a. PostgreSQL: Insert detection record
7. PostgreSQL: Insert inference_session record
8. Update image record: has_detections=TRUE/FALSE, processing_status='completed'
9. LakeFS: Commit changes (batch commits for efficiency)
10. Return DetectionResponse to edge device
11. Update Prometheus metrics
```

**Notes:**
- Edge device sends preprocessed images (already resized, normalized)
- Server only validates and runs inference
- Database transaction ensures atomicity (all or nothing)
- LakeFS commits batched for performance (configurable)

### 7.2 Batch Processing Flow (Backfill from Edge)

**Scenario:** Edge device has stored images locally during server downtime and needs to backfill.

```
1. Edge device POSTs batch of images to /api/v1/detect/batch
2. Server processes each image sequentially or in small batches (5-10)
3. For each image in batch:
   a. Validate image
   b. Run inference
   c. Store in LakeFS
   d. Update database
4. LakeFS: Single commit for entire batch (atomic)
5. Return BatchDetectionResponse with summary:
   - total_processed
   - total_failed
   - failed_images (list of filenames that failed)
6. Edge device can retry failed images individually
```

**Benefits:**
- Single commit in LakeFS (faster, atomic)
- Reduced HTTP overhead
- Edge device can efficiently backfill after network recovery

**Limitations:**
- Batch size limited to 50 images (configurable)
- Timeout set to 5 minutes per batch
- Failed images don't block entire batch

### 7.3 Database Query Flow (For Dataset Creation/Experiments)

```
1. Data scientist connects to PostgreSQL directly (Jupyter, psql, Python script)
2. Execute SQL query to find images matching criteria:
   - Time range
   - Device ID
   - Confidence threshold
   - Detection count
3. Retrieve lakefs_path for each image
4. Use LakeFS client/API to download images
5. Process locally for dataset creation, analysis, or training
```

**Example Workflow:**

```python
# 1. Query database for high-quality detections
query = """
SELECT i.id, i.lakefs_path, i.captured_at, COUNT(d.id) as num_birds
FROM images i
JOIN detections d ON d.image_id = i.id
WHERE d.confidence > 0.8
  AND i.captured_at > '2026-01-01'
GROUP BY i.id
HAVING COUNT(d.id) BETWEEN 1 AND 5
ORDER BY i.captured_at DESC
LIMIT 1000;
"""

# 2. Download images from LakeFS
lakefs = LakeFSClient(...)
for row in results:
    image_bytes = lakefs.download(row.lakefs_path)
    # Process for training dataset
```

---

## 8. Configuration

**Example `config.yaml`:**

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

model:
  name: "grounding-dino-base"
  checkpoint_path: "/models/groundingdino_swint_ogc.pth"
  config_path: "/models/GroundingDINO_SwinT_OGC.py"
  device: "cuda"  # or "cpu"
  text_prompt: "bird"
  box_threshold: 0.3
  text_threshold: 0.25
  nms_threshold: 0.5

validation:
  max_image_size_mb: 10
  allowed_formats: ["image/jpeg", "image/png"]
  min_width: 256
  min_height: 256

lakefs:
  endpoint: "http://lakefs:8000"
  access_key: "${LAKEFS_ACCESS_KEY}"
  secret_key: "${LAKEFS_SECRET_KEY}"
  repository: "bird-detection"
  branch: "main"
  commit_strategy: "batch"  # or "immediate"
  batch_size: 10
  batch_timeout_sec: 300  # 5 minutes

database:
  host: "postgres"
  port: 5432
  database: "bird_detection"
  user: "${POSTGRES_USER}"
  password: "${POSTGRES_PASSWORD}"
  pool_size: 10
  max_overflow: 20
  echo: false

auth:
  enabled: true
  token_header: "X-API-Token"
  tokens:
    device-001: "${DEVICE_001_TOKEN}"
    device-002: "${DEVICE_002_TOKEN}"

batch_processing:
  enabled: true
  max_batch_size: 50
  batch_timeout_sec: 300  # 5 minutes per batch

observability:
  metrics:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    endpoint: "http://jaeger:4318"
  logging:
    level: "INFO"
    format: "json"
    
storage:
  max_total_gb: 100              # Maximum total storage
  retention_days: 90
  cleanup_enabled: true
  cleanup_schedule: "0 2 * * *"  # 2 AM daily
  warning_threshold_gb: 85       # Alert when 85% full
```

---

## 9. Deployment

### 9.1 Docker Compose

**`docker-compose.yml`:**

```yaml
version: "3.9"

services:
  inference-server:
    build: ./inference-server
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - ./models:/models:ro
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - LAKEFS_ACCESS_KEY=${LAKEFS_ACCESS_KEY}
      - LAKEFS_SECRET_KEY=${LAKEFS_SECRET_KEY}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - DEVICE_001_TOKEN=${DEVICE_001_TOKEN}
      - DEVICE_002_TOKEN=${DEVICE_002_TOKEN}
    depends_on:
      - postgres
      - lakefs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=bird_detection
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d:ro

  lakefs:
    image: treeverse/lakefs:latest
    ports:
      - "8001:8000"
    environment:
      - LAKEFS_DATABASE_TYPE=postgres
      - LAKEFS_DATABASE_POSTGRES_CONNECTION_STRING=postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/lakefs?sslmode=disable
      - LAKEFS_AUTH_ENCRYPT_SECRET_KEY=${LAKEFS_ENCRYPT_KEY}
      - LAKEFS_BLOCKSTORE_TYPE=s3
      - LAKEFS_BLOCKSTORE_S3_ENDPOINT=http://minio:9000
      - LAKEFS_BLOCKSTORE_S3_CREDENTIALS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY}
      - LAKEFS_BLOCKSTORE_S3_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY}
    depends_on:
      - postgres
      - minio

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
      - MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro

volumes:
  postgres_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

---

## 10. Failure Modes & Recovery

| Failure | Impact | Mitigation | Recovery |
|---------|--------|------------|----------|
| **Model load failure** | No inference | Fail fast on startup, log error | Restart container, check model files |
| **GPU OOM** | Inference fails | Reduce batch size, use CPU fallback | Auto-retry with CPU, monitor VRAM |
| **LakeFS unavailable** | No image storage | Return 503, edge retries | Edge stores locally, retries with backoff |
| **PostgreSQL down** | No metadata storage | Connection pooling, retry logic | Wait for DB recovery, edge retries |
| **Network timeout** | Edge device retry | Request timeout limits (10s) | Edge implements exponential backoff |
| **Invalid image** | Request fails | Validation before inference | Return 400 Bad Request with details |
| **Disk full (100GB limit)** | Cannot save images | Monitor disk usage, auto-cleanup | Delete old images, alert admin |
| **Concurrent writes** | Data corruption | Database transactions, locks | PostgreSQL handles ACID |
| **Authentication failure** | Unauthorized access | Token validation | Return 401, edge checks token config |
| **Batch upload partial failure** | Some images lost | Per-image error tracking | Return failed list, edge retries failed only |

---

## 11. Testing Strategy

### Unit Tests
- Pydantic schema validation
- Detection filtering (NMS, confidence)
- Database queries (mocked)
- Token authentication logic
- Image validation (size, format)

### Integration Tests
- End-to-end API tests with real database
- LakeFS upload/retrieval
- Model inference on sample images
- Authentication (valid/invalid tokens)
- Batch upload with partial failures
- Error handling (malformed requests, missing files)

### Performance Tests
- Inference latency benchmarks (single image)
- Concurrent request handling (10-50 clients)
- Batch upload performance (10-50 images)
- Database query performance
- Memory/GPU usage profiling
- Storage cleanup efficiency

### Contract Tests
- API schema compatibility with edge devices
- Database schema migrations
- LakeFS path format validation

**Test Data:**
- Synthetic bird images (generated)
- Preprocessed images from edge (realistic)
- Edge cases: blurry, dark, no birds, multiple birds
- Malformed requests (corrupt files, invalid JSON, missing auth)
- Large batches for backfill testing

---

## 12. Open Questions

1. **Batch Commits vs Immediate Commits in LakeFS?**
   - Immediate: Better data durability, more overhead
   - Batch: Better performance, risk of data loss on crash
   - **Recommendation:** Batch commits every 10 images or 5 minutes for normal flow, single commit for batch uploads

2. **GPU Sharing Strategy?**
   - Single process per GPU
   - Multiple workers with GPU lock
   - **Recommendation:** Single worker per GPU for M2, scale horizontally with multiple servers

3. **Image Storage Format in LakeFS?**
   - JPEG (compressed, lossy, smaller files)
   - PNG (lossless, larger files)
   - **Recommendation:** JPEG for storage efficiency (100GB limit), edge already compressed

4. **Database Partitioning?**
   - Partition `images` table by month for query performance?
   - **Recommendation:** Start unpartitioned, monitor performance, partition at 1M+ rows

5. **Batch Upload Error Handling?**
   - Fail entire batch on first error (atomic)
   - Continue processing, return partial success
   - **Recommendation:** Continue processing (better for edge backfill), return detailed error list

6. **Storage Cleanup Strategy?**
   - Delete oldest by timestamp
   - Delete images without detections first
   - **Recommendation:** Hybrid - keep high-confidence detections, delete old no-detection images first

7. **Authentication Token Rotation?**
   - Static tokens (simple, less secure)
   - Rotating tokens (more secure, complex)
   - **Recommendation:** Static tokens for M2, add rotation in M3

---

## 13. Success Criteria

### Functional
- ✅ Accepts preprocessed images from authenticated edge devices
- ✅ Token-based authentication working
- ✅ Runs Grounding DINO inference on preprocessed images
- ✅ Stores images in LakeFS with commits
- ✅ Persists metadata in PostgreSQL
- ✅ Returns detections with bounding boxes
- ✅ Batch upload endpoint for edge backfill
- ✅ Database accessible for experiments/dataset creation
- ✅ Exposes Prometheus metrics

### Performance
- ✅ p95 latency < 500ms for 1024x1024 images
- ✅ Handles 10+ concurrent requests
- ✅ No memory leaks over 24h continuous operation

### Reliability
- ✅ Graceful degradation when LakeFS/DB unavailable
- ✅ No data loss on server restart
- ✅ All database operations transactional

### Observability
- ✅ All requests traced end-to-end
- ✅ Metrics exposed and scraped by Prometheus
- ✅ Structured logs queryable in Grafana Loki

### Developer Experience
- ✅ Single command deployment (`docker-compose up`)
- ✅ API documentation auto-generated (Swagger/ReDoc)
- ✅ Database migrations automated (Alembic)
- ✅ Comprehensive test coverage (>80%)

---

## 14. Migration Path from M1

**Edge Device Changes:**
- Update `server.url` in `config.yaml` to inference server endpoint
- Add authentication token to config and HTTP headers (`X-API-Token`)
- Ensure images are preprocessed before sending (already done in M1)
- Handle new response format (DetectionResponse)
- Implement batch upload for backfilling stored images
- Add retry logic for 401 (auth failure) and 503 (server unavailable)

**Server Setup:**
- Deploy PostgreSQL with schema migrations
- Deploy LakeFS with MinIO backend
- Deploy inference server with Grounding DINO model
- Configure authentication tokens for each device
- Set up monitoring (Prometheus, Grafana)

**Data Migration:**
- Backfill existing edge images from M1 local storage using batch endpoint
- Verify data integrity (LakeFS paths, database records)
- Validate detections against manual labels (if available)

**Validation:**
- Test single image upload
- Test batch upload with 10-50 images
- Test authentication (valid/invalid tokens)
- Verify database queries return expected results
- Check LakeFS commits and branches

---

## 15. Future Enhancements (Post-M2)

### M3: Model Training Pipeline
- Export datasets from LakeFS branches
- Train lightweight models (YOLO, RT-DETR)
- Model versioning and A/B testing

### M4: Web UI
- Real-time detection dashboard
- Image annotation tool
- Model performance monitoring

### M5: Advanced Features
- Multi-modal fusion (audio + visual)
- Species-level classification
- Temporal tracking (bird behavior analysis)
- Federated learning across edge devices

---

## 16. References

- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [LakeFS Documentation](https://docs.lakefs.io/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Prometheus Metric Naming](https://prometheus.io/docs/practices/naming/)

---

**End of Design Doc**