#!/usr/bin/env python3
"""
Grounding DINO Inference Server

FastAPI server for bird detection using Grounding DINO model.
Receives images from edge devices and returns detection results.
"""

import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

# Import Grounding DINO modules
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration manager for the inference server."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    @property
    def server_host(self) -> str:
        return self.config['server']['host']

    @property
    def server_port(self) -> int:
        return self.config['server']['port']

    @property
    def model_config_file(self) -> str:
        return self.config['model']['config_file']

    @property
    def model_checkpoint_path(self) -> str:
        return self.config['model']['checkpoint_path']

    @property
    def device(self) -> str:
        return self.config['model']['device']

    @property
    def text_prompt(self) -> str:
        return self.config['inference']['text_prompt']

    @property
    def box_threshold(self) -> float:
        return self.config['inference']['box_threshold']

    @property
    def text_threshold(self) -> float:
        return self.config['inference']['text_threshold']

    @property
    def max_image_size(self) -> int:
        return self.config['inference']['max_image_size']

    @property
    def min_image_size(self) -> int:
        return self.config['inference']['min_image_size']

    @property
    def with_logits(self) -> bool:
        return self.config['inference']['with_logits']

    @property
    def save_detections(self) -> bool:
        return self.config.get('storage', {}).get('save_detections', False)

    @property
    def output_dir(self) -> str:
        return self.config.get('storage', {}).get('output_dir', '/resources/outputs')


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Create a custom registry for better control
registry = CollectorRegistry()

# Request metrics
http_requests_total = Counter(
    'inference_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# Detection metrics
detections_total = Counter(
    'inference_detections_total',
    'Total number of bird detections',
    registry=registry
)

detection_requests_total = Counter(
    'inference_detection_requests_total',
    'Total number of detection requests',
    registry=registry
)

# Inference timing metrics
inference_duration_seconds = Histogram(
    'inference_duration_seconds',
    'Time spent on model inference',
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

request_duration_seconds = Histogram(
    'inference_request_duration_seconds',
    'Total request processing time',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=registry
)

# Model status
model_loaded = Gauge(
    'inference_model_loaded',
    'Whether the model is loaded (1) or not (0)',
    registry=registry
)

model_load_time_seconds = Gauge(
    'inference_model_load_time_seconds',
    'Time taken to load the model',
    registry=registry
)

# Error metrics
errors_total = Counter(
    'inference_errors_total',
    'Total number of errors',
    ['error_type'],
    registry=registry
)

# Image processing metrics
image_size_bytes = Histogram(
    'inference_image_size_bytes',
    'Size of uploaded images in bytes',
    buckets=[10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
    registry=registry
)

# Detection confidence metrics
detection_confidence = Histogram(
    'inference_detection_confidence',
    'Confidence scores of detections',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=registry
)


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(self, config: Config):
        """Initialize model manager."""
        self.config = config
        self.model = None
        self.device = None
        self.transform = None
        self._inference_count = 0
        self._total_inference_time = 0.0

    def load_model(self):
        """Load the Grounding DINO model."""
        logging.info("Loading Grounding DINO model...")
        logging.info(f"Config file: {self.config.model_config_file}")
        logging.info(f"Checkpoint: {self.config.model_checkpoint_path}")

        load_start_time = time.time()

        try:
            # Load model configuration
            args = SLConfig.fromfile(self.config.model_config_file)
            args.device = self.config.device

            # Build model
            self.model = build_model(args)

            # Load checkpoint
            checkpoint = torch.load(
                self.config.model_checkpoint_path,
                map_location="cpu"
            )
            load_res = self.model.load_state_dict(
                clean_state_dict(checkpoint["model"]),
                strict=False
            )
            logging.info(f"Model load result: {load_res}")

            # Set to evaluation mode
            self.model.eval()

            # Move to device
            self.device = torch.device(self.config.device)
            self.model = self.model.to(self.device)

            # Initialize transform
            self.transform = T.Compose([
                T.RandomResize([self.config.min_image_size], max_size=self.config.max_image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # Update metrics
            load_time = time.time() - load_start_time
            model_load_time_seconds.set(load_time)
            model_loaded.set(1)

            logging.info(f"Model loaded successfully in {load_time:.2f} seconds!")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            model_loaded.set(0)
            errors_total.labels(error_type='model_load_error').inc()
            raise

    def preprocess_image(self, image_pil: Image.Image):
        """Preprocess image for inference."""
        image_tensor, _ = self.transform(image_pil, None)
        return image_tensor

    def run_inference(
        self,
        image_pil: Image.Image,
        text_prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            image_pil: PIL Image
            text_prompt: Text prompt for detection (default: from config)
            box_threshold: Box confidence threshold (default: from config)
            text_threshold: Text matching threshold (default: from config)

        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use defaults from config if not provided
        text_prompt = text_prompt or self.config.text_prompt
        box_threshold = box_threshold or self.config.box_threshold
        text_threshold = text_threshold or self.config.text_threshold

        # Preprocess caption
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."

        # Preprocess image
        image_tensor = self.preprocess_image(image_pil)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])

        inference_time = time.time() - start_time
        self._inference_count += 1
        self._total_inference_time += inference_time

        # Track inference time in Prometheus
        inference_duration_seconds.observe(inference_time)

        # Extract predictions
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # Filter by confidence
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # Get phrases and confidences
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        pred_confidences = []

        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                tokenizer
            )
            confidence = logit.max().item()
            pred_phrases.append(pred_phrase)
            pred_confidences.append(confidence)

            # Track confidence distribution
            detection_confidence.observe(confidence)

        # Track total detections
        num_detections = len(pred_phrases)
        detections_total.inc(num_detections)

        # Convert boxes from normalized to pixel coordinates
        W, H = image_pil.size
        boxes_pixel = boxes_filt * torch.Tensor([W, H, W, H])
        # Convert from xywh to xyxy
        boxes_pixel[:, :2] -= boxes_pixel[:, 2:] / 2
        boxes_pixel[:, 2:] += boxes_pixel[:, :2]

        return {
            "num_detections": num_detections,
            "boxes": boxes_pixel.tolist(),
            "labels": pred_phrases,
            "confidences": pred_confidences,
            "inference_time_ms": inference_time * 1000,
            "image_size": {"width": W, "height": H}
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = (self._total_inference_time / self._inference_count
                   if self._inference_count > 0 else 0)
        return {
            "total_inferences": self._inference_count,
            "total_time_seconds": self._total_inference_time,
            "average_time_ms": avg_time * 1000
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Grounding DINO Inference Server",
    description="Bird detection inference server using Grounding DINO",
    version="1.0.0"
)

# Global instances
config: Optional[Config] = None
model_manager: Optional[ModelManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global config, model_manager

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("=" * 70)
    logging.info("Grounding DINO Inference Server Starting")
    logging.info("=" * 70)

    # Load configuration
    config = Config("config.yaml")
    logging.info("Configuration loaded")

    # Initialize and load model
    model_manager = ModelManager(config)
    model_manager.load_model()

    logging.info("Server startup complete!")


@app.get("/")
async def root():
    """Root endpoint."""
    http_requests_total.labels(method='GET', endpoint='/', status='200').inc()
    return {
        "service": "Grounding DINO Inference Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_is_loaded = model_manager is not None and model_manager.model is not None
    status_code = "200" if model_is_loaded else "503"

    http_requests_total.labels(method='GET', endpoint='/health', status=status_code).inc()

    return {
        "status": "healthy" if model_is_loaded else "unhealthy",
        "model_loaded": model_is_loaded,
        "device": config.device if config else "unknown",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    timestamp: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None),
    box_threshold: Optional[float] = Form(None),
    text_threshold: Optional[float] = Form(None)
):
    """
    Detect birds in an uploaded image.

    Args:
        image: Image file to process
        timestamp: Capture timestamp (optional)
        device_id: ID of the device that captured the image (optional)
        text_prompt: Custom text prompt (optional, uses config default)
        box_threshold: Box confidence threshold (optional, uses config default)
        text_threshold: Text matching threshold (optional, uses config default)

    Returns:
        JSON response with detection results
    """
    request_start = time.time()

    # Track detection request
    detection_requests_total.inc()

    if model_manager is None or model_manager.model is None:
        http_requests_total.labels(method='POST', endpoint='/detect', status='503').inc()
        errors_total.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        contents = await image.read()
        image_size_bytes.observe(len(contents))
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        logging.info(f"Processing image from device: {device_id or 'unknown'}")

        # Run inference
        results = model_manager.run_inference(
            image_pil,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Add metadata
        results["metadata"] = {
            "device_id": device_id,
            "timestamp": timestamp or datetime.now().isoformat(),
            "server_timestamp": datetime.now().isoformat()
        }

        logging.info(f"Detection complete: {results['num_detections']} birds found")
        
        # TODO: If more than one detection, save the image. 
        # if no detections, save them depending on the probability present in the config file.

        # Track successful request
        http_requests_total.labels(method='POST', endpoint='/detect', status='200').inc()
        request_duration_seconds.labels(endpoint='/detect').observe(time.time() - request_start)

        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error during detection: {e}", exc_info=True)
        http_requests_total.labels(method='POST', endpoint='/detect', status='500').inc()
        errors_total.labels(error_type='inference_error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get inference statistics."""
    if model_manager is None:
        http_requests_total.labels(method='GET', endpoint='/stats', status='503').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats = model_manager.get_stats()
    stats["model_config"] = {
        "text_prompt": config.text_prompt,
        "box_threshold": config.box_threshold,
        "text_threshold": config.text_threshold,
        "device": config.device
    }

    http_requests_total.labels(method='GET', endpoint='/stats', status='200').inc()

    return stats


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if config is None:
        http_requests_total.labels(method='GET', endpoint='/config', status='503').inc()
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    http_requests_total.labels(method='GET', endpoint='/config', status='200').inc()

    return {
        "inference": {
            "text_prompt": config.text_prompt,
            "box_threshold": config.box_threshold,
            "text_threshold": config.text_threshold,
            "max_image_size": config.max_image_size,
            "min_image_size": config.min_image_size
        },
        "model": {
            "device": config.device
        }
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    # Track metrics endpoint access
    http_requests_total.labels(method='GET', endpoint='/metrics', status='200').inc()

    # Generate metrics in Prometheus format
    metrics_output = generate_latest(registry)

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Load config for server settings
    temp_config = Config("config.yaml")

    uvicorn.run(
        app,
        host=temp_config.server_host,
        port=temp_config.server_port,
        log_level="info"
    )
