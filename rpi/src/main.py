#!/usr/bin/env python3
"""
Bird Detection Edge Capture Agent - Main Entry Point

This module implements the edge capture agent for the bird detection project.
It handles:
- Image capture from Raspberry Pi Camera during operating hours (9 AM - 5 PM)
- Frame filtering to reduce redundancy
- Asynchronous communication with remote inference server
- Local storage fallback when server is unavailable
- HTTP metrics endpoint for health monitoring

Architecture: See docs/architecture.md and docs/design_docs/m1_edge.md
Author: Bird Detection Project
Last Updated: 2026-01-03
"""

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import yaml
from flask import Flask, jsonify
from picamera2 import Picamera2


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class Config:
    """
    Configuration data class for the edge capture agent.

    This class encapsulates all configuration parameters loaded from YAML.
    See rpi/configs/image_capture.yaml for example configuration.
    """
    # Capture settings
    capture_interval_sec: int

    # Sensor settings
    sensor_mode: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    exposure_time_us: int
    gain: float
    white_balance: int
    fps: int
    resolution: str
    format: str
    sensor_format: str

    # Operating hours
    operating_hours_enabled: bool
    start_hour: int
    end_hour: int
    timezone: str

    # Filter settings
    filter_enabled: bool
    frame_diff_threshold: int

    # Storage settings
    max_disk_mb: int
    retention_days: int
    storage_path: str = "/var/lib/bird-capture/images"

    # Server settings
    server_url: str
    server_timeout_sec: int
    retry_attempts: int
    retry_backoff_sec: int

    # Metrics settings
    metrics_enabled: bool
    metrics_port: int
    metrics_host: str

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configuration
        sensor = data.get('sensor_settings', {})
        crop = sensor.get('crop_settings', {})
        operating_hours = data.get('operating_hours', {})
        filter_config = data.get('filter', {})
        storage = data.get('storage', {})
        server = data.get('server', {})
        metrics = data.get('metrics', {})

        return cls(
            # Capture
            capture_interval_sec=data.get('capture_interval_sec', 5),

            # Sensor
            sensor_mode=sensor.get('sensor_mode', 7),
            crop_x=crop.get('x', 0),
            crop_y=crop.get('y', 0),
            crop_width=crop.get('width', 1024),
            crop_height=crop.get('height', 1024),
            exposure_time_us=sensor.get('exposure_time_us', 100000),
            gain=sensor.get('gain', 1.0),
            white_balance=sensor.get('white_balance', 4000),
            fps=sensor.get('fps', 30),
            resolution=sensor.get('resolution', '1024x1024'),
            format=sensor.get('format', 'RGB24'),
            sensor_format=sensor.get('sensor_format', 'RGB24'),

            # Operating hours
            operating_hours_enabled=operating_hours.get('enabled', True),
            start_hour=operating_hours.get('start_hour', 9),
            end_hour=operating_hours.get('end_hour', 17),
            timezone=operating_hours.get('timezone', 'local'),

            # Filter
            filter_enabled=filter_config.get('enabled', True),
            frame_diff_threshold=filter_config.get('frame_diff_threshold', 25),

            # Storage
            max_disk_mb=storage.get('max_disk_mb', 3072),
            retention_days=storage.get('retention_days', 14),

            # Server
            server_url=server.get('url', 'http://inference-server:8000/detect'),
            server_timeout_sec=server.get('timeout_sec', 3),
            retry_attempts=server.get('retry_attempts', 3),
            retry_backoff_sec=server.get('retry_backoff_sec', 5),

            # Metrics
            metrics_enabled=metrics.get('enabled', True),
            metrics_port=metrics.get('port', 8080),
            metrics_host=metrics.get('host', '0.0.0.0'),
        )


# ============================================================================
# Metrics & State Management
# ============================================================================

class MetricsCollector:
    """
    Thread-safe metrics collector for monitoring service health and statistics.

    Tracks:
    - Frames captured, filtered, sent, stored
    - Service uptime
    - Disk usage
    - Error counts
    - Operating hours state
    """

    def __init__(self):
        """Initialize metrics with zero values."""
        self._lock = threading.Lock()
        self.start_time = time.time()

        # Counters
        self.frames_captured_total = 0
        self.frames_filtered_total = 0
        self.frames_sent_total = 0
        self.frames_stored_locally_total = 0
        self.send_failures_total = 0

        # State
        self.last_capture_time: Optional[str] = None
        self.last_successful_send: Optional[str] = None
        self.camera_status = "ok"
        self.disk_usage_mb = 0
        self.current_operating_state = "inactive"  # active|inactive|sleeping

    def increment_captured(self):
        """Increment frames captured counter."""
        with self._lock:
            self.frames_captured_total += 1
            self.last_capture_time = datetime.now().isoformat()

    def increment_filtered(self):
        """Increment frames filtered counter."""
        with self._lock:
            self.frames_filtered_total += 1

    def increment_sent(self):
        """Increment frames sent counter."""
        with self._lock:
            self.frames_sent_total += 1
            self.last_successful_send = datetime.now().isoformat()

    def increment_stored_locally(self):
        """Increment frames stored locally counter."""
        with self._lock:
            self.frames_stored_locally_total += 1

    def increment_send_failures(self):
        """Increment send failures counter."""
        with self._lock:
            self.send_failures_total += 1

    def set_camera_status(self, status: str):
        """Set camera status (ok|error)."""
        with self._lock:
            self.camera_status = status

    def set_operating_state(self, state: str):
        """Set operating state (active|inactive|sleeping)."""
        with self._lock:
            self.current_operating_state = state

    def update_disk_usage(self, usage_mb: float):
        """Update disk usage in MB."""
        with self._lock:
            self.disk_usage_mb = usage_mb

    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get thread-safe snapshot of all metrics.

        Returns:
            Dictionary containing all current metrics
        """
        with self._lock:
            return {
                'uptime_seconds': self.get_uptime_seconds(),
                'frames_captured_total': self.frames_captured_total,
                'frames_filtered_total': self.frames_filtered_total,
                'frames_sent_total': self.frames_sent_total,
                'frames_stored_locally_total': self.frames_stored_locally_total,
                'send_failures_total': self.send_failures_total,
                'last_capture_time': self.last_capture_time,
                'last_successful_send': self.last_successful_send,
                'camera_status': self.camera_status,
                'disk_usage_mb': self.disk_usage_mb,
                'current_operating_state': self.current_operating_state,
            }


# Global metrics instance
metrics = MetricsCollector()


# ============================================================================
# Metrics HTTP Server
# ============================================================================

def create_metrics_app(config: Config) -> Flask:
    """
    Create Flask app for metrics HTTP server.

    Exposes three endpoints:
    - /health: Service health check
    - /metrics: Prometheus-compatible metrics
    - /stats: Human-readable JSON statistics

    Args:
        config: Configuration object

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Disable Flask default logging to avoid cluttering logs
    import logging as flask_logging
    flask_log = flask_logging.getLogger('werkzeug')
    flask_log.setLevel(flask_logging.ERROR)

    @app.route('/health', methods=['GET'])
    def health():
        """
        Health check endpoint.

        Returns JSON with service health status and key metrics.
        """
        snapshot = metrics.get_snapshot()

        # Determine overall health status
        status = "healthy"
        if snapshot['camera_status'] == "error":
            status = "unhealthy"
        elif snapshot['send_failures_total'] > 10:
            status = "degraded"

        return jsonify({
            'status': status,
            'uptime_seconds': snapshot['uptime_seconds'],
            'last_capture_time': snapshot['last_capture_time'],
            'camera_status': snapshot['camera_status'],
            'disk_usage_mb': snapshot['disk_usage_mb'],
            'disk_quota_mb': config.max_disk_mb,
            'current_operating_state': snapshot['current_operating_state'],
        })

    @app.route('/metrics', methods=['GET'])
    def prometheus_metrics():
        """
        Prometheus-compatible metrics endpoint.

        Returns metrics in Prometheus text format.
        """
        snapshot = metrics.get_snapshot()

        lines = [
            '# HELP frames_captured_total Total frames captured from camera',
            '# TYPE frames_captured_total counter',
            f'frames_captured_total {snapshot["frames_captured_total"]}',
            '',
            '# HELP frames_filtered_total Frames discarded by filter',
            '# TYPE frames_filtered_total counter',
            f'frames_filtered_total {snapshot["frames_filtered_total"]}',
            '',
            '# HELP frames_sent_total Frames successfully sent to server',
            '# TYPE frames_sent_total counter',
            f'frames_sent_total {snapshot["frames_sent_total"]}',
            '',
            '# HELP frames_stored_locally_total Frames stored to local disk',
            '# TYPE frames_stored_locally_total counter',
            f'frames_stored_locally_total {snapshot["frames_stored_locally_total"]}',
            '',
            '# HELP send_failures_total Failed attempts to send to server',
            '# TYPE send_failures_total counter',
            f'send_failures_total {snapshot["send_failures_total"]}',
            '',
            '# HELP disk_usage_bytes Current disk usage in bytes',
            '# TYPE disk_usage_bytes gauge',
            f'disk_usage_bytes {int(snapshot["disk_usage_mb"] * 1024 * 1024)}',
            '',
            '# HELP service_uptime_seconds Service uptime in seconds',
            '# TYPE service_uptime_seconds gauge',
            f'service_uptime_seconds {snapshot["uptime_seconds"]}',
            '',
            '# HELP operating_hours_active Is service in active hours (1=yes, 0=no)',
            '# TYPE operating_hours_active gauge',
            f'operating_hours_active {1 if snapshot["current_operating_state"] == "active" else 0}',
            '',
        ]

        return '\n'.join(lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}

    @app.route('/stats', methods=['GET'])
    def stats():
        """
        Human-readable JSON statistics endpoint.

        Returns comprehensive statistics in JSON format.
        """
        snapshot = metrics.get_snapshot()

        # Calculate next state change time
        now = datetime.now()
        current_hour = now.hour
        if config.operating_hours_enabled:
            if current_hour < config.start_hour:
                next_change = now.replace(hour=config.start_hour, minute=0, second=0, microsecond=0)
            elif current_hour >= config.end_hour:
                next_change = (now + timedelta(days=1)).replace(
                    hour=config.start_hour, minute=0, second=0, microsecond=0
                )
            else:
                next_change = now.replace(hour=config.end_hour, minute=0, second=0, microsecond=0)
        else:
            next_change = None

        return jsonify({
            'service': {
                'uptime_seconds': snapshot['uptime_seconds'],
                'status': 'healthy' if snapshot['camera_status'] == 'ok' else 'unhealthy',
                'version': '1.0.0',
            },
            'capture': {
                'frames_captured': snapshot['frames_captured_total'],
                'frames_filtered': snapshot['frames_filtered_total'],
                'frames_sent': snapshot['frames_sent_total'],
                'frames_stored_locally': snapshot['frames_stored_locally_total'],
                'last_capture_time': snapshot['last_capture_time'],
            },
            'operating_hours': {
                'enabled': config.operating_hours_enabled,
                'currently_active': snapshot['current_operating_state'] == 'active',
                'start_hour': config.start_hour,
                'end_hour': config.end_hour,
                'next_state_change': next_change.isoformat() if next_change else None,
            },
            'storage': {
                'disk_usage_mb': snapshot['disk_usage_mb'],
                'disk_quota_mb': config.max_disk_mb,
                'usage_percent': (snapshot['disk_usage_mb'] / config.max_disk_mb * 100)
                                 if config.max_disk_mb > 0 else 0,
            },
            'network': {
                'send_failures': snapshot['send_failures_total'],
                'last_successful_send': snapshot['last_successful_send'],
            },
        })

    return app


def start_metrics_server(config: Config):
    """
    Start the metrics HTTP server in a background thread.

    The server runs on the configured host and port, exposing
    health and metrics endpoints for monitoring.

    Args:
        config: Configuration object with metrics settings
    """
    if not config.metrics_enabled:
        logging.info("Metrics server disabled in configuration")
        return

    app = create_metrics_app(config)

    def run_server():
        """Run Flask server (non-blocking)."""
        app.run(
            host=config.metrics_host,
            port=config.metrics_port,
            debug=False,
            use_reloader=False,
        )

    # Start server in daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logging.info("Metrics HTTP server started on %s:%d", config.metrics_host, config.metrics_port)


# ============================================================================
# Operating Hours Management
# ============================================================================

def is_within_operating_hours(config: Config) -> bool:
    """
    Check if current time is within configured operating hours.

    Operating hours are configured as start_hour and end_hour in 24-hour format.
    Default operating hours: 9 AM to 5 PM (09:00 - 17:00).

    Args:
        config: Configuration object with operating hours settings

    Returns:
        True if within operating hours or if operating hours are disabled,
        False otherwise
    """
    if not config.operating_hours_enabled:
        return True

    current_hour = datetime.now().hour

    # Check if current hour is within range
    # Note: end_hour is exclusive (e.g., 17 means up to 16:59:59)
    within_hours = config.start_hour <= current_hour < config.end_hour

    return within_hours


def wait_for_operating_hours(config: Config):
    """
    Sleep until the next operating hours period begins.

    This function calculates the time until the next operating period
    and sleeps for that duration. It checks periodically to avoid
    oversleeping.

    Args:
        config: Configuration object with operating hours settings
    """
    if not config.operating_hours_enabled:
        return

    now = datetime.now()
    current_hour = now.hour

    # Calculate next start time
    if current_hour < config.start_hour:
        # Wait until start_hour today
        next_start = now.replace(hour=config.start_hour, minute=0, second=0, microsecond=0)
    else:
        # Wait until start_hour tomorrow
        next_start = (now + timedelta(days=1)).replace(
            hour=config.start_hour, minute=0, second=0, microsecond=0
        )

    wait_seconds = (next_start - now).total_seconds()

    logging.info("Outside operating hours. Waiting %.0f seconds until %s", wait_seconds, next_start)
    metrics.set_operating_state("sleeping")

    # Sleep in chunks to allow for graceful shutdown
    sleep_chunk = 60  # Wake up every minute to check
    while wait_seconds > 0:
        time.sleep(min(sleep_chunk, wait_seconds))
        wait_seconds -= sleep_chunk

        # Re-check in case config was reloaded
        if is_within_operating_hours(config):
            break


# ============================================================================
# Image Capture Service
# ============================================================================

class CaptureService:
    """
    Manages Raspberry Pi Camera for image capture.

    Handles camera initialization, configuration, and image capture
    with support for ROI cropping and sensor settings.
    """

    def __init__(self, config: Config):
        """
        Initialize the capture service.

        Args:
            config: Configuration object with sensor settings
        """
        self.config = config
        self.camera: Optional[Picamera2] = None
        self.last_frame: Optional[np.ndarray] = None

    def initialize_camera(self):
        """
        Initialize and configure the Raspberry Pi camera.

        Sets up the camera with configured sensor mode, resolution,
        and other sensor parameters.

        Raises:
            RuntimeError: If camera initialization fails
        """
        try:
            logging.info("Initializing Raspberry Pi camera...")

            self.camera = Picamera2()

            # Get available sensor modes
            sensor_modes = self.camera.sensor_modes

            if self.config.sensor_mode >= len(sensor_modes):
                logging.warning(
                    "Sensor mode %d not available. Using mode 0 instead.",
                    self.config.sensor_mode
                )
                self.config.sensor_mode = 0

            # Set sensor mode
            selected_mode = sensor_modes[self.config.sensor_mode]
            self.camera.sensor_mode = selected_mode

            # Create capture configuration
            capture_config = self.camera.create_still_configuration(
                raw={
                    'format': selected_mode.get('format', self.config.sensor_format),
                    'bit_depth': selected_mode.get('bit_depth', 10),
                    'fps': self.config.fps,
                }
            )

            self.camera.configure(capture_config)
            self.camera.start()

            logging.info("Camera initialized with sensor mode %d", self.config.sensor_mode)
            metrics.set_camera_status("ok")

        except Exception as e:
            logging.error("Failed to initialize camera: %s", e)
            metrics.set_camera_status("error")
            raise RuntimeError(f"Camera initialization failed: {e}") from e

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Applies ROI cropping if configured.

        Returns:
            NumPy array containing the captured image (BGR format),
            or None if capture fails
        """
        try:
            if self.camera is None:
                raise RuntimeError("Camera not initialized")

            # Capture frame as NumPy array
            frame = self.camera.capture_array()

            # Convert RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Apply ROI cropping if configured
            if (self.config.crop_width > 0 and self.config.crop_height > 0 and
                (self.config.crop_x > 0 or self.config.crop_y > 0 or
                 self.config.crop_width < frame.shape[1] or
                 self.config.crop_height < frame.shape[0])):

                x1 = self.config.crop_x
                y1 = self.config.crop_y
                x2 = min(x1 + self.config.crop_width, frame.shape[1])
                y2 = min(y1 + self.config.crop_height, frame.shape[0])

                frame = frame[y1:y2, x1:x2]
                logging.debug("Applied ROI crop: (%d,%d) to (%d,%d)", x1, y1, x2, y2)

            metrics.increment_captured()
            self.last_frame = frame

            return frame

        except Exception as e:
            logging.error("Failed to capture frame: %s", e)
            metrics.set_camera_status("error")
            return None

    def cleanup(self):
        """Stop the camera and release resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
                logging.info("Camera stopped and cleaned up")
            except Exception as e:
                logging.error("Error during camera cleanup: %s", e)


# ============================================================================
# Frame Filtering
# ============================================================================

def filter_frame(
    current_frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    config: Config
) -> Tuple[bool, str]:
    """
    Filter frame based on motion/change detection.

    This is a PLACEHOLDER implementation. The actual filtering logic
    will be implemented in a future iteration.

    The filter compares the current frame with the previous frame
    using grayscale difference. If the change exceeds configured
    thresholds, the frame is accepted; otherwise it's filtered out.

    Args:
        current_frame: Current frame to evaluate (BGR format)
        previous_frame: Previous accepted frame for comparison (BGR format),
                       or None if this is the first frame
        config: Configuration object with filter settings

    Returns:
        Tuple of (should_accept: bool, reason: str)
        - should_accept: True if frame should be accepted, False to filter
        - reason: Human-readable reason for the decision

    TODO: Implement actual frame differencing logic
    TODO: Add support for multiple filter strategies
    TODO: Add configuration for different filter algorithms
    """
    if not config.filter_enabled:
        return True, "filter_disabled"

    # First frame is always accepted
    if previous_frame is None:
        return True, "first_frame"
    
    image_diff = np.abs(current_frame.astype(np.float32) - previous_frame.astype(np.float32))
    image_diff = cv2.GaussianBlur(image_diff, (5, 5), 0)
    image_diff = image_diff.astype(np.uint8)
    
    if np.max(image_diff) > config.frame_diff_threshold:
        return True, "motion_detected"
    else:
        return False, "no_motion_detected"


# ============================================================================
# Server Communication
# ============================================================================

def send_frame_to_server(
    frame: np.ndarray,
    timestamp: datetime,
    config: Config
) -> bool:
    """
    Send captured frame to the remote inference server.

    Encodes the frame as JPEG and sends via HTTP POST to the
    configured server URL. Implements retry logic with exponential
    backoff on failure.

    Args:
        frame: Image frame to send (NumPy array, BGR format)
        timestamp: Capture timestamp
        config: Configuration object with server settings

    Returns:
        True if frame was successfully sent, False otherwise
    """
    try:
        # Encode frame as JPEG
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            logging.error("Failed to encode frame as JPEG")
            return False

        # Prepare multipart form data
        files = {
            'image': ('image.jpg', encoded_image.tobytes(), 'image/jpeg')
        }

        data = {
            'timestamp': timestamp.isoformat(),
            'device_id': 'rpi-edge-001',  # TODO: Make this configurable
        }

        # Attempt to send with retry logic
        for attempt in range(config.retry_attempts):
            try:
                logging.debug(
                    "Sending frame to server (attempt %d/%d)",
                    attempt + 1,
                    config.retry_attempts
                )

                response = requests.post(
                    config.server_url,
                    files=files,
                    data=data,
                    timeout=config.server_timeout_sec
                )

                if response.status_code == 200:
                    logging.info("Frame sent successfully to server: %s", response.json())
                    metrics.increment_sent()
                    return True

                logging.warning(
                    "Server returned error %d: %s",
                    response.status_code,
                    response.text
                )

            except requests.exceptions.Timeout:
                logging.warning("Server request timed out (attempt %d)", attempt + 1)

            except requests.exceptions.ConnectionError as e:
                logging.warning("Failed to connect to server: %s", e)

            except Exception as e:
                logging.error("Unexpected error sending to server: %s", e)

            # Exponential backoff before retry
            if attempt < config.retry_attempts - 1:
                backoff_time = config.retry_backoff_sec * (2 ** attempt)
                logging.debug("Retrying in %d seconds...", backoff_time)
                time.sleep(backoff_time)

        # All retries failed
        logging.error("Failed to send frame after %d attempts", config.retry_attempts)
        metrics.increment_send_failures()
        return False

    except Exception as e:
        logging.error("Error in send_frame_to_server: %s", e)
        metrics.increment_send_failures()
        return False


# ============================================================================
# Local Storage Management
# ============================================================================

class StorageManager:
    """
    Manages local disk storage for captured images.

    Handles:
    - Storing images to disk when server is unavailable
    - Enforcing disk quota
    - Automatic cleanup based on retention policy
    - Directory structure management
    """

    def __init__(self, config: Config):
        """
        Initialize storage manager.

        Args:
            config: Configuration object with storage settings
        """
        self.config = config
        self.storage_path = Path(config.storage_path)
        self._current_disk_usage_bytes = 0
        self._ensure_storage_directory()
        self._calculate_initial_disk_usage()

    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logging.info("Storage directory: %s", self.storage_path)
        except Exception as e:
            logging.error("Failed to create storage directory: %s", e)
            raise

    def _calculate_initial_disk_usage(self):
        """
        Calculate disk usage once on startup.

        This walks through all existing files to get the initial disk usage.
        After this, we maintain a running total in memory.
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.storage_path):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)

            self._current_disk_usage_bytes = total_size
            usage_mb = total_size / (1024 * 1024)
            metrics.update_disk_usage(usage_mb)
            logging.info("Initial disk usage: %.2f MB / %d MB", usage_mb, self.config.max_disk_mb)

        except Exception as e:
            logging.error("Error calculating initial disk usage: %s", e)
            self._current_disk_usage_bytes = 0

    def store_frame(self, frame: np.ndarray, timestamp: datetime) -> bool:
        """
        Store frame to local disk.

        Organizes images by date in directory structure:
        /storage_path/YYYY-MM-DD/HH-MM-SS-microseconds.jpg

        Args:
            frame: Image frame to store (NumPy array, BGR format)
            timestamp: Capture timestamp

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Create date-based subdirectory
            date_str = timestamp.strftime("%Y-%m-%d")
            day_dir = self.storage_path / date_str
            day_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            filename = timestamp.strftime("%H-%M-%S-%f") + ".jpg"
            filepath = day_dir / filename

            # Write image to disk
            success = cv2.imwrite(str(filepath), frame)

            if success:
                # Get the file size that was just written
                file_size = filepath.stat().st_size

                # Update running total
                self._current_disk_usage_bytes += file_size

                logging.info("Frame stored locally: %s (%.2f KB)", filepath, file_size / 1024)
                metrics.increment_stored_locally()

                # Update metrics with current usage
                usage_mb = self._current_disk_usage_bytes / (1024 * 1024)
                metrics.update_disk_usage(usage_mb)

                # Enforce quota
                self._enforce_quota()

                return True
            else:
                logging.error("Failed to write image to %s", filepath)
                return False

        except Exception as e:
            logging.error("Error storing frame: %s", e)
            return False

    def _enforce_quota(self):
        """
        Enforce disk quota by deleting oldest data.

        Optimized algorithm:
        1. Check if current usage exceeds quota
        2. If exceeded, first delete entire folders older than retention_days
        3. Recalculate usage after folder deletion
        4. If still exceeded, delete oldest individual files until under quota

        This is efficient because:
        - Early exit if under quota (no file enumeration)
        - Folder deletion is faster than individual file deletion
        - Only enumerate files if absolutely necessary
        """
        try:
            max_bytes = self.config.max_disk_mb * 1024 * 1024

            # Early exit: Check if quota is even exceeded
            if self._current_disk_usage_bytes <= max_bytes:
                return

            logging.info(
                "Disk quota exceeded: %.2f MB / %d MB",
                self._current_disk_usage_bytes / (1024*1024),
                self.config.max_disk_mb
            )

            now = datetime.now()
            cutoff_date = now - timedelta(days=self.config.retention_days)

            # Step 1: Delete entire folders older than retention_days
            folders_deleted = False
            for item in sorted(self.storage_path.iterdir()):
                if item.is_dir():
                    try:
                        # Folder names are in YYYY-MM-DD format
                        folder_date = datetime.strptime(item.name, "%Y-%m-%d")

                        if folder_date < cutoff_date:
                            # Calculate folder size before deletion
                            folder_size = sum(
                                f.stat().st_size for f in item.rglob('*.jpg') if f.is_file()
                            )

                            # Delete entire folder
                            shutil.rmtree(item)

                            # Update running total
                            self._current_disk_usage_bytes -= folder_size
                            folders_deleted = True

                            logging.info(
                                "Deleted old folder: %s (freed %.2f MB)",
                                item.name,
                                folder_size / (1024*1024)
                            )

                    except (ValueError, OSError) as e:
                        # Skip folders that don't match date format or can't be deleted
                        logging.warning("Skipping folder %s: %s", item.name, e)
                        continue

            # Update metrics after folder deletion
            if folders_deleted:
                usage_mb = self._current_disk_usage_bytes / (1024 * 1024)
                metrics.update_disk_usage(usage_mb)
                logging.info("After folder cleanup: %.2f MB / %d MB", usage_mb, self.config.max_disk_mb)

            # Step 2: Check if still over quota
            if self._current_disk_usage_bytes <= max_bytes:
                logging.info("Quota satisfied after folder deletion")
                return

            # Step 3: Still over quota - need to delete individual files
            bytes_to_free = self._current_disk_usage_bytes - max_bytes
            logging.info("Need to free %.2f MB more", bytes_to_free / (1024*1024))

            # Enumerate all image files with metadata (only when necessary)
            image_files = []
            for dirpath, dirnames, filenames in os.walk(self.storage_path):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        filepath = Path(dirpath) / filename
                        try:
                            stat = filepath.stat()
                            image_files.append((filepath, stat.st_mtime, stat.st_size))
                        except OSError:
                            continue

            # Sort by modification time (oldest first)
            image_files.sort(key=lambda x: x[1])

            # Delete oldest files until we've freed enough space
            bytes_freed = 0
            for filepath, mtime, size in image_files:
                if bytes_freed >= bytes_to_free:
                    break

                try:
                    filepath.unlink()
                    bytes_freed += size
                    self._current_disk_usage_bytes -= size

                    logging.info(
                        "Deleted old image: %s (freed %.2f KB, total freed: %.2f MB)",
                        filepath.name,
                        size / 1024,
                        bytes_freed / (1024*1024)
                    )

                except Exception as e:
                    logging.error("Failed to delete %s: %s", filepath, e)

            # Final metrics update
            usage_mb = self._current_disk_usage_bytes / (1024 * 1024)
            metrics.update_disk_usage(usage_mb)
            logging.info("Final disk usage: %.2f MB / %d MB", usage_mb, self.config.max_disk_mb)

        except Exception as e:
            logging.error("Error enforcing quota: %s", e)


# ============================================================================
# Main Capture Loop
# ============================================================================

class EdgeCaptureAgent:
    """
    Main edge capture agent orchestrating all components.

    Coordinates:
    - Operating hours management
    - Image capture
    - Frame filtering
    - Server communication
    - Local storage fallback
    """

    def __init__(self, config: Config):
        """
        Initialize the edge capture agent.

        Args:
            config: Configuration object
        """
        self.config = config
        self.capture_service = CaptureService(config)
        self.storage_manager = StorageManager(config)
        self.running = False
        self.previous_accepted_frame: Optional[np.ndarray] = None

    def start(self):
        """
        Start the capture agent main loop.

        This is the main entry point that coordinates all components:
        1. Initialize camera
        2. Check operating hours
        3. Capture frames at configured interval
        4. Apply filtering
        5. Send to server or store locally
        6. Enforce storage quota
        """
        self.running = True

        try:
            # Initialize camera
            self.capture_service.initialize_camera()

            logging.info("Starting capture loop...")
            logging.info("Capture interval: %d seconds", self.config.capture_interval_sec)
            logging.info("Operating hours: %d:00 - %d:00", self.config.start_hour, self.config.end_hour)

            while self.running:
                # Check operating hours
                if not is_within_operating_hours(self.config):
                    metrics.set_operating_state("sleeping")
                    wait_for_operating_hours(self.config)
                    continue

                metrics.set_operating_state("active")

                # Capture frame
                capture_start = time.time()
                frame = self.capture_service.capture_frame()

                if frame is None:
                    logging.warning("Failed to capture frame, skipping this iteration")
                    time.sleep(self.config.capture_interval_sec)
                    continue

                timestamp = datetime.now()

                # Apply filter
                should_accept, reason = filter_frame(
                    frame,
                    self.previous_accepted_frame,
                    self.config
                )

                if not should_accept:
                    logging.debug("Frame filtered: %s", reason)
                    metrics.increment_filtered()

                    # Sleep for remaining interval time
                    elapsed = time.time() - capture_start
                    sleep_time = max(0, self.config.capture_interval_sec - elapsed)
                    time.sleep(sleep_time)
                    continue

                # Frame accepted - update previous frame
                self.previous_accepted_frame = frame.copy()

                # Try to send to server
                sent = send_frame_to_server(frame, timestamp, self.config)

                # If send failed, store locally
                if not sent:
                    logging.info("Server unavailable, storing frame locally")
                    self.storage_manager.store_frame(frame, timestamp)

                # Sleep for remaining interval time
                elapsed = time.time() - capture_start
                sleep_time = max(0, self.config.capture_interval_sec - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logging.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logging.error("Fatal error in capture loop: %s", e, exc_info=True)
        finally:
            self.cleanup()

    def stop(self):
        """Stop the capture agent."""
        logging.info("Stopping capture agent...")
        self.running = False

    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        self.capture_service.cleanup()
        logging.info("Shutdown complete")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_level: str = "INFO"):
    """
    Configure structured logging for the service.

    Logs are written to stdout in a structured format suitable
    for systemd journald.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Logging initialized at %s level", log_level)


# ============================================================================
# Signal Handling
# ============================================================================

agent: Optional[EdgeCaptureAgent] = None

def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully.

    Ensures clean shutdown on SIGINT and SIGTERM.
    """
    logging.info("Received signal %d, initiating graceful shutdown...", signum)
    if agent is not None:
        agent.stop()


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Bird Detection Edge Capture Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run with default config
        python main.py

        # Run with custom config
        python main.py --config /path/to/config.yaml

        # Run with debug logging
        python main.py --log-level DEBUG
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../configs/image_capture.yaml',
        help='Path to YAML configuration file (default: ../configs/image_capture.yaml)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the edge capture agent.

    Workflow:
    1. Parse arguments
    2. Setup logging
    3. Load configuration
    4. Start metrics server
    5. Create and start capture agent
    6. Handle graceful shutdown
    """
    global agent

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    logging.info("=" * 70)
    logging.info("Bird Detection Edge Capture Agent v1.0.0")
    logging.info("=" * 70)

    try:
        # Load configuration
        logging.info("Loading configuration from: %s", args.config)
        config = Config.from_yaml(args.config)
        logging.info("Configuration loaded successfully")

        # Start metrics server
        start_metrics_server(config)

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Create and start agent
        agent = EdgeCaptureAgent(config)
        agent.start()

    except FileNotFoundError:
        logging.error("Configuration file not found: %s", args.config)
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error("Failed to parse configuration file: %s", e)
        sys.exit(1)
    except Exception as e:
        logging.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
