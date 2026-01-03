#!/usr/bin/env python3
"""
Metrics Collection and HTTP Server for Bird Detection Edge Capture Agent

This module implements metrics collection and HTTP endpoints for monitoring:
- Service health and uptime
- Frame capture statistics
- Disk usage tracking
- Network status
- Operating hours state
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from flask import Flask, jsonify


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


def create_metrics_app(config, metrics: MetricsCollector) -> Flask:
    """
    Create Flask app for metrics HTTP server.

    Exposes three endpoints:
    - /health: Service health check
    - /metrics: Prometheus-compatible metrics
    - /stats: Human-readable JSON statistics

    Args:
        config: Configuration object
        metrics: MetricsCollector instance

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


def start_metrics_server(config, metrics: MetricsCollector):
    """
    Start the metrics HTTP server in a background thread.

    The server runs on the configured host and port, exposing
    health and metrics endpoints for monitoring.

    Args:
        config: Configuration object with metrics settings
        metrics: MetricsCollector instance
    """
    if not config.metrics_enabled:
        logging.info("Metrics server disabled in configuration")
        return

    app = create_metrics_app(config, metrics)

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
