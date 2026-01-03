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
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

# Import from local modules
from capture import CaptureService
from metrics import MetricsCollector, start_metrics_server
from storage import StorageManager
import threading
from utils import (
    filter_frame,
    is_within_operating_hours,
    send_frame_to_server,
    setup_logging,
    wait_for_operating_hours,
)


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
    storage_path: str 

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
            capture_interval_sec=int(data.get('capture_interval_sec', 5)),

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
            storage_path=storage.get('path', '/home/rpi5/bird-detection/data/images'),
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
# Global Metrics Instance
# ============================================================================

# Global metrics instance (initialized in main)
metrics: Optional[MetricsCollector] = None



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

    def __init__(self, config: Config, metrics: MetricsCollector):
        """
        Initialize the edge capture agent.

        Args:
            config: Configuration object
            metrics: MetricsCollector instance for tracking statistics
        """
        self.config = config
        self.metrics = metrics
        self.capture_service = CaptureService(config, metrics)
        self.storage_manager = StorageManager(config, metrics)
        self.shutdown_event = threading.Event()  # Changed from self.running
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
        try:
            # Initialize camera
            self.capture_service.initialize_camera()

            logging.info("Starting capture loop...")
            logging.info("Capture interval: %d seconds", self.config.capture_interval_sec)
            logging.info("Operating hours: %d:00 - %d:00", self.config.start_hour, self.config.end_hour)

            while not self.shutdown_event.is_set():
                # Check operating hours
                if not is_within_operating_hours(self.config):
                    self.metrics.set_operating_state("sleeping")
                    wait_for_operating_hours(
                        self.config, 
                        self.metrics,
                        shutdown_event=self.shutdown_event
                    )
                    
                    # Check if shutdown was requested during wait
                    if self.shutdown_event.is_set():
                        break
                        
                    continue
                logging.debug("Operating hours: %s", is_within_operating_hours(self.config))

                self.metrics.set_operating_state("active")
                logging.debug("Operating state set to active")
                # Capture frame
                capture_start = time.time()
                logging.debug("Capturing frame...")
                frame = self.capture_service.capture_frame()
                logging.debug("Frame captured")

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
                    self.metrics.increment_filtered()

                    # Sleep for remaining interval time
                    elapsed = time.time() - capture_start
                    sleep_time = max(0, self.config.capture_interval_sec - elapsed)
                    
                    # Use interruptible sleep
                    self.shutdown_event.wait(timeout=sleep_time)
                    continue

                # Frame accepted - update previous frame
                self.previous_accepted_frame = frame.copy()
                # Try to send to server
                logging.debug("Sending frame to server...")
                sent = send_frame_to_server(frame, timestamp, self.config, self.metrics)
                logging.debug("Frame sent to server: %s", sent)

                # If send failed, store locally
                if not sent:
                    logging.info("Server unavailable, storing frame locally")
                    self.storage_manager.store_frame(frame, timestamp)
                    logging.debug("Frame stored locally")

                # Sleep for remaining interval time
                elapsed = time.time() - capture_start
                logging.debug("Elapsed time: %.2f seconds", elapsed)
                sleep_time = max(0, self.config.capture_interval_sec - elapsed)
                logging.debug("Sleep time: %.2f seconds", sleep_time)
                
                # Use interruptible sleep
                self.shutdown_event.wait(timeout=sleep_time)
                logging.debug("Sleep completed")

        except KeyboardInterrupt:
            logging.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logging.error("Fatal error in capture loop: %s", e, exc_info=True)
        finally:
            self.cleanup()

    def stop(self):
        """Stop the capture agent."""
        logging.info("Stopping capture agent...")
        self.shutdown_event.set()

    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        self.capture_service.cleanup()
        logging.info("Shutdown complete")


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
        required=True,
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
    4. Initialize metrics
    5. Start metrics server
    6. Create and start capture agent
    7. Handle graceful shutdown
    """
    global agent, metrics

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

        # Initialize metrics
        metrics = MetricsCollector()

        # Start metrics server
        start_metrics_server(config, metrics)

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Create and start agent
        agent = EdgeCaptureAgent(config, metrics)
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
