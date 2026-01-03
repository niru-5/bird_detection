#!/usr/bin/env python3
"""
Utility Functions for Bird Detection Edge Capture Agent

This module contains utility functions for:
- Operating hours management
- Frame filtering
- Server communication
- Logging setup
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


# ============================================================================
# Operating Hours Management
# ============================================================================

def is_within_operating_hours(config) -> bool:
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


def wait_for_operating_hours(config, metrics, shutdown_event=None):
    """
    Sleep until the next operating hours period begins.

    This function calculates the time until the next operating period
    and sleeps for that duration. It checks periodically to avoid
    oversleeping and responds to shutdown signals.

    Args:
        config: Configuration object with operating hours settings
        metrics: MetricsCollector instance for updating state
        shutdown_event: threading.Event that signals shutdown when set
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
    elapsed = 0
    
    while elapsed < wait_seconds:
        # Check if shutdown was requested
        if shutdown_event and shutdown_event.is_set():
            logging.info("Shutdown requested during wait period")
            return
        
        # Sleep for the minimum of remaining time or chunk size
        remaining = wait_seconds - elapsed
        sleep_time = min(sleep_chunk, remaining)
        
        if shutdown_event:
            # Use event.wait() for interruptible sleep
            shutdown_event.wait(timeout=sleep_time)
        else:
            time.sleep(sleep_time)
        
        elapsed += sleep_time

        # Re-check in case we're now in operating hours
        if is_within_operating_hours(config):
            break


# ============================================================================
# Frame Filtering
# ============================================================================

def filter_frame(
    current_frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    config
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
    config,
    metrics
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
        metrics: MetricsCollector instance for updating counters

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
