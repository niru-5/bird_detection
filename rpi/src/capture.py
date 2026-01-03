#!/usr/bin/env python3
"""
Image Capture Service for Bird Detection Edge Capture Agent

This module implements the camera capture service for the Raspberry Pi.
Handles camera initialization, configuration, and image capture with support
for ROI cropping and sensor settings.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2


class CaptureService:
    """
    Manages Raspberry Pi Camera for image capture.

    Handles camera initialization, configuration, and image capture
    with support for ROI cropping and sensor settings.
    """

    def __init__(self, config, metrics):
        """
        Initialize the capture service.

        Args:
            config: Configuration object with sensor settings
            metrics: MetricsCollector instance for updating status
        """
        self.config = config
        self.metrics = metrics
        self.camera: Optional[Picamera2] = None
        self.last_frame: Optional[np.ndarray] = None
        self.figure_out_crop_settings()
        
    def figure_out_crop_settings(self):
        """
        Figure out the crop settings for the camera.
        """
        self.crop_x = max(0, self.config.crop_x)
        self.crop_y = max(0, self.config.crop_y)
        self.crop_width = self.config.crop_width
        self.crop_height = self.config.crop_height
        self.apply_crop = True
        

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
            self.metrics.set_camera_status("ok")

        except Exception as e:
            logging.error("Failed to initialize camera: %s", e)
            self.metrics.set_camera_status("error")
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
            if self.apply_crop:

                x1 = self.config.crop_x
                y1 = self.config.crop_y
                x2 = min(x1 + self.config.crop_width, frame.shape[1])
                y2 = min(y1 + self.config.crop_height, frame.shape[0])

                frame = frame[y1:y2, x1:x2]
                logging.debug("Applied ROI crop: (%d,%d) to (%d,%d)", x1, y1, x2, y2)

            self.metrics.increment_captured()
            self.last_frame = frame

            return frame

        except Exception as e:
            logging.error("Failed to capture frame: %s", e)
            self.metrics.set_camera_status("error")
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
