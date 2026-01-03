#!/usr/bin/env python3
"""
Local Storage Management for Bird Detection Edge Capture Agent

This module manages local disk storage for captured images.
Handles storage, quota enforcement, and automatic cleanup based on retention policy.
"""

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


class StorageManager:
    """
    Manages local disk storage for captured images.

    Handles:
    - Storing images to disk when server is unavailable
    - Enforcing disk quota
    - Automatic cleanup based on retention policy
    - Directory structure management
    """

    def __init__(self, config, metrics):
        """
        Initialize storage manager.

        Args:
            config: Configuration object with storage settings
            metrics: MetricsCollector instance for updating disk usage
        """
        self.config = config
        self.metrics = metrics
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
            self.metrics.update_disk_usage(usage_mb)
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
                self.metrics.increment_stored_locally()

                # Update metrics with current usage
                usage_mb = self._current_disk_usage_bytes / (1024 * 1024)
                self.metrics.update_disk_usage(usage_mb)

                # Enforce quota
                self._enforce_quota()
                
            
                logging.debug("Current disk usage: %.2f MB", usage_mb)

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
                self.metrics.update_disk_usage(usage_mb)
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
            self.metrics.update_disk_usage(usage_mb)
            logging.info("Final disk usage: %.2f MB / %d MB", usage_mb, self.config.max_disk_mb)

        except Exception as e:
            logging.error("Error enforcing quota: %s", e)
