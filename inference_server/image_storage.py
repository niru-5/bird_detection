#!/usr/bin/env python3
"""
Image Storage Manager for Bird Detection Inference Server

Manages image storage using MinIO (S3-compatible) with LakeFS versioning.
Saves detection results for later analysis and model improvement.
"""

import io
import logging
import os
import random
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, Any

import boto3
from PIL import Image
import lakefs_sdk
from lakefs_sdk.client import LakeFSClient
from lakefs_sdk.models import RepositoryCreation
from apscheduler.schedulers.background import BackgroundScheduler


logger = logging.getLogger(__name__)


class ImageStorage:
    """Manages image storage in MinIO with LakeFS versioning."""

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_bucket_name: str,
        lakefs_host: str,
        lakefs_access_key: str,
        lakefs_secret_key: str,
        lakefs_repo_name: str,
        save_probability_no_detections: float = 0.01,
        commit_interval: int = 100,
        daily_commit_hour: int = 18,
        daily_commit_minute: int = 0
    ):
        """
        Initialize image storage manager.

        Args:
            minio_endpoint: MinIO endpoint (e.g., 'localhost:9000' or 'minio:9000')
            minio_access_key: MinIO access key
            minio_secret_key: MinIO secret key
            minio_bucket_name: MinIO bucket name for storing images
            lakefs_host: LakeFS host (e.g., 'localhost:8000' or 'lakefs:8000')
            lakefs_access_key: LakeFS access key
            lakefs_secret_key: LakeFS secret key
            lakefs_repo_name: LakeFS repository name
            save_probability_no_detections: Probability (0-1) of saving images without detections
            commit_interval: Number of images to save before auto-commit (0 = disabled)
            daily_commit_hour: Hour (0-23) for daily commit
            daily_commit_minute: Minute (0-59) for daily commit
        """
        self.minio_endpoint = minio_endpoint
        self.minio_bucket_name = minio_bucket_name
        self.lakefs_repo_name = lakefs_repo_name
        self.save_probability_no_detections = save_probability_no_detections
        self.commit_interval = commit_interval
        self.daily_commit_hour = daily_commit_hour
        self.daily_commit_minute = daily_commit_minute

        # Tracking for auto-commits
        self.uncommitted_count = 0
        self.commit_lock = threading.Lock()
        self.scheduler = None

        # Initialize MinIO client (S3-compatible)
        logger.info(f"Initializing MinIO client: {minio_endpoint}")
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'http://{minio_endpoint}',
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key
        )

        # Initialize LakeFS client
        logger.info(f"Initializing LakeFS client: {lakefs_host}")
        configuration = lakefs_sdk.Configuration(
            host=f'http://{lakefs_host}',
            username=lakefs_access_key,
            password=lakefs_secret_key
        )
        self.lakefs_client = LakeFSClient(configuration)

        logger.info("Image storage manager initialized")

    def setup_infrastructure(self):
        """
        Initialize storage buckets and repositories.
        Should be called once during server startup.
        """
        logger.info("Setting up storage infrastructure...")

        # Create MinIO bucket
        try:
            self.s3_client.create_bucket(Bucket=self.minio_bucket_name)
            logger.info(f"MinIO bucket '{self.minio_bucket_name}' created")
        except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"MinIO bucket '{self.minio_bucket_name}' already exists")
        except Exception as e:
            logger.warning(f"Error creating MinIO bucket: {e}")

        # Create LakeFS repository
        try:
            repo = RepositoryCreation(
                name=self.lakefs_repo_name,
                storage_namespace=f's3://{self.minio_bucket_name}/',
                default_branch='main'
            )
            self.lakefs_client.repositories_api.create_repository(repo)
            logger.info(f"LakeFS repository '{self.lakefs_repo_name}' created")
        except Exception as e:
            if 'already exists' in str(e).lower():
                logger.info(f"LakeFS repository '{self.lakefs_repo_name}' already exists")
            else:
                logger.warning(f"Error creating LakeFS repository: {e}")

        logger.info("Storage infrastructure setup complete")

        # Start the daily commit scheduler
        self._start_scheduler()

    def should_save_image(self, num_detections: int) -> bool:
        """
        Determine if an image should be saved based on detection count.

        Args:
            num_detections: Number of detections in the image

        Returns:
            True if image should be saved, False otherwise
        """
        if num_detections > 0:
            # Always save images with detections
            return True
        else:
            # Probabilistically save images without detections
            return random.random() < self.save_probability_no_detections

    def save_detection_image(
        self,
        image_pil: Image.Image,
        detection_results: Dict[str, Any],
        device_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        branch: str = 'main'
    ) -> Optional[str]:
        """
        Save image with detection results to LakeFS.

        Args:
            image_pil: PIL Image object
            detection_results: Detection results dictionary
            device_id: ID of the device that captured the image
            timestamp: Capture timestamp
            branch: LakeFS branch to save to

        Returns:
            Path where image was saved, or None if not saved
        """
        num_detections = detection_results.get('num_detections', 0)

        # Decide whether to save
        if not self.should_save_image(num_detections):
            logger.debug(f"Skipping save for image with {num_detections} detections")
            return None

        # Generate filename with metadata
        if timestamp:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                # Fallback if parsing fails
                dt = datetime.now()
        else:
            dt = datetime.now()

        device_str = device_id or 'unknown'

        # Create path: dataset/day_dir/device_str/filename
        date_str = dt.strftime("%Y-%m-%d")
        filename = dt.strftime("%H-%M-%S-%f") + "_" + str(num_detections) + ".jpg"
        path = f'dataset/{date_str}/{device_str}/{filename}'

        # Convert PIL image to bytes
        image_buffer = io.BytesIO()
        image_pil.save(image_buffer, format='JPEG', quality=95)
        image_bytes = image_buffer.getvalue()

        # Upload to LakeFS
        try:
            self.upload_to_lakefs(image_bytes, path, branch)
            logger.info(
                f"Saved image: {path} "
                f"(detections={num_detections}, device={device_str})"
            )

            # Track uncommitted images and auto-commit if needed
            with self.commit_lock:
                self.uncommitted_count += 1
                if self.commit_interval > 0 and self.uncommitted_count >= self.commit_interval:
                    self._auto_commit(branch)

            return path
        except Exception as e:
            logger.error(f"Failed to save image to LakeFS: {e}", exc_info=True)
            return None

    def upload_to_lakefs(
        self,
        image_data: bytes,
        path: str,
        branch: str = 'main'
    ):
        """
        Upload image data to LakeFS.

        Args:
            image_data: Image bytes
            path: Path within repository
            branch: LakeFS branch to upload to
        """
        self.lakefs_client.objects_api.upload_object(
            repository=self.lakefs_repo_name,
            branch=branch,
            path=path,
            content=image_data
        )
        logger.debug(f"Uploaded to LakeFS branch '{branch}': {path}")

    def commit_changes(self, branch: str, message: str) -> str:
        """
        Commit changes to LakeFS branch.

        Args:
            branch: Branch to commit
            message: Commit message

        Returns:
            Commit ID
        """
        commit = self.lakefs_client.commits_api.commit(
            repository=self.lakefs_repo_name,
            branch=branch,
            commit_creation={'message': message}
        )
        logger.info(f"Committed to {branch}: {commit.id[:8]} - {message}")
        return commit.id

    def create_branch(self, branch_name: str, source_branch: str = 'main'):
        """
        Create a new branch in LakeFS.

        Args:
            branch_name: Name of the new branch
            source_branch: Source branch to branch from
        """
        from lakefs_sdk.models import BranchCreation

        branch = BranchCreation(
            name=branch_name,
            source=source_branch
        )
        self.lakefs_client.branches_api.create_branch(
            repository=self.lakefs_repo_name,
            branch_creation=branch
        )
        logger.info(f"Created branch: {branch_name} from {source_branch}")

    def _auto_commit(self, branch: str = 'main'):
        """
        Perform auto-commit with uncommitted images.
        Should be called with commit_lock held.

        Args:
            branch: Branch to commit to
        """
        if self.uncommitted_count == 0:
            logger.debug("No uncommitted images to commit")
            return

        count = self.uncommitted_count
        try:
            commit_msg = f"Auto-commit: {count} images saved"
            commit_id = self.commit_changes(branch, commit_msg)
            self.uncommitted_count = 0
            logger.info(f"Auto-committed {count} images: {commit_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to auto-commit: {e}", exc_info=True)

    def _scheduled_commit(self):
        """
        Scheduled daily commit job.
        """
        logger.info("Running scheduled daily commit...")
        with self.commit_lock:
            self._auto_commit(branch='main')

    def _start_scheduler(self):
        """
        Start the background scheduler for daily commits.
        """
        if self.scheduler is not None:
            logger.warning("Scheduler already running")
            return

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._scheduled_commit,
            'cron',
            hour=self.daily_commit_hour,
            minute=self.daily_commit_minute,
            id='daily_commit'
        )
        self.scheduler.start()
        logger.info(
            f"Started daily commit scheduler: "
            f"{self.daily_commit_hour:02d}:{self.daily_commit_minute:02d}"
        )

    def shutdown(self):
        """
        Shutdown the image storage manager.
        Commits any remaining uncommitted images and stops the scheduler.
        """
        logger.info("Shutting down image storage...")

        # Commit any remaining images
        with self.commit_lock:
            if self.uncommitted_count > 0:
                logger.info(f"Committing {self.uncommitted_count} remaining images...")
                self._auto_commit(branch='main')

        # Stop scheduler
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

        logger.info("Image storage shutdown complete")
