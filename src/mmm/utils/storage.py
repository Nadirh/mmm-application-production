"""
Simple client-based storage management for Chunk 2.
Files are stored in client-specific directories (not encrypted yet).
"""
import os
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger()


class ClientStorageManager:
    """Manages client-specific file storage paths."""

    def __init__(self, base_path: str = None):
        """Initialize storage manager with base path."""
        if base_path is None:
            # Default to static/uploads for backward compatibility
            base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "uploads")

        self.base_path = Path(base_path).resolve()
        logger.info(f"Storage manager initialized with base path: {self.base_path}")

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_client_upload_dir(self, client_id: str = "default") -> Path:
        """
        Get the upload directory for a specific client.
        Creates the directory if it doesn't exist.

        Args:
            client_id: The client identifier (default: "default")

        Returns:
            Path object for the client's upload directory
        """
        # Create client-specific directory
        client_dir = self.base_path / client_id
        client_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Client directory: {client_dir} (exists: {client_dir.exists()})")
        return client_dir

    def get_client_file_path(self, client_id: str, filename: str) -> Path:
        """
        Get the full path for a client's file.

        Args:
            client_id: The client identifier
            filename: The filename (typically upload_id.csv)

        Returns:
            Full path to the file
        """
        client_dir = self.get_client_upload_dir(client_id)
        file_path = client_dir / filename

        logger.info(f"File path for client {client_id}: {file_path}")
        return file_path

    def ensure_file_exists(self, client_id: str, filename: str) -> bool:
        """
        Check if a file exists for a client.

        Args:
            client_id: The client identifier
            filename: The filename to check

        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_client_file_path(client_id, filename)
        exists = file_path.exists()

        if not exists:
            logger.warning(f"File not found: {file_path}")

        return exists

    def migrate_existing_files(self):
        """
        Migrate existing files from root upload directory to default client directory.
        This ensures backward compatibility with existing uploads.
        """
        # Get list of files in root directory
        root_files = [f for f in self.base_path.iterdir() if f.is_file() and f.suffix == '.csv']

        if root_files:
            logger.info(f"Found {len(root_files)} files to migrate to default client directory")

            default_dir = self.get_client_upload_dir("default")

            for file_path in root_files:
                # Move file to default client directory
                new_path = default_dir / file_path.name
                if not new_path.exists():
                    file_path.rename(new_path)
                    logger.info(f"Migrated {file_path.name} to default client directory")
                else:
                    logger.info(f"File {file_path.name} already exists in default directory, skipping")
        else:
            logger.info("No files to migrate")


# Global instance
storage_manager = ClientStorageManager()