"""
Simple client context management for multi-tenant isolation.
Extracts client_id from request headers or uses default.
"""
from fastapi import Request, HTTPException
from typing import Optional
import structlog

logger = structlog.get_logger()


class ClientContext:
    """Manages client context for multi-tenant requests."""

    @staticmethod
    def get_client_id(request: Request = None) -> str:
        """
        Extract client_id from request headers or use default.

        Headers checked (in order):
        1. X-Client-ID: Direct client identifier
        2. X-Organization-ID: Organization identifier (future use)

        Args:
            request: FastAPI request object

        Returns:
            client_id string (defaults to "default" if not provided)
        """
        # Check for client ID in headers
        if not request:
            return "default"

        client_id = request.headers.get("X-Client-ID")

        if client_id:
            # Validate client_id format (alphanumeric and hyphens only)
            if not all(c.isalnum() or c == '-' for c in client_id):
                logger.warning(f"Invalid client_id format: {client_id}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid client ID format. Use only alphanumeric characters and hyphens."
                )

            logger.info(f"Request for client: {client_id}")
            return client_id

        # Default to "default" client for backward compatibility
        return "default"

    @staticmethod
    def get_organization_id(request: Request = None) -> str:
        """
        Extract organization_id from request headers or use default.

        Args:
            request: FastAPI request object

        Returns:
            organization_id string (defaults to "default" if not provided)
        """
        if not request:
            return "default"

        org_id = request.headers.get("X-Organization-ID")

        if org_id:
            # Validate organization_id format
            if not all(c.isalnum() or c == '-' for c in org_id):
                logger.warning(f"Invalid organization_id format: {org_id}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid organization ID format. Use only alphanumeric characters and hyphens."
                )

            return org_id

        return "default"


# Global instance
client_context = ClientContext()