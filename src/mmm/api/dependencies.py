"""
FastAPI dependencies for request handling.
"""
from fastapi import Header, HTTPException
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger()


async def get_client_context(
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID")
) -> Tuple[str, str]:
    """
    Extract client context from request headers.

    This is a FastAPI dependency that can be injected into route handlers.

    Args:
        x_client_id: Client ID from X-Client-ID header
        x_organization_id: Organization ID from X-Organization-ID header

    Returns:
        Tuple of (client_id, organization_id), defaults to ("default", "default")

    Raises:
        HTTPException: If client ID or organization ID format is invalid
    """
    # Default values
    client_id = "default"
    organization_id = "default"

    # Process client ID if provided
    if x_client_id:
        # Validate client_id format (alphanumeric and hyphens only)
        if not all(c.isalnum() or c == '-' for c in x_client_id):
            logger.warning(f"Invalid client_id format: {x_client_id}")
            raise HTTPException(
                status_code=400,
                detail="Invalid client ID format. Use only alphanumeric characters and hyphens."
            )
        client_id = x_client_id
        logger.debug(f"Request for client: {client_id}")

    # Process organization ID if provided
    if x_organization_id:
        # Validate organization_id format
        if not all(c.isalnum() or c == '-' for c in x_organization_id):
            logger.warning(f"Invalid organization_id format: {x_organization_id}")
            raise HTTPException(
                status_code=400,
                detail="Invalid organization ID format. Use only alphanumeric characters and hyphens."
            )
        organization_id = x_organization_id
        logger.debug(f"Request for organization: {organization_id}")

    return client_id, organization_id


async def get_client_id(
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID")
) -> str:
    """
    Simple dependency to get just the client ID.

    Args:
        x_client_id: Client ID from X-Client-ID header

    Returns:
        client_id string, defaults to "default"
    """
    client_id, _ = await get_client_context(x_client_id, None)
    return client_id