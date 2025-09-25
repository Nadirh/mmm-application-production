"""
Multi-tenant data isolation and security module for MMM application.
Ensures complete data separation between clients and implements privacy controls.
"""
import os
import hashlib
import secrets
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, UTC, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import structlog
from pathlib import Path
import json
import shutil
from enum import Enum
from dataclasses import dataclass
import uuid

logger = structlog.get_logger()


class DataClassification(Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "public"           # Non-sensitive metadata
    INTERNAL = "internal"       # Internal metrics
    CONFIDENTIAL = "confidential"  # Client-specific data
    RESTRICTED = "restricted"   # Highly sensitive financial data


class AccessLevel(Enum):
    """User access levels for data."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class ClientContext:
    """Client context for data isolation."""
    client_id: str
    organization_id: str
    user_id: str
    session_id: str
    access_level: AccessLevel
    data_classification: DataClassification
    created_at: datetime
    expires_at: datetime


@dataclass
class DataAccessAudit:
    """Audit record for data access."""
    audit_id: str
    client_id: str
    user_id: str
    action: str  # upload, read, process, export, delete
    resource_type: str  # file, model, results
    resource_id: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None


class DataIsolationManager:
    """Manages multi-tenant data isolation and security."""

    def __init__(self, base_storage_path: str = "/secure/client_data"):
        self.base_storage_path = Path(base_storage_path)
        self.encryption_keys: Dict[str, bytes] = {}  # Client-specific encryption keys
        self.active_contexts: Dict[str, ClientContext] = {}  # Active client sessions
        self.audit_log: List[DataAccessAudit] = []

        # Initialize secure storage
        self._initialize_secure_storage()

    def _initialize_secure_storage(self):
        """Initialize secure storage directories with proper permissions."""
        self.base_storage_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Create subdirectories for different data types
        subdirs = ["uploads", "processed", "models", "results", "temp", "audit"]
        for subdir in subdirs:
            path = self.base_storage_path / subdir
            path.mkdir(exist_ok=True, mode=0o700)

    def generate_client_encryption_key(self, client_id: str, master_secret: str) -> bytes:
        """Generate a unique encryption key for each client."""
        # Use PBKDF2 to derive a key from client_id and master secret
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=client_id.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_secret.encode()))

        # Cache the key (in production, store in secure key management service)
        self.encryption_keys[client_id] = key

        logger.info("Generated encryption key for client", client_id=client_id)
        return key

    def get_client_storage_path(self, client_id: str, data_type: str = "uploads") -> Path:
        """Get isolated storage path for a specific client."""
        # Create client-specific directory with hashed name for obfuscation
        client_hash = hashlib.sha256(client_id.encode()).hexdigest()[:16]
        client_path = self.base_storage_path / data_type / client_hash

        # Ensure directory exists with restricted permissions
        client_path.mkdir(parents=True, exist_ok=True, mode=0o700)

        return client_path

    def encrypt_client_data(self, client_id: str, data: bytes) -> bytes:
        """Encrypt data using client-specific key."""
        if client_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for client {client_id}")

        fernet = Fernet(self.encryption_keys[client_id])
        encrypted_data = fernet.encrypt(data)

        return encrypted_data

    def decrypt_client_data(self, client_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt data using client-specific key."""
        if client_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for client {client_id}")

        fernet = Fernet(self.encryption_keys[client_id])
        decrypted_data = fernet.decrypt(encrypted_data)

        return decrypted_data

    def store_client_file(self,
                         client_id: str,
                         file_data: bytes,
                         filename: str,
                         metadata: Dict[str, Any]) -> Tuple[str, str]:
        """
        Securely store a client file with encryption and isolation.

        Returns:
            Tuple of (file_id, storage_path)
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Encrypt file data
        encrypted_data = self.encrypt_client_data(client_id, file_data)

        # Get isolated storage path
        storage_path = self.get_client_storage_path(client_id, "uploads")

        # Store encrypted file with obfuscated name
        file_hash = hashlib.sha256(file_id.encode()).hexdigest()[:16]
        encrypted_file_path = storage_path / f"{file_hash}.enc"

        with open(encrypted_file_path, 'wb') as f:
            f.write(encrypted_data)

        # Store metadata separately (also encrypted)
        metadata_with_original = {
            **metadata,
            "original_filename": filename,
            "file_id": file_id,
            "uploaded_at": datetime.now(UTC).isoformat(),
            "encryption_version": "1.0",
            "data_classification": DataClassification.RESTRICTED.value
        }

        encrypted_metadata = self.encrypt_client_data(
            client_id,
            json.dumps(metadata_with_original).encode()
        )

        metadata_path = storage_path / f"{file_hash}.meta"
        with open(metadata_path, 'wb') as f:
            f.write(encrypted_metadata)

        # Audit the file storage
        self._audit_data_access(
            client_id=client_id,
            user_id=metadata.get("user_id", "system"),
            action="upload",
            resource_type="file",
            resource_id=file_id,
            success=True
        )

        logger.info("Stored encrypted file for client",
                   client_id=client_id,
                   file_id=file_id,
                   classification=DataClassification.RESTRICTED.value)

        return file_id, str(encrypted_file_path)

    def retrieve_client_file(self,
                           client_id: str,
                           file_id: str,
                           user_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Retrieve and decrypt a client file with access control.

        Returns:
            Tuple of (decrypted_data, metadata)
        """
        # Get storage path
        storage_path = self.get_client_storage_path(client_id, "uploads")

        # Find encrypted file
        file_hash = hashlib.sha256(file_id.encode()).hexdigest()[:16]
        encrypted_file_path = storage_path / f"{file_hash}.enc"
        metadata_path = storage_path / f"{file_hash}.meta"

        if not encrypted_file_path.exists():
            self._audit_data_access(
                client_id=client_id,
                user_id=user_id,
                action="read",
                resource_type="file",
                resource_id=file_id,
                success=False,
                error_message="File not found"
            )
            raise FileNotFoundError(f"File {file_id} not found for client {client_id}")

        # Read and decrypt file
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()

        decrypted_data = self.decrypt_client_data(client_id, encrypted_data)

        # Read and decrypt metadata
        with open(metadata_path, 'rb') as f:
            encrypted_metadata = f.read()

        metadata = json.loads(
            self.decrypt_client_data(client_id, encrypted_metadata).decode()
        )

        # Audit successful access
        self._audit_data_access(
            client_id=client_id,
            user_id=user_id,
            action="read",
            resource_type="file",
            resource_id=file_id,
            success=True
        )

        return decrypted_data, metadata

    def delete_client_data(self,
                          client_id: str,
                          file_id: Optional[str] = None,
                          user_id: str = "system",
                          permanent: bool = False) -> bool:
        """
        Delete client data with secure erasure.

        Args:
            client_id: Client identifier
            file_id: Specific file to delete (None for all client data)
            user_id: User performing the deletion
            permanent: If True, securely overwrite data before deletion
        """
        try:
            if file_id:
                # Delete specific file
                storage_path = self.get_client_storage_path(client_id, "uploads")
                file_hash = hashlib.sha256(file_id.encode()).hexdigest()[:16]

                files_to_delete = [
                    storage_path / f"{file_hash}.enc",
                    storage_path / f"{file_hash}.meta"
                ]

                for file_path in files_to_delete:
                    if file_path.exists():
                        if permanent:
                            self._secure_delete_file(file_path)
                        else:
                            file_path.unlink()

                action_detail = f"delete_file:{file_id}"
            else:
                # Delete all client data
                for data_type in ["uploads", "processed", "models", "results"]:
                    client_path = self.get_client_storage_path(client_id, data_type)

                    if client_path.exists():
                        if permanent:
                            self._secure_delete_directory(client_path)
                        else:
                            shutil.rmtree(client_path)

                action_detail = "delete_all_data"

            # Audit the deletion
            self._audit_data_access(
                client_id=client_id,
                user_id=user_id,
                action=action_detail,
                resource_type="file" if file_id else "all",
                resource_id=file_id or "all",
                success=True
            )

            logger.info("Deleted client data",
                       client_id=client_id,
                       file_id=file_id,
                       permanent=permanent)

            return True

        except Exception as e:
            self._audit_data_access(
                client_id=client_id,
                user_id=user_id,
                action="delete",
                resource_type="file" if file_id else "all",
                resource_id=file_id or "all",
                success=False,
                error_message=str(e)
            )
            logger.error("Failed to delete client data",
                        client_id=client_id,
                        file_id=file_id,
                        error=str(e))
            return False

    def _secure_delete_file(self, file_path: Path):
        """Securely overwrite and delete a file."""
        if not file_path.exists():
            return

        file_size = file_path.stat().st_size

        # Overwrite with random data 3 times (DOD 5220.22-M standard)
        with open(file_path, 'rb+') as f:
            for _ in range(3):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())

        # Finally delete the file
        file_path.unlink()

    def _secure_delete_directory(self, dir_path: Path):
        """Securely delete all files in a directory."""
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                self._secure_delete_file(file_path)

        shutil.rmtree(dir_path)

    def create_client_context(self,
                            client_id: str,
                            organization_id: str,
                            user_id: str,
                            access_level: AccessLevel = AccessLevel.READ,
                            session_duration_hours: int = 8) -> ClientContext:
        """Create a new client context for session management."""
        session_id = str(uuid.uuid4())

        context = ClientContext(
            client_id=client_id,
            organization_id=organization_id,
            user_id=user_id,
            session_id=session_id,
            access_level=access_level,
            data_classification=DataClassification.RESTRICTED,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=session_duration_hours)
        )

        self.active_contexts[session_id] = context

        logger.info("Created client context",
                   client_id=client_id,
                   user_id=user_id,
                   session_id=session_id,
                   access_level=access_level.value)

        return context

    def validate_client_access(self,
                              session_id: str,
                              required_access: AccessLevel,
                              resource_client_id: str) -> bool:
        """Validate that a session has appropriate access to a resource."""
        if session_id not in self.active_contexts:
            logger.warning("Invalid session attempted access",
                          session_id=session_id,
                          resource_client_id=resource_client_id)
            return False

        context = self.active_contexts[session_id]

        # Check session expiry
        if datetime.now(UTC) > context.expires_at:
            logger.warning("Expired session attempted access",
                          session_id=session_id,
                          client_id=context.client_id)
            del self.active_contexts[session_id]
            return False

        # Check client ID match (strict isolation)
        if context.client_id != resource_client_id:
            logger.error("Cross-client access attempt blocked",
                        session_client_id=context.client_id,
                        resource_client_id=resource_client_id,
                        user_id=context.user_id)
            return False

        # Check access level
        access_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.DELETE: 3,
            AccessLevel.ADMIN: 4
        }

        if access_hierarchy[context.access_level] < access_hierarchy[required_access]:
            logger.warning("Insufficient access level",
                          session_id=session_id,
                          required=required_access.value,
                          actual=context.access_level.value)
            return False

        return True

    def _audit_data_access(self,
                          client_id: str,
                          user_id: str,
                          action: str,
                          resource_type: str,
                          resource_id: str,
                          success: bool,
                          error_message: Optional[str] = None,
                          ip_address: str = "0.0.0.0",
                          user_agent: str = "system"):
        """Record data access in audit log."""
        audit_record = DataAccessAudit(
            audit_id=str(uuid.uuid4()),
            client_id=client_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now(UTC),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )

        self.audit_log.append(audit_record)

        # Also write to persistent audit log
        audit_path = self.base_storage_path / "audit" / f"{client_id}.jsonl"
        with open(audit_path, 'a') as f:
            f.write(json.dumps({
                "audit_id": audit_record.audit_id,
                "client_id": audit_record.client_id,
                "user_id": audit_record.user_id,
                "action": audit_record.action,
                "resource_type": audit_record.resource_type,
                "resource_id": audit_record.resource_id,
                "timestamp": audit_record.timestamp.isoformat(),
                "ip_address": audit_record.ip_address,
                "user_agent": audit_record.user_agent,
                "success": audit_record.success,
                "error_message": audit_record.error_message
            }) + "\n")

    def get_audit_log(self,
                     client_id: str,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[DataAccessAudit]:
        """Retrieve audit log for a client."""
        logs = [log for log in self.audit_log if log.client_id == client_id]

        if start_date:
            logs = [log for log in logs if log.timestamp >= start_date]

        if end_date:
            logs = [log for log in logs if log.timestamp <= end_date]

        return logs

    def export_audit_report(self, client_id: str) -> Dict[str, Any]:
        """Generate audit report for compliance."""
        logs = self.get_audit_log(client_id)

        report = {
            "client_id": client_id,
            "report_generated": datetime.now(UTC).isoformat(),
            "total_accesses": len(logs),
            "successful_accesses": sum(1 for log in logs if log.success),
            "failed_accesses": sum(1 for log in logs if not log.success),
            "unique_users": len(set(log.user_id for log in logs)),
            "actions_breakdown": {},
            "resource_types": {},
            "access_timeline": []
        }

        # Breakdown by action
        for log in logs:
            if log.action not in report["actions_breakdown"]:
                report["actions_breakdown"][log.action] = 0
            report["actions_breakdown"][log.action] += 1

            if log.resource_type not in report["resource_types"]:
                report["resource_types"][log.resource_type] = 0
            report["resource_types"][log.resource_type] += 1

        # Access timeline (last 30 days)
        thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
        recent_logs = [log for log in logs if log.timestamp >= thirty_days_ago]

        for log in recent_logs[:100]:  # Limit to last 100 entries
            report["access_timeline"].append({
                "timestamp": log.timestamp.isoformat(),
                "user": log.user_id,
                "action": log.action,
                "resource": f"{log.resource_type}:{log.resource_id}",
                "success": log.success
            })

        return report


# Singleton instance
data_isolation_manager = DataIsolationManager()