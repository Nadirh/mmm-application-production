"""
Multi-tenant authentication and authorization module.
Implements client-specific authentication with role-based access control.
"""
import secrets
import jwt
import bcrypt
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog
import json
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis

logger = structlog.get_logger()


class UserRole(Enum):
    """User roles within an organization."""
    VIEWER = "viewer"           # Read-only access to results
    ANALYST = "analyst"         # Can upload data and run models
    MANAGER = "manager"         # Can manage team access and settings
    ADMIN = "admin"            # Full access to client organization
    SUPER_ADMIN = "super_admin" # System-level admin (internal use only)


class Permission(Enum):
    """Granular permissions for access control."""
    # Data permissions
    DATA_UPLOAD = "data:upload"
    DATA_VIEW = "data:view"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Model permissions
    MODEL_TRAIN = "model:train"
    MODEL_VIEW = "model:view"
    MODEL_DELETE = "model:delete"
    MODEL_EXPORT = "model:export"

    # Optimization permissions
    OPTIMIZATION_RUN = "optimization:run"
    OPTIMIZATION_VIEW = "optimization:view"
    OPTIMIZATION_EXPORT = "optimization:export"

    # Admin permissions
    USER_MANAGE = "user:manage"
    SETTINGS_MANAGE = "settings:manage"
    AUDIT_VIEW = "audit:view"
    BILLING_VIEW = "billing:view"


@dataclass
class ClientOrganization:
    """Client organization details."""
    organization_id: str
    organization_name: str
    subscription_tier: str  # enterprise, mid_market, small_business, trial
    data_retention_days: int
    max_users: int
    max_monthly_runs: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    features: List[str] = None  # Enabled features


@dataclass
class User:
    """User account with client association."""
    user_id: str
    email: str
    organization_id: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    mfa_enabled: bool = False
    password_hash: str = None


@dataclass
class AuthSession:
    """Authenticated session with client context."""
    session_id: str
    user_id: str
    organization_id: str
    client_id: str  # Specific client within organization
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_valid: bool = True


class MultiTenantAuthManager:
    """Manages multi-tenant authentication and authorization."""

    def __init__(self,
                 jwt_secret: str = None,
                 jwt_algorithm: str = "HS256",
                 session_duration_hours: int = 8):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.jwt_algorithm = jwt_algorithm
        self.session_duration_hours = session_duration_hours

        # In-memory stores (in production, use database)
        self.organizations: Dict[str, ClientOrganization] = {}
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, AuthSession] = {}

        # Role-permission mapping
        self.role_permissions = {
            UserRole.VIEWER: [
                Permission.DATA_VIEW,
                Permission.MODEL_VIEW,
                Permission.OPTIMIZATION_VIEW
            ],
            UserRole.ANALYST: [
                Permission.DATA_UPLOAD,
                Permission.DATA_VIEW,
                Permission.DATA_EXPORT,
                Permission.MODEL_TRAIN,
                Permission.MODEL_VIEW,
                Permission.MODEL_EXPORT,
                Permission.OPTIMIZATION_RUN,
                Permission.OPTIMIZATION_VIEW,
                Permission.OPTIMIZATION_EXPORT
            ],
            UserRole.MANAGER: [
                # All analyst permissions plus:
                Permission.DATA_UPLOAD,
                Permission.DATA_VIEW,
                Permission.DATA_DELETE,
                Permission.DATA_EXPORT,
                Permission.MODEL_TRAIN,
                Permission.MODEL_VIEW,
                Permission.MODEL_DELETE,
                Permission.MODEL_EXPORT,
                Permission.OPTIMIZATION_RUN,
                Permission.OPTIMIZATION_VIEW,
                Permission.OPTIMIZATION_EXPORT,
                Permission.USER_MANAGE,
                Permission.SETTINGS_MANAGE,
                Permission.AUDIT_VIEW
            ],
            UserRole.ADMIN: [
                # All permissions for the organization
                perm for perm in Permission
            ],
            UserRole.SUPER_ADMIN: [
                # System-level permissions (internal use)
                perm for perm in Permission
            ]
        }

    def create_organization(self,
                          organization_name: str,
                          subscription_tier: str = "trial",
                          admin_email: str = None) -> Tuple[ClientOrganization, User]:
        """Create a new client organization with admin user."""
        organization_id = f"org_{secrets.token_urlsafe(16)}"

        # Set tier-based limits
        tier_configs = {
            "enterprise": {
                "data_retention_days": 730,  # 2 years
                "max_users": 100,
                "max_monthly_runs": 1000,
                "features": ["advanced_optimization", "api_access", "custom_reports", "sso"]
            },
            "mid_market": {
                "data_retention_days": 365,
                "max_users": 25,
                "max_monthly_runs": 100,
                "features": ["standard_optimization", "api_access"]
            },
            "small_business": {
                "data_retention_days": 180,
                "max_users": 10,
                "max_monthly_runs": 20,
                "features": ["basic_optimization"]
            },
            "trial": {
                "data_retention_days": 30,
                "max_users": 3,
                "max_monthly_runs": 5,
                "features": ["basic_optimization"]
            }
        }

        config = tier_configs.get(subscription_tier, tier_configs["trial"])

        org = ClientOrganization(
            organization_id=organization_id,
            organization_name=organization_name,
            subscription_tier=subscription_tier,
            data_retention_days=config["data_retention_days"],
            max_users=config["max_users"],
            max_monthly_runs=config["max_monthly_runs"],
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30) if subscription_tier == "trial" else None,
            is_active=True,
            features=config["features"]
        )

        self.organizations[organization_id] = org

        # Create admin user
        admin_user = None
        if admin_email:
            admin_user = self.create_user(
                email=admin_email,
                organization_id=organization_id,
                role=UserRole.ADMIN,
                password=secrets.token_urlsafe(16)  # Generate initial password
            )

        logger.info("Created organization",
                   organization_id=organization_id,
                   name=organization_name,
                   tier=subscription_tier)

        return org, admin_user

    def create_user(self,
                   email: str,
                   organization_id: str,
                   role: UserRole,
                   password: str) -> User:
        """Create a new user within an organization."""
        if organization_id not in self.organizations:
            raise ValueError(f"Organization {organization_id} not found")

        org = self.organizations[organization_id]

        # Check user limit
        org_users = [u for u in self.users.values() if u.organization_id == organization_id]
        if len(org_users) >= org.max_users:
            raise ValueError(f"Organization has reached maximum user limit ({org.max_users})")

        user_id = f"user_{secrets.token_urlsafe(16)}"

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        user = User(
            user_id=user_id,
            email=email,
            organization_id=organization_id,
            role=role,
            permissions=self.role_permissions.get(role, []),
            created_at=datetime.now(UTC),
            is_active=True,
            password_hash=password_hash
        )

        self.users[user_id] = user

        logger.info("Created user",
                   user_id=user_id,
                   email=email,
                   organization_id=organization_id,
                   role=role.value)

        return user

    def authenticate_user(self,
                         email: str,
                         password: str,
                         client_id: str,
                         ip_address: str = "0.0.0.0",
                         user_agent: str = "unknown") -> Optional[str]:
        """
        Authenticate user and create session.

        Returns:
            JWT token if authentication successful, None otherwise
        """
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email and u.is_active:
                user = u
                break

        if not user:
            logger.warning("Authentication failed - user not found", email=email)
            return None

        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            logger.warning("Authentication failed - invalid password",
                          email=email,
                          user_id=user.user_id)
            return None

        # Check organization is active
        org = self.organizations.get(user.organization_id)
        if not org or not org.is_active:
            logger.warning("Authentication failed - inactive organization",
                          email=email,
                          organization_id=user.organization_id)
            return None

        # Check organization expiry
        if org.expires_at and datetime.now(UTC) > org.expires_at:
            logger.warning("Authentication failed - expired organization",
                          email=email,
                          organization_id=user.organization_id)
            return None

        # Create session
        session_id = secrets.token_urlsafe(32)
        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            organization_id=user.organization_id,
            client_id=client_id,
            role=user.role,
            permissions=user.permissions,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=self.session_duration_hours),
            ip_address=ip_address,
            user_agent=user_agent,
            is_valid=True
        )

        self.sessions[session_id] = session

        # Update last login
        user.last_login = datetime.now(UTC)

        # Generate JWT token
        token_payload = {
            "session_id": session_id,
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "client_id": client_id,
            "role": user.role.value,
            "exp": session.expires_at.timestamp(),
            "iat": session.created_at.timestamp()
        }

        token = jwt.encode(token_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        logger.info("User authenticated successfully",
                   user_id=user.user_id,
                   organization_id=user.organization_id,
                   client_id=client_id,
                   session_id=session_id)

        return token

    def verify_token(self, token: str) -> Optional[AuthSession]:
        """Verify JWT token and return session."""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            session_id = payload.get("session_id")
            if not session_id or session_id not in self.sessions:
                logger.warning("Invalid session in token", session_id=session_id)
                return None

            session = self.sessions[session_id]

            # Check session validity
            if not session.is_valid:
                logger.warning("Invalid session", session_id=session_id)
                return None

            # Check expiry
            if datetime.now(UTC) > session.expires_at:
                logger.warning("Expired session", session_id=session_id)
                session.is_valid = False
                return None

            return session

        except jwt.ExpiredSignatureError:
            logger.warning("Expired token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None

    def check_permission(self,
                        session: AuthSession,
                        required_permission: Permission,
                        resource_client_id: Optional[str] = None) -> bool:
        """Check if session has required permission."""
        # Super admin has all permissions
        if session.role == UserRole.SUPER_ADMIN:
            return True

        # Check client isolation if resource_client_id provided
        if resource_client_id and session.client_id != resource_client_id:
            logger.error("Cross-client access attempt",
                        session_client_id=session.client_id,
                        resource_client_id=resource_client_id,
                        user_id=session.user_id)
            return False

        # Check permission
        if required_permission not in session.permissions:
            logger.warning("Permission denied",
                          user_id=session.user_id,
                          required=required_permission.value,
                          user_permissions=[p.value for p in session.permissions])
            return False

        return True

    def revoke_session(self, session_id: str):
        """Revoke a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_valid = False
            logger.info("Session revoked", session_id=session_id)

    def revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a user."""
        for session in self.sessions.values():
            if session.user_id == user_id:
                session.is_valid = False

        logger.info("All sessions revoked for user", user_id=user_id)

    def get_organization_usage(self, organization_id: str) -> Dict[str, Any]:
        """Get usage statistics for an organization."""
        if organization_id not in self.organizations:
            return None

        org = self.organizations[organization_id]

        # Count active users
        active_users = sum(1 for u in self.users.values()
                          if u.organization_id == organization_id and u.is_active)

        # Count active sessions
        active_sessions = sum(1 for s in self.sessions.values()
                            if s.organization_id == organization_id and s.is_valid)

        return {
            "organization_id": organization_id,
            "organization_name": org.organization_name,
            "subscription_tier": org.subscription_tier,
            "active_users": active_users,
            "max_users": org.max_users,
            "active_sessions": active_sessions,
            "data_retention_days": org.data_retention_days,
            "max_monthly_runs": org.max_monthly_runs,
            "features": org.features,
            "is_active": org.is_active,
            "expires_at": org.expires_at.isoformat() if org.expires_at else None
        }


# FastAPI dependency for authentication
security = HTTPBearer()


async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> AuthSession:
    """FastAPI dependency to get current authenticated session."""
    auth_manager = request.app.state.auth_manager  # Assume auth_manager in app state

    session = auth_manager.verify_token(credentials.credentials)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


async def require_permission(permission: Permission):
    """FastAPI dependency to require specific permission."""
    async def permission_checker(session: AuthSession = Depends(get_current_session)):
        auth_manager = session.app.state.auth_manager

        if not auth_manager.check_permission(session, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )

        return session

    return permission_checker