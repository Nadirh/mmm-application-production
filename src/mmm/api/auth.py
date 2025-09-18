"""
Authentication module for MMM application.
"""
import os
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse, HTMLResponse
import base64
import structlog

logger = structlog.get_logger()

# Security instance for HTTP Basic Auth
security = HTTPBasic(auto_error=False)

# Simple in-memory session store (in production, consider using Redis)
sessions = {}

# Configuration from environment variables
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "mmm_admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "SecureMMM2024!@#")
SESSION_COOKIE_NAME = "mmm_session"
SESSION_DURATION_HOURS = int(os.getenv("SESSION_DURATION_HOURS", "8"))


def verify_credentials(credentials: HTTPBasicCredentials) -> bool:
    """Verify username and password."""
    if not credentials:
        return False

    correct_username = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)

    return correct_username and correct_password


def create_session() -> str:
    """Create a new session token."""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "created_at": secrets.randbits(32),  # Simple timestamp substitute
        "valid": True
    }
    return token


def verify_session(token: str) -> bool:
    """Verify if a session token is valid."""
    return token in sessions and sessions[token].get("valid", False)


async def auth_middleware(request: Request, call_next):
    """
    Authentication middleware that protects all routes except health and static assets.
    """
    path = request.url.path

    # Skip authentication for health check, training progress, cancel, and admin list endpoints
    if (path.startswith("/api/health") or
        "/training/progress/" in path or
        "/training/cancel/" in path or
        "/training/force-cancel/" in path or
        path == "/api/admin/training/list"):
        return await call_next(request)

    # Skip authentication for static files (CSS, JS, etc.)
    if path.startswith("/static"):
        return await call_next(request)

    # Check for session cookie
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token and verify_session(session_token):
        return await call_next(request)

    # For API endpoints, return 401
    if path.startswith("/api") or path.startswith("/ws"):
        # Check for Basic Auth header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Basic "):
            try:
                # Decode basic auth
                encoded_credentials = auth_header[6:]
                decoded = base64.b64decode(encoded_credentials).decode("utf-8")
                username, password = decoded.split(":", 1)

                credentials = HTTPBasicCredentials(username=username, password=password)
                if verify_credentials(credentials):
                    # Create session for valid credentials
                    token = create_session()
                    response = await call_next(request)
                    response.set_cookie(
                        key=SESSION_COOKIE_NAME,
                        value=token,
                        max_age=SESSION_DURATION_HOURS * 3600,
                        httponly=True,
                        samesite="lax"
                    )
                    return response
            except Exception as e:
                logger.error(f"Error processing auth header: {e}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic realm='MMM Application'"},
        )

    # For the main page, show login form
    if path == "/" or not path.startswith("/"):
        # Check if this is a form submission
        if request.method == "POST":
            form = await request.form()
            username = form.get("username")
            password = form.get("password")

            credentials = HTTPBasicCredentials(username=username, password=password)
            if verify_credentials(credentials):
                # Create session
                token = create_session()
                response = RedirectResponse(url="/", status_code=303)
                response.set_cookie(
                    key=SESSION_COOKIE_NAME,
                    value=token,
                    max_age=SESSION_DURATION_HOURS * 3600,
                    httponly=True,
                    samesite="lax"
                )
                return response
            else:
                # Show login form with error
                return HTMLResponse(get_login_page(error="Invalid credentials"), status_code=401)

        # Show login form
        return HTMLResponse(get_login_page(), status_code=401)

    return await call_next(request)


def get_login_page(error: Optional[str] = None) -> str:
    """Generate the login page HTML."""
    error_html = f'<div style="color: red; margin-bottom: 10px;">{error}</div>' if error else ""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MMM Dashboard - Login</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 0;
            }}
            .login-container {{
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .subtitle {{
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            input {{
                width: 100%;
                padding: 12px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }}
            input:focus {{
                outline: none;
                border-color: #667eea;
            }}
            button {{
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s;
            }}
            button:hover {{
                transform: translateY(-1px);
            }}
            .info {{
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
                font-size: 13px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="login-container">
            <h1>Media Mix Modeling</h1>
            <div class="subtitle">Please sign in to continue</div>
            {error_html}
            <form method="POST" action="/">
                <input type="text" name="username" placeholder="Username" required autofocus>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Sign In</button>
            </form>
            <div class="info">
                <strong>Note:</strong> This is a protected production environment.
                Contact your administrator for access credentials.
            </div>
        </div>
    </body>
    </html>
    """