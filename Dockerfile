# Multi-stage Dockerfile for MMM Application
# Stage 1: Base Python environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r mmm && useradd -r -g mmm mmm

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy dependency files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Development image
FROM dependencies as development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R mmm:mmm /app

# Switch to app user
USER mmm

# Expose port
EXPOSE 8000

# Development command
CMD ["python", "-m", "mmm", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production build
FROM dependencies as builder

# Copy source code
COPY . .

# Build wheel package
RUN pip install build && \
    python -m build

# Stage 5: Production image
FROM base as production

# Copy only production dependencies
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/
COPY static/ /app/static/

# Create necessary directories
RUN mkdir -p /app/logs /app/static/uploads /app/data

# Change ownership to app user
RUN chown -R mmm:mmm /app

# Switch to app user
USER mmm

# Set Python path
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "mmm", "--host", "0.0.0.0", "--port", "8000"]