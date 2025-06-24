# Multi-stage build for OpenGuidance
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=latest

# Add metadata labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="OpenGuidance" \
      org.label-schema.description="Advanced AI assistant framework" \
      org.label-schema.url="https://openguidance.ai" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/openguidance/openguidance" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      maintainer="Nik Jois <nikjois@llamasearch.ai>"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directory
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install --no-deps -r requirements-dev.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENGUIDANCE_ENV=production \
    OPENGUIDANCE_LOG_LEVEL=INFO \
    OPENGUIDANCE_HOST=0.0.0.0 \
    OPENGUIDANCE_PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY openguidance/ ./openguidance/
COPY setup.py ./
COPY README.md ./
COPY CHANGELOG.md ./

# Install the application
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R appuser:appuser /app

# Copy configuration files
COPY docker-compose.yml ./
COPY nginx/ ./nginx/

# Health check script
COPY <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import asyncio
import sys
import httpx

async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8000/health', timeout=10.0)
            if response.status_code == 200:
                print('Health check passed')
                sys.exit(0)
            else:
                print(f'Health check failed with status {response.status_code}')
                sys.exit(1)
    except Exception as e:
        print(f'Health check failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(health_check())
EOF

RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

# Default command
CMD ["uvicorn", "openguidance.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage
FROM production as development

USER root

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Development command
CMD ["uvicorn", "openguidance.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--debug"]