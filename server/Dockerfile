# Medical Diagnostic Environment - Production Dockerfile
#
# This Dockerfile containerizes the complete Medical Diagnostic Environment.
# It can be deployed to Docker Hub, GitHub Container Registry, or Hugging Face Spaces.

FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for layer caching)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy application code
COPY models.py .
COPY client.py .
COPY server/ ./server/

# Create __init__ files for Python packages
RUN touch __init__.py && \
    touch server/__init__.py

# Health check - validates the server is running and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
