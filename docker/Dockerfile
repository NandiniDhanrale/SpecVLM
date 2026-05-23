# =============================================================================
# SpecVLM — Production Docker Image
# =============================================================================
# Multi-stage build for minimal production image size.
#
# Stage 1: Build environment
# Stage 2: Runtime with CUDA dependencies
# Stage 3: Production image
#
# Build:
#   docker build -t specvlm:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 -p 8001:8001 specvlm:latest
#
# Environment variables:
#   SPECVLM_CONFIG=/app/config/model_config.yaml
#   SPECVLM__INFERENCE__ENGINE=vllm
#   SPECVLM__INFERENCE__USE_SPECULATIVE_DECODING=true

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    SPECVLM_LOG_LEVEL=INFO

# Install system runtime deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data/images /app/results /app/models

# Expose ports
# 8000: FastAPI inference server
# 8001: Prometheus metrics
EXPOSE 8000 8001

# Default command (override with docker-compose or command line)
CMD ["uvicorn", "specvlm.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
