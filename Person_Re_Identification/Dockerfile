# ==============================================================================
# PERSON RE-IDENTIFICATION SYSTEM DOCKERFILE
# ==============================================================================
# Multi-stage build for optimized image size and security

# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# DEVELOPMENT STAGE (Optional - for development)
# ==============================================================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest \
    pytest-cov

# Create non-root user for development
RUN useradd -m -s /bin/bash developer && \
    chown -R developer:developer /app
USER developer

# ==============================================================================
# PRODUCTION STAGE
# ==============================================================================
FROM base as production

# Create necessary directories
RUN mkdir -p /app/data/input_videos \
    /app/data/processed_output/tracked_persons \
    /app/data/processed_output/detected_faces \
    /app/data/processed_output/gait_cycles \
    /app/db/vector_db \
    /app/models/gait \
    /app/RobustVideoMatting/pretrained

# Copy application code
COPY . /app/

# Download required models (if not present)
RUN python -c "
import torch
from ultralytics import YOLO
import os

# Download YOLOv8 models if they don't exist
models = ['yolov8n.pt', 'yolov8m-pose.pt']
for model in models:
    if not os.path.exists(model):
        print(f'Downloading {model}...')
        YOLO(model)
    else:
        print(f'{model} already exists')
"

# Create non-root user for production
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables for production
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "scripts/process_feed_optimized.py", "--help"]

# ==============================================================================
# RUNTIME CONFIGURATION
# ==============================================================================

# Expose ports if needed for web interface
# EXPOSE 8000

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/db", "/app/models"]

# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================
# 
# Build the image:
# docker build -t person-reid-system .
#
# Run with GPU support:
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/db:/app/db person-reid-system
#
# Run specific script:
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/db:/app/db person-reid-system \
#   python scripts/process_feed_optimized.py --video_path data/input_videos/video.mp4
#
# Development mode:
# docker build --target development -t person-reid-dev .
# docker run --gpus all -it -v $(pwd):/app person-reid-dev bash 