#!/bin/bash

# Setup script for Hybrid Person Tracking System on Jetson Orin NX
# This script installs all necessary dependencies and sets up the environment

echo "Setting up Hybrid Person Tracking System for Jetson Orin NX..."

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Error: This script is intended to run on NVIDIA Jetson devices only."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv jetson_env
source jetson_env/bin/activate

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libopenblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Install CUDA-enabled packages
echo "Installing CUDA-enabled packages..."
pip install pycuda

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# Create symbolic links to system OpenCV (which has CUDA support)
echo "Setting up OpenCV with CUDA support..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
sudo ln -sf /usr/lib/python3/dist-packages/cv2 $SITE_PACKAGES/cv2

# Download model weights if needed
echo "Checking for model weights..."
if [ ! -d "weights" ]; then
    echo "Downloading model weights..."
    bash download.sh
fi

# Set environment variables for optimal performance
echo "Setting environment variables for optimal performance..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> jetson_env/bin/activate
echo 'export OMP_NUM_THREADS=4' >> jetson_env/bin/activate
echo 'export OPENBLAS_NUM_THREADS=4' >> jetson_env/bin/activate

# Make the script executable
chmod +x main.py

echo "Setup complete! Activate the environment with: source jetson_env/bin/activate"
echo "Run the application with: python main.py --source <video_file> --performance-mode high" 