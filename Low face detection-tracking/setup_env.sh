#!/bin/bash

echo "Setting up virtual environment for Face Recognition System"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install screeninfo for display functionality
pip install screeninfo>=0.8.0

echo "Setup complete. Activate the virtual environment with: source .venv/bin/activate" 