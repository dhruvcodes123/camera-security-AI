#!/bin/bash
# Cleanup script for the Hybrid Person Tracking System project

echo "Cleaning up project directory..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# Remove log files if any
echo "Removing log files..."
rm -f app.log

# Clean the logs directory but keep it
echo "Cleaning logs directory..."
mkdir -p logs
rm -f logs/*.log

# Clean face_logs database if requested
read -p "Do you want to clean the face_logs database? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Cleaning face_logs database..."
    rm -f database/face_logs.db
fi

# Clean tracked faces directory but keep it
echo "Cleaning tracked_faces directory..."
mkdir -p tracked_faces
rm -f tracked_faces/*.jpg

echo "Cleanup complete!" 