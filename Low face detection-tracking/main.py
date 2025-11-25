#!/usr/bin/env python
"""
Hybrid Person Tracking System

A modular implementation of a person tracking system using:
- ByteTrack for motion-based tracking
- ArcFace for face embeddings and re-identification
- Door line crossing detection
- Face image saving

Usage:
    python main.py --source <video_file> [options]
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import and run the main function from our modular implementation
from src.main import main

if __name__ == "__main__":
    main() 