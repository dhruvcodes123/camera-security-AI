"""
Hybrid Person Tracking System Configuration
Combines ByteTrack (motion) + ArcFace (embeddings) + Re-identification
"""

import os

# Model Paths
DET_WEIGHT = "weights/det_2.5g.onnx"  # Detection model (good balance of speed/accuracy)
REC_WEIGHT = "weights/w600k_mbf.onnx"  # ArcFace model for embeddings (lightweight)

# Alternative model combinations for different performance requirements:
# FAST: det_500m.onnx + w600k_mbf.onnx (Real-time performance)
# BALANCED: det_2.5g.onnx + w600k_mbf.onnx (Default - good speed/accuracy)
# ACCURATE: det_10g.onnx + w600k_r50.onnx (Best accuracy, slower)

# Face Detection Thresholds
CONFIDENCE_THRESH = 0.6  # Face detection confidence threshold

# ByteTrack Parameters (Motion Tracking)
TRACK_THRESH = 0.4      # High confidence detection threshold for creating new tracks
MATCH_THRESH = 0.35     # IoU threshold for matching detections to existing tracks (lowered for hybrid)
TRACK_BUFFER = 30       # Number of frames to keep lost tracks before removal
LOW_THRESH = 0.1        # Low confidence detection threshold (auto-calculated as track_thresh - 0.1)

# Re-identification Parameters (ArcFace + FAISS)
SIMILARITY_THRESH = 0.4          # Face embedding similarity threshold (0.3-0.6, lower = more permissive)
REID_CONFIDENCE_THRESH = 0.7     # Minimum detection confidence for re-identification
REID_FRAME_INTERVAL = 5          # Only attempt re-ID every N frames (performance optimization)
EMBEDDING_DIM = 512              # ArcFace embedding dimension
MAX_REID_DISTANCE = 50           # Max frames between re-identification attempts

# Database Settings
DB_PATH = "database/person_database"  # Path for saving person database
FORCE_EMPTY_DB = True                 # Always start with empty database
LOAD_EXISTING_DB = False              # Load existing database on startup

# Filtering Parameters
ASPECT_RATIO_THRESH = 2.0   # Maximum aspect ratio (width/height) for valid detections
MIN_BOX_AREA = 100          # Minimum bounding box area to be considered valid

# Video Processing Settings
MAX_NUM_FACES = 10         # Maximum number of faces to detect per frame
VIDEO_SOURCE = "assets/demo.mp4"  # Default video source (or 0 for webcam)

# Display Settings
AUTO_DISPLAY_MAX_WIDTH = 1200
AUTO_DISPLAY_MAX_HEIGHT = 800
SHOW_STATISTICS = True     # Show person count, FPS, and frame info on video

# Output Settings
OUTPUT_VIDEO = "output.mp4"
SAVE_TRACKED_FACES = True   # Always save cropped face images by default

# Door Line Settings  
ENABLE_DOOR_LINE = True     # Always enable door line crossing detection by default
DOOR_LINE_POSITION = 200    # Door line position in pixels from top

# Logging
LOG_LEVEL = "INFO"
LOG_TO_FILE = True

# Person ID Settings
PERSON_ID_PREFIX = "Person"   # Prefix for generated person IDs (Person_001, Person_002, etc.)
ID_PADDING = 3

# Performance Optimization
HYBRID_MODE = True                    # Enable hybrid tracking (ByteTrack + Re-ID)
REALTIME_OPTIMIZATION = True          # Enable real-time performance optimizations

# Performance Settings
FRAME_RATE = 30            # Expected frame rate for optimal tracking performance

# ByteTrack Advanced Settings
# These control the internal behavior of the tracking algorithm
KALMAN_PROCESS_NOISE = 0.1      # Process noise covariance for Kalman filter
KALMAN_MEASUREMENT_NOISE = 0.1  # Measurement noise covariance for Kalman filter

# Tracking Quality Settings
MIN_HITS_TO_TRACK = 1      # Minimum consecutive detections before showing track
MAX_AGE_WITHOUT_UPDATE = 1  # Maximum frames without update before hiding track

# Association Settings
IOU_THRESHOLD_HIGH = 0.7   # IoU threshold for high confidence association
IOU_THRESHOLD_LOW = 0.5    # IoU threshold for low confidence association

# Debug Settings
ENABLE_DEBUG_LOGGING = False  # Enable detailed debug logs
SAVE_TRACKING_STATS = False   # Save tracking statistics to file

# Comparison with Face Re-ID System
"""
ByteTrack vs Face Re-ID Comparison:

PERFORMANCE:
- ByteTrack: ~50-80 FPS (much faster)
- Face Re-ID: ~20-30 FPS (slower due to embedding computation)

ROBUSTNESS:
- ByteTrack: Works with partial occlusions, profile views, poor lighting
- Face Re-ID: Requires clear frontal face views, good lighting

ACCURACY:
- ByteTrack: May assign new IDs after long occlusions
- Face Re-ID: Can re-identify after hours/days if face quality is good

COMPUTATIONAL REQUIREMENTS:
- ByteTrack: Only detection model (~3-16MB), minimal CPU/GPU usage
- Face Re-ID: Detection + recognition models (~13-182MB), higher GPU usage

MEMORY USAGE:
- ByteTrack: ~1KB per active track
- Face Re-ID: ~2KB per registered person + FAISS index

USE CASES:
ByteTrack is better for:
- Real-time applications
- Crowded scenes
- Poor lighting conditions
- Edge/mobile devices
- When people don't leave and re-enter

Face Re-ID is better for:
- Long-term identification
- Security applications
- When people frequently enter/exit
- When high accuracy is more important than speed
""" 