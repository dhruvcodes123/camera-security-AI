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
CONFIDENCE_THRESH = 0.3  # Face detection confidence threshold (lowered for debugging)

# ByteTrack Parameters (Motion Tracking)
TRACK_THRESH = 0.4      # High confidence detection threshold for creating new tracks
MATCH_THRESH = 0.35     # IoU threshold for matching detections to existing tracks (lowered for hybrid)
TRACK_BUFFER = 30       # Number of frames to keep lost tracks before removal
LOW_THRESH = 0.1        # Low confidence detection threshold (auto-calculated as track_thresh - 0.1)

# Re-identification Parameters (ArcFace + FAISS)
SIMILARITY_THRESH = 0.45         # Face embedding similarity threshold (optimized via ROC analysis)
REID_CONFIDENCE_THRESH = 0.75    # Minimum detection confidence for re-identification
REID_FRAME_INTERVAL = 1          # Check every frame for best quality face
EMBEDDING_DIM = 512             # ArcFace embedding dimension (512-d face embeddings)
MAX_REID_DISTANCE = 30          # Max frames between re-identification attempts
MIN_FACE_SIZE = 64             # Minimum face size for quality embeddings
FACE_QUALITY_THRESH = 0.8      # Minimum face detection quality score

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

# Performance Mode Settings
# These settings are adjusted based on the --performance-mode argument
# Low: Best quality, slower processing
# Balanced: Good balance between quality and speed
# High: Best speed, may sacrifice some accuracy
PERFORMANCE_MODES = {
    "low": {
        "reid_frame_interval": 1,     # Check every frame for best quality
        "max_faces_per_frame": 20,    # Process more faces
        "face_quality_thresh": 0.7,   # Higher quality threshold
        "batch_commit_size": 1,       # Commit database changes immediately
    },
    "balanced": {
        "reid_frame_interval": 5,     # Check every 5 frames
        "max_faces_per_frame": 10,    # Process moderate number of faces
        "face_quality_thresh": 0.5,   # Medium quality threshold
        "batch_commit_size": 10,      # Batch database commits
    },
    "high": {
        "reid_frame_interval": 15,    # Check less frequently
        "max_faces_per_frame": 5,     # Process fewer faces
        "face_quality_thresh": 0.3,   # Lower quality threshold
        "batch_commit_size": 20,      # Larger batch commits
    }
}

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

# RTSP Stream Settings
RTSP_BUFFER = 3  # Buffer size for RTSP streams

# Face Log Database
SQL_DB_PATH = "database/face_logs.db"  # SQLite database path for face logs 