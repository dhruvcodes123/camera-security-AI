# Hybrid Person Tracking System

A modular implementation of a person tracking system that combines multiple tracking techniques:

- ByteTrack for motion-based tracking
- ArcFace for face embeddings
- Re-identification for long-term identity consistency
- Door line crossing detection
- Smart face image saving

## Features

- **Motion-based Tracking**: Uses ByteTrack algorithm with Kalman filtering for robust tracking even during occlusions
- **Appearance-based Re-identification**: Uses ArcFace face embeddings and FAISS for fast similarity search
- **Door Line Detection**: Only tracks people who enter from the top of the frame
- **Smart Face Saving**: Saves face images at key events (NEW, REIDENTIFIED, PERSIST)
- **Database Logging**: Records person entries, tracking events, and statistics
- **Real-time Performance**: Optimized for real-time processing on modern hardware

## Code Structure

The project has been refactored into a modular structure for better organization and debugging:

```
.
├── main.py                  # Main entry point
├── src/                     # Source directory
│   ├── config/              # Configuration and argument parsing
│   │   ├── config.py        # Configuration parameters
│   │   └── args.py          # Command line argument parsing
│   ├── core/                # Core application logic
│   │   └── app.py           # Main application class
│   ├── video/               # Video processing
│   │   ├── video_processor.py  # Video input/output handling
│   │   └── frame_processor.py  # Face detection and tracking
│   ├── visualization/       # Visualization utilities
│   │   └── visualization.py # Bounding box and statistics drawing
│   └── utils/               # Utility functions
│       ├── helpers.py       # Face alignment and drawing helpers
│       ├── face_quality.py  # Face quality assessment
│       └── logging.py       # Logging configuration
├── database/                # Database components
│   ├── hybrid_tracker.py    # Hybrid tracking implementation
│   ├── bytetrack.py         # ByteTrack algorithm implementation
│   └── face_log_db.py       # SQLite face event logging
├── models/                  # Deep learning models
│   ├── arcface.py           # ArcFace face recognition model
│   └── scrfd.py             # SCRFD face detection model
└── weights/                 # Model weights (ONNX format)
    ├── det_2.5g.onnx        # SCRFD detection model
    └── w600k_mbf.onnx       # ArcFace recognition model
```

## Technologies Used

- **ByteTrack**: Multi-object tracking using Kalman filtering and IoU matching
- **ArcFace**: Face recognition using additive angular margin loss
- **FAISS**: Fast similarity search for face embeddings
- **SCRFD**: Sample and Computation Redistribution for Face Detection
- **OpenCV**: Computer vision operations and visualization
- **ONNX**: Open Neural Network Exchange format for ML models

## Usage

### Basic Usage

```bash
python main.py --source <video_file>
```

### Advanced Options

```bash
# Use a webcam as input
python main.py --source 0

# Adjust confidence thresholds
python main.py --source <video_file> --confidence-thresh 0.7 --similarity-thresh 0.6

# Adjust door line position (in pixels from top)
python main.py --source <video_file> --door-line 300

# Disable face saving or door line
python main.py --source <video_file> --no-save-faces --no-enable-door-line

# Process video without displaying (headless mode)
python main.py --source <video_file> --no-display
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Video file path or camera index | "assets/demo.mp4" |
| `--output` | Output video path | "output.mp4" |
| `--det-weight` | Detection model path | "./weights/det_2.5g.onnx" |
| `--rec-weight` | Recognition model path | "./weights/w600k_mbf.onnx" |
| `--confidence-thresh` | Detection confidence threshold | 0.3 |
| `--similarity-thresh` | Embedding similarity threshold | 0.45 |
| `--door-line` | Door line position from top | 200 |
| `--enable-door-line` | Enable door line crossing | True |
| `--save-faces` | Save face images | True |
| `--no-display` | Disable video display | False |

## Keyboard Controls

When the video display is active, the following keyboard controls are available:

- **Q**: Quit the application
- **C**: Clear all tracking data and start fresh

## Database Structure

The system maintains two databases:

1. **Vector Database** (FAISS): Stores face embeddings for re-identification
2. **Event Database** (SQLite): Records tracking events, person entries/exits, and statistics

## Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy
- FAISS
- ONNX Runtime
- SQLite3

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv face_reid_env
   source face_reid_env/bin/activate   # Linux/Mac
   face_reid_env\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download model weights (if not already present):
```bash
   bash download.sh
```

## Performance Considerations

- **GPU**: For optimal performance, a CUDA-capable GPU is recommended
- **Video Resolution**: Processing time scales with resolution; consider resizing large videos
- **Detection Frequency**: Re-identification is performed every N frames to optimize performance
- **Door Line**: Using the door line feature can reduce processing load by only tracking relevant individuals

## Troubleshooting

- **Low detection confidence**: Try lowering `--confidence-thresh` to 0.3-0.5
- **False re-identifications**: Adjust `--similarity-thresh` (higher = stricter matching)
- **Poor performance**: Use lightweight models (det_500m.onnx + w600k_mbf.onnx)
- **Memory issues**: Reduce `--max-num` to limit maximum detections per frame 