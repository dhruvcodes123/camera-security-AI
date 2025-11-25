"""
Command line argument parsing for Hybrid Person Tracking System.
"""

import argparse
from . import config


def parse_args():
    """
    Parse command line arguments for the Hybrid Person Tracking System.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Hybrid Person Tracking System with ByteTrack + ArcFace Re-identification")

    # Model weights
    parser.add_argument("--det-weight", type=str, default=config.DET_WEIGHT, 
                        help=f"Path to detection model (default: {config.DET_WEIGHT})")
    parser.add_argument("--rec-weight", type=str, default=config.REC_WEIGHT, 
                        help=f"Path to ArcFace recognition model (default: {config.REC_WEIGHT})")
    
    # Detection and tracking thresholds
    parser.add_argument("--confidence-thresh", type=float, default=config.CONFIDENCE_THRESH, 
                        help=f"Confidence threshold for face detection (default: {config.CONFIDENCE_THRESH})")
    parser.add_argument("--similarity-thresh", type=float, default=config.SIMILARITY_THRESH, 
                        help=f"Similarity threshold for re-identification (default: {config.SIMILARITY_THRESH})")
    parser.add_argument("--reid-confidence-thresh", type=float, default=config.REID_CONFIDENCE_THRESH, 
                        help=f"Minimum confidence for re-identification (default: {config.REID_CONFIDENCE_THRESH})")
    
    # ByteTrack parameters
    parser.add_argument("--track-thresh", type=float, default=config.TRACK_THRESH, 
                        help=f"Tracking threshold for ByteTracker (default: {config.TRACK_THRESH})")
    parser.add_argument("--match-thresh", type=float, default=config.MATCH_THRESH, 
                        help=f"Matching threshold for ByteTracker (default: {config.MATCH_THRESH})")
    parser.add_argument("--track-buffer", type=int, default=config.TRACK_BUFFER, 
                        help=f"Track buffer frames (default: {config.TRACK_BUFFER})")
    
    # Video source and output
    parser.add_argument("--source", type=str, default=config.VIDEO_SOURCE, 
                        help=f"Video file or webcam source (default: {config.VIDEO_SOURCE})")
    parser.add_argument("--output", type=str, default=config.OUTPUT_VIDEO, 
                        help=f"Output path for annotated video (default: {config.OUTPUT_VIDEO})")
    
    # Face detection parameters
    parser.add_argument("--max-num", type=int, default=config.MAX_NUM_FACES, 
                        help=f"Maximum number of face detections from a frame (default: {config.MAX_NUM_FACES})")
    
    # Display options
    parser.add_argument("--no-display", action="store_true", 
                        help="Disable video display window")
    
    # Face saving
    parser.add_argument("--save-faces", action="store_true", default=config.SAVE_TRACKED_FACES, 
                        help=f"Save cropped face images (default: {config.SAVE_TRACKED_FACES})")
    
    # Door line detection
    parser.add_argument("--door-line", type=int, default=config.DOOR_LINE_POSITION, 
                        help=f"Door line position in pixels from top (default: {config.DOOR_LINE_POSITION})")
    parser.add_argument("--enable-door-line", action="store_true", default=config.ENABLE_DOOR_LINE, 
                        help=f"Enable door line crossing detection (default: {config.ENABLE_DOOR_LINE})")
    
    # Database settings
    parser.add_argument("--db-path", type=str, default=config.DB_PATH, 
                        help=f"Person database path (default: {config.DB_PATH})")
    parser.add_argument("--force-empty-db", action="store_true", default=config.FORCE_EMPTY_DB, 
                        help=f"Start with empty database (default: {config.FORCE_EMPTY_DB})")
    parser.add_argument("--sql-db-path", type=str, default=config.SQL_DB_PATH, 
                        help=f"SQLite database path for face logs (default: {config.SQL_DB_PATH})")
    parser.add_argument("--clear-face-logs", action="store_true", default=False,
                        help="Clear face logs database on startup")
    
    # RTSP stream options
    parser.add_argument("--rtsp-buffer", type=int, default=config.RTSP_BUFFER, 
                        help=f"Buffer size for RTSP streams (1-10, higher means more latency but smoother video) (default: {config.RTSP_BUFFER})")
                        
    # Performance optimization
    parser.add_argument("--performance-mode", choices=["low", "balanced", "high"], default="balanced",
                        help="Performance optimization level: low (best quality), balanced, high (best speed)")

    return parser.parse_args() 