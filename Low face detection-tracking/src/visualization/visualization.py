"""
Visualization module for the Hybrid Person Tracking System.
Handles drawing of bounding boxes, statistics, and annotations.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any


class Visualizer:
    """
    Handles visualization of tracking results and statistics.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize visualizer.
        
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def draw_statistics(self, 
                       frame: np.ndarray, 
                       stats: Dict[str, Any], 
                       current_fps: float, 
                       frame_count: int,
                       crossing_counts: Dict[str, int]) -> np.ndarray:
        """
        Draw statistics on the frame.
        
        Args:
            frame: Input video frame
            stats: Statistics from tracker
            current_fps: Current frames per second
            frame_count: Current frame count
            crossing_counts: Door line crossing counts
            
        Returns:
            np.ndarray: Frame with statistics overlay
        """
        stats_y_start = 30
        stats_x = self.frame_width - 300
        
        # Person statistics
        cv2.putText(frame, f"Total Persons: {stats['total_persons']}", 
                   (stats_x, stats_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {stats['active_tracks']}", 
                   (stats_x, stats_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                   (stats_x, stats_y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (stats_x, stats_y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Entry/exit counts
        cv2.putText(frame, f"Entries: {crossing_counts['entries']}", 
                   (stats_x, stats_y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"DB Size: {stats['total_embeddings']}", 
                   (stats_x, stats_y_start + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_bbox(self, 
                 frame: np.ndarray, 
                 bbox: List[float], 
                 color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 2) -> None:
        """
        Draw a bounding box on a frame.
        
        Args:
            frame: Input video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            color: RGB color tuple
            thickness: Line thickness
        """
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    def draw_bbox_info(self, 
                      frame: np.ndarray, 
                      bbox: List[float], 
                      confidence: float, 
                      label_text: str,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw a bounding box with label and confidence.
        
        Args:
            frame: Input video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence
            label_text: Label text to display
            color: RGB color tuple
        """
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate label position (above the bounding box)
        label_y = max(y1 - 10, 20)
        
        # Draw label background
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, label_y - 20), (x1 + text_size[0], label_y), color, -1)
        
        # Draw label text (white)
        cv2.putText(frame, label_text, (x1, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw confidence below the box
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                   
    def draw_detections(self, 
                        frame: np.ndarray,
                        detections: List[Dict],
                        active_zone_detections: List[Dict] = None,
                        pre_door_detections: List[Dict] = None) -> None:
        """
        Draw all detections on the frame.
        
        Args:
            frame: Input video frame
            detections: List of detection dictionaries
            active_zone_detections: List of detections in active zone
            pre_door_detections: List of detections near the door
        """
        # Default empty lists if None
        active_zone_detections = active_zone_detections or []
        pre_door_detections = pre_door_detections or []
        
        # Draw all detections
        for det in detections:
            if 'bbox' in det:
                bbox = det['bbox']
                
                # Choose color based on detection type
                color = (0, 255, 0)  # Default green
                
                # Check if this detection is in active zone (blue)
                if any(d.get('track_id') == det.get('track_id') for d in active_zone_detections):
                    color = (255, 0, 0)  # Blue
                
                # Check if this detection is near door (yellow)
                if any(d.get('track_id') == det.get('track_id') for d in pre_door_detections):
                    color = (0, 255, 255)  # Yellow
                
                # Get label text
                label = det.get('label', f"ID:{det.get('track_id', '?')}")
                confidence = det.get('confidence', 0.0)
                
                # Draw the box with label and confidence
                self.draw_bbox_info(frame, bbox, confidence, label, color) 