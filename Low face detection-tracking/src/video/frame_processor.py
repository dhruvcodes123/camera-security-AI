"""
Frame processor module for the Hybrid Person Tracking System.
Handles face detection, tracking, and door line crossing detection.
"""

import cv2
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime
import os

from models import SCRFD
from database.hybrid_tracker import HybridPersonTracker
from database.face_log_db import FaceLogDatabase
from ..config import config


class FrameProcessor:
    """
    Processes video frames for face detection, tracking, and re-identification.
    """
    
    def __init__(self, 
                 detector: SCRFD,
                 tracker: HybridPersonTracker,
                 face_db: Optional[FaceLogDatabase] = None,
                 door_line_y: Optional[int] = None,
                 faces_output_dir: Optional[str] = None,
                 max_faces: int = 10):
        """
        Initialize frame processor.
        
        Args:
            detector: SCRFD face detector
            tracker: Hybrid person tracker
            face_db: Face log database
            door_line_y: Door line position from top (None to disable)
            faces_output_dir: Directory to save face images (None to disable)
            max_faces: Maximum number of faces to detect per frame
        """
        self.detector = detector
        self.tracker = tracker
        self.face_db = face_db
        self.door_line_y = door_line_y
        self.faces_output_dir = faces_output_dir
        self.max_faces = max_faces
        
        # Tracking state
        self.colors = {}  # person_id -> color
        self.saved_persons = set()  # person_id already saved
        self.track_positions = {}  # track_id -> previous position
        self.crossing_counts = {"entries": 0, "exits": 0}  # Door line counters
        self.pre_door_detections = {}  # Detections above door line
        
        logging.info(f"Frame processor initialized - Door line enabled: {door_line_y is not None}")
        
    def process_frame(self, frame: np.ndarray, frame_count: int, params: Any) -> np.ndarray:
        """
        Process a video frame for face detection and tracking.
        
        Args:
            frame: Input video frame
            frame_count: Current frame count
            params: Processing parameters
            
        Returns:
            np.ndarray: Processed frame with annotations
        """
        # Detect all faces in the frame (use self.max_faces from performance settings)
        bboxes, kpss = self.detector.detect(frame, self.max_faces)
        
        # Process detections and separate by door line
        pre_door_dets = []  # Faces above door line (not yet tracked)
        active_zone_dets = []  # Faces below door line (can be tracked)
        active_zone_kps = []   # Corresponding keypoints
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, confidence = bbox.astype(np.float32)
            if confidence >= params['confidence_thresh']:
                center_y = (y1 + y2) / 2
                
                if self.door_line_y is not None and center_y < self.door_line_y:
                    # Face is above door line - store for crossing detection
                    pre_door_dets.append({
                        'bbox': [x1, y1, x2, y2, confidence],
                        'center_y': center_y,
                        'kps': kpss[i] if i < len(kpss) else None
                    })
                else:
                    # Face is below door line or door line is disabled - can be actively tracked
                    active_zone_dets.append([x1, y1, x2, y2, confidence])
                    active_zone_kps.append(kpss[i] if i < len(kpss) else None)
        
        # Debug logging
        if frame_count % 50 == 0:  # Log every 50 frames to avoid spam
            logging.info(f"Frame {frame_count}: Total detections: {len(bboxes)}, "
                         f"Active zone: {len(active_zone_dets)}, Pre-door: {len(pre_door_dets)}")
            if len(active_zone_dets) > 0:
                logging.info(f"Active zone detections: {active_zone_dets[:2]}...")  # Show first 2 detections
        
        # Check for door line crossings from pre-door detections
        new_entries = []
        if self.door_line_y is not None:
            for detection in pre_door_dets:
                bbox = detection['bbox']
                center_y = detection['center_y']
                det_id = f"{bbox[0]:.0f}_{bbox[1]:.0f}"  # Simple ID based on position
                
                # Check if this detection was seen before and has now crossed
                if det_id in self.pre_door_detections:
                    prev_y = self.pre_door_detections[det_id]['prev_y']
                    # If detection moved from above to below door line
                    if prev_y < self.door_line_y and center_y >= self.door_line_y:
                        new_entries.append(detection)
                        self.crossing_counts["entries"] += 1
                        logging.info(f"New person ENTERED through door line")
                        del self.pre_door_detections[det_id]  # Remove from pre-door tracking
                else:
                    # Store new pre-door detection
                    self.pre_door_detections[det_id] = {
                        'prev_y': center_y,
                        'bbox': bbox,
                        'kps': detection['kps']
                    }
        
        # Add new entries to active tracking zone
        for entry in new_entries:
            active_zone_dets.append(entry['bbox'])
            active_zone_kps.append(entry['kps'])
        
        # Use Hybrid Tracker for tracking and re-identification
        results = self.tracker.update(frame, active_zone_dets, active_zone_kps, frame_count, self.face_db)
        
        # Process results and draw annotations
        self._process_tracking_results(frame, results, frame_count)
        
        # Draw door line if enabled
        if self.door_line_y is not None:
            self._draw_door_line(frame)
            
        return frame
    
    def _process_tracking_results(self, frame: np.ndarray, results: List[Dict], frame_count: int) -> None:
        """
        Process tracking results and draw annotations.
        
        Args:
            frame: Input video frame
            results: Tracking results from hybrid tracker
            frame_count: Current frame count
        """
        from src.utils.helpers import draw_bbox_info
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            track_id = result['track_id']
            person_id = result['person_id']
            confidence = result['confidence']
            reid_event = result['reid_event']
            similarity_score = result['similarity_score']
            
            # Generate color for this person (consistent across re-appearances)
            if person_id is not None:
                if person_id not in self.colors:
                    self.colors[person_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = self.colors[person_id]
                
                # Create display text
                if reid_event == "NEW":
                    label_text = f"Person_{person_id:03d} (NEW)"
                elif reid_event == "REIDENTIFIED":
                    label_text = f"Person_{person_id:03d} (RE-ID: {similarity_score:.2f})"
                else:
                    label_text = f"Person_{person_id:03d}"
                    
            else:
                # No person ID assigned yet
                color = (128, 128, 128)  # Gray for unidentified
                label_text = f"Track_{track_id}"
            
            # Draw bounding box and label
            draw_bbox_info(frame, [x1, y1, x2, y2], confidence, label_text, color)
            
            # Handle face events and database recording
            self._handle_face_events(result, frame, frame_count)
    
    def _handle_face_events(self, result: Dict, frame: np.ndarray, frame_count: int) -> None:
        """
        Handle face events such as new person or re-identification.
        
        Args:
            result: Tracking result for a single face
            frame: Input video frame
            frame_count: Current frame count
        """
        person_id = result['person_id']
        if person_id is None:
            return  # Skip if no person ID assigned
            
        track_id = result['track_id']
        confidence = result['confidence']
        reid_event = result['reid_event']
        similarity_score = result['similarity_score']
        
        current_time = datetime.now().isoformat()
        
        # Handle NEW and REIDENTIFIED events
        if reid_event == "NEW":
            self.saved_persons.add(person_id)
            
            # Record person entry in database
            if self.face_db:
                entry_id = self.face_db.record_person_entry(person_id)
                # Record event in database with entry time
                self.face_db.record_face_event(
                    face_id=person_id,
                    event_type=reid_event,
                    confidence=confidence,
                    similarity_score=None,
                    frame_number=frame_count,
                    track_id=track_id,
                    entry_time=current_time,
                    exit_time=None
                )
                logging.info(f"New person registered: Person_{person_id:03d}")
                
                # Save face image if enabled
                self._save_face_image(frame, result, "NEW")
                
        elif reid_event == "REIDENTIFIED":
            # Record event in database
            if self.face_db:
                self.face_db.record_face_event(
                    face_id=person_id,
                    event_type=reid_event,
                    confidence=confidence,
                    similarity_score=similarity_score,
                    frame_number=frame_count,
                    track_id=track_id,
                    entry_time=current_time,
                    exit_time=None
                )
                logging.info(f"Person re-identified: Person_{person_id:03d}")
                
                # Save face image if enabled
                self._save_face_image(frame, result, "REIDENTIFIED")
                
        # Periodically save persistent tracks (every 100 frames)
        elif reid_event.startswith("PERSIST") and frame_count % 100 == 0:
            # Save face image if enabled
            self._save_face_image(frame, result, "PERSIST")
    
    def _save_face_image(self, frame: np.ndarray, result: Dict, save_reason: str) -> None:
        """
        Save a cropped face image.
        
        Args:
            frame: Input video frame
            result: Tracking result for a single face
            save_reason: Reason for saving (NEW, REIDENTIFIED, PERSIST)
        """
        if self.faces_output_dir is None:
            return  # Face saving disabled
            
        try:
            # Extract data
            person_id = result['person_id']
            x1, y1, x2, y2 = [int(coord) for coord in result['bbox']]
            confidence = result['confidence']
            
            # Crop face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                filename = f"Person_{person_id:03d}_{save_reason}_{confidence:.2f}.jpg"
                filepath = os.path.join(self.faces_output_dir, filename)
                cv2.imwrite(filepath, face_crop)
        except Exception as e:
            logging.warning(f"Failed to save face crop: {e}")
    
    def _draw_door_line(self, frame: np.ndarray) -> None:
        """
        Draw door line and zone labels.
        
        Args:
            frame: Input video frame
        """
        if self.door_line_y is None:
            return
            
        height, width = frame.shape[:2]
        
        # Draw horizontal door line
        cv2.line(frame, (0, self.door_line_y), (width, self.door_line_y), (0, 255, 255), 3)
        
        # Add door line label
        cv2.putText(frame, "DOOR LINE", (10, self.door_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add zone labels
        cv2.putText(frame, "DETECTION ZONE", (width - 200, self.door_line_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "TRACKING ZONE", (width - 200, self.door_line_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def get_crossing_counts(self) -> Dict[str, int]:
        """
        Get door line crossing counts.
        
        Returns:
            Dict[str, int]: Entry and exit counts
        """
        return self.crossing_counts
        
    def clear_tracking_state(self) -> None:
        """
        Clear tracking state.
        """
        self.colors.clear()
        self.saved_persons.clear()
        self.track_positions.clear()
        self.pre_door_detections.clear()
        self.crossing_counts["entries"] = 0
        self.crossing_counts["exits"] = 0 