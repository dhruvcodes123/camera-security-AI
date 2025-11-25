"""
Core application for the Hybrid Person Tracking System.
"""

import os
import cv2
import time
import shutil
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import traceback

from models import get_detector, get_recognizer
from database.hybrid_tracker import HybridPersonTracker
from database.face_log_db import FaceLogDatabase
from ..video import VideoProcessor, FrameProcessor
from ..visualization import Visualizer
from ..config import config


class HybridPersonTrackingApp:
    """
    Main application for hybrid person tracking system.
    """
    
    def __init__(self, args):
        """
        Initialize hybrid person tracking application.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.detector = None
        self.tracker = None
        self.face_db = None
        self.video_processor = None
        self.frame_processor = None
        self.visualizer = None
        self.frame_count = 0
        
        # Directory for saving face images
        self.faces_output_dir = "tracked_faces" if args.save_faces else None
        
        # Door line position
        self.door_line_y = args.door_line if args.enable_door_line else None
        
    def initialize(self) -> bool:
        """
        Initialize models, trackers, and databases.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize detection model (automatically uses TensorRT on Jetson)
            self.detector = get_detector(
                self.args.det_weight, 
                input_size=(640, 640), 
                conf_thres=self.args.confidence_thresh
            )
            logging.info("Detection model loaded successfully")
            
            # Initialize hybrid tracker with performance settings
            # Apply performance mode settings
            performance_settings = config.PERFORMANCE_MODES[self.args.performance_mode]
            logging.info(f"Using performance mode: {self.args.performance_mode}")
            
            self.tracker = HybridPersonTracker(
                arcface_model_path=self.args.rec_weight,
                track_thresh=self.args.track_thresh,
                match_thresh=self.args.match_thresh,
                track_buffer=self.args.track_buffer,
                similarity_thresh=self.args.similarity_thresh,
                reid_confidence_thresh=self.args.reid_confidence_thresh,
                face_quality_thresh=performance_settings["face_quality_thresh"]
            )
            
            # Set reid_frame_interval based on performance mode
            self.tracker.reid_frame_interval = performance_settings["reid_frame_interval"]
            logging.info(f"Hybrid tracker initialized with similarity_thresh: {self.args.similarity_thresh}")
            logging.info(f"Re-identification interval: {self.tracker.reid_frame_interval} frames")
            
            # Database management
            if self.args.force_empty_db:
                self.tracker.clear_database()
                logging.info("Started with empty person database")
            else:
                if self.tracker.load_database(self.args.db_path):
                    logging.info("Loaded existing person database")
                else:
                    logging.info("Starting with empty database (no existing database found)")
            
            # Initialize SQLite database for face logs
            try:
                # Check if we need to clear the face logs database
                if hasattr(self.args, 'clear_face_logs') and self.args.clear_face_logs and os.path.exists(self.args.sql_db_path):
                    logging.info(f"Clearing face logs database at {self.args.sql_db_path}")
                    os.remove(self.args.sql_db_path)
                    logging.info("Face logs database cleared")
                
                self.face_db = FaceLogDatabase(self.args.sql_db_path)
                # Set batch commit size based on performance mode
                self.face_db.max_pending_operations = performance_settings["batch_commit_size"]
                logging.info(f"Face log database initialized at {self.args.sql_db_path}")
                logging.info(f"Database batch commit size: {self.face_db.max_pending_operations}")
                
                # Start a new session
                self.session_id = self.face_db.start_new_session(self.args.source)
                logging.info(f"Started new tracking session with ID: {self.session_id}")
            except Exception as e:
                logging.error(f"Failed to initialize face log database: {e}")
                logging.warning("Continuing without database logging")
                self.face_db = None
            
            # Initialize face saving directory if enabled
            if self.faces_output_dir:
                self._setup_faces_directory()
            
            # Initialize video processor
            self.video_processor = VideoProcessor(
                source=self.args.source,
                output_path=self.args.output,
                rtsp_buffer=self.args.rtsp_buffer,
                display=not self.args.no_display
            )
            
            # Open video source
            if not self.video_processor.open():
                logging.error("Failed to open video source")
                return False
                
            # Initialize visualizer
            frame_width, frame_height = self.video_processor.get_dimensions()
            self.visualizer = Visualizer(frame_width, frame_height)
            
            # Initialize frame processor with performance settings
            self.frame_processor = FrameProcessor(
                detector=self.detector,
                tracker=self.tracker,
                face_db=self.face_db,
                door_line_y=self.door_line_y,
                faces_output_dir=self.faces_output_dir,
                max_faces=performance_settings["max_faces_per_frame"]  # Apply performance setting
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def _setup_faces_directory(self) -> None:
        """
        Set up directory for saving face images.
        """
        # Clear the directory if it exists
        if os.path.exists(self.faces_output_dir):
            shutil.rmtree(self.faces_output_dir)
            logging.info(f"Cleared existing '{self.faces_output_dir}/' directory")
        
        # Create fresh directory
        os.makedirs(self.faces_output_dir, exist_ok=True)
        logging.info(f"Created fresh '{self.faces_output_dir}/' directory for saving faces")
    
    def _draw_door_line(self, frame: np.ndarray) -> None:
        """
        Draw door line on the frame.
        
        Args:
            frame: Video frame to draw on
        """
        if self.door_line_y is None:
            return
            
        height, width = frame.shape[:2]
        # Draw horizontal door line
        cv2.line(frame, (0, self.door_line_y), (width, self.door_line_y), (0, 255, 255), 3)
        # Add door line label
        cv2.putText(frame, "DOOR LINE", (10, self.door_line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _safe_method_call(self, obj, method_name, default_func=None, *args, **kwargs):
        """
        Safely call a method on an object if it exists.
        
        Args:
            obj: Object to call method on
            method_name: Name of the method to call
            default_func: Default function to call if method doesn't exist
            *args, **kwargs: Arguments to pass to the method
            
        Returns:
            Result of the method call or None if method doesn't exist
        """
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            return method(*args, **kwargs)
        elif default_func:
            return default_func(*args, **kwargs)
        return None
    
    def run(self):
        """Run the application"""
        try:
            logging.info("Starting hybrid person tracking...")
            self.frame_count = 0
            last_fps_update = time.time()
            fps = 0
            
            # Flag to keep the application running
            keep_running = True
            
            while keep_running:
                try:
                    # Read frame - handle different VideoProcessor implementations
                    if hasattr(self.video_processor, "read_frame"):
                        ret, frame = self.video_processor.read_frame()
                    else:
                        ret, frame = self.video_processor.cap.read()
                    
                    # Handle RTSP stream errors or end of stream
                    if not ret:
                        logging.info("Video source error or end of stream, attempting to reconnect...")
                        # Close and reopen the video source
                        if hasattr(self.video_processor, "release_cap"):
                            self.video_processor.release_cap()
                        elif hasattr(self.video_processor, "cap") and self.video_processor.cap:
                            self.video_processor.cap.release()
                            
                        time.sleep(2)  # Wait before reconnecting
                        
                        # Reopen the video source
                        if hasattr(self.video_processor, "reopen_cap"):
                            self.video_processor.reopen_cap()
                        elif hasattr(self.video_processor, "open"):
                            self.video_processor.open()
                        else:
                            self.video_processor.cap = cv2.VideoCapture(self.video_processor.source)
                        
                        # Check if reconnection was successful
                        is_opened = False
                        if hasattr(self.video_processor, "is_opened"):
                            is_opened = self.video_processor.is_opened()
                        elif hasattr(self.video_processor, "cap"):
                            is_opened = self.video_processor.cap.isOpened()
                            
                        if not is_opened:
                            logging.warning("Failed to reconnect to video source, retrying...")
                            continue
                        
                        logging.info("Successfully reconnected to video source")
                        continue
                    
                    # Process frame with error handling
                    try:
                        # Pass args as a dict to avoid attribute access issues
                        process_result = self.frame_processor.process_frame(
                            frame, 
                            frame_count=self.frame_count, 
                            params={"confidence_thresh": self.args.confidence_thresh}
                        )
                        
                        # Handle different return types from process_frame
                        if isinstance(process_result, tuple) and len(process_result) == 3:
                            # If process_frame returns (detections, active_zone_detections, pre_door_detections)
                            detections, active_zone_detections, pre_door_detections = process_result
                            processed_frame = frame  # Use original frame
                        else:
                            # If process_frame returns just the processed frame
                            processed_frame = process_result
                            detections = []
                            active_zone_detections = []
                            pre_door_detections = []
                            
                        # Log frame info periodically
                        if self.frame_count % 50 == 0:
                            logging.info(f"Frame {self.frame_count}: Total detections: {len(detections)}, "
                                        f"Active zone: {len(active_zone_detections)}, "
                                        f"Pre-door: {len(pre_door_detections)}")
                    except Exception as process_error:
                        logging.error(f"Error processing frame: {process_error}")
                        processed_frame = frame  # Use original frame on error
                        detections = []
                        active_zone_detections = []
                        pre_door_detections = []
                    
                    # Get tracker statistics
                    stats = self.tracker.get_person_stats()
                    
                    # Draw door line directly instead of relying on visualizer
                    self._draw_door_line(processed_frame)
                    
                    # Draw detections and tracks - safely call methods
                    try:
                        if hasattr(self.visualizer, "draw_detections"):
                            self.visualizer.draw_detections(processed_frame, detections, active_zone_detections, pre_door_detections)
                    except Exception as vis_error:
                        logging.warning(f"Error drawing detections: {vis_error}")
                    
                    # Draw statistics - safely call methods
                    try:
                        crossing_counts = {"entries": 0, "exits": 0}
                        if hasattr(self.frame_processor, "get_crossing_counts"):
                            crossing_counts = self.frame_processor.get_crossing_counts()
                            
                        if hasattr(self.visualizer, "draw_statistics"):
                            # Try different parameter combinations based on what might be expected
                            try:
                                self.visualizer.draw_statistics(
                                    processed_frame, stats, fps, self.frame_count, crossing_counts
                                )
                            except TypeError:
                                try:
                                    self.visualizer.draw_statistics(
                                        processed_frame, stats, self.frame_count, fps, 
                                        door_line_y=self.door_line_y
                                    )
                                except TypeError:
                                    # Fallback to minimal parameters
                                    self.visualizer.draw_statistics(processed_frame, stats)
                    except Exception as stats_error:
                        logging.warning(f"Error drawing statistics: {stats_error}")
                        # Draw basic stats directly on the frame as fallback
                        cv2.putText(processed_frame, f"FPS: {fps:.1f} | Persons: {stats.get('total_persons', 0)}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save output video if specified
                    try:
                        if hasattr(self.video_processor, "is_writing") and self.video_processor.is_writing():
                            if hasattr(self.video_processor, "write_frame"):
                                self.video_processor.write_frame(processed_frame)
                            elif hasattr(self.video_processor, "write"):
                                self.video_processor.write(processed_frame)
                        else:
                            # Use write_frame method if it exists
                            if hasattr(self.video_processor, "write_frame"):
                                self.video_processor.write_frame(processed_frame)
                    except Exception as write_error:
                        logging.warning(f"Error writing output frame: {write_error}")
                    
                    # Display the frame using the video processor's display method
                    key = -1
                    if hasattr(self.video_processor, "display_frame"):
                        key = self.video_processor.display_frame(processed_frame, "Hybrid Person Tracking")
                    else:
                        # Fallback to direct display
                        cv2.imshow("Hybrid Person Tracking", processed_frame)
                        key = cv2.waitKey(1) & 0xFF
                    
                    # Check for key press to exit
                    if key == 27 or key == ord('q'):  # ESC or 'q' key
                        keep_running = False
                        break
                    
                    self.frame_count += 1
                    
                    # Update FPS every 100 frames
                    if self.frame_count % 100 == 0:
                        current_time = time.time()
                        fps = 100 / (current_time - last_fps_update) if (current_time - last_fps_update) > 0 else 0
                        last_fps_update = current_time
                        logging.info(f"Processed {self.frame_count} frames, Current FPS: {fps:.2f}, "
                                    f"Total persons: {stats['total_persons']}, "
                                    f"Active tracks: {stats['active_tracks']}")
                        
                        # Force commit database changes periodically
                        if self.face_db and hasattr(self.face_db, "force_commit"):
                            self.face_db.force_commit()
                
                except Exception as frame_error:
                    logging.error(f"Error processing frame {self.frame_count}: {frame_error}")
                    logging.error(traceback.format_exc())
                    time.sleep(0.1)  # Avoid tight loop on repeated errors
            
            # Finish processing
            self._finish_processing(self.frame_count)
                
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            self._cleanup()
    
    def _finish_processing(self, frame_count: int) -> None:
        """
        Complete processing and save final statistics.
        
        Args:
            frame_count: Total frames processed
        """
        logging.info(f"Processing completed. Total frames: {frame_count}")
        
        try:
            # Get final statistics
            final_stats = self.tracker.get_person_stats()
            
            # Get crossing counts if available
            crossing_counts = {"entries": 0, "exits": 0}
            if hasattr(self.frame_processor, "get_crossing_counts"):
                crossing_counts = self.frame_processor.get_crossing_counts()
            
            logging.info(f"Final statistics - Total persons: {final_stats['total_persons']}, "
                        f"Total embeddings: {final_stats['total_embeddings']}")
            
            # Save database for future use
            if not self.args.force_empty_db:
                self.tracker.save_database(self.args.db_path)
                logging.info("Person database saved for future sessions")
            
            # Log final crossing counts
            logging.info(f"Final crossing counts - Entries: {crossing_counts['entries']}, Exits: {crossing_counts['exits']}")
            logging.info(f"Net people count: {crossing_counts['entries'] - crossing_counts['exits']}")
            
            # Save final statistics to database
            if self.face_db and hasattr(self.face_db, "end_session"):
                try:
                    self.face_db.end_session(self.session_id, frame_count, final_stats['total_persons'])
                    logging.info("Face log database updated with final statistics")
                except Exception as db_error:
                    logging.error(f"Failed to update database with final statistics: {db_error}")
        except Exception as e:
            logging.error(f"Error during finish processing: {e}")
    
    def _cleanup(self) -> None:
        """
        Clean up resources.
        """
        # Release video resources
        if self.video_processor:
            if hasattr(self.video_processor, "release"):
                self.video_processor.release()
            elif hasattr(self.video_processor, "cap") and self.video_processor.cap:
                self.video_processor.cap.release()
        
        # Close database connections
        if self.face_db and hasattr(self.face_db, "close"):
            self.face_db.close()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logging.info("Hybrid Person Tracking session completed")
