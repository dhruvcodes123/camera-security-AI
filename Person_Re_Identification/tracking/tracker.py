import cv2
import torch
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker, get_tracker_config
import sys
import threading
import queue
import yaml
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from face_embedding.embedder import FaceEmbedder
from db.database import VectorDatabase
from db.person_tracking_db import PersonTrackingSQLiteDB
from gpu_memory_manager import GPUMemoryManager
from performance_monitor import update_fps

# Add project root to sys.path to allow for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.reid_system import UnifiedReIdentificationSystem

class PersonTracker:
    """
    A class to track persons in a video stream using YOLOv8 and a chosen tracker from boxmot,
    with real-time re-identification to assign persistent IDs.
    Re-identification only triggers when a person is 100% inside the defined zone.
    """

    def __init__(self,
                 output_dir="data/processed_output/tracked_persons",
                 yolo_model='yolov8n.pt',
                 tracker_type='bytetrack',
                 face_det_thresh=0.5):
        """
        Initializes the PersonTracker.

        Args:
            output_dir (str): Directory to save cropped images of tracked persons.
            yolo_model (str): The YOLO model to use for detection.
            tracker_type (str): The type of tracker to use.
            face_det_thresh (float): The confidence threshold for face detection.
        """
        self.pytorch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tracker_device = '0' if torch.cuda.is_available() else 'cpu'

        # print(f"Using PyTorch device: {self.pytorch_device}")
        # print(f"Using Tracker device: {self.tracker_device}")

        # --- GPU MEMORY MANAGEMENT ---
        self.gpu_memory_manager = GPUMemoryManager()
        self.batch_processing_enabled = True
        self.max_concurrent_persons = 6  # Limit concurrent processing
        
        # --- PERSON TRACKING SQLITE DATABASE ---
        self.person_tracking_db = PersonTrackingSQLiteDB()
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create the requested output directories
        self.detected_faces_dir = "data/processed_output/detected_faces"
        self.tracked_persons_dir = "data/processed_output/tracked_persons"
        os.makedirs(self.detected_faces_dir, exist_ok=True)
        os.makedirs(self.tracked_persons_dir, exist_ok=True)
        
        # Gait cycle directory for storing gait cycle frames
        self.gait_cycle_output_dir = "data/processed_output/gait_cycles"
        os.makedirs(self.gait_cycle_output_dir, exist_ok=True)

        self.face_output_dir = "data/processed_output/detected_faces"
        os.makedirs(self.face_output_dir, exist_ok=True)

        # Handle PyTorch 2.6+ compatibility for YOLO model loading
        try:
            self.detector = YOLO(yolo_model)
        except Exception as e:
            if "weights_only" in str(e):
                # print("⚠️ PyTorch 2.6+ compatibility issue detected. Attempting to fix...")
                # Use a simpler approach - download the model again
                if os.path.exists(yolo_model):
                    os.remove(yolo_model)
                    # print(f"Removed corrupted model file: {yolo_model}")
                self.detector = YOLO(yolo_model)
            else:
                raise e
        self.detector.to(self.pytorch_device)

        tracker_config_path = get_tracker_config(tracker_type)
        with open(tracker_config_path, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        # Flatten the parameters and override the track_buffer
        tracker_params = {param: details['default'] for param, details in yaml_config.items()}
        tracker_params['track_buffer'] = 120

        self.tracker = create_tracker(
            tracker_type,
            tracker_config=tracker_config_path, # Still pass the original path
            reid_weights=None,
            device=self.tracker_device,
            half=torch.cuda.is_available(),
            evolve_param_dict=tracker_params
        )

        self.frame_count = 0

        # --- Asynchronous Re-ID Integration ---
        # print("Initializing Re-Identification system for real-time matching...")
        # Initialize the Re-ID system
        self.reid_system = UnifiedReIdentificationSystem(
            face_det_thresh=0.5  # Lowered from 0.5 to be more sensitive
        )
        self.track_id_details = {}
        self.tracklet_buffer = {}

        # --- Asynchronous Re-ID using parallel processing ---
        self.reid_queue = queue.Queue()
        self.reid_lock = threading.Lock()
        self.reid_worker_thread = threading.Thread(target=self._reid_worker, daemon=True)
        self.reid_worker_thread.start()
        self.reid_executor = ThreadPoolExecutor(max_workers=2)  # Parallel processing
        # print("Re-ID worker with parallel processing started.")

        # --- Zone-based Re-ID Logic ---
        # Define rectangular zone boundaries for 100% inside detection
        # These coordinates define the zone where re-identification will be triggered
        self.ZONE_LEFT = 50     # X minimum (expanded from 100)
        self.ZONE_RIGHT = 2500  # X maximum (broadened to 1800)
        self.ZONE_TOP = 100     # Y minimum (starts at y = 300)
        self.ZONE_BOTTOM = 2000 # Y maximum (extended to 2157)
        
        self.min_frames_for_fast_reid = 0  # Minimal frames for face-only check
        self.min_frames_for_full_reid = 40  # Minimal frames for full re-identification
        self.reid_triggered_for_track = set()
        self.fast_reid_triggered_for_track = set()
        self.zone_track_ids = set()  # Tracks that are 100% inside the zone
        self.person_save_count = {}
        
        # Track persons based on zone status
        self.visibility_counter = {}
        self.visibility_confirm_frames = 3  # Frames to confirm visibility in zone
        self.stability_counter = {}  # Track stability in zone
        self.stability_required_frames = 5  # Frames needed to confirm stability

        # Grace period for lost tracks
        self.lost_track_frames = {}
        self.grace_period_frames = 150
        # --- End Logic ---

    def _is_bbox_100_percent_in_zone(self, x1, y1, x2, y2):
        """
        Check if a bounding box is 100% inside the defined rectangular zone.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            bool: True if the entire bounding box is inside the zone
        """
        # Check if all corners of the bounding box are within zone boundaries
        return (x1 >= self.ZONE_LEFT and 
                x2 <= self.ZONE_RIGHT and 
                y1 >= self.ZONE_TOP and 
                y2 <= self.ZONE_BOTTOM)

    def _draw_detection_zone(self, frame):
        """
        Draw the rectangular zone for visualization.
        
        Args:
            frame: The frame to draw on
        """
        # Draw the rectangular zone in blue
        cv2.rectangle(frame, (self.ZONE_LEFT, self.ZONE_TOP), 
                     (self.ZONE_RIGHT, self.ZONE_BOTTOM), (255, 0, 0), 2)
        cv2.putText(frame, "Re-ID Zone (100% Inside Required)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    def _reid_worker(self):
        """Processes Re-ID requests from a queue in the background."""
        while True:
            try:
                track_id, image_sequence, capture_crop, reid_type = self.reid_queue.get(timeout=1.0)
                if track_id is None: # Sentinel for stopping
                    # print("Re-ID worker received stop signal.")
                    break

                # print(f"Re-ID worker processing track {track_id} with type '{reid_type}'...")

                # Updated to handle new return signature with both face and gait scores
                if reid_type == 'fast':
                    persistent_id, match_type, face_score, gait_score, returned_frames, fusion_similarity = self.reid_system.reidentify_track_fast(image_sequence)
                else: # 'full'
                    # ✅ NEW: Use unified fused embedding approach
                    persistent_id, match_type, face_score, gait_score, returned_frames, fusion_similarity = self.reid_system.reidentify_track_with_fused(image_sequence, fused_embedding=None)

                # ✅ NEW: Use the similarity score from re-identification process
                if fusion_similarity is None:
                    fusion_similarity = 0.0

                if persistent_id is not None:
                    with self.reid_lock:
                        # ✅ NEW: Get unified fused embedding from re-identification system
                        try:
                            # Get the unified fused embedding directly from re-id system
                            fused_result = self.reid_system._create_unified_fused_embedding(image_sequence)
                            unified_fused_embedding, face_embedding, gait_embedding, shoe_embedding, view_type, gait_cycle_frames, face_crop, shoe_crop = fused_result
                            
                            if unified_fused_embedding is not None:
                                # print(f"✅ Track {track_id}: Unified 3840D fused embedding created - shape: {unified_fused_embedding.shape}")
                                pass
                            else:
                                # print(f"❌ Track {track_id}: Failed to create unified fused embedding")
                                pass
                                
                        except Exception as e:
                            # print(f"❌ Track {track_id}: Unified fused embedding error - {e}")
                            unified_fused_embedding = None
                            face_embedding = None
                            gait_embedding = None
                            shoe_embedding = None
                            face_crop = None
                            gait_cycle_frames = None
                        
                        # ✅ VALIDATION: Log all embedding details
                                            # print(f"🔍 Track {track_id} - RE-ID SUCCESS VALIDATION:")
                    # print(f"   - Person ID: {persistent_id}")
                    # print(f"   - Match Type: {match_type}")
                    # print(f"   - Face Score: {face_score:.3f}")
                    # print(f"   - Gait Score: {gait_score:.3f}")
                    # print(f"   - Face Embedding: {face_embedding.shape if face_embedding is not None else 'None (512D zero)'}")
                    # print(f"   - Gait Embedding: {gait_embedding.shape if gait_embedding is not None else 'None (1024D zero)'}")
                    # print(f"   - Shoe Embedding: {shoe_embedding.shape if shoe_embedding is not None else 'None (2048D zero)'}")
                    # print(f"   - Unified Fused Embedding: {unified_fused_embedding.shape if unified_fused_embedding is not None else 'None (3840D)'}")
                    # print(f"   - All embeddings stored in track_id_details[{track_id}]")
                        
                        details_to_store = {
                            "person_id": persistent_id,
                            "match_type": match_type,
                            "face_score": face_score,
                            "gait_score": gait_score,
                            "face_embedding": face_embedding,  # ✅ NEW: Store face embedding
                            "gait_embedding": gait_embedding,  # ✅ NEW: Store gait embedding
                            "shoe_embedding": shoe_embedding,  # ✅ NEW: Store shoe embedding
                            "unified_fused_embedding": unified_fused_embedding  # ✅ NEW: Store unified fused embedding
                        }
                        self.track_id_details[track_id] = details_to_store
                        
                        if track_id in self.tracklet_buffer:
                            del self.tracklet_buffer[track_id]

                    # print(f"Track {track_id} assigned Person ID {persistent_id} ({match_type}) with face_score: {face_score:.2f}, gait_score: {gait_score:.2f}")

                    # --- RECORD IN SQLITE DATABASE (ALL ENTRIES) ---
                    try:
                        # Record in SQLite database (ALL entries - both new and re-identified)
                        self.person_tracking_db.record_person_entry(
                            person_id=persistent_id
                        )
                        
                        # print(f"🚶 Person ID {persistent_id} recorded in SQLite database")
                            
                    except Exception as e:
                        # print(f"❌ Error recording in SQLite database: {e}")
                        pass

                    # --- Image Saving Disabled ---
                    # Note: Image saving to detected_faces, tracked_persons, and gait_cycles directories has been disabled
                    # Uncomment the sections below if you want to re-enable image saving
                    
                    # # Save tracked person image (DISABLED)
                    # if capture_crop is not None:
                    #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    #     persons_filename = f"ID_{persistent_id}_fusion_score_{fusion_similarity:.4f}_{timestamp}.jpg"
                    #     persons_save_path = os.path.join(self.tracked_persons_dir, persons_filename)
                    #     cv2.imwrite(persons_save_path, capture_crop)
                    #     print(f"Saved person image to {persons_save_path}")

                    # # Save face image (DISABLED)
                    # if face_crop is not None:
                    #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    #     faces_filename = f"ID_{persistent_id}_fusion_score_{fusion_similarity:.4f}_{timestamp}.jpg"
                    #     faces_save_path = os.path.join(self.detected_faces_dir, faces_filename)
                    #     cv2.imwrite(faces_save_path, face_crop)
                    #     print(f"Saved face image to {faces_save_path}")

                    # # Save gait cycle frames (DISABLED)
                    # if returned_frames is not None:
                    #     if match_type == 'fused' and isinstance(returned_frames, list):
                    #         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    #         gait_cycles_dir = os.path.join("data/processed_output/gait_cycles", f"ID_{persistent_id}_fusion_score_{fusion_similarity:.4f}_{timestamp}")
                    #         os.makedirs(gait_cycles_dir, exist_ok=True)
                    #         
                    #         for i, frame in enumerate(returned_frames):
                    #             frame_filename = f"frame_{i:03d}.jpg"
                    #             frame_path = os.path.join(gait_cycles_dir, frame_filename)
                    #             cv2.imwrite(frame_path, frame)
                    #         
                    #         print(f"Saved {len(returned_frames)} gait cycle frames to {gait_cycles_dir}")
                    #     elif match_type == 'fused' and returned_frames is not None:
                    #         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    #         frame_filename = f"ID_{persistent_id}_fusion_score_{fusion_similarity:.4f}_{timestamp}.jpg"
                    #         frame_path = os.path.join("data/processed_output/tracked_persons", frame_filename)
                    #         cv2.imwrite(frame_path, returned_frames)
                    #         print(f"Saved single frame to {frame_path}")

                else:
                    # print(f"❌ Track {track_id}: Re-identification failed")
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                # print(f"❌ Error in Re-ID worker: {e}")
                import traceback
                traceback.print_exc()

    def _process_tracks_batch(self, active_tracks, clean_frame, frame):
        """Process multiple tracks efficiently in batches to prevent video freezing."""
        
        for track, x1, y1, x2, y2, track_id in active_tracks:
            # Check if this track is within the detection region
            is_in_zone = self._is_bbox_100_percent_in_zone(x1, y1, x2, y2)
            
            # Check if bounding box is of reasonable size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            min_person_width = 30
            min_person_height = 80
            is_full_person = (bbox_width >= min_person_width and bbox_height >= min_person_height)
            
            # Only track persons currently in the region
            if is_in_zone and is_full_person:
                if track_id not in self.zone_track_ids:
                    # print(f"Track {track_id} entered detection zone. Starting re-identification workflow.")
                    self.zone_track_ids.add(track_id)
                    self.visibility_counter[track_id] = 1
                    self.stability_counter[track_id] = 1
                else:
                    self.visibility_counter[track_id] = self.visibility_counter.get(track_id, 0) + 1
                    self.stability_counter[track_id] = self.stability_counter.get(track_id, 0) + 1
            else:
                # Person is outside zone or not fully visible - stop tracking them
                if track_id in self.zone_track_ids:
                    if not is_in_zone:
                        # print(f"Track {track_id} left detection zone. Stopping tracking.")
                        pass
                    else:
                        # print(f"Track {track_id} not fully visible (size: {bbox_width}x{bbox_height}). Stopping tracking.")
                        pass
                    
                    # Remove from all tracking data structures
                    self.zone_track_ids.discard(track_id)
                    self.visibility_counter.pop(track_id, None)
                    self.stability_counter.pop(track_id, None)
                    
                    # Also remove from buffers to stop re-identification
                    if track_id in self.tracklet_buffer:
                        del self.tracklet_buffer[track_id]
                    with self.reid_lock:
                        self.reid_triggered_for_track.discard(track_id)
                        self.fast_reid_triggered_for_track.discard(track_id)
                
                # Draw bounding box for tracks outside zone (gray color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)  # Gray color
                if not is_in_zone:
                    label_text = f"Track: {track_id} (Outside Zone)"
                elif not is_full_person:
                    label_text = f"Track: {track_id} (Partial {bbox_width}x{bbox_height})"
                else:
                    label_text = f"Track: {track_id} (Unknown)"
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
                continue  # Skip re-identification logic for tracks outside zone
            
            # Check if person is fully visible and stable in zone
            visibility_count = self.visibility_counter.get(track_id, 0)
            stability_count = self.stability_counter.get(track_id, 0)
            is_confirmed = (visibility_count >= self.visibility_confirm_frames and 
                           stability_count >= self.stability_required_frames)

            with self.reid_lock:
                track_details = self.track_id_details.get(track_id)
                is_identifying = track_id in self.reid_triggered_for_track

            # Determine label and handle buffering - ONLY if person is 100% inside zone
            if track_details:
                face_score = track_details.get('face_score', 0.0)
                gait_score = track_details.get('gait_score', 0.0)
                label = f"ID: {track_details['person_id']} ({track_details['match_type']}) - F:{face_score:.2f} G:{gait_score:.2f}"
            elif is_identifying:
                label = "ID: Identifying..."
            elif is_confirmed:
                # CONFIRMED: Person is 100% inside zone and stable - start Re-ID process
                if track_id not in self.tracklet_buffer:
                    self.tracklet_buffer[track_id] = []
                
                # Use the clean_frame to get a crop without any drawings
                crop = clean_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    self.tracklet_buffer[track_id].append(crop)

                buffer_len = len(self.tracklet_buffer.get(track_id, []))
                
                with self.reid_lock:
                    fast_reid_triggered = track_id in self.fast_reid_triggered_for_track

                # Re-ID Trigger Logic (Only when 100% inside zone)
                if buffer_len >= self.min_frames_for_full_reid:
                    # print(f"✅ Track {track_id} is 100% inside zone with {buffer_len} frames. Triggering Re-ID.")
                    
                    image_sequence = list(self.tracklet_buffer[track_id])
                    # Capture the crop for saving from the clean frame as well
                    capture_crop = clean_frame[y1:y2, x1:x2] if image_sequence else None
                    self.reid_queue.put((track_id, image_sequence, capture_crop, 'full'))
                    
                    with self.reid_lock:
                        self.reid_triggered_for_track.add(track_id)
                    label = "ID: Identifying... (100% In Zone)"
                else:
                    # Update label to show progress towards re-identification
                    target_frames = self.min_frames_for_full_reid
                    label = f"Track: {track_id} - Zone Ready ({buffer_len}/{target_frames})"

            else:
                # Person is in zone but not yet confirmed - show progress
                v_count = self.visibility_counter.get(track_id, 0)
                s_count = self.stability_counter.get(track_id, 0)
                v_req = self.visibility_confirm_frames
                s_req = self.stability_required_frames
                
                if v_count < v_req:
                    label = f"Track: {track_id} - Zone Visibility: {v_count}/{v_req}"
                elif s_count < s_req:
                    label = f"Track: {track_id} - Zone Stability: {s_count}/{s_req}"
                else:
                    label = f"Track: {track_id} - Zone Confirming..."

            # Draw bounding box and label - GREEN for persons in zone ready for re-ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cleanup for lost tracks with grace period
        current_track_ids = {int(t[4]) for t in active_tracks}
        lost_track_ids = set(self.track_id_details.keys()) | self.zone_track_ids - current_track_ids
        
        for i_id in lost_track_ids:
            self.lost_track_frames[i_id] = self.lost_track_frames.get(i_id, 0) + 1

        tracks_to_delete = {i_id for i_id, count in self.lost_track_frames.items() if count > self.grace_period_frames}

        if tracks_to_delete:
            # print(f"Permanently cleaning up lost tracks: {list(tracks_to_delete)}")
            for i_id in tracks_to_delete:
                if i_id in self.tracklet_buffer:
                    del self.tracklet_buffer[i_id]
                self.zone_track_ids.discard(i_id)
                self.visibility_counter.pop(i_id, None)
                self.stability_counter.pop(i_id, None)
                with self.reid_lock:
                    self.reid_triggered_for_track.discard(i_id)
                    if i_id in self.track_id_details:
                        del self.track_id_details[i_id]
                del self.lost_track_frames[i_id]

        reappeared_tracks = current_track_ids.intersection(self.lost_track_frames.keys())
        for i_id in reappeared_tracks:
            self.lost_track_frames.pop(i_id)

        self.frame_count += 1

    def process_frame(self, frame: np.ndarray):
        """
        Processes a single video frame for person tracking and re-identification.
        All operations are restricted to persons within the detection region.
        """
        # --- PERFORMANCE TRACKING ---
        start_time = time.time()
        
        # --- GPU MEMORY MANAGEMENT ---
        self.gpu_memory_manager.monitor_memory_usage()
        
        frame_height, frame_width = frame.shape[:2]
        clean_frame = frame.copy() # Create a clean copy for pristine crops

        # --- 1. Run detection on the full frame ---
        # Draw the detection region on the original frame for visualization
        self._draw_detection_zone(frame)

        raw_detections = self.detector(frame, classes=[0], verbose=False)[0].boxes
        
        # --- 2. Update tracker with ALL detections (for proper tracking continuity) ---
        if raw_detections.xyxy.numel() > 0:
            dets_for_tracker = np.hstack((
                raw_detections.xyxy.cpu().numpy(),
                raw_detections.conf.cpu().numpy()[:, np.newaxis],
                raw_detections.cls.cpu().numpy()[:, np.newaxis]
            ))
            tracked_objects = self.tracker.update(dets_for_tracker, frame)
        else:
            # If no detections, update the tracker with an empty array
            tracked_objects = self.tracker.update(np.empty((0, 6)), frame)

        # --- BATCH PROCESSING OPTIMIZATION ---
        active_tracks = []
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            
            # Check if this track is within the detection region
            is_in_zone = self._is_bbox_100_percent_in_zone(x1, y1, x2, y2)
            
            # Check if bounding box is of reasonable size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            min_person_width = 30
            min_person_height = 80
            is_full_person = (bbox_width >= min_person_width and bbox_height >= min_person_height)
            
            if is_in_zone and is_full_person:
                active_tracks.append((track, x1, y1, x2, y2, track_id))
        
        # --- BATCH RE-ID PROCESSING ---
        if len(active_tracks) > 0:
            # Limit concurrent processing to prevent GPU OOM
            if len(active_tracks) > self.max_concurrent_persons:
                # print(f"⚠️ Limiting concurrent persons from {len(active_tracks)} to {self.max_concurrent_persons}")
                active_tracks = active_tracks[:self.max_concurrent_persons]
            
            # Process tracks in batches for better GPU utilization
            self._process_tracks_batch(active_tracks, clean_frame, frame)
        
        # --- GPU MEMORY CLEANUP ---
        if len(active_tracks) > 2:  # Cleanup when multiple persons detected
            self.gpu_memory_manager.cleanup_gpu_memory()
        
        # --- FPS TRACKING ---
        processing_time = time.time() - start_time
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            update_fps(current_fps)
        
        return frame

    def save_track_details(self, filename="track_id_details.pkl"):
        """Save track_id_details dictionary to pickle file for validation."""
        import pickle
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.track_id_details, f)
                    # print(f"✅ Track details saved to {filename}")
        # print(f"   - Total tracks: {len(self.track_id_details)}")
        # for track_id, details in self.track_id_details.items():
        #     print(f"   - Track {track_id}: {details.get('person_id', 'Unknown')} ({details.get('match_type', 'Unknown')})")
        except Exception as e:
            # print(f"❌ Failed to save track details: {e}")
            pass

    def close(self):
        """Shuts down the tracker and background threads."""
        # print("Shutting down PersonTracker...")
        
        # ✅ VALIDATION: Save track details before closing
        self.save_track_details()
        
        self.reid_queue.put((None, None, None, None))
        self.reid_worker_thread.join()
        
        # Close only the fused database
        self.reid_system.fused_db.close()
        # print("PersonTracker shut down.")
