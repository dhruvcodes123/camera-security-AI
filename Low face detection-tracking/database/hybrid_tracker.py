import numpy as np
import cv2
import faiss
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import json
import os

from .bytetrack import ByteTracker
from models import get_recognizer
from src.utils.helpers import face_alignment
from src.utils.face_quality import assess_face_quality


class HybridPersonTracker:
    """
    Hybrid Person Tracker combining ByteTrack + ArcFace + Re-identification
    
    Features:
    - ByteTrack for smooth short-term tracking (motion-based)
    - ArcFace embeddings for long-term re-identification (appearance-based)
    - FAISS for fast similarity search
    - Real-time optimized performance
    """
    
    def __init__(self, 
                 arcface_model_path: str,
                 track_thresh: float = 0.4,
                 match_thresh: float = 0.35,
                 track_buffer: int = 30,
                 similarity_thresh: float = 0.45,
                 embedding_dim: int = 512,
                 max_reid_distance: int = 30,
                 reid_confidence_thresh: float = 0.75,
                 min_face_size: int = 64,
                 face_quality_thresh: float = 0.1):  # Lowered for debugging
        """
        Initialize Hybrid Tracker
        
        Args:
            arcface_model_path: Path to ArcFace model
            track_thresh: ByteTrack tracking threshold
            match_thresh: ByteTrack matching threshold  
            track_buffer: ByteTrack buffer frames
            similarity_thresh: Face embedding similarity threshold (0.3-0.6)
            embedding_dim: ArcFace embedding dimension (512)
            max_reid_distance: Max frames between re-identification attempts
            reid_confidence_thresh: Minimum confidence for re-identification
        """
        
        # Initialize ByteTracker for motion tracking
        self.bytetracker = ByteTracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh, 
            track_buffer=track_buffer,
            aspect_ratio_thresh=2.0,
            min_box_area=100
        )
        
        # Initialize ArcFace for embeddings (automatically uses TensorRT on Jetson)
        try:
            self.arcface = get_recognizer(arcface_model_path)
            logging.info(f"Face recognition model loaded: {arcface_model_path}")
        except Exception as e:
            logging.error(f"Failed to load face recognition model: {e}")
            raise
            
        # Re-identification parameters
        self.similarity_thresh = similarity_thresh
        self.max_reid_distance = max_reid_distance  
        self.reid_confidence_thresh = reid_confidence_thresh
        
        # Face quality parameters
        self.min_face_size = min_face_size
        self.face_quality_thresh = face_quality_thresh
        
        # FAISS index for fast similarity search
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        
        # Person database
        self.person_embeddings = []  # List of embeddings
        self.person_metadata = []    # List of metadata dicts
        self.next_person_id = 0
        
        # Tracking state
        self.track_to_person_map = {}     # track_id -> person_id mapping
        self.person_last_seen = {}        # person_id -> frame_count
        self.active_track_ids = set()     # Currently active ByteTrack IDs
        self.track_best_embedding = {}    # track_id -> (embedding, quality_score)
        self.track_embedding_frame = {}   # track_id -> frame number of best embedding
        
        # Performance optimization
        self.reid_frame_interval = 10   # Only attempt re-ID every N frames (increased for better performance)
        self.last_reid_frame = 0
        
        logging.info(f"HybridPersonTracker initialized - similarity_thresh: {similarity_thresh}")
        
    def _extract_face_embedding(self, frame: np.ndarray, bbox: List[float],
                               keypoints: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract face embedding using ArcFace with quality assessment
        Returns:
            Tuple[Optional[np.ndarray], float]: (embedding, quality_score)
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
            h, w = frame.shape[:2]
            # Expand box slightly for context
            x1a = max(0, x1 - 10)
            y1a = max(0, y1 - 10)
            x2a = min(w, x2 + 10)
            y2a = min(h, y2 + 10)
            face_crop = frame[y1a:y2a, x1a:x2a]

            # Debug logging for first few attempts
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 5:  # Only log first 5 attempts
                logging.info(f"Debug {self._debug_count}: bbox={bbox[:4]}, crop_shape={face_crop.shape}, frame_shape={frame.shape}")

            # Validate face crop
            if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                if self._debug_count <= 5:
                    logging.warning(f"Debug {self._debug_count}: Invalid face crop - size={face_crop.size}, shape={face_crop.shape}")
                return None, 0.0

            # Assess face quality
            quality_score, metrics = assess_face_quality(
                face_crop, bbox,
                min_size=self.min_face_size,
                min_confidence=self.face_quality_thresh
            )
            if quality_score < self.face_quality_thresh:
                if self._debug_count <= 5:
                    logging.warning(f"Debug {self._debug_count}: Low quality face - score={quality_score:.3f}")
                return None, quality_score

            # Try keypoint alignment if available
            embedding = None
            if keypoints is not None and len(keypoints) >= 10:
                try:
                    kps_reshaped = keypoints.reshape(5, 2)
                    embedding = self.arcface.get_embedding(face_crop, kps_reshaped, normalized=True)
                    if self._debug_count <= 5:
                        logging.info(f"Debug {self._debug_count}: Keypoint embedding successful")
                except Exception as kp_error:
                    if self._debug_count <= 5:
                        logging.warning(f"Debug {self._debug_count}: Keypoint alignment failed: {kp_error}")
                    embedding = None
            # Fallback: use simple resize if keypoints not available or failed
            if embedding is None:
                try:
                    face_resized = cv2.resize(face_crop, (112, 112))
                    face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                    embedding = self.arcface.session.run(self.arcface.output_names, {self.arcface.input_name: face_blob})[0]
                    embedding = embedding.flatten()
                    if self._debug_count <= 5:
                        logging.info(f"Debug {self._debug_count}: Fallback embedding successful")
                except Exception as e:
                    if self._debug_count <= 5:
                        logging.warning(f"Debug {self._debug_count}: Fallback embedding failed: {e}")
                    return None, 0.0
            # Ensure embedding is valid and normalized
            if embedding is not None:
                embedding = embedding.astype(np.float32)
                if not np.all(np.isfinite(embedding)):
                    if self._debug_count <= 5:
                        logging.warning(f"Debug {self._debug_count}: Non-finite embedding detected - min={np.min(embedding)}, max={np.max(embedding)}, has_nan={np.any(np.isnan(embedding))}, has_inf={np.any(np.isinf(embedding))}")
                    return None, 0.0
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    if self._debug_count <= 5:
                        logging.info(f"Debug {self._debug_count}: Valid embedding created - norm={norm:.3f}")
                    return embedding, quality_score
                else:
                    if self._debug_count <= 5:
                        logging.warning(f"Debug {self._debug_count}: Zero norm embedding")
                    return None, 0.0
            return None, 0.0
        except Exception as e:
            if hasattr(self, '_debug_count') and self._debug_count <= 5:
                logging.warning(f"Debug {self._debug_count}: Failed to extract embedding: {e}")
            return None, 0.0
    
    def _find_matching_person(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Find matching person using FAISS similarity search"""
        if self.index.ntotal == 0:
            return None, 0.0
            
        try:
            # Validate embedding
            if embedding is None or not np.isfinite(embedding).all():
                logging.warning("Invalid embedding for FAISS search")
                return None, 0.0
                
            # Search for most similar embedding
            embedding = embedding.reshape(1, -1)
            similarities, indices = self.index.search(embedding, 1)
            
            if len(similarities[0]) > 0:
                similarity = float(similarities[0][0])
                person_idx = int(indices[0][0])
                
                # Validate similarity and index
                if np.isfinite(similarity) and 0 <= person_idx < len(self.person_metadata):
                    if similarity >= self.similarity_thresh:
                        person_id = self.person_metadata[person_idx]['person_id']
                        return person_id, similarity
                    
        except Exception as e:
            logging.warning(f"FAISS search failed: {e}")
            
        return None, 0.0
    
    def _register_new_person(self, embedding: np.ndarray, confidence: float) -> int:
        """Register new person in the database"""
        try:
            # Validate embedding
            if embedding is None:
                logging.warning("Cannot register person with None embedding")
                return -1
                
            # Ensure embedding is float32 and check for invalid values
            try:
                embedding = np.asarray(embedding, dtype=np.float32)
                if not np.all(np.isfinite(embedding)):
                    logging.warning("Cannot register person with non-finite embedding values")
                    return -1
            except Exception as e:
                logging.warning(f"Embedding validation failed: {e}")
                return -1
                
            person_id = self.next_person_id
            self.next_person_id += 1
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            self.person_embeddings.append(embedding.copy())
            
            # Add metadata
            metadata = {
                'person_id': person_id,
                'first_seen': time.time(),
                'total_appearances': 1,
                'confidence': float(confidence),
                'last_updated': time.time()
            }
            self.person_metadata.append(metadata)
            
            logging.info(f"Registered new Person_{person_id:03d} (confidence: {confidence:.2f})")
            return person_id
            
        except Exception as e:
            logging.error(f"Failed to register new person: {e}")
            return -1
    
    def _update_person_appearance(self, person_id: int, embedding: np.ndarray, confidence: float):
        """Update person's appearance with new embedding"""
        try:
            # Validate embedding
            if embedding is None or not np.isfinite(embedding).all():
                logging.warning(f"Cannot update Person_{person_id:03d} with invalid embedding")
                return
                
            # Find person in metadata
            for i, metadata in enumerate(self.person_metadata):
                if metadata['person_id'] == person_id:
                    # Update metadata
                    metadata['total_appearances'] += 1
                    metadata['last_updated'] = time.time()
                    metadata['confidence'] = max(metadata['confidence'], float(confidence))
                    
                    # Update embedding (exponential moving average for better representation)
                    alpha = 0.3  # Learning rate
                    old_embedding = self.person_embeddings[i]
                    new_embedding = alpha * embedding + (1 - alpha) * old_embedding
                    
                    # Normalize and validate
                    norm = np.linalg.norm(new_embedding)
                    if norm > 0:
                        new_embedding = new_embedding / norm
                    else:
                        logging.warning(f"Zero norm embedding for Person_{person_id:03d}, keeping old embedding")
                        return
                    
                    self.person_embeddings[i] = new_embedding
                    
                    # Update FAISS index (Note: reconstruct doesn't exist, we'd need to rebuild index)
                    # For now, skip FAISS update to avoid crashes
                    # self.index.reconstruct(i, new_embedding.reshape(1, -1))
                    
                    logging.info(f"Updated Person_{person_id:03d} (appearances: {metadata['total_appearances']}, confidence: {metadata['confidence']:.2f})")
                    break
                    
        except Exception as e:
            logging.error(f"Failed to update person {person_id}: {e}")
    
    def update(self, frame: np.ndarray, detections: List[List], 
               keypoints_list: Optional[List] = None, frame_count: int = 0,
               face_db=None) -> List[Dict]:
        """
        Update tracking with new frame detections
        
        Args:
            frame: Current frame
            detections: List of [x1, y1, x2, y2, confidence] 
            keypoints_list: Optional face keypoints for alignment
            frame_count: Current frame number
            
        Returns:
            List of track results with person IDs
        """
        
        # Step 1: ByteTrack motion tracking 
        bytetrack_results = self.bytetracker.update(detections)
        
        # Step 2: Re-identification (optimized - not every frame)
        # Increase re-identification interval for better performance
        should_do_reid = (frame_count - self.last_reid_frame) >= self.reid_frame_interval
        
        # Skip re-identification if there are too many detections (performance optimization)
        if len(detections) > 10:  # If more than 10 faces, process only every 10 frames
            should_do_reid = should_do_reid and (frame_count % 10 == 0)
        
        results = []
        current_track_ids = set()
        
        for i, track in enumerate(bytetrack_results):
            x1, y1, x2, y2, track_id, confidence = track
            current_track_ids.add(track_id)
            
            person_id = None
            reid_event = None
            similarity_score = 0.0
            
            # Check if this track already has a person ID
            if track_id in self.track_to_person_map:
                person_id = self.track_to_person_map[track_id]
                self.person_last_seen[person_id] = frame_count
                reid_event = "TRACKED"
                
            # Attempt re-identification for new tracks or periodically
            elif should_do_reid and confidence >= self.reid_confidence_thresh:
                
                # Extract face embedding
                kps = keypoints_list[i] if keypoints_list and i < len(keypoints_list) else None
                embedding_result = self._extract_face_embedding(frame, [x1, y1, x2, y2], kps)
                
                if embedding_result is not None:
                    embedding, quality_score = embedding_result
                    # Search for matching person
                    matched_person_id, similarity_score = self._find_matching_person(embedding)
                    
                    if matched_person_id is not None:
                        # Re-identified existing person
                        person_id = matched_person_id
                        self.track_to_person_map[track_id] = person_id
                        self.person_last_seen[person_id] = frame_count
                        self._update_person_appearance(person_id, embedding, confidence)
                        reid_event = "REIDENTIFIED"
                        logging.info(f"Re-identified Person_{person_id:03d} as Track_{track_id} (similarity: {similarity_score:.3f})")
                        
                    else:
                        # Register new person
                        person_id = self._register_new_person(embedding, confidence)
                        if person_id >= 0:  # Valid person ID
                            self.track_to_person_map[track_id] = person_id
                            self.person_last_seen[person_id] = frame_count
                            reid_event = "NEW"
                        else:
                            # Registration failed, treat as unidentified track
                            person_id = None
                            reid_event = None
            
            # Build result
            result = {
                'bbox': [x1, y1, x2, y2],
                'track_id': track_id,
                'person_id': person_id,
                'confidence': confidence,
                'reid_event': reid_event,
                'similarity_score': similarity_score
            }
            results.append(result)
        
        # Update active tracks
        self.active_track_ids = current_track_ids
        
        # Clean up old track mappings
        tracks_to_remove = []
        for track_id in self.track_to_person_map:
            if track_id not in current_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            # Get person_id before removing from map
            person_id = self.track_to_person_map[track_id]
            
            # Record exit event in database if available
            if face_db:
                from datetime import datetime
                current_time = datetime.now().isoformat()
                # Record exit event
                face_db.record_face_event(
                    face_id=person_id,
                    event_type="EXIT",
                    confidence=None,
                    similarity_score=None,
                    frame_number=frame_count,
                    track_id=track_id,
                    entry_time=None,
                    exit_time=current_time
                )
                # Update person presence record
                face_db.record_person_exit(person_id)
                logging.info(f"Person_{person_id:03d} exited the scene (Track_{track_id})")
            else:
                # Just log the event if no database
                logging.info(f"Track_{track_id} with Person_{person_id} is no longer active")
                
            # Remove from map
            del self.track_to_person_map[track_id]
        
        if should_do_reid:
            self.last_reid_frame = frame_count
        
        return results
    
    def get_person_stats(self) -> Dict:
        """Get statistics about tracked persons"""
        return {
            'total_persons': len(self.person_metadata),
            'active_tracks': len(self.active_track_ids),
            'total_embeddings': self.index.ntotal
        }
    
    def clear_database(self):
        """Clear all person data"""
        self.index.reset()
        self.person_embeddings = []
        self.person_metadata = []
        self.track_to_person_map = {}
        self.person_last_seen = {}
        self.active_track_ids = set()
        self.next_person_id = 0
        logging.info("Cleared person database")
    
    def save_database(self, db_path: str):
        """Save person database to disk"""
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Save embeddings and metadata
            data = {
                'embeddings': [emb.tolist() for emb in self.person_embeddings],
                'metadata': self.person_metadata,
                'next_person_id': self.next_person_id
            }
            
            with open(f"{db_path}.json", 'w') as f:
                json.dump(data, f)
                
            # Save FAISS index
            faiss.write_index(self.index, f"{db_path}.faiss")
            
            logging.info(f"Saved person database: {len(self.person_metadata)} persons")
            
        except Exception as e:
            logging.error(f"Failed to save database: {e}")
    
    def load_database(self, db_path: str) -> bool:
        """Load person database from disk"""
        try:
            # Load metadata
            with open(f"{db_path}.json", 'r') as f:
                data = json.load(f)
            
            self.person_metadata = data['metadata']
            self.next_person_id = data['next_person_id']
            self.person_embeddings = [np.array(emb, dtype=np.float32) for emb in data['embeddings']]
            
            # Load FAISS index
            self.index = faiss.read_index(f"{db_path}.faiss")
            
            logging.info(f"Loaded person database: {len(self.person_metadata)} persons")
            return True
            
        except Exception as e:
            logging.warning(f"Failed to load database: {e}")
            return False 