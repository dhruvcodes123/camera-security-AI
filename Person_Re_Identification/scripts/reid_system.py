import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from db.database import VectorDatabase
from face_embedding.embedder import FaceEmbedder
from hybrid_embedding.embedder import GaitEmbedder, GaitAndShoeEmbedder
from shoe_detection.shoe_detector import ShoeDetector
from shoe_embedding.embedder import ShoeEmbedder
from modules.body_shape_embedding import get_body_shape_embedding
import numpy as np
import cv2
import os
import torch
from gpu_memory_manager import GPUMemoryManager
from performance_monitor import update_fps

class UnifiedReIdentificationSystem:
    """
    Handles person re-identification using a three-stage pipeline:
    1. Face Recognition (Primary)
    2. Gait + Shoe Recognition (Fallback)
    """
    def __init__(self, face_det_thresh=0.5, fused_threshold=0.70):
        # print("Initializing Unified Re-ID System...")
        
        # --- GPU MEMORY MANAGEMENT ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_manager = GPUMemoryManager()
        
        # --- FACE RECOGNITION ENABLED ---
        # print("Initializing Face Components...")
        self.face_embedder = FaceEmbedder(det_thresh=face_det_thresh)
        # Face enhancement removed - using original face quality

        # Initialize Gait Components
        # print("Initializing Gait Components...")
        self.gait_embedder = GaitEmbedder()
        
        # --- SHOE DETECTION ENABLED ---
        # print("Initializing Shoe Components...")
        self.shoe_detector = ShoeDetector(model_path='yolov8n.pt', confidence_threshold=0.25)
        self.shoe_embedder = ShoeEmbedder()
        
        # --- UNIFIED FUSED EMBEDDING SYSTEM ---
        # print("Initializing Unified Fused Embedding System...")
        # Fused embedding: Face(512D) + Gait(1024D) + Shoe(2048D) + Body Shape(256D) = 3840D
        self.fused_db = VectorDatabase(db_type='fused', embedding_dim=3840)
        self.fused_threshold = fused_threshold  # Increased from 0.65 to 0.85 for stricter matching
        
        # --- PERFORMANCE OPTIMIZATIONS ---
        self.batch_size = 4  # Process multiple persons simultaneously
        self.max_concurrent_persons = 8  # Limit concurrent processing
        self.gpu_memory_threshold = 0.8  # Clear GPU memory when 80% full
        
        # print("--- Unified Re-ID System Initialized with 3840D Fused Embedding ---")
        # print(f"   - Fused Threshold: {self.fused_threshold} (Strict matching)")
        # print(f"   - GPU Memory Management: Enabled")
        # print(f"   - Batch Processing: {self.batch_size} persons")
        # print(f"   - Max Concurrent Persons: {self.max_concurrent_persons}")

    def _calculate_image_clarity(self, image: np.ndarray):
        """Calculates the clarity of an image using the variance of the Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _try_get_face_embedding(self, image_sequence: list):
        """
        Attempts to find the best face in a sequence and get an embedding.
        INDEPENDENT MODE: Returns zero embeddings if no face detected (no fallback).
        OPTIMIZED FOR REAL-TIME: Process every 3rd frame for speed.
        """
        best_face_crop = None
        best_embedding = None
        max_clarity = 0
        faces_detected = 0
        
        # OPTIMIZATION: Process every 3rd frame for speed
        step = max(1, len(image_sequence) // 8)  # Process max 8 frames
        frames_to_process = image_sequence[::step]
        
        for i, image in enumerate(frames_to_process):
            # Try to get face embedding and crop from the full image
            embedding, face_crop = self.face_embedder.get_face_embedding(image)
            
            if embedding is not None and face_crop is not None:
                faces_detected += 1
                clarity = self._calculate_image_clarity(face_crop)
                
                if clarity > max_clarity:
                    max_clarity = clarity
                    best_face_crop = face_crop
                    best_embedding = embedding
        
        # print(f"Face detection: Found {faces_detected} faces in {len(frames_to_process)} processed frames")
        
        if best_face_crop is not None:
            # Use original face crop without enhancement
            # print(f"Using original face crop (no enhancement)")
            return best_embedding, best_face_crop

        # INDEPENDENT MODE: Return zero embeddings instead of None
        # print("⚠️ No face detected - creating zero embeddings")
        zero_embedding = np.zeros(512, dtype=np.float32)  # 512D zero vector
        return zero_embedding, None

    def _try_get_shoe_embedding(self, image_sequence: list):
        """
        Attempts to find the best shoe in a sequence and get an embedding.
        Processes multiple frames to find the best shoe detection.
        """
        best_shoe_crop = None
        best_embedding = None
        max_confidence = 0
        shoes_detected = 0
        
        # Process every 4th frame for shoe detection (less frequent than face)
        step = max(1, len(image_sequence) // 6)  # Process max 6 frames
        frames_to_process = image_sequence[::step]
        
        for i, image in enumerate(frames_to_process):
            try:
                # Get the best shoe crop from this frame
                shoe_crop = self.shoe_detector.get_best_shoe_crop(image)
                
                if shoe_crop is not None and shoe_crop.size > 0:
                    shoes_detected += 1
                    
                    # Get shoe embedding
                    shoe_embedding = self.shoe_embedder.create_fused_embedding(shoe_crop)
                    
                    if shoe_embedding is not None:
                        # For now, use the first valid shoe embedding we find
                        # In a more sophisticated approach, we could compare multiple detections
                        best_shoe_crop = shoe_crop
                        best_embedding = shoe_embedding
                        # print(f"Found valid shoe embedding from frame {i}")
                        break  # Use the first good detection
                        
            except Exception as e:
                # print(f"Error processing shoe detection in frame {i}: {e}")
                continue
        
        # print(f"Shoe detection: Found {shoes_detected} shoes in {len(frames_to_process)} processed frames")
        return best_embedding, best_shoe_crop

    def _create_unified_fused_embedding(self, image_sequence: list):
        """
        Creates a unified 3840D fused embedding by concatenating:
        - Face embedding (512D)
        - Gait embedding (1024D) 
        - Shoe embedding (2048D)
        - Body shape embedding (256D)
        
        Returns the fused embedding and individual components for debugging.
        """
        # --- GPU MEMORY MANAGEMENT ---
        self.gpu_memory_manager.monitor_memory_usage()
        
        # --- Step 1: Generate Face Embedding (512D) ---
        face_embedding, face_crop = self._try_get_face_embedding(image_sequence)
        
        # --- Step 2: Generate Gait Embedding (1024D) ---
        gait_embedding = None
        gait_cycle_frames = None
        view_type = "default"
        
        gait_result = self.gait_embedder.get_embedding(image_sequence)
        if gait_result:
            gait_embedding, view_type, gait_cycle_frames = gait_result
            # print(f"✅ Gait embedding generated and normalized, shape: {gait_embedding.shape}")
        else:
            # print("❌ Failed to generate gait embedding")
            # Use zero embedding for gait
            gait_embedding = np.zeros(1024, dtype=np.float32)
        
        # --- Step 3: Generate Shoe Embedding (2048D) ---
        shoe_embedding, shoe_crop = self._try_get_shoe_embedding(image_sequence)
        if shoe_embedding is not None:
            # print(f"✅ Shoe embedding generated and normalized, shape: {shoe_embedding.shape}")
            pass
        else:
            # print("❌ Failed to generate shoe embedding")
            # Use zero embedding for shoe
            shoe_embedding = np.zeros(2048, dtype=np.float32)
        
        # --- Step 4: Generate Body Shape Embedding (256D) ---
        body_shape_embedding = None
        body_shape_crop = None
        
        # Use the first frame for body shape analysis (pose detection)
        if len(image_sequence) > 0:
            first_frame = image_sequence[0]
            body_shape_embedding = get_body_shape_embedding(first_frame)
            if body_shape_embedding is not None:
                # print(f"✅ Body shape embedding generated and normalized, shape: {body_shape_embedding.shape}")
                pass
            else:
                # print("❌ Failed to generate body shape embedding")
                # Use zero embedding for body shape
                body_shape_embedding = np.zeros(256, dtype=np.float32)
        else:
            # print("❌ No frames available for body shape embedding.")
            body_shape_embedding = np.zeros(256, dtype=np.float32)

        # --- Step 5: Create Unified Fused Embedding (3840D) ---
        # Concatenate: Face(512D) + Gait(1024D) + Shoe(2048D) + Body Shape(256D) = 3840D
        if face_embedding is not None and gait_embedding is not None and shoe_embedding is not None and body_shape_embedding is not None:
            # Ensure all embeddings are 1D arrays
            face_emb = face_embedding.flatten() if face_embedding.ndim > 1 else face_embedding
            gait_emb = gait_embedding.flatten() if gait_embedding.ndim > 1 else gait_embedding
            shoe_emb = shoe_embedding.flatten() if shoe_embedding.ndim > 1 else shoe_embedding
            body_shape_emb = body_shape_embedding.flatten() if body_shape_embedding.ndim > 1 else body_shape_embedding
            
            # Ensure all individual embeddings are L2 normalized to unit length
            # Face embedding normalization check
            face_norm = np.linalg.norm(face_emb)
            if face_norm > 0:
                face_emb = face_emb / face_norm
            
            # Gait embedding normalization check
            gait_norm = np.linalg.norm(gait_emb)
            if gait_norm > 0:
                gait_emb = gait_emb / gait_norm
            
            # Shoe embedding normalization check
            shoe_norm = np.linalg.norm(shoe_emb)
            if shoe_norm > 0:
                shoe_emb = shoe_emb / shoe_norm
            
            # Body shape embedding normalization check
            body_shape_norm = np.linalg.norm(body_shape_emb)
            if body_shape_norm > 0:
                body_shape_emb = body_shape_emb / body_shape_norm
            
            # Concatenate all normalized embeddings
            fused_embedding = np.concatenate([face_emb, gait_emb, shoe_emb, body_shape_emb])
            
            # Final normalization of the fused embedding to unit length
            embedding_norm = np.linalg.norm(fused_embedding)
            if embedding_norm > 0:
                fused_embedding = fused_embedding / embedding_norm
            
            # print(f"✅ 3840D Unified Fused embedding created: {fused_embedding.shape}")
            # print(f"   📊 Face: {face_emb.shape}, Gait: {gait_emb.shape}, Shoe: {shoe_emb.shape}, Body Shape: {body_shape_emb.shape}")
            # print(f"   🔍 All features L2 normalized to unit length before fusion")
            
            # --- GPU MEMORY CLEANUP ---
            self.gpu_memory_manager.cleanup_gpu_memory()
            
            return fused_embedding, face_embedding, gait_embedding, shoe_embedding, view_type, gait_cycle_frames, face_crop, shoe_crop
        else:
            # print("❌ Failed to create unified fused embedding - missing components")
            return None, None, None, None, "default", None, None, None

    def check_face_quality(self, image: np.ndarray):
        """
        Checks if a face is detectable by attempting to get an embedding.
        """
        # This is a more direct way to check for a usable face. If we can get
        # an embedding, the face is considered "good enough".
        embedding, _ = self.face_embedder.get_face_embedding(image)
        return embedding is not None

    def reidentify_track_fast(self, image_sequence: list):
        """
        Fast re-identification using unified fused embedding only.
        Creates 3840D fused embedding and performs similarity search.
        """
        # --- Step 1: Generate Unified Fused Embedding (3840D) ---
        fused_result = self._create_unified_fused_embedding(image_sequence)
        fused_embedding, face_embedding, gait_embedding, shoe_embedding, view_type, gait_cycle_frames, face_crop, shoe_crop = fused_result
        
        # --- Step 2: FUSED EMBEDDING SIMILARITY CHECK (ONLY) ---
        if fused_embedding is not None:
            # Query the fused database for similarity
            fused_matched_id, fused_similarity = self.fused_db.query(fused_embedding)
            
            if fused_similarity is not None and fused_similarity >= self.fused_threshold:
                        # print(f"🔍 FAST FUSED 3840D SIMILARITY: {fused_similarity:.4f} for Person ID {fused_matched_id}")
        # print(f"   - Threshold: {self.fused_threshold:.4f}")
        # print(f"   - Match: ✅ YES")
        # print(f"✅ FAST FUSED 3840D MATCH: Found Person ID {fused_matched_id} with similarity {fused_similarity:.4f}")
                
                # Store individual embeddings for debugging
                self.track_id_details[track_id] = {
                    'person_id': fused_matched_id,
                    'match_type': 'fused',
                    'face_score': 0.0,  # Not used in unified system
                    'gait_score': 0.0,  # Not used in unified system
                    'face_embedding': face_embedding,
                    'gait_embedding': gait_embedding,
                    'shoe_embedding': shoe_embedding,
                    'unified_fused_embedding': fused_embedding,
                    'view_type': view_type,
                    'gait_cycle_frames': gait_cycle_frames
                }
                
                return fused_matched_id, 'fused', 0.0, 0.0, gait_cycle_frames, fused_similarity
            else:
                # No match found - create new person
                new_id = self.fused_db.get_next_person_id()
                self.fused_db.add_person(new_id, fused_embedding)
                
                        # print(f"🆕 NEW PERSON (with Fast Unified Fused 3840D): Assigning Person ID {new_id}")
        # print(f"   - Similarity: {fused_similarity:.4f} (below threshold {self.fused_threshold:.4f})")
        # print(f"✅ Stored new fast unified fused 3840D embedding for Person ID {new_id}.")
                
                # Store individual embeddings for debugging
                self.track_id_details[track_id] = {
                    'person_id': new_id,
                    'match_type': 'fused',
                    'face_score': 0.0,
                    'gait_score': 0.0,
                    'face_embedding': face_embedding,
                    'gait_embedding': gait_embedding,
                    'shoe_embedding': shoe_embedding,
                    'unified_fused_embedding': fused_embedding,
                    'view_type': view_type,
                    'gait_cycle_frames': gait_cycle_frames
                }
                
                return new_id, 'fused', 0.0, 0.0, gait_cycle_frames, 0.0  # New person, no similarity
        else:
            # print("❌ Failed to create unified fused embedding")
            return None, None, 0.0, 0.0, None, 0.0

    def reidentify_track(self, image_sequence: list):
        """
        Full re-identification using unified fused embedding only.
        Creates 3840D fused embedding and performs similarity search.
        """
        # --- Step 1: Generate Unified Fused Embedding (3840D) ---
        fused_result = self._create_unified_fused_embedding(image_sequence)
        fused_embedding, face_embedding, gait_embedding, shoe_embedding, view_type, gait_cycle_frames, face_crop, shoe_crop = fused_result
        
        # --- Step 2: FUSED EMBEDDING SIMILARITY CHECK (ONLY) ---
        if fused_embedding is not None:
            fused_matched_id, fused_similarity = self.fused_db.query(fused_embedding)
            
            # Enhanced logging for debugging
            if fused_matched_id is not None and fused_similarity is not None:
                fused_score = fused_similarity
                # print(f"🔍 FUSED 3840D SIMILARITY: {fused_similarity:.4f} for Person ID {fused_matched_id}")
                # print(f"   - Threshold: {self.fused_threshold:.4f}")
                # print(f"   - Match: {'✅ YES' if fused_similarity >= self.fused_threshold else '❌ NO'}")
                
                if fused_similarity >= self.fused_threshold:
                    # print(f"✅ FUSED 3840D MATCH: Found Person ID {fused_matched_id} with similarity {fused_similarity:.4f}")
                    
                    # Log reappearance in fused database
                    self.fused_db.log_reappearance(fused_matched_id, fused_embedding)
                    
                    # Return fused match with gait cycle frames for saving
                    return fused_matched_id, 'fused', face_score, gait_score, gait_cycle_frames, fused_similarity
                else:
                    # print(f"❌ SIMILARITY TOO LOW: {fused_similarity:.4f} < {self.fused_threshold:.4f} - Assigning new ID")
                    pass
            else:
                # print(f"❌ NO EXISTING MATCH FOUND - Assigning new ID")
                pass

        # --- Step 3: Assign New Identity (FUSED ONLY) ---
        # If we reach here, no existing person was matched or similarity was too low.
        new_id = self.fused_db.get_next_person_id()
        
        # Store only fused embedding
        if fused_embedding is not None:
            # print(f"🆕 NEW PERSON (with Unified Fused 3840D): Assigning Person ID {new_id}")
            self.fused_db.add_person(new_id, fused_embedding)
            # print(f"✅ Stored new unified fused 3840D embedding for Person ID {new_id}.")
            fused_score = 1.0  # New person, perfect match
            
            # Return appropriate data with gait cycle frames for saving
            return new_id, 'fused', face_score, gait_score, gait_cycle_frames, fused_score
        
        # --- Failure Case ---
        # print("❌ Could not generate unified fused embedding for the track.")
        return None, None, 0.0, 0.0, None, 0.0

    def reidentify_track_with_fused(self, image_sequence: list, fused_embedding=None):
        """
        UNIFIED FUSED RE-IDENTIFICATION: Only uses 3840D fused embedding for re-identification.
        UPDATED LOGIC:
        1. Generate unified fused embedding (face + gait + shoe + body shape = 3840D)
        2. Check fused embedding similarity only
        3. Assign ID based on fused match or create new one
        4. Store only in fused database
        """
        # Initialize scores
        face_score = 0.0
        gait_score = 0.0
        fused_score = 0.0
        
        # --- Step 1: Generate or Use Unified Fused Embedding (3840D) ---
        if fused_embedding is not None:
            # print(f"✅ Using provided 3840D fused embedding: {fused_embedding.shape}")
            # Use the provided fused embedding directly
            unified_fused_embedding = fused_embedding
        else:
            # Generate unified fused embedding
            fused_result = self._create_unified_fused_embedding(image_sequence)
            final_fused_embedding, face_embedding, gait_embedding, shoe_embedding, view_type, gait_cycle_frames, face_crop, shoe_crop = fused_result
            
            if final_fused_embedding is not None:
                unified_fused_embedding = final_fused_embedding
            else:
                # print("❌ Failed to generate unified fused embedding")
                return None, None, 0.0, 0.0, None, 0.0
        
        # --- Step 2: FUSED EMBEDDING SIMILARITY CHECK (ONLY) ---
        if unified_fused_embedding is not None:
            # Query the fused database for similarity
            fused_matched_id, fused_similarity = self.fused_db.query(unified_fused_embedding)
            
            if fused_similarity is not None and fused_similarity >= self.fused_threshold:
                # print(f"🔍 FUSED 3840D SIMILARITY: {fused_similarity:.4f} for Person ID {fused_matched_id}")
                # print(f"   - Threshold: {self.fused_threshold:.4f}")
                # print(f"   - Match: ✅ YES")
                # print(f"✅ FUSED 3840D MATCH: Found Person ID {fused_matched_id} with similarity {fused_similarity:.4f}")
                
                return fused_matched_id, 'fused', face_score, gait_score, None, fused_similarity
            else:
                # No match found - create new person
                new_id = self.fused_db.get_next_person_id()
                self.fused_db.add_person(new_id, unified_fused_embedding)
                
                # print(f"🆕 NEW PERSON (with Unified Fused 3840D): Assigning Person ID {new_id}")
                # print(f"   - Similarity: {fused_similarity:.4f} (below threshold {self.fused_threshold:.4f})")
                # print(f"✅ Stored new unified fused 3840D embedding for Person ID {new_id}.")
                
                return new_id, 'fused', face_score, gait_score, None, 0.0  # New person, no similarity
        else:
            # print("❌ Failed to create unified fused embedding")
            return None, None, 0.0, 0.0, None, 0.0

def save_image(image, filepath):
    """
    Helper function to save images with proper error handling.
    
    Args:
        image: Image array to save
        filepath: Path where to save the image
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, image)
        # print(f"Image saved to {filepath}")
        return True
    except Exception as e:
        # print(f"Error saving image to {filepath}: {e}")
        return False

if __name__ == '__main__':
    # print("This script provides the UnifiedReIdentificationSystem class.")
    # You can add example usage here if needed, similar to the new db/database.py
    # For example:
    # 1. Initialize the system.
    # 2. Create a dummy image sequence.
    # 3. Call reidentify_track and print the results.
    pass 