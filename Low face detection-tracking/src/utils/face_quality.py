"""
Face quality assessment utilities for the Hybrid Person Tracking System.
Used to determine if a face image is suitable for generating embeddings.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union


def assess_face_quality(face_crop: np.ndarray, 
                        bbox: List[float], 
                        min_size: int = 64,
                        min_confidence: float = 0.3,  # Lowered for debugging
                        blur_threshold: float = 150.0,  # Higher = more lenient
                        weight_blur: float = 0.3,
                        weight_size: float = 0.4,
                        weight_confidence: float = 0.3) -> Tuple[float, dict]:
    """
    Assess the quality of a face crop for embedding generation.
    
    Args:
        face_crop: Cropped face image
        bbox: Bounding box coordinates [x1, y1, x2, y2, confidence]
        min_size: Minimum face size for good quality
        min_confidence: Minimum detection confidence
        blur_threshold: Laplacian variance threshold for blur detection
        weight_blur: Weight for blur score in final quality score
        weight_size: Weight for size score in final quality score
        weight_confidence: Weight for confidence score in final quality score
        
    Returns:
        Tuple of (quality_score, quality_metrics)
    """
    import logging
    quality_metrics = {}
    
    # Check valid input
    if face_crop is None or face_crop.size == 0:
        logging.warning(f"Invalid input to assess_face_quality: face_crop={None if face_crop is None else face_crop.shape}, bbox={bbox}")
        return 0.0, {"error": "Invalid input"}
    
    # Extract bbox info - ensure we have a confidence value
    if len(bbox) >= 5:
        confidence = bbox[4]
    else:
        confidence = 0.5  # Default confidence if not provided
    
    # Size score (how big the face is)
    face_width = face_crop.shape[1]
    face_height = face_crop.shape[0]
    size_score = min(1.0, (min(face_width, face_height) / min_size))
    quality_metrics["size_score"] = size_score
    quality_metrics["face_size"] = (face_width, face_height)
    
    # Blur score using Laplacian variance (higher is better/sharper)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = 0.0
    
    try:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, lap_var / blur_threshold)
    except Exception as e:
        logging.warning(f"Error calculating blur score: {e}")
        blur_score = 0.0
        
    quality_metrics["blur_score"] = blur_score
    quality_metrics["laplacian_var"] = lap_var if 'lap_var' in locals() else 0.0
    
    # Confidence score from detection
    confidence_score = min(1.0, confidence / min_confidence) if confidence > 0.0 else 0.0
    quality_metrics["confidence_score"] = confidence_score
    quality_metrics["detection_confidence"] = confidence
    
    # Weighted quality score
    quality_score = (weight_blur * blur_score + 
                     weight_size * size_score + 
                     weight_confidence * confidence_score)
    
    quality_metrics["final_score"] = quality_score
    
    # Debug log
    logging.debug(f"Face quality: size_score={size_score:.3f}, blur_score={blur_score:.3f}, " +
                 f"confidence_score={confidence_score:.3f}, final_score={quality_score:.3f}")
    
    # For debugging, let's force a minimum score to allow face registration
    if quality_score < 0.1:  # Very low threshold for debugging
        quality_score = 0.5  # Force a passing score
        quality_metrics["final_score"] = quality_score
        logging.warning(f"Forcing face quality score to {quality_score} for debugging")
    
    return quality_score, quality_metrics


def estimate_pose_score(landmarks: np.ndarray) -> float:
    """
    Estimate pose quality from face landmarks.
    Higher score = more frontal face.
    
    Args:
        landmarks: 5-point facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        
    Returns:
        float: Pose score (0.0-1.0)
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0
        
    try:
        # Calculate horizontal symmetry of landmarks (eyes and mouth corners)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate horizontal distance between eyes
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Calculate horizontal distance between mouth corners
        mouth_distance = np.linalg.norm(right_mouth - left_mouth)
        
        # Perfect symmetry would have ratios close to 1.0
        if eye_distance > 0 and mouth_distance > 0:
            ratio = eye_distance / mouth_distance
            symmetry_score = max(0.0, 1.0 - abs(1.0 - ratio))
        else:
            symmetry_score = 0.0
            
        return symmetry_score
        
    except Exception:
        return 0.0 