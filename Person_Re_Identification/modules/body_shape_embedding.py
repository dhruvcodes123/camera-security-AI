#!/usr/bin/env python3
"""
Body Shape Embedding Module for Person Re-Identification

This module extracts structural body shape features from person images using pose estimation
and generates normalized embeddings suitable for re-identification tasks.

Dependencies:
- ultralytics (for YOLOv10-pose-M)
- numpy
- cv2
- scipy
- sklearn (for normalization and PCA)

Usage:
    from modules.body_shape_embedding import get_body_shape_embedding
    
    # Extract body shape embedding from image
    embedding = get_body_shape_embedding(image)
    print(f"Body shape embedding shape: {embedding.shape}")
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import math
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

# Try to import ultralytics for pose estimation
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("ultralytics not available. Pose estimation will be disabled.")

# Try to import silhouette extraction
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gait_extraction.rvm_only_silhouette import create_rvm_only_silhouette_extractor
    SILHOUETTE_AVAILABLE = True
except ImportError:
    SILHOUETTE_AVAILABLE = False
    warnings.warn("RVM silhouette extraction not available. Silhouette features will be disabled.")

# COCO keypoint mapping for YOLO pose
COCO_KEYPOINTS = {
    0: 'nose',
    1: 'left_eye', 2: 'right_eye',
    3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

# Required keypoints for body shape analysis
REQUIRED_KEYPOINTS = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Optional keypoints (neck is approximated)
OPTIONAL_KEYPOINTS = {
    'nose': 0,  # Used to approximate neck position
}

class PoseEstimator:
    """Pose estimation using YOLOv10-pose-M"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the pose estimation model"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        try:
            # Load YOLOv8-pose model (using medium for better accuracy)
            self.model = YOLO('yolov8m-pose.pt')  # Better accuracy than nano
            self.initialized = True
            print("✅ Pose estimation model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load pose estimation model: {e}")
            self.initialized = False
    
    def extract_keypoints(self, image: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extract pose keypoints from image
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary mapping keypoint names to (x, y) coordinates
        """
        if not self.initialized:
            self.initialize()
        
        if not self.initialized:
            return None
        
        try:
            # Run pose estimation
            results = self.model(image, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            # Get the first person detected
            result = results[0]
            
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                return None
            
            keypoints = result.keypoints.data[0]  # Shape: (17, 3) - x, y, confidence
            
            # Convert to dictionary format
            keypoint_dict = {}
            
            # Extract required keypoints
            for name, idx in REQUIRED_KEYPOINTS.items():
                if idx < len(keypoints):
                    x, y, conf = keypoints[idx]
                    if conf > 0.2:  # Lowered confidence threshold for better detection
                        keypoint_dict[name] = (float(x), float(y))
            
            # Extract optional keypoints
            for name, idx in OPTIONAL_KEYPOINTS.items():
                if idx < len(keypoints):
                    x, y, conf = keypoints[idx]
                    if conf > 0.2:  # Lowered confidence threshold
                        keypoint_dict[name] = (float(x), float(y))
            
            # Approximate neck position if not available
            if 'nose' in keypoint_dict and 'left_shoulder' in keypoint_dict and 'right_shoulder' in keypoint_dict:
                nose_x, nose_y = keypoint_dict['nose']
                left_shoulder_x, left_shoulder_y = keypoint_dict['left_shoulder']
                right_shoulder_x, right_shoulder_y = keypoint_dict['right_shoulder']
                
                # Neck is between shoulders, slightly above nose
                neck_x = (left_shoulder_x + right_shoulder_x) / 2
                neck_y = min(nose_y, (left_shoulder_y + right_shoulder_y) / 2) - 10
                keypoint_dict['neck'] = (neck_x, neck_y)
            
            return keypoint_dict if len(keypoint_dict) >= 6 else None  # Reduced minimum keypoints for better detection
            
        except Exception as e:
            print(f"❌ Error in pose estimation: {e}")
            return None

class SilhouetteExtractor:
    """Silhouette extraction using RVM-HQ"""
    
    def __init__(self):
        self.extractor = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the silhouette extraction model"""
        if not SILHOUETTE_AVAILABLE:
            raise ImportError("RVM silhouette extraction not available")
        
        try:
            self.extractor = create_rvm_only_silhouette_extractor()
            self.initialized = True
            print("✅ Silhouette extraction model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load silhouette extraction model: {e}")
            self.initialized = False
    
    def extract_silhouette(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract silhouette mask from image
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Binary silhouette mask (H, W) with values 0 or 255
        """
        if not self.initialized:
            self.initialize()
        
        if not self.initialized:
            return None
        
        try:
            silhouette = self.extractor.extract_silhouette(image)
            return silhouette
        except Exception as e:
            print(f"❌ Error in silhouette extraction: {e}")
            return None

def compute_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points"""
    return euclidean(point1, point2)

def compute_body_shape_features(keypoints: Dict[str, Tuple[float, float]], 
                              silhouette_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute structural body shape features from keypoints
    
    Args:
        keypoints: Dictionary mapping keypoint names to (x, y) coordinates
        silhouette_mask: Optional binary silhouette mask
        
    Returns:
        Feature array with body measurements
    """
    features = []
    
    # 1. Shoulder width
    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        shoulder_width = compute_distance(keypoints['left_shoulder'], keypoints['right_shoulder'])
        features.append(shoulder_width)
    else:
        features.append(0.0)
    
    # 2. Torso height (neck to hip midpoint)
    if 'neck' in keypoints and 'left_hip' in keypoints and 'right_hip' in keypoints:
        neck_x, neck_y = keypoints['neck']
        left_hip_x, left_hip_y = keypoints['left_hip']
        right_hip_x, right_hip_y = keypoints['right_hip']
        
        hip_midpoint_x = (left_hip_x + right_hip_x) / 2
        hip_midpoint_y = (left_hip_y + right_hip_y) / 2
        
        torso_height = compute_distance((neck_x, neck_y), (hip_midpoint_x, hip_midpoint_y))
        features.append(torso_height)
    else:
        features.append(0.0)
    
    # 3. Left arm length (shoulder -> elbow -> wrist)
    left_arm_length = 0.0
    if all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        left_upper_arm = compute_distance(keypoints['left_shoulder'], keypoints['left_elbow'])
        left_forearm = compute_distance(keypoints['left_elbow'], keypoints['left_wrist'])
        left_arm_length = left_upper_arm + left_forearm
    features.append(left_arm_length)
    
    # 4. Right arm length (shoulder -> elbow -> wrist)
    right_arm_length = 0.0
    if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        right_upper_arm = compute_distance(keypoints['right_shoulder'], keypoints['right_elbow'])
        right_forearm = compute_distance(keypoints['right_elbow'], keypoints['right_wrist'])
        right_arm_length = right_upper_arm + right_forearm
    features.append(right_arm_length)
    
    # 5. Left leg length (hip -> knee -> ankle)
    left_leg_length = 0.0
    if all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
        left_thigh = compute_distance(keypoints['left_hip'], keypoints['left_knee'])
        left_shin = compute_distance(keypoints['left_knee'], keypoints['left_ankle'])
        left_leg_length = left_thigh + left_shin
    features.append(left_leg_length)
    
    # 6. Right leg length (hip -> knee -> ankle)
    right_leg_length = 0.0
    if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
        right_thigh = compute_distance(keypoints['right_hip'], keypoints['right_knee'])
        right_shin = compute_distance(keypoints['right_knee'], keypoints['right_ankle'])
        right_leg_length = right_thigh + right_shin
    features.append(right_leg_length)
    
    # 7. Body height (neck to ankle average)
    body_height = 0.0
    if 'neck' in keypoints:
        neck_x, neck_y = keypoints['neck']
        ankle_distances = []
        
        if 'left_ankle' in keypoints:
            ankle_distances.append(compute_distance((neck_x, neck_y), keypoints['left_ankle']))
        if 'right_ankle' in keypoints:
            ankle_distances.append(compute_distance((neck_x, neck_y), keypoints['right_ankle']))
        
        if ankle_distances:
            body_height = np.mean(ankle_distances)
    features.append(body_height)
    
    # 8. Hip width
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        hip_width = compute_distance(keypoints['left_hip'], keypoints['right_hip'])
        features.append(hip_width)
    else:
        features.append(0.0)
    
    # 9. Shoulder to hip ratio
    if features[0] > 0 and features[7] > 0:  # shoulder_width and hip_width
        shoulder_hip_ratio = features[0] / features[7]
        features.append(shoulder_hip_ratio)
    else:
        features.append(1.0)  # Default ratio
    
    # 10. Arm to leg ratio (average)
    if features[2] > 0 and features[4] > 0 and features[3] > 0 and features[5] > 0:
        left_arm_leg_ratio = features[2] / features[4]
        right_arm_leg_ratio = features[3] / features[5]
        avg_arm_leg_ratio = (left_arm_leg_ratio + right_arm_leg_ratio) / 2
        features.append(avg_arm_leg_ratio)
    else:
        features.append(1.0)  # Default ratio
    
    # 11. Torso to height ratio
    if features[1] > 0 and features[6] > 0:  # torso_height and body_height
        torso_height_ratio = features[1] / features[6]
        features.append(torso_height_ratio)
    else:
        features.append(0.4)  # Default ratio
    
    # 12. Left-right symmetry (arm length difference)
    if features[2] > 0 and features[3] > 0:
        arm_symmetry = abs(features[2] - features[3]) / max(features[2], features[3])
        features.append(arm_symmetry)
    else:
        features.append(0.0)
    
    # 13. Left-right symmetry (leg length difference)
    if features[4] > 0 and features[5] > 0:
        leg_symmetry = abs(features[4] - features[5]) / max(features[4], features[5])
        features.append(leg_symmetry)
    else:
        features.append(0.0)
    
    # 14. Silhouette-based features (if available)
    if silhouette_mask is not None:
        # Silhouette area ratio
        total_pixels = silhouette_mask.size
        person_pixels = np.count_nonzero(silhouette_mask)
        silhouette_area_ratio = person_pixels / total_pixels if total_pixels > 0 else 0.0
        features.append(silhouette_area_ratio)
        
        # Silhouette aspect ratio
        if person_pixels > 0:
            person_coords = np.where(silhouette_mask > 0)
            if len(person_coords[0]) > 0 and len(person_coords[1]) > 0:
                height = np.max(person_coords[0]) - np.min(person_coords[0])
                width = np.max(person_coords[1]) - np.min(person_coords[1])
                aspect_ratio = height / width if width > 0 else 1.0
                features.append(aspect_ratio)
            else:
                features.append(1.0)
        else:
            features.append(1.0)
    else:
        features.extend([0.0, 1.0])  # Default values
    
    return np.array(features, dtype=np.float32)

def normalize_and_embed(features: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """
    Normalize features and project to target dimension
    
    Args:
        features: Raw feature array
        target_dim: Target embedding dimension
        
    Returns:
        Normalized embedding vector (L2 normalized to unit length)
    """
    # Check if we have any non-zero features
    if np.all(features == 0):
        # If all features are zero, return zero embedding
        # This ensures consistent zero embeddings for failed pose detection
        zero_embedding = np.zeros(target_dim, dtype=np.float32)
        return zero_embedding
    
    # If we have fewer features than target dimension, pad with zeros
    if len(features) < target_dim:
        embedding = np.zeros(target_dim, dtype=np.float32)
        embedding[:len(features)] = features
    else:
        # If we have more features, use PCA to reduce dimension
        if len(features) > target_dim:
            pca = PCA(n_components=target_dim)
            embedding = pca.fit_transform(features.reshape(1, -1)).flatten()
        else:
            embedding = features
    
    # Apply L2 normalization to unit length (consistent with other features)
    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm > 0:
        embedding = embedding / embedding_norm
    
    return embedding.astype(np.float32)

# Global instances
pose_estimator = PoseEstimator()
silhouette_extractor = SilhouetteExtractor()

def extract_pose_keypoints(image: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Extract pose keypoints from image
    
    Args:
        image: Input image as numpy array (H, W, C)
        
    Returns:
        Dictionary mapping keypoint names to (x, y) coordinates
    """
    return pose_estimator.extract_keypoints(image)

def extract_silhouette_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract silhouette mask from image
    
    Args:
        image: Input image as numpy array (H, W, C)
        
    Returns:
        Binary silhouette mask (H, W) with values 0 or 255
    """
    return silhouette_extractor.extract_silhouette(image)

def get_body_shape_embedding(image: np.ndarray, target_dim: int = 256) -> np.ndarray:  # Updated to 256D
    """
    Extract body shape embedding from image
    
    Args:
        image: Input image as numpy array (H, W, C)
        target_dim: Target embedding dimension (default: 256)  # Updated default
        
    Returns:
        Body shape embedding vector of shape (target_dim,)
    """
    try:
        # Step 1: Extract pose keypoints
        keypoints = extract_pose_keypoints(image)
        
        # Step 2: Extract silhouette (optional)
        silhouette_mask = None
        if SILHOUETTE_AVAILABLE:
            try:
                silhouette_mask = extract_silhouette_mask(image)
            except Exception as e:
                print(f"⚠️ Silhouette extraction failed: {e}")
        
        # Step 3: Handle different keypoint scenarios
        if keypoints is None or len(keypoints) < 4:
            print("⚠️ Insufficient pose keypoints detected - creating zero embeddings")
            # Create zero embeddings when pose detection fails
            zero_embedding = np.zeros(target_dim, dtype=np.float32)
            print(f"✅ Zero body shape embedding created: {zero_embedding.shape}")
            return zero_embedding
        else:
            # Compute body shape features from keypoints
            features = compute_body_shape_features(keypoints, silhouette_mask)
            
            # Step 4: Normalize and embed
            embedding = normalize_and_embed(features, target_dim)
            
            print(f"✅ Body shape embedding extracted: {embedding.shape}")
            return embedding
        
    except Exception as e:
        print(f"❌ Error in body shape embedding: {e}")
        return np.zeros(target_dim, dtype=np.float32)

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample image
    import matplotlib.pyplot as plt
    
    # Create a test image (you can replace this with actual image loading)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("🧪 Testing Body Shape Embedding Module")
    print("=" * 50)
    
    # Test the embedding extraction
    embedding = get_body_shape_embedding(test_image)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding values: {embedding[:10]}...")  # Show first 10 values
    
    print("\n✅ Body shape embedding module is ready for integration!") 