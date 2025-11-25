"""
Helper functions for the Hybrid Person Tracking System.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union


def face_alignment(img: np.ndarray, 
                   landmarks: np.ndarray, 
                   image_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Align face image using 5-point facial landmarks.
    
    Args:
        img: Original image
        landmarks: 5-point facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        image_size: Output image size
        
    Returns:
        Aligned face image
    """
    if landmarks is None or len(landmarks) < 5:
        # If no landmarks provided, just resize the image
        if img.shape[0] > 0 and img.shape[1] > 0:
            return cv2.resize(img, image_size)
        else:
            return None
            
    try:
        # Standard 5 facial landmarks positions
        std_landmarks = np.array([
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041]   # right mouth
        ], dtype=np.float32)
        
        # Scale standard landmarks to target size
        std_landmarks[:, 0] *= (image_size[0] / 112.0)
        std_landmarks[:, 1] *= (image_size[1] / 112.0)
        
        # Convert input landmarks to float32
        landmarks = landmarks.astype(np.float32)
        
        # Calculate transformation matrix
        transform = cv2.estimateAffinePartial2D(landmarks, std_landmarks)[0]
        
        # Apply transformation
        warped = cv2.warpAffine(img, transform, image_size, borderValue=0.0)
        
        return warped
        
    except Exception as e:
        # Fallback to simple resize on failure
        if img.shape[0] > 0 and img.shape[1] > 0:
            return cv2.resize(img, image_size)
        else:
            return None


def draw_bbox(frame: np.ndarray, 
              bbox: List[float], 
              color: Tuple[int, int, int] = (0, 255, 0), 
              thickness: int = 2) -> None:
    """
    Draw a bounding box on a frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: RGB color tuple
        thickness: Line thickness
    """
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_bbox_info(frame: np.ndarray, 
                   bbox: List[float], 
                   confidence: float, 
                   label_text: str,
                   color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Draw a bounding box with label and confidence.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
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


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance prediction to bounding box.
    
    Args:
        points: Anchor points [N, 2], typically the grid cell centers
        distance: Distance from anchor points to four boundaries [N, 4]
        
    Returns:
        np.ndarray: Decoded bounding boxes [N, 4]
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Decode distance prediction to landmarks.
    
    Args:
        points: Anchor points [N, 2], typically the grid cell centers
        distance: Distance from anchor points to landmarks [N, K*2]
        
    Returns:
        np.ndarray: Decoded landmarks [N, K*2]
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1) 