#!/usr/bin/env python3
"""
RVM (Robust Video Matting) Module for Silhouette Extraction
A completely separate, modular implementation for fast silhouette extraction.
This module is isolated and can be toggled without affecting existing logic.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Add RobustVideoMatting to path
rvm_path = Path(__file__).parent / "RobustVideoMatting"
sys.path.append(str(rvm_path))

# Import RVM model
try:
    from model import MattingNetwork
    RVM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RVM model not available: {e}")
    RVM_AVAILABLE = False

class RVMSilhouetteExtractor:
    """
    RVM-based silhouette extractor for fast person segmentation.
    Completely separate from existing P3MNet-HQ logic.
    """
    
    def __init__(self, 
                 model_path: str = "RobustVideoMatting/pretrained/rvm_mobilenetv3.pth",
                 device: str = "auto"):
        """
        Initialize the RVM silhouette extractor.
        
        Args:
            model_path: Path to the pretrained RVM model
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.rec = None  # Recurrent states for temporal consistency
        self.downsample_ratio = 0.25  # Downsample ratio for speed
        
        # Load model
        self._load_model()
        
        # Initialize recurrent states
        self.reset_recurrent_states()
        
        print(f"✅ RVM Silhouette Extractor initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the pretrained RVM model."""
        if not RVM_AVAILABLE:
            raise ImportError("RVM model not available. Check installation.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RVM model not found at {self.model_path}")
        
        try:
            # Load model
            self.model = MattingNetwork('mobilenetv3').eval()
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model = self.model.to(self.device)
            
            print(f"✅ RVM MobileNetV3 model loaded from {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load RVM model: {e}")
    
    def reset_recurrent_states(self):
        """Reset recurrent states for new video sequence."""
        self.rec = [None] * 4
    
    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """
        Extract binary silhouette from a single frame using RVM.
        
        Args:
            image: Input RGB image (H, W, 3) in BGR format (OpenCV)
            
        Returns:
            Binary silhouette mask (H, W) with values 0 or 255
        """
        if image is None or image.size == 0:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with RVM
            alpha_matte = self._process_with_rvm(rgb_image)
            
            # Convert to binary silhouette
            binary_silhouette = self._alpha_to_binary(alpha_matte)
            
            return binary_silhouette
            
        except Exception as e:
            print(f"❌ RVM silhouette extraction failed: {e}")
            return None
    
    def _process_with_rvm(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Process image with RVM to extract alpha matte.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            
        Returns:
            Alpha matte (H, W) with values 0-1
        """
        # Original dimensions
        original_h, original_w = rgb_image.shape[:2]
        
        # FIXED RESIZE TO 256x256 - Eliminates tensor mismatch errors
        target_size = (256, 256)  # Fixed size for stability
        target_h, target_w = target_size
        
        # Resize image to fixed size
        resized_image = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
        src = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        src = src.unsqueeze(0).to(self.device)
        
        # Ensure tensor has correct format for RVM (RGB channels)
        if src.shape[1] != 3:
            # If not 3 channels, duplicate the channel to make RGB
            if src.shape[1] == 1:
                src = src.repeat(1, 3, 1, 1)
            else:
                # Take only first 3 channels if more than 3
                src = src[:, :3, :, :]
        
        # Inference with robust error handling
        with torch.no_grad():
            try:
                if self.rec[0] is None:
                    # First frame - no recurrent states
                    fgr, pha, *self.rec = self.model(src, *([None] * 4), 0.25)
                else:
                    # Use previous recurrent states
                    fgr, pha, *self.rec = self.model(src, *self.rec, 0.25)
            except Exception as e:
                # Reset recurrent states and try again
                print(f"⚠️ RVM tensor mismatch detected, resetting states: {e}")
                self.rec = [None] * 4
                try:
                    fgr, pha, *self.rec = self.model(src, *([None] * 4), 0.25)
                except Exception as e2:
                    # Final fallback without downsample ratio
                    print(f"⚠️ Trying without downsample ratio")
                    fgr, pha, *self.rec = self.model(src, *([None] * 4))
        
        # Get alpha matte
        alpha_matte = pha[0, 0].cpu().numpy()
        
        # Resize back to original size
        alpha_matte = cv2.resize(alpha_matte, (original_w, original_h), 
                               interpolation=cv2.INTER_LINEAR)
        
        return alpha_matte
    
    def _alpha_to_binary(self, alpha_matte: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert alpha matte to binary silhouette.
        
        Args:
            alpha_matte: Alpha matte (H, W) with values 0-1
            threshold: Threshold for binarization
            
        Returns:
            Binary silhouette (H, W) with values 0 or 255
        """
        # Threshold the alpha matte
        binary_mask = (alpha_matte > threshold).astype(np.uint8) * 255
        
        # Optional: Apply morphological operations for cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    def extract_and_save_silhouettes(self, 
                                   image_sequence: list, 
                                   person_id: int,
                                   output_dir: str = "gait_cycles",
                                   prefix: str = "person") -> list:
        """
        Extract silhouettes from image sequence and save to gait_cycles folder.
        
        Args:
            image_sequence: List of RGB images
            person_id: Person ID for filename
            output_dir: Output directory for silhouettes
            prefix: Filename prefix
            
        Returns:
            List of saved filenames
        """
        if not image_sequence:
            return []
        
        # Reset recurrent states for new sequence
        self.reset_recurrent_states()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for frame_idx, image in enumerate(image_sequence):
            # Extract silhouette
            silhouette = self.extract_silhouette(image)
            
            if silhouette is not None:
                # Generate filename
                filename = f"{prefix}_{person_id:03d}_frame_{frame_idx:04d}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save silhouette
                cv2.imwrite(filepath, silhouette)
                saved_files.append(filename)
                
                if (frame_idx + 1) % 10 == 0:
                    print(f"   📁 Saved {frame_idx + 1}/{len(image_sequence)} RVM silhouettes")
        
        print(f"✅ RVM extracted and saved {len(saved_files)} silhouettes to {output_dir}/")
        return saved_files


# Global RVM instance (singleton pattern)
_rvm_extractor = None

def load_rvm_model(model_path: str = "RobustVideoMatting/pretrained/rvm_mobilenetv3.pth") -> RVMSilhouetteExtractor:
    """
    Load RVM model (singleton pattern).
    
    Args:
        model_path: Path to the pretrained RVM model
        
    Returns:
        RVMSilhouetteExtractor instance
    """
    global _rvm_extractor
    
    if _rvm_extractor is None:
        _rvm_extractor = RVMSilhouetteExtractor(model_path)
    
    return _rvm_extractor

def process_with_rvm(rvm_model: RVMSilhouetteExtractor, 
                     person_frame: np.ndarray) -> np.ndarray:
    """
    Process a single frame with RVM to get binary silhouette.
    
    Args:
        rvm_model: RVM model instance
        person_frame: Input RGB/BGR image
        
    Returns:
        Binary silhouette mask (0, 255)
    """
    return rvm_model.extract_silhouette(person_frame)

def extract_rvm_silhouettes_for_person(person_id: int, 
                                      image_sequence: list,
                                      output_dir: str = "gait_cycles") -> list:
    """
    Complete RVM silhouette extraction pipeline for a person.
    
    Args:
        person_id: Person ID
        image_sequence: List of RGB images
        output_dir: Output directory
        
    Returns:
        List of saved filenames
    """
    # Load RVM model
    rvm_model = load_rvm_model()
    
    # Extract and save silhouettes
    saved_files = rvm_model.extract_and_save_silhouettes(
        image_sequence, person_id, output_dir
    )
    
    return saved_files

# Test function
def test_rvm_module():
    """Test the RVM module with a sample image."""
    print("🧪 Testing RVM Module...")
    
    try:
        # Load RVM model
        rvm_model = load_rvm_model()
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract silhouette
        silhouette = rvm_model.extract_silhouette(dummy_image)
        
        if silhouette is not None:
            print(f"✅ RVM test successful! Silhouette shape: {silhouette.shape}")
            return True
        else:
            print("❌ RVM test failed - no silhouette returned")
            return False
            
    except Exception as e:
        print(f"❌ RVM test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    test_rvm_module() 