#!/usr/bin/env python3
"""
Enhanced RVM Module with Improved Quality and Error Handling
Fixes tensor mismatch issues and improves silhouette quality for gait analysis.
"""

import cv2
import numpy as np
import torch
import os
from typing import Optional

# Import RVM model
try:
    import sys
    sys.path.append('RobustVideoMatting')
    from model import MattingNetwork
    RVM_AVAILABLE = True
except ImportError:
    print("❌ RVM model not available")
    RVM_AVAILABLE = False

class EnhancedRVMSilhouetteExtractor:
    """Enhanced RVM extractor with better quality and stability."""
    
    def __init__(self, 
                 model_path: str = "RobustVideoMatting/pretrained/rvm_mobilenetv3.pth",
                 device: str = "auto"):
        """Initialize enhanced RVM extractor."""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.rec = None
        self.downsample_ratio = 0.5  # Increased for better quality
        
        # Enhanced quality parameters
        self.min_person_area = 0.02  # Minimum person area ratio
        self.max_person_area = 0.6   # Maximum person area ratio
        self.quality_threshold = 0.3  # Higher threshold for better quality
        self.morphology_enabled = True
        self.preprocessing_enabled = True
        
        # Stability tracking
        self.last_successful_size = None
        self.consecutive_failures = 0
        self.max_failures = 3
        
        self._load_model()
        self.reset_recurrent_states()
        
        print(f"✅ Enhanced RVM Silhouette Extractor initialized on {self.device}")
        print(f"   Quality threshold: {self.quality_threshold}")
        print(f"   Downsample ratio: {self.downsample_ratio}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the RVM model."""
        if not RVM_AVAILABLE:
            raise ImportError("RVM model not available")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RVM model not found at {self.model_path}")
        
        try:
            self.model = MattingNetwork('mobilenetv3').eval()
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            print(f"✅ Enhanced RVM MobileNetV3 model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load RVM model: {e}")
    
    def reset_recurrent_states(self):
        """Reset recurrent states."""
        self.rec = [None] * 4
        self.consecutive_failures = 0
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better silhouette quality."""
        if not self.preprocessing_enabled:
            return image
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction for better contrast
        gamma = 1.2
        img_float = np.power(img_float, gamma)
        
        # Enhance contrast using CLAHE on each channel
        lab = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _process_with_rvm(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Process image with RVM using enhanced error handling and fixed resizing.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            
        Returns:
            Alpha matte (H, W) with values 0-1, or None if failed
        """
        original_h, original_w = rgb_image.shape[:2]
        current_size = (original_h, original_w)
        
        # FIXED RESIZE TO 256x256 - Eliminates tensor mismatch errors
        target_size = (256, 256)  # Fixed size for stability
        target_h, target_w = target_size
        
        # Resize image to fixed size
        resized_image = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor with proper normalization
        src = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        src = src.unsqueeze(0).to(self.device)
        
        # Ensure 3 channels
        if src.shape[1] != 3:
            if src.shape[1] == 1:
                src = src.repeat(1, 3, 1, 1)
            else:
                src = src[:, :3, :, :]
        
        try:
            with torch.no_grad():
                if self.rec[0] is None:
                    fgr, pha, *self.rec = self.model(src, *([None] * 4), 0.25)
                else:
                    fgr, pha, *self.rec = self.model(src, *self.rec, 0.25)
            
            # Track successful processing
            self.last_successful_size = current_size
            self.consecutive_failures = 0
            
        except Exception as e:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_failures:
                print(f"⚠️ Too many consecutive RVM failures, resetting completely")
                self.reset_recurrent_states()
            else:
                print(f"⚠️ RVM error (attempt {self.consecutive_failures}), resetting states: {str(e)[:100]}")
                self.rec = [None] * 4
            
            try:
                with torch.no_grad():
                    fgr, pha, *self.rec = self.model(src, *([None] * 4), 0.25)
            except Exception as e2:
                print(f"❌ RVM complete failure: {str(e2)[:100]}")
                return None
        
        # Extract alpha matte and resize back
        alpha_matte = pha.squeeze().cpu().numpy()
        alpha_matte = cv2.resize(alpha_matte, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return alpha_matte
    
    def _enhanced_alpha_to_binary(self, alpha_matte: np.ndarray) -> np.ndarray:
        """Enhanced conversion with better quality control."""
        if alpha_matte is None:
            return None
        
        # Use adaptive thresholding for better results
        alpha_uint8 = (alpha_matte * 255).astype(np.uint8)
        
        # Try Otsu's method first
        otsu_thresh, otsu_binary = cv2.threshold(alpha_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if Otsu result is reasonable
        person_area_ratio = np.count_nonzero(otsu_binary) / otsu_binary.size
        
        if self.min_person_area <= person_area_ratio <= self.max_person_area:
            binary_mask = otsu_binary
            print(f"✅ Using Otsu threshold: {otsu_thresh:.1f}, area ratio: {person_area_ratio:.3f}")
        else:
            # Fallback to quality threshold
            binary_mask = (alpha_matte > self.quality_threshold).astype(np.uint8) * 255
            person_area_ratio = np.count_nonzero(binary_mask) / binary_mask.size
            print(f"🔄 Using quality threshold: {self.quality_threshold}, area ratio: {person_area_ratio:.3f}")
        
        # Apply morphological operations for cleanup
        if self.morphology_enabled:
            binary_mask = self._apply_morphological_cleanup(binary_mask)
        
        # Final quality check
        final_area_ratio = np.count_nonzero(binary_mask) / binary_mask.size
        if final_area_ratio < self.min_person_area:
            print(f"⚠️ Silhouette too small (area: {final_area_ratio:.3f}), may be poor quality")
        elif final_area_ratio > self.max_person_area:
            print(f"⚠️ Silhouette too large (area: {final_area_ratio:.3f}), may be noisy")
        
        return binary_mask
    
    def _apply_morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations for cleaner silhouettes."""
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
        
        # Smooth edges
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_smooth)
        
        return cleaned
    
    def extract_silhouette(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract high-quality silhouette."""
        if image is None or image.size == 0:
            return None
        
        try:
            # Preprocess for better quality
            if self.preprocessing_enabled:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # Convert BGR to RGB if needed
            if len(processed_image.shape) == 3:
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = processed_image
            
            # Process with RVM
            alpha_matte = self._process_with_rvm(rgb_image)
            
            if alpha_matte is None:
                return None
            
            # Convert to binary with enhanced quality
            binary_silhouette = self._enhanced_alpha_to_binary(alpha_matte)
            
            return binary_silhouette
            
        except Exception as e:
            print(f"❌ Enhanced RVM silhouette extraction failed: {e}")
            return None

def load_enhanced_rvm_model() -> EnhancedRVMSilhouetteExtractor:
    """Load enhanced RVM model."""
    return EnhancedRVMSilhouetteExtractor()

# Test function
def test_enhanced_rvm():
    """Test enhanced RVM module."""
    print("🧪 Testing Enhanced RVM Module...")
    
    try:
        extractor = load_enhanced_rvm_model()
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create a mock person silhouette for testing
        test_image[200:400, 280:360] = [255, 255, 255]  # White rectangle as person
        
        silhouette = extractor.extract_silhouette(test_image)
        
        if silhouette is not None:
            print(f"✅ Enhanced RVM test successful! Silhouette shape: {silhouette.shape}")
            print(f"   Silhouette area ratio: {np.count_nonzero(silhouette) / silhouette.size:.3f}")
            return True
        else:
            print("❌ Enhanced RVM test failed - no silhouette returned")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced RVM test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_rvm() 