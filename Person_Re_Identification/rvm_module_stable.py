#!/usr/bin/env python3
"""
Stable RVM Module - Eliminates Tensor Mismatch Errors
Uses stateless processing to avoid recurrent state corruption.
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

class StableRVMSilhouetteExtractor:
    """Stable RVM extractor that eliminates tensor mismatch errors."""
    
    def __init__(self, 
                 model_path: str = "RobustVideoMatting/pretrained/rvm_mobilenetv3.pth",
                 device: str = "auto"):
        """Initialize stable RVM extractor."""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        
        # Stable processing parameters
        self.use_stateless_mode = True  # Always use stateless for stability
        self.fixed_input_size = (256, 256)  # Fixed size to avoid dimension issues
        self.quality_threshold = 0.3
        self.morphology_enabled = True
        
        # Quality control
        self.min_person_area = 0.02
        self.max_person_area = 0.65
        
        self._load_model()
        print(f"✅ Stable RVM Silhouette Extractor initialized on {self.device}")
        print(f"   Mode: {'Stateless (No Recurrent States)' if self.use_stateless_mode else 'Stateful'}")
        print(f"   Fixed input size: {self.fixed_input_size}")
    
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
            print(f"✅ Stable RVM MobileNetV3 model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load RVM model: {e}")
    
    def reset_recurrent_states(self):
        """Reset method for compatibility - not needed in stateless mode."""
        if not self.use_stateless_mode:
            print("🔄 Recurrent states reset (stateless mode active)")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image with enhanced quality."""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma = 1.2
        img_float = np.power(img_float, gamma)
        
        # Convert back to uint8 for CLAHE
        img_uint8 = (img_float * 255).astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _process_with_rvm_stateless(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Process with RVM in stateless mode - eliminates tensor mismatch errors."""
        original_h, original_w = rgb_image.shape[:2]
        
        # Always use fixed input size for stability
        target_h, target_w = self.fixed_input_size
        
        # Resize to fixed size
        resized_image = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
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
                # ALWAYS use stateless mode - no recurrent states
                # This completely eliminates tensor mismatch errors
                fgr, pha, *_ = self.model(src, *([None] * 4), 0.25)
                
        except Exception as e:
            print(f"⚠️ RVM processing failed: {str(e)[:100]}")
            return None
        
        # Extract alpha matte and resize back to original size
        alpha_matte = pha.squeeze().cpu().numpy()
        alpha_matte = cv2.resize(alpha_matte, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        return alpha_matte
    
    def _enhanced_thresholding(self, alpha_matte: np.ndarray) -> np.ndarray:
        """Apply enhanced thresholding with quality control."""
        if alpha_matte is None:
            return None
        
        # Convert to uint8
        alpha_uint8 = (alpha_matte * 255).astype(np.uint8)
        
        # Try Otsu's method first
        try:
            otsu_thresh, otsu_binary = cv2.threshold(alpha_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            person_area_ratio = np.count_nonzero(otsu_binary) / otsu_binary.size
            
            # Check if Otsu result is reasonable
            if self.min_person_area <= person_area_ratio <= self.max_person_area:
                binary_mask = otsu_binary
                # print(f"✅ Otsu threshold: {otsu_thresh:.0f}, area: {person_area_ratio:.3f}")
            else:
                # Fallback to fixed threshold
                binary_mask = (alpha_matte > self.quality_threshold).astype(np.uint8) * 255
                person_area_ratio = np.count_nonzero(binary_mask) / binary_mask.size
                # print(f"🔄 Fixed threshold: {self.quality_threshold}, area: {person_area_ratio:.3f}")
                
        except Exception as e:
            # Final fallback
            binary_mask = (alpha_matte > self.quality_threshold).astype(np.uint8) * 255
            person_area_ratio = np.count_nonzero(binary_mask) / binary_mask.size
            print(f"⚠️ Fallback threshold used, area: {person_area_ratio:.3f}")
        
        # Apply morphological cleanup
        if self.morphology_enabled:
            binary_mask = self._apply_morphology(binary_mask)
        
        # Final quality warning
        final_area_ratio = np.count_nonzero(binary_mask) / binary_mask.size
        if final_area_ratio > self.max_person_area:
            print(f"⚠️ Large silhouette (area: {final_area_ratio:.3f}), may be noisy")
        elif final_area_ratio < self.min_person_area:
            # print(f"⚠️ Small silhouette (area: {final_area_ratio:.3f}), may be poor quality")
            pass
        return binary_mask
    
    def _apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations for cleanup."""
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
        
        # Smooth edges
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_smooth)
        
        return cleaned
    
    def extract_silhouette(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract silhouette using stable stateless processing."""
        if image is None or image.size == 0:
            return None
        
        try:
            # Preprocess for better quality
            processed_image = self._preprocess_image(image)
            
            # Convert BGR to RGB if needed
            if len(processed_image.shape) == 3:
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = processed_image
            
            # Process with stable RVM (stateless)
            alpha_matte = self._process_with_rvm_stateless(rgb_image)
            
            if alpha_matte is None:
                return None
            
            # Apply enhanced thresholding
            binary_silhouette = self._enhanced_thresholding(alpha_matte)
            
            return binary_silhouette
            
        except Exception as e:
            print(f"❌ Stable RVM extraction failed: {e}")
            return None

def load_stable_rvm_model() -> StableRVMSilhouetteExtractor:
    """Load stable RVM model."""
    return StableRVMSilhouetteExtractor()

# Test function
def test_stable_rvm():
    """Test stable RVM module."""
    print("🧪 Testing Stable RVM Module...")
    
    try:
        extractor = load_stable_rvm_model()
        
        # Test with multiple different sized images to ensure stability
        test_sizes = [(480, 640), (360, 480), (720, 1280), (240, 320)]
        
        for i, (h, w) in enumerate(test_sizes):
            print(f"\n📏 Testing size {h}x{w}...")
            test_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            
            # Create a mock person region for better testing
            test_image[h//4:3*h//4, w//3:2*w//3] = [255, 255, 255]
            
            silhouette = extractor.extract_silhouette(test_image)
            
            if silhouette is not None:
                area_ratio = np.count_nonzero(silhouette) / silhouette.size
                print(f"✅ Size {h}x{w} successful! Output: {silhouette.shape}, area: {area_ratio:.3f}")
            else:
                print(f"❌ Size {h}x{w} failed")
                return False
        
        print("\n✅ All size tests passed - stable processing confirmed!")
        return True
        
    except Exception as e:
        print(f"❌ Stable RVM test failed: {e}")
        return False

if __name__ == "__main__":
    test_stable_rvm() 