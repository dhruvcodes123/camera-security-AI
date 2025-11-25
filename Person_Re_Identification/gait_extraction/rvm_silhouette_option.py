#!/usr/bin/env python3
"""
RVM Silhouette Option for Gait Extraction
RVM integration for silhouette extraction.
This module provides RVM-based silhouette extraction.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import RVM module
try:
    from rvm_module import load_rvm_model, RVMSilhouetteExtractor
    RVM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RVM module not available: {e}")
    RVM_AVAILABLE = False

class RVMSilhouetteOption:
    """
    RVM silhouette extractor for gait analysis.
    """
    
    def __init__(self, use_rvm: bool = True):
        """
        Initialize the RVM silhouette option.
        
        Args:
            use_rvm: Whether to use RVM for silhouette extraction (default: True)
        """
        self.use_rvm = use_rvm and RVM_AVAILABLE
        self.rvm_extractor = None
        
        if self.use_rvm:
            self._initialize_rvm()
        else:
            raise RuntimeError("RVM is required for silhouette extraction")
        
        print(f"🔄 RVM Silhouette Option: {'ENABLED' if self.use_rvm else 'DISABLED'}")
    
    def _initialize_rvm(self):
        """Initialize RVM extractor."""
        try:
            self.rvm_extractor = load_rvm_model()
            print("✅ RVM silhouette extractor ready for gait analysis")
        except Exception as e:
            print(f"❌ Failed to initialize RVM: {e}")
            self.use_rvm = False
    
    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """
        Extract silhouette using RVM.
        This method signature matches the existing interface.
        
        Args:
            image: Input RGB/BGR image
            
        Returns:
            Binary silhouette mask
        """
        if self.use_rvm and self.rvm_extractor is not None:
            return self._extract_with_rvm(image)
        else:
            raise RuntimeError("RVM extractor not available")
    
    def _extract_with_rvm(self, image: np.ndarray) -> np.ndarray:
        """Extract silhouette using RVM."""
        return self.rvm_extractor.extract_silhouette(image)
    
    def extract_silhouettes_for_gait_cycle(self, 
                                         image_sequence: List[np.ndarray],
                                         person_id: Optional[int] = None,
                                         save_to_folder: bool = False) -> List[np.ndarray]:
        """
        Extract silhouettes for gait cycle analysis.
        
        Args:
            image_sequence: List of RGB images
            person_id: Person ID (for saving)
            save_to_folder: Whether to save silhouettes to gait_cycles folder
            
        Returns:
            List of binary silhouette masks
        """
        silhouettes = []
        
        if self.use_rvm and self.rvm_extractor is not None:
            # Reset recurrent states for new sequence
            self.rvm_extractor.reset_recurrent_states()
        
        for i, image in enumerate(image_sequence):
            silhouette = self.extract_silhouette(image)
            if silhouette is not None:
                silhouettes.append(silhouette)
                
                # Save to gait_cycles folder if requested
                if save_to_folder and person_id is not None:
                    self._save_silhouette(silhouette, person_id, i)
            
            if (i + 1) % 10 == 0:
                print(f"   🔄 Processed {i + 1}/{len(image_sequence)} frames with RVM")
        
        print(f"✅ Extracted {len(silhouettes)} silhouettes using RVM")
        return silhouettes
    
    def _save_silhouette(self, silhouette: np.ndarray, person_id: int, frame_idx: int):
        """Save silhouette to gait_cycles folder."""
        output_dir = "gait_cycles"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"person_{person_id:03d}_frame_{frame_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, silhouette)
    
    def is_rvm_enabled(self) -> bool:
        """Check if RVM is currently enabled."""
        return self.use_rvm
    
    def get_method_name(self) -> str:
        """Get the current silhouette extraction method name."""
        return "RVM"


# Factory function for easy integration
def create_rvm_silhouette_extractor(use_rvm: bool = True) -> RVMSilhouetteOption:
    """
    Factory function to create RVM silhouette extractor.
    
    Args:
        use_rvm: Whether to enable RVM (default: True)
        
    Returns:
        RVMSilhouetteOption instance
    """
    return RVMSilhouetteOption(use_rvm=use_rvm)

# Convenience functions that match existing interface
def create_rvm_p3mnet_silhouette_extractor():
    """
    Create RVM-enabled silhouette extractor that matches existing interface.
    This function can be used as a drop-in replacement.
    """
    return create_rvm_silhouette_extractor(use_rvm=True)

# Test function
def test_rvm_integration():
    """Test RVM integration."""
    print("🧪 Testing RVM Integration...")
    
    # Test with RVM enabled
    print("\n1. Testing with RVM ENABLED:")
    rvm_extractor = create_rvm_silhouette_extractor(use_rvm=True)
    
    # Test with dummy images
    dummy_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
    ]
    
    if rvm_extractor.is_rvm_enabled():
        try:
            silhouettes = rvm_extractor.extract_silhouettes_for_gait_cycle(dummy_images)
            print(f"✅ RVM test successful! Extracted {len(silhouettes)} silhouettes")
            return True
        except Exception as e:
            print(f"❌ RVM test failed: {e}")
            return False
    else:
        print("❌ RVM not enabled")
        return False

if __name__ == "__main__":
    test_rvm_integration() 