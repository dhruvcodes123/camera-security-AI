#!/usr/bin/env python3
"""
RVM-Only Silhouette Extractor
Exclusively uses RVM for silhouette extraction with no fallback methods.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import RVM modules
try:
    from rvm_module import load_rvm_model
    from rvm_module_stable import load_stable_rvm_model
    from rvm_module_enhanced import load_enhanced_rvm_model
    RVM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RVM modules not available: {e}")
    RVM_AVAILABLE = False

# Configuration for RVM module selection
USE_STABLE = True      # Use stable RVM module
USE_ENHANCED = False   # Use enhanced RVM module

class RVMOnlySilhouetteExtractor:
    """
    RVM-Only silhouette extractor that uses RVM exclusively.
    No fallback methods - uses RVM exclusively.
    """
    
    def __init__(self):
        """Initialize RVM-only silhouette extractor."""
        if not RVM_AVAILABLE:
            raise ImportError("RVM module is required for RVM-only mode")
        
        self.rvm_extractor = None
        self._initialize_rvm()
        
        print("🎯 RVM-ONLY MODE ACTIVATED")
        print("   - P3MNet-HQ: DISABLED")
        print("   - Fallbacks: DISABLED")
        print("   - Primary: RVM ONLY")
    
    def _initialize_rvm(self):
        """Initialize RVM extractor with the best available version."""
        try:
            if USE_STABLE:
                self.rvm_extractor = load_stable_rvm_model()
                print("✅ Stable RVM-only silhouette extractor initialized (no tensor errors)")
            elif USE_ENHANCED:
                self.rvm_extractor = load_enhanced_rvm_model()
                print("✅ Enhanced RVM-only silhouette extractor initialized successfully")
            else:
                self.rvm_extractor = load_rvm_model()
                print("✅ Standard RVM-only silhouette extractor initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RVM in RVM-only mode: {e}")
    
    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """
        Extract silhouette using RVM ONLY.
        No fallback methods available.
        
        Args:
            image: Input RGB/BGR image
            
        Returns:
            Binary silhouette mask from RVM
            
        Raises:
            RuntimeError: If RVM fails (no fallbacks available)
        """
        if self.rvm_extractor is None:
            raise RuntimeError("RVM extractor not initialized")
        
        try:
            silhouette = self.rvm_extractor.extract_silhouette(image)
            
            if silhouette is None:
                # Try resetting recurrent states and retry once
                print("⚠️ RVM returned None, resetting states and retrying...")
                self.rvm_extractor.reset_recurrent_states()
                silhouette = self.rvm_extractor.extract_silhouette(image)
                
                if silhouette is None:
                    raise RuntimeError("RVM returned None silhouette after retry")
            
            return silhouette
            
        except Exception as e:
            # Try one more time with state reset
            try:
                print(f"⚠️ RVM error occurred, attempting recovery: {e}")
                self.rvm_extractor.reset_recurrent_states()
                silhouette = self.rvm_extractor.extract_silhouette(image)
                if silhouette is not None:
                    print("✅ RVM recovery successful")
                    return silhouette
            except:
                pass
            
            raise RuntimeError(f"RVM-only silhouette extraction failed: {e}")
    
    def reset_recurrent_states(self):
        """Reset RVM recurrent states for new video sequence."""
        if self.rvm_extractor is not None:
            self.rvm_extractor.reset_recurrent_states()

def create_rvm_only_silhouette_extractor():
    """
    Factory function to create RVM-only silhouette extractor.
    This replaces the P3MNet-HQ extractor completely.
    
    Returns:
        RVMOnlySilhouetteExtractor instance
    """
    return RVMOnlySilhouetteExtractor()

# Test function
def test_rvm_only_mode():
    """Test RVM-only mode."""
    print("🧪 Testing RVM-Only Mode...")
    
    try:
        # Create RVM-only extractor
        extractor = create_rvm_only_silhouette_extractor()
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract silhouette
        silhouette = extractor.extract_silhouette(dummy_image)
        
        if silhouette is not None:
            print(f"✅ RVM-only test successful! Silhouette shape: {silhouette.shape}")
            return True
        else:
            print("❌ RVM-only test failed - no silhouette returned")
            return False
            
    except Exception as e:
        print(f"❌ RVM-only test failed: {e}")
        return False

if __name__ == "__main__":
    test_rvm_only_mode() 