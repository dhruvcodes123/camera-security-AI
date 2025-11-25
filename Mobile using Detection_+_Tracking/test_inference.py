#!/usr/bin/env python3
"""
Simple test script for the Mobile Detection Inference
This script demonstrates how to use the inference class
"""

import os
import sys
from inference import MobileDetectionInference

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    model_path = "model/weights_mobile_using_detection.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Initialize inference with GPU support
        inference = MobileDetectionInference(
            model_path=model_path,
            device='auto',  # Will automatically use GPU if available
            conf_threshold=0.50,
            iou_threshold=0.50
        )
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_gpu_utilization():
    """Test GPU utilization"""
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✓ GPU memory allocation successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
            return False
    else:
        print("⚠ CUDA is not available, will use CPU")
    
    return True

def main():
    print("=== Mobile Detection Inference Test ===\n")
    
    # Test GPU utilization
    if not test_gpu_utilization():
        print("GPU test failed!")
        return
    
    # Test model loading
    if not test_model_loading():
        print("Model loading test failed!")
        return
    
    print("\n=== All tests passed! ===")
    print("\nYou can now use the inference script:")
    print("\nFor image inference:")
    print("python inference.py --input path/to/image.jpg --output result.jpg")
    
    print("\nFor video inference:")
    print("python inference.py --input path/to/video.mp4 --output result.mp4")
    
    print("\nAdditional options:")
    print("--device auto/cuda/cpu: Choose device")
    print("--conf-threshold 0.25: Set confidence threshold")
    print("--iou-threshold 0.45: Set IoU threshold")
    print("--no-display: Don't show results")

if __name__ == "__main__":
    main() 