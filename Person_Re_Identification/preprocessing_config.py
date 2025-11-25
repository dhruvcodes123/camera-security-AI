#!/usr/bin/env python3
"""
Configuration file for CLAHE preprocessing parameters.
Simplified configuration for basic image enhancement.
"""

import numpy as np
import cv2

# CLAHE Parameters
CLAHE_CONFIG = {
    "clip_limit": 2.0,        # Contrast limit for CLAHE (higher = more contrast)
    "tile_grid_size": (8, 8), # Grid size for CLAHE (smaller = more local enhancement)
    "enabled": True           # Enable/disable CLAHE preprocessing
}

# Overall Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "enabled": True,          # Enable/disable entire preprocessing pipeline
    "save_intermediate": False, # Save intermediate results for debugging
    "verbose_logging": True,   # Enable detailed logging
    "performance_mode": False  # Optimize for speed over quality
}

# Performance Optimization Settings
PERFORMANCE_CONFIG = {
    "use_gpu": True,          # Use GPU acceleration if available
    "batch_size": 1,          # Batch size for processing (1 for real-time)
    "cache_size": 100,        # Number of processed frames to cache
    "parallel_processing": False  # Enable parallel processing for batch operations
}

# Quality Control Settings
QUALITY_CONFIG = {
    "min_enhancement_ratio": 0.8,  # Minimum acceptable enhancement ratio
    "max_processing_time": 0.1,    # Maximum processing time per frame (seconds)
    "quality_threshold": 0.7,      # Quality threshold for accepting enhanced frames
    "auto_adjust": True            # Automatically adjust parameters based on image quality
}

def get_preprocessing_config():
    """
    Get the complete preprocessing configuration.
    
    Returns:
        dict: Complete configuration dictionary
    """
    return {
        "clahe": CLAHE_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "quality": QUALITY_CONFIG
    }

def update_preprocessing_config(updates):
    """
    Update preprocessing configuration with new values.
    
    Args:
        updates (dict): Dictionary with configuration updates
    """
    global CLAHE_CONFIG, PREPROCESSING_CONFIG, PERFORMANCE_CONFIG, QUALITY_CONFIG
    
    if "clahe" in updates:
        CLAHE_CONFIG.update(updates["clahe"])
    if "preprocessing" in updates:
        PREPROCESSING_CONFIG.update(updates["preprocessing"])
    if "performance" in updates:
        PERFORMANCE_CONFIG.update(updates["performance"])
    if "quality" in updates:
        QUALITY_CONFIG.update(updates["quality"])

def get_optimized_config_for_scenario(scenario):
    """
    Get optimized configuration for specific scenarios.
    
    Args:
        scenario (str): Scenario type ('realtime', 'quality', 'balanced')
    
    Returns:
        dict: Optimized configuration for the scenario
    """
    if scenario == "realtime":
        return {
            "clahe": {"clip_limit": 1.5, "tile_grid_size": (4, 4), "enabled": True},
            "preprocessing": {"enabled": True, "save_intermediate": False, "verbose_logging": False, "performance_mode": True},
            "performance": {"use_gpu": True, "batch_size": 1, "cache_size": 50, "parallel_processing": False},
            "quality": {"min_enhancement_ratio": 0.6, "max_processing_time": 0.05, "quality_threshold": 0.5, "auto_adjust": False}
        }
    elif scenario == "quality":
        return {
            "clahe": {"clip_limit": 3.0, "tile_grid_size": (16, 16), "enabled": True},
            "preprocessing": {"enabled": True, "save_intermediate": True, "verbose_logging": True, "performance_mode": False},
            "performance": {"use_gpu": True, "batch_size": 1, "cache_size": 200, "parallel_processing": False},
            "quality": {"min_enhancement_ratio": 0.9, "max_processing_time": 0.2, "quality_threshold": 0.8, "auto_adjust": True}
        }
    elif scenario == "balanced":
        return {
            "clahe": {"clip_limit": 2.0, "tile_grid_size": (8, 8), "enabled": True},
            "preprocessing": {"enabled": True, "save_intermediate": False, "verbose_logging": True, "performance_mode": False},
            "performance": {"use_gpu": True, "batch_size": 1, "cache_size": 100, "parallel_processing": False},
            "quality": {"min_enhancement_ratio": 0.8, "max_processing_time": 0.1, "quality_threshold": 0.7, "auto_adjust": True}
        }
    else:
        # Default configuration
        return get_preprocessing_config()

def apply_clahe_enhancement(image: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Apply CLAHE enhancement to an image.
    
    Args:
        image: Input image as numpy array
        config: CLAHE configuration dictionary
    
    Returns:
        Enhanced image as numpy array
    """
    if config is None:
        config = CLAHE_CONFIG
    
    if not config.get("enabled", True):
        return image
    
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=config.get("clip_limit", 2.0),
            tileGridSize=config.get("tile_grid_size", (8, 8))
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    except Exception as e:
        print(f"⚠️ CLAHE enhancement failed: {e}")
        return image

def print_current_config():
    """
    Print the current preprocessing configuration.
    """
    print("📋 Current Preprocessing Configuration:")
    print(f"   CLAHE Enabled: {CLAHE_CONFIG['enabled']}")
    print(f"   CLAHE Clip Limit: {CLAHE_CONFIG['clip_limit']}")
    print(f"   CLAHE Tile Grid: {CLAHE_CONFIG['tile_grid_size']}")
    print(f"   Preprocessing Enabled: {PREPROCESSING_CONFIG['enabled']}")
    print(f"   Performance Mode: {PREPROCESSING_CONFIG['performance_mode']}")
    print(f"   Verbose Logging: {PREPROCESSING_CONFIG['verbose_logging']}")

if __name__ == "__main__":
    # Test the configuration
    print("🧪 Testing Preprocessing Configuration...")
    print_current_config()
    
    # Test different scenarios
    scenarios = ["realtime", "quality", "balanced"]
    for scenario in scenarios:
        print(f"\n📊 {scenario.upper()} Configuration:")
        config = get_optimized_config_for_scenario(scenario)
        print(f"   CLAHE: {config['clahe']['enabled']}")
        print(f"   Performance Mode: {config['preprocessing']['performance_mode']}")
    
    print("\n✅ Preprocessing configuration test completed!") 