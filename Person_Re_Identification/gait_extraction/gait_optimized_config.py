#!/usr/bin/env python3
"""
Gait-Optimized Configuration for P3MNet-HQ Silhouette Extraction
Optimized parameters for best gait analysis accuracy
"""

# Gait Analysis Optimization Parameters
GAIT_CONFIG = {
    # Thresholding parameters optimized for person silhouettes
    'thresholding': {
        'otsu_area_range': (0.15, 0.40),  # Optimal area ratio for person silhouettes
        'otsu_intensity_max': 150,         # Maximum intensity for Otsu acceptance
        'adaptive_std_factor': 0.4,        # Standard deviation factor for adaptive threshold
        'gait_std_factor': 0.6,            # More conservative factor for gait
        'max_area_ratio': 0.5,             # Maximum acceptable area ratio
        'max_intensity': 200,              # Maximum acceptable intensity
    },
    
    # Post-processing parameters for gait analysis
    'postprocessing': {
        'min_contour_area': 50,            # Minimum contour area for gait analysis
        'morphological_kernel_size': 3,    # Kernel size for morphological operations
        'use_morphological_ops': False,    # Disable morphological ops for now
    },
    
    # Resize parameters for gait analysis
    'resize': {
        'target_size': (64, 44),           # Standard gait analysis size
        'interpolation': 'scipy_nearest',  # Use scipy for better binary preservation
        'force_binary': True,              # Force binary conversion after resize
    },
    
    # Quality control parameters
    'quality_control': {
        'min_noise_ratio': 0.1,            # Minimum acceptable noise ratio
        'max_noise_ratio': 0.6,            # Maximum acceptable noise ratio
        'min_intensity': 20,               # Minimum acceptable intensity
        'max_intensity': 180,              # Maximum acceptable intensity
        'force_binary_output': True,       # Force pure binary (0, 255) output
    },
    
    # Performance parameters
    'performance': {
        'gpu_preferred': True,             # Prefer GPU processing
        'batch_size': 1,                   # Process one image at a time for quality
        'warmup_frames': 3,                # Number of warmup frames
    }
}

# Gait Analysis Quality Metrics
GAIT_QUALITY_METRICS = {
    'excellent': {
        'noise_ratio': (0.1, 0.3),        # 10-30% white pixels
        'intensity': (20, 80),             # Low to moderate intensity
        'area_ratio': (0.15, 0.45),       # Reasonable person silhouette area
        'binary_score': 1.0,               # Perfect binary output
    },
    'good': {
        'noise_ratio': (0.3, 0.5),        # 30-50% white pixels
        'intensity': (80, 120),            # Moderate intensity
        'area_ratio': (0.1, 0.6),         # Acceptable silhouette area
        'binary_score': 0.8,               # Good binary output
    },
    'acceptable': {
        'noise_ratio': (0.5, 0.7),        # 50-70% white pixels
        'intensity': (120, 160),           # Higher intensity
        'area_ratio': (0.05, 0.8),        # Wide acceptable range
        'binary_score': 0.6,               # Acceptable binary output
    }
}

def get_gait_quality_score(silhouette):
    """
    Calculate gait quality score for a silhouette
    
    Args:
        silhouette: numpy array of silhouette (H, W)
        
    Returns:
        dict: Quality metrics and overall score
    """
    import numpy as np
    
    # Calculate basic metrics
    noise_ratio = np.count_nonzero(silhouette) / silhouette.size
    intensity = np.mean(silhouette)
    unique_values = np.unique(silhouette)
    
    # Calculate binary score (how close to pure binary)
    if set(unique_values).issubset({0, 255}):
        binary_score = 1.0
    else:
        # Penalize for intermediate values
        non_binary_count = len([v for v in unique_values if v not in {0, 255}])
        binary_score = max(0, 1.0 - non_binary_count / 10)
    
    # Determine quality level
    quality_level = 'poor'
    for level, criteria in GAIT_QUALITY_METRICS.items():
        if (criteria['noise_ratio'][0] <= noise_ratio <= criteria['noise_ratio'][1] and
            criteria['intensity'][0] <= intensity <= criteria['intensity'][1] and
            binary_score >= criteria['binary_score']):
            quality_level = level
            break
    
    return {
        'noise_ratio': noise_ratio,
        'intensity': intensity,
        'binary_score': binary_score,
        'unique_values': len(unique_values),
        'quality_level': quality_level,
        'is_binary': binary_score == 1.0
    }

def optimize_for_gait_accuracy(silhouette):
    """
    Apply final optimizations for gait accuracy
    
    Args:
        silhouette: numpy array of silhouette (H, W)
        
    Returns:
        numpy array: Optimized silhouette for gait analysis
    """
    import numpy as np
    
    # Force pure binary output
    silhouette = np.where(silhouette > 0, 255, 0).astype(np.uint8)
    
    # Ensure correct size for gait analysis
    if silhouette.shape != GAIT_CONFIG['resize']['target_size']:
        from scipy.ndimage import zoom
        h, w = silhouette.shape
        scale_h, scale_w = 44 / h, 64 / w
        silhouette = zoom(silhouette, (scale_h, scale_w), order=0, mode='nearest')
        silhouette = silhouette.astype(np.uint8)
        # Force binary again after resize
        silhouette = np.where(silhouette > 0, 255, 0).astype(np.uint8)
    
    return silhouette 