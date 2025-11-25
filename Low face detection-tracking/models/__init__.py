from .arcface import ArcFace
from .scrfd import SCRFD

# TensorRT optimized models for Jetson
try:
    from .tensorrt_model import TensorRTModel
    from .scrfd_tensorrt import SCRFD_TensorRT
    from .arcface_tensorrt import ArcFace_TensorRT
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Function to determine if we're running on Jetson
def is_jetson():
    """Check if we're running on a Jetson device"""
    import os
    return os.path.exists('/etc/nv_tegra_release')

# Factory function to get appropriate model based on platform
def get_detector(model_path, **kwargs):
    """Get appropriate detector model based on platform"""
    if is_jetson() and TENSORRT_AVAILABLE:
        return SCRFD_TensorRT(model_path, **kwargs)
    else:
        return SCRFD(model_path, **kwargs)
        
def get_recognizer(model_path, **kwargs):
    """Get appropriate recognizer model based on platform"""
    if is_jetson() and TENSORRT_AVAILABLE:
        return ArcFace_TensorRT(model_path, **kwargs)
    else:
        return ArcFace(model_path, **kwargs)
