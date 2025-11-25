#!/usr/bin/env python3
"""
Optimize ONNX models for TensorRT on Jetson Orin NX
This script converts ONNX models to TensorRT format for faster inference
"""

import os
import argparse
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_jetson_device_info():
    """Get Jetson device information"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            tegra_info = f.read().strip()
            logger.info(f"Jetson device: {tegra_info}")
        
        # Get CUDA capability
        device = cuda.Device(0)  # Use first GPU
        cuda_capability = device.compute_capability()
        logger.info(f"CUDA capability: {cuda_capability[0]}.{cuda_capability[1]}")
        
        # Get memory information
        mem_info = cuda.mem_get_info()
        free_memory = mem_info[0] / (1024 * 1024 * 1024)  # Convert to GB
        total_memory = mem_info[1] / (1024 * 1024 * 1024)  # Convert to GB
        logger.info(f"GPU memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")
        
        return {
            'tegra_info': tegra_info,
            'cuda_capability': cuda_capability,
            'free_memory': free_memory,
            'total_memory': total_memory
        }
    except Exception as e:
        logger.warning(f"Failed to get Jetson device info: {e}")
        return None

def build_engine(onnx_file_path, engine_file_path, precision="fp16", workspace_size=1, device_info=None):
    """
    Build TensorRT engine from ONNX file
    
    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        precision: Precision mode (fp32, fp16, int8)
        workspace_size: Maximum workspace size in GB
        device_info: Jetson device information
    """
    logger.info(f"Building TensorRT engine for {onnx_file_path}")
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Set workspace size based on available memory
        if device_info and device_info.get('free_memory'):
            # Use up to 75% of available memory, but not more than workspace_size
            memory_to_use = min(device_info['free_memory'] * 0.75, workspace_size)
            workspace_bytes = int(memory_to_use * (1 << 30))  # Convert to bytes
            logger.info(f"Setting workspace size to {memory_to_use:.2f}GB")
        else:
            workspace_bytes = workspace_size * (1 << 30)  # Default
        
        config.max_workspace_size = workspace_bytes
        
        # Enable Tensor Core usage if available
        if hasattr(config, 'set_flag'):
            config.set_flag(trt.BuilderFlag.TF32)  # TensorFloat-32 acceleration
        
        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision with Tensor Cores")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Using INT8 precision")
        else:
            logger.info("Using FP32 precision")
        
        # Set device type to Jetson
        if hasattr(config, 'set_device_type'):
            config.set_device_type(builder.device_type, 0)
        
        # Set DLA (Deep Learning Accelerator) core if available
        if hasattr(builder, 'num_DLA_cores') and builder.num_DLA_cores > 0:
            logger.info(f"Found {builder.num_DLA_cores} DLA cores")
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = 0
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            logger.info("Enabled DLA acceleration with GPU fallback")
        
        # Parse ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise ValueError(f"Failed to parse ONNX file: {onnx_file_path}")
        
        # Set optimization profiles for dynamic shapes if needed
        if network.has_implicit_batch_dimension:
            logger.info("Network uses implicit batch dimension")
        else:
            logger.info("Network uses explicit batch dimension")
            profile = builder.create_optimization_profile()
            
            # Check for dynamic inputs
            has_dynamic_shapes = False
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                if -1 in input_tensor.shape:
                    has_dynamic_shapes = True
                    break
            
            if has_dynamic_shapes:
                logger.info("Setting up optimization profiles for dynamic shapes")
                for i in range(network.num_inputs):
                    input_tensor = network.get_input(i)
                    name = input_tensor.name
                    shape = input_tensor.shape
                    
                    # Handle dynamic dimensions
                    if -1 in shape:
                        # Find dynamic dimensions and set appropriate values
                        min_shape = []
                        opt_shape = []
                        max_shape = []
                        
                        for dim in shape:
                            if dim == -1:  # Dynamic dimension
                                min_shape.append(1)    # Minimum
                                opt_shape.append(16)   # Optimal
                                max_shape.append(64)   # Maximum
                            else:
                                min_shape.append(dim)
                                opt_shape.append(dim)
                                max_shape.append(dim)
                        
                        profile.set_shape(name, min_shape, opt_shape, max_shape)
                        logger.info(f"Dynamic input {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                
                config.add_optimization_profile(profile)
        
        # Enable preview features for additional optimizations
        if hasattr(config, 'set_preview_feature'):
            config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
        
        # Build and save engine
        logger.info("Building TensorRT engine (this might take a while)...")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise ValueError("Failed to build TensorRT engine")
        
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved to {engine_file_path}")
        return True

def optimize_models(models_dir="weights", precision="fp16", workspace_size=1):
    """
    Optimize all ONNX models in the specified directory
    
    Args:
        models_dir: Directory containing ONNX models
        precision: Precision mode (fp32, fp16, int8)
        workspace_size: Maximum workspace size in GB
    """
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    # Check if running on Jetson
    is_jetson = os.path.exists('/etc/nv_tegra_release')
    if not is_jetson:
        logger.warning("Not running on a Jetson device. Optimization may not be optimal.")
    
    # Get device information if running on Jetson
    device_info = get_jetson_device_info() if is_jetson else None
    
    # Find all ONNX models
    onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    if not onnx_models:
        logger.error(f"No ONNX models found in {models_dir}")
        return False
    
    logger.info(f"Found {len(onnx_models)} ONNX models to optimize")
    
    # Optimize each model
    for model in onnx_models:
        onnx_path = os.path.join(models_dir, model)
        engine_path = os.path.join(models_dir, model.replace('.onnx', '.engine'))
        
        # Skip if engine already exists and is newer than ONNX file
        if os.path.exists(engine_path) and os.path.getmtime(engine_path) > os.path.getmtime(onnx_path):
            logger.info(f"Skipping {model}, engine is up to date")
            continue
        
        try:
            build_engine(onnx_path, engine_path, precision, workspace_size, device_info)
        except Exception as e:
            logger.error(f"Failed to optimize {model}: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX models for TensorRT on Jetson Orin NX")
    parser.add_argument("--models_dir", type=str, default="weights", help="Directory containing ONNX models")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp16", 
                        help="Precision mode for TensorRT (default: fp16)")
    parser.add_argument("--workspace_size", type=int, default=1, 
                        help="Maximum workspace size in GB (default: 1)")
    parser.add_argument("--force", action="store_true", 
                        help="Force regeneration of engines even if they exist")
    args = parser.parse_args()
    
    logger.info("Starting model optimization for Jetson Orin NX")
    logger.info(f"System: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    
    # Force regeneration of engines if requested
    if args.force:
        logger.info("Forcing regeneration of all TensorRT engines")
        for engine_file in os.listdir(args.models_dir):
            if engine_file.endswith('.engine'):
                os.remove(os.path.join(args.models_dir, engine_file))
                logger.info(f"Removed existing engine: {engine_file}")
    
    optimize_models(args.models_dir, args.precision, args.workspace_size)
    logger.info("Model optimization complete")

if __name__ == "__main__":
    main() 