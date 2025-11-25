"""
TensorRT wrapper for ONNX models on Jetson Orin NX
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging

class TensorRTModel:
    """
    TensorRT model wrapper for optimized inference on Jetson
    """
    
    def __init__(self, engine_path=None, onnx_path=None):
        """
        Initialize TensorRT model
        
        Args:
            engine_path: Path to TensorRT engine file
            onnx_path: Path to ONNX model file (used if engine_path not found)
        """
        self.logger = logging.getLogger("TensorRTModel")
        self.engine = None
        self.context = None
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.input_name = None
        self.output_names = []
        self.bindings = []
        self.input_shape = None
        self.output_shapes = []
        self.stream = cuda.Stream()
        
        # Load engine or convert from ONNX
        if engine_path and os.path.exists(engine_path):
            self._load_engine(engine_path)
        elif onnx_path and os.path.exists(onnx_path):
            # Generate engine path from ONNX path
            if not engine_path:
                engine_path = onnx_path.replace('.onnx', '.engine')
            
            # Check if engine exists, if not convert ONNX to TensorRT
            if os.path.exists(engine_path):
                self._load_engine(engine_path)
            else:
                self._convert_onnx_to_engine(onnx_path, engine_path)
        else:
            raise ValueError("Either engine_path or onnx_path must be provided")
        
        # Initialize execution context
        self._initialize_context()
        
    def _load_engine(self, engine_path):
        """
        Load TensorRT engine from file
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        self.logger.info(f"Loading TensorRT engine from {engine_path}")
        
        # Create runtime and load engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
            
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise ValueError(f"Failed to load TensorRT engine from {engine_path}")
            
        self.logger.info("TensorRT engine loaded successfully")
        
    def _convert_onnx_to_engine(self, onnx_path, engine_path):
        """
        Convert ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model file
            engine_path: Path to save TensorRT engine
        """
        self.logger.info(f"Converting ONNX model {onnx_path} to TensorRT engine")
        
        # Create builder and network
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                raise ValueError(f"Failed to parse ONNX file: {onnx_path}")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Enable FP16 precision if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("Using FP16 precision")
        
        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            raise ValueError("Failed to build TensorRT engine")
        
        # Save engine to file
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        
        self.logger.info(f"TensorRT engine saved to {engine_path}")
        
        # Set engine
        self.engine = engine
        
    def _initialize_context(self):
        """
        Initialize execution context and allocate buffers
        """
        if not self.engine:
            raise ValueError("Engine not loaded")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input and output information
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate device memory
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            
            # Store binding information
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.input_name = name
                self.input_shape = shape
            else:
                self.output_names.append(name)
                self.output_shapes.append(shape)
        
        self.logger.info(f"Model initialized with input shape {self.input_shape} and {len(self.output_names)} outputs")
        
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: Input data as numpy array
            
        Returns:
            List of output arrays
        """
        # Prepare input
        input_data = np.ascontiguousarray(input_data)
        input_ptr = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod_async(input_ptr, input_data, self.stream)
        
        # Prepare output buffers
        outputs = []
        output_ptrs = []
        
        for shape in self.output_shapes:
            output = np.empty(shape, dtype=np.float32)
            output_ptr = cuda.mem_alloc(output.nbytes)
            outputs.append(output)
            output_ptrs.append(output_ptr)
        
        # Set bindings
        bindings = [int(input_ptr)]
        bindings.extend([int(ptr) for ptr in output_ptrs])
        
        # Execute inference
        self.context.execute_async_v2(bindings, self.stream.handle, None)
        
        # Copy outputs from device to host
        for i, output_ptr in enumerate(output_ptrs):
            cuda.memcpy_dtoh_async(outputs[i], output_ptr, self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Free memory
        input_ptr.free()
        for ptr in output_ptrs:
            ptr.free()
        
        return outputs
    
    def __del__(self):
        """
        Clean up resources
        """
        # Free CUDA memory
        for binding in self.bindings:
            if binding:
                cuda.Context.pop()
                
        # Delete context and engine
        self.context = None
        self.engine = None 