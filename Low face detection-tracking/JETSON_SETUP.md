# Setting Up Hybrid Person Tracking System on Jetson Orin NX

This guide provides instructions for setting up and running the Hybrid Person Tracking System on the NVIDIA Jetson Orin NX platform.

## System Requirements

- NVIDIA Jetson Orin NX (tested on 8GB model)
- JetPack 6.0 or newer (L4T 36.3.0+)
- CUDA 12.2+
- TensorRT 10.0+
- Python 3.10+
- At least 10GB of free storage space

## Hardware Specifications (Jetson Orin NX)

- CPU: 6-core Arm Cortex-A78AE v8.2
- GPU: NVIDIA Ampere architecture with 1024 CUDA cores
- Memory: 8GB LPDDR5
- Storage: microSD card or NVMe SSD
- Power: 10W-15W

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hybrid-person-tracking.git
cd hybrid-person-tracking
```

### 2. Run the Setup Script

We've provided a dedicated setup script for Jetson devices that will:
- Create a Python virtual environment
- Install required dependencies
- Set up TensorRT optimizations
- Download model weights if needed

```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### 3. Activate the Environment

```bash
source jetson_env/bin/activate
```

### 4. Optimize Models for TensorRT (Optional but Recommended)

This step converts the ONNX models to TensorRT format for faster inference:

```bash
python optimize_models.py --precision fp16
```

You can choose between:
- `fp32`: Higher accuracy, slower performance
- `fp16`: Good balance between accuracy and speed (recommended)
- `int8`: Fastest performance, slightly lower accuracy

## Running the Application

### Basic Usage

```bash
python main.py --source <video_file> --performance-mode high
```

### Performance Modes

The system supports three performance modes:

1. **low**: Highest quality detection and recognition, but slower processing
2. **balanced**: Good balance between quality and speed
3. **high**: Fastest processing, may sacrifice some accuracy

For Jetson Orin NX, we recommend using the `high` performance mode for real-time processing.

### Monitoring Performance

You can monitor Jetson performance while running the application using `jtop`:

```bash
# In a separate terminal
sudo jtop
```

This will show CPU, GPU, memory usage, and power consumption.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `max_faces_per_frame` in config.py
   - Use `high` performance mode
   - Close other applications

2. **Slow Performance**
   - Ensure models are optimized with TensorRT
   - Use a smaller input resolution
   - Process fewer faces per frame

3. **TensorRT Errors**
   - Make sure you have the correct TensorRT version installed
   - Try rebuilding the TensorRT engines: `rm weights/*.engine && python optimize_models.py`

4. **OpenCV Errors**
   - The setup script links to the system OpenCV with CUDA support
   - If you see errors, try reinstalling: `sudo apt install --reinstall python3-opencv`

### Power Management

Jetson Orin NX supports different power modes. For better performance:

```bash
sudo nvpmodel -m 0  # Set to 15W mode
sudo jetson_clocks  # Max out clock speeds
```

For power efficiency:

```bash
sudo nvpmodel -m 1  # Set to 10W mode
```

## Additional Resources

- [Jetson Orin NX Documentation](https://developer.nvidia.com/embedded/jetson-orin-nx)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 