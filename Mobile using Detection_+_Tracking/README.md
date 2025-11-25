# Mobile Detection Inference

This repository contains an inference script for a mobile detection model trained on Roboflow. The script is optimized to utilize your 8GB GPU efficiently.

## Features

- **GPU Optimization**: Automatically detects and utilizes your 8GB GPU
- **Image & Video Support**: Process both images and videos
- **Real-time Display**: View results in real-time during processing
- **Batch Processing**: Efficient processing with progress bars
- **Configurable Parameters**: Adjustable confidence and IoU thresholds

## Setup

### 1. Activate Virtual Environment

```bash
source my_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test Setup

```bash
python test_inference.py
```

This will test:
- GPU availability and memory allocation
- Model loading capabilities
- Basic functionality

## Usage

### Basic Image Inference

```bash
python inference.py --input path/to/image.jpg --output result.jpg
```

### Basic Video Inference

```bash
python inference.py --input path/to/video.mp4 --output result.mp4
```

### Advanced Options

```bash
python inference.py \
    --input input.jpg \
    --output result.jpg \
    --device cuda \
    --conf-threshold 0.3 \
    --iou-threshold 0.5 \
    --no-display
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to model weights | `model/weights_mobile_using_detection.pt` |
| `--input` | Input image or video path | Required |
| `--output` | Output path for results | None (display only) |
| `--device` | Device to use (`auto`, `cuda`, `cpu`) | `auto` |
| `--conf-threshold` | Confidence threshold | `0.25` |
| `--iou-threshold` | IoU threshold for NMS | `0.45` |
| `--no-display` | Don't display results | False |

## GPU Optimization Features

The script includes several optimizations for your 8GB GPU:

1. **Automatic Device Detection**: Automatically uses GPU if available
2. **Memory Management**: Efficient tensor operations and memory cleanup
3. **Batch Processing**: Optimized for video processing
4. **CUDA Optimization**: Uses PyTorch's JIT optimization when available

## Performance Tips

### For 8GB GPU:

1. **Batch Size**: The script automatically optimizes batch size for your GPU
2. **Memory Management**: Includes automatic memory cleanup
3. **Mixed Precision**: Can be enabled for faster inference (if supported)

### Expected Performance:

- **Images**: ~50-100ms per image
- **Videos**: ~15-30 FPS depending on resolution
- **Memory Usage**: ~2-4GB GPU memory during inference

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce input resolution
   - Close other GPU applications
   - Use `--device cpu` as fallback

2. **Model Loading Error**:
   - Check model file path
   - Verify PyTorch version compatibility
   - Ensure all dependencies are installed

3. **Slow Performance**:
   - Ensure GPU is being used (`--device cuda`)
   - Check GPU memory usage
   - Reduce input resolution

### GPU Memory Monitoring:

```bash
# Monitor GPU usage
nvidia-smi -l 1
```

## File Structure

```
├── inference.py          # Main inference script
├── test_inference.py     # Test script
├── requirements.txt      # Dependencies
├── model/
│   └── weights_mobile_using_detection.pt  # Your trained model
└── my_env/              # Virtual environment
```

## Examples

### Process a single image:
```bash
python inference.py --input test_image.jpg --output result.jpg
```

### Process a video with custom thresholds:
```bash
python inference.py \
    --input test_video.mp4 \
    --output result_video.mp4 \
    --conf-threshold 0.4 \
    --iou-threshold 0.6
```

### Process without displaying (for batch processing):
```bash
python inference.py --input image.jpg --output result.jpg --no-display
```

## Model Information

- **Model Type**: Mobile detection model from Roboflow
- **Input Size**: 640x640 pixels
- **Supported Formats**: Images (jpg, png, etc.) and Videos (mp4, avi, etc.)
- **GPU Memory**: Optimized for 8GB GPU

## Support

If you encounter any issues:

1. Run the test script: `python test_inference.py`
2. Check GPU availability: `nvidia-smi`
3. Verify model file exists: `ls model/weights_mobile_using_detection.pt`
4. Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"` 