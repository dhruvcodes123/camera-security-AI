#!/bin/bash

echo "=== Mobile Detection Inference Setup ==="
echo ""

# Check if virtual environment exists
if [ ! -d "my_env" ]; then
    echo "Error: Virtual environment 'my_env' not found!"
    echo "Please create the virtual environment first."
    exit 1
fi

# Activate virtual environment
echo "1. Activating virtual environment..."
source my_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment!"
    exit 1
fi

echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies!"
    exit 1
fi

echo "✓ Dependencies installed"

# Check if model file exists
echo ""
echo "3. Checking model file..."
if [ ! -f "model/weights_mobile_using_detection.pt" ]; then
    echo "Warning: Model file not found at model/weights_mobile_using_detection.pt"
    echo "Please ensure your model file is in the correct location."
else
    echo "✓ Model file found"
fi

# Test GPU availability
echo ""
echo "4. Testing GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name()}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ CUDA not available, will use CPU')
"

# Run inference test
echo ""
echo "5. Running inference test..."
python test_inference.py

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now use the inference script:"
echo "  python inference.py --input your_image.jpg --output result.jpg"
echo ""
echo "For more options, see README.md" 