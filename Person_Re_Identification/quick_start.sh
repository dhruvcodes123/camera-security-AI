#!/bin/bash

# ==============================================================================
# PERSON RE-IDENTIFICATION SYSTEM - QUICK START SCRIPT
# ==============================================================================
# This script helps you quickly set up and run the Person Re-ID system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            print_success "NVIDIA GPU detected"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
            return 0
        else
            print_warning "NVIDIA GPU not available"
            return 1
        fi
    else
        print_warning "nvidia-smi not found. GPU acceleration may not be available."
        return 1
    fi
}

# Function to check Docker
check_docker() {
    if command_exists docker; then
        if docker info >/dev/null 2>&1; then
            print_success "Docker is running"
            return 0
        else
            print_error "Docker is not running. Please start Docker first."
            return 1
        fi
    else
        print_error "Docker is not installed. Please install Docker first."
        return 1
    fi
}

# Function to check Docker Compose
check_docker_compose() {
    if command_exists docker-compose; then
        print_success "Docker Compose is available"
        return 0
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        return 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/input_videos
    mkdir -p data/processed_output/tracked_persons
    mkdir -p data/processed_output/detected_faces
    mkdir -p data/processed_output/gait_cycles
    mkdir -p db/vector_db
    mkdir -p models/gait
    mkdir -p RobustVideoMatting/pretrained
    
    print_success "Directories created successfully"
}

# Function to download sample video (if needed)
download_sample_video() {
    if [ ! -f "data/input_videos/sample.mp4" ]; then
        print_status "No sample video found. You can add your own videos to data/input_videos/"
        print_warning "Please add a video file to data/input_videos/ before running the system"
    else
        print_success "Sample video found"
    fi
}

# Function to build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    if docker build -t person-reid-system .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run with Docker Compose
run_with_compose() {
    print_status "Starting system with Docker Compose..."
    
    if docker-compose up --build -d; then
        print_success "System started successfully"
        print_status "You can now run processing commands:"
        echo "  docker-compose run person-reid-system python scripts/process_feed_optimized.py --help"
    else
        print_error "Failed to start system with Docker Compose"
        exit 1
    fi
}

# Function to run with Docker directly
run_with_docker() {
    print_status "Starting system with Docker..."
    
    if docker run --gpus all -d \
        --name person-reid-system \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/db:/app/db" \
        -v "$(pwd)/models:/app/models" \
        person-reid-system; then
        print_success "System started successfully"
        print_status "You can now run processing commands:"
        echo "  docker exec person-reid-system python scripts/process_feed_optimized.py --help"
    else
        print_error "Failed to start system with Docker"
        exit 1
    fi
}

# Function to show usage examples
show_usage_examples() {
    echo ""
    print_status "Usage Examples:"
    echo ""
    echo "1. Process a video file:"
    echo "   docker-compose run person-reid-system python scripts/process_feed_optimized.py \\"
    echo "     --video_path data/input_videos/your_video.mp4 \\"
    echo "     --output_path data/processed_output/output.mp4"
    echo ""
    echo "2. Process with real-time display:"
    echo "   docker-compose run person-reid-system python scripts/process_feed_optimized.py \\"
    echo "     --video_path data/input_videos/your_video.mp4 \\"
    echo "     --output_path data/processed_output/output.mp4 \\"
    echo "     --show_display True"
    echo ""
    echo "3. Access the container shell:"
    echo "   docker-compose run --service-ports person-reid-system bash"
    echo ""
    echo "4. View logs:"
    echo "   docker-compose logs -f person-reid-system"
    echo ""
    echo "5. Stop the system:"
    echo "   docker-compose down"
    echo ""
}

# Function to show system status
show_system_status() {
    echo ""
    print_status "System Status:"
    echo ""
    
    # Check if containers are running
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q person-reid-system; then
        print_success "Person Re-ID system is running"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep person-reid
    else
        print_warning "Person Re-ID system is not running"
    fi
    
    # Check GPU usage
    if command_exists nvidia-smi; then
        echo ""
        print_status "GPU Usage:"
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
    
    # Check disk usage
    echo ""
    print_status "Disk Usage:"
    df -h . | tail -1
}

# Main function
main() {
    echo "=============================================================================="
    echo "PERSON RE-IDENTIFICATION SYSTEM - QUICK START"
    echo "=============================================================================="
    echo ""
    
    # Check system requirements
    print_status "Checking system requirements..."
    
    if ! check_docker; then
        exit 1
    fi
    
    if ! check_docker_compose; then
        exit 1
    fi
    
    check_gpu
    
    # Create directories
    create_directories
    
    # Check for sample video
    download_sample_video
    
    # Build Docker image
    build_docker_image
    
    # Ask user for deployment method
    echo ""
    print_status "Choose deployment method:"
    echo "1. Docker Compose (recommended)"
    echo "2. Docker directly"
    echo "3. Show usage examples"
    echo "4. Show system status"
    echo "5. Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            run_with_compose
            show_usage_examples
            ;;
        2)
            run_with_docker
            show_usage_examples
            ;;
        3)
            show_usage_examples
            ;;
        4)
            show_system_status
            ;;
        5)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 