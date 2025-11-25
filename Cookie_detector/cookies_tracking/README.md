# Cookie Tracker Project

This project is a high-performance, real-time cookie tracking system that uses YOLOv8 for object detection, ByteTrack for tracking, and a Re-ID model for re-identification of objects. It's designed for efficiency and includes features for performance monitoring, database logging of tracking sessions, and Docker support for easy deployment.

## 🚀 Features

- **Real-time Object Tracking**: Utilizes YOLOv8 and ByteTrack for accurate and fast tracking.
- **Re-Identification**: Employs a Re-ID model to re-identify tracks that are lost and reappear.
- **Performance Optimized**: The video processing pipeline is heavily optimized to prevent video lag and ensure smooth real-time performance. See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for details.
- **Database Logging**: Records detailed tracking sessions (entry/exit times, duration, cookie type) into a SQLite database. See [DATABASE_README.md](DATABASE_README.md) for more info.
- **Performance Monitoring**: Includes a script to monitor FPS, CPU, and memory usage.
- **Dockerized**: Comes with a `Dockerfile` for easy setup and deployment.

## 📁 Project Structure

```
.
├── config.py                  # Model path configuration
├── run_tracker.py             # Main script to run the tracker
├── database.py                # Database connection and functions
├── view_database.py           # Script to view database contents
├── performance_monitor.py     # Script to monitor performance
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── weights/                   # YOLOv8 model weights
│   └── best.pt
├── cookie_tracking.db         # SQLite database file (created on run)
├── DATABASE_README.md         # Documentation for the database
└── PERFORMANCE_OPTIMIZATION.md  # Documentation for performance optimizations
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.10
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Running the Application

### 1. Run the Tracker
To process a video file, use `run_tracker.py`:
```bash
python run_tracker.py /path/to/your/video.mp4
```

### 2. View the Database
To see the tracking data stored in the database:
```bash
python view_database.py
```

### 3. Monitor Performance
To run the tracker with real-time performance monitoring:
```bash
python performance_monitor.py /path/to/your/video.mp4
```

## 🐳 Docker Setup

### Prerequisites
- Docker

### Build the Docker Image
```bash
docker build -t cookie-tracker .
```

### Run the Docker Container

To run the tracker on a video file, you need to mount the video file into the container.

1.  **Place your video file** in the project's root directory.
2.  **Run the container:**
    ```bash
    docker run --rm -v $(pwd)/your_video.mp4:/app/video.mp4 cookie-tracker python run_tracker.py video.mp4
    ```
    *   `--rm`: Automatically removes the container when it exits.
    *   `-v`: Mounts the video file from your local machine to the container.

Since the application uses `cv2.imshow` to display the video, running it in a Docker container will not show the UI by default. For UI, you would need to set up an X11 server, which is more complex. The Docker setup is primarily for processing the video and populating the database.

To get the database file from the container after processing:
```bash
# 1. Get the container ID
docker ps -a

# 2. Copy the database file from the container to your local machine
docker cp <container-id>:/app/cookie_tracking.db .
```

## ⚙️ Configuration

-   **Model Path**: The path to the YOLOv8 model is configured in `config.py`.
-   **Performance Tuning**: Various performance parameters (e.g., `FRAME_SKIP_REID`, `DB_UPDATE_INTERVAL`) can be adjusted in `run_tracker.py`. See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for more details. 