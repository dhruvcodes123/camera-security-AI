# 🚀 Optimized Video Processing Usage Guide

## **Issues Fixed:**

✅ **Low FPS (0.0)** - Added frame skipping and performance optimizations  
✅ **High CPU Usage** - Added memory management and garbage collection  
✅ **Process killed** - Added resource limits and better error handling  
✅ **Command line arguments** - Fixed argument parsing  

## **Quick Start - Test Mode**

For quick testing with limited frames:

```bash
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" output_test.mp4 --test-mode
```

## **Full Processing - Optimized**

For full video processing with performance optimizations:

```bash
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" output_full.mp4 --skip-frames 2 --max-frames 1000
```

## **Command Line Options**

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Input video file path | Required |
| `output` | Output video file path | Required |
| `--face-det-thresh` | Face detection threshold | 0.5 |
| `--skip-frames` | Skip N frames for speed | 1 |
| `--max-frames` | Maximum frames to process | All |
| `--show-display` | Show display window | False |
| `--test-mode` | Test mode (100 frames, skip 2) | False |

## **Performance Optimization Tips**

### **1. For Fast Testing**
```bash
# Process only 100 frames, skip every 2nd frame
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" test_output.mp4 --test-mode
```

### **2. For Medium Performance**
```bash
# Skip every 2nd frame, limit to 1000 frames
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" medium_output.mp4 --skip-frames 2 --max-frames 1000
```

### **3. For Full Processing**
```bash
# Process all frames but skip every 3rd frame
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" full_output.mp4 --skip-frames 3
```

### **4. With Display (for debugging)**
```bash
# Show display window during processing
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" debug_output.mp4 --show-display --test-mode
```

## **Output Files Generated**

After processing, you'll get:

1. **`output_video.mp4`** - Processed video with tracking overlays
2. **`final_tracking_data.csv`** - Tracking database with UUIDs and durations
3. **`db/tracking_database.sqlite`** - SQLite database file
4. **`track_id_details.pkl`** - Track details for validation

## **Sample Tracking Data**

The CSV file will contain:
```csv
unique_id,person_id,entry_timestamp,exit_timestamp,duration
91b456d1-1542-49b9-be5b-e634d90b1393,1,2025-07-30 11:51:49.324804,,
eb3c62cf-d9ea-4858-8116-91d4e829895f,1,2025-07-30 11:51:47.316909,2025-07-30 11:51:49.321218,2.004978060722351
```

## **Troubleshooting**

### **If you get "Low FPS" warnings:**
- Use `--skip-frames 2` or `--skip-frames 3`
- Use `--max-frames 500` to limit processing
- Use `--test-mode` for quick testing

### **If you get "High CPU Usage" warnings:**
- The system will automatically manage memory
- Use `--skip-frames` to reduce processing load
- The script includes automatic garbage collection

### **If the process gets killed:**
- Use `--max-frames` to limit processing
- Use `--skip-frames` to reduce memory usage
- Try `--test-mode` first to verify everything works

## **Example Workflow**

### **Step 1: Test with small video**
```bash
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" test_output.mp4 --test-mode
```

### **Step 2: Check tracking data**
```bash
cat final_tracking_data.csv
```

### **Step 3: Process full video**
```bash
python scripts/process_feed_optimized.py "data/input_videos/video  test 2.mp4" full_output.mp4 --skip-frames 2
```

## **Performance Monitoring**

The script includes built-in performance monitoring:
- FPS tracking
- Memory usage monitoring
- Automatic cleanup
- Progress reporting

## **Integration Status**

✅ **Tracking Database Integration Complete**
- UUID-based unique IDs
- Automatic entry/exit recording
- Duration calculation
- CSV export functionality
- Thread-safe operations

The tracking database is now fully integrated and will automatically record person entries/exits with UUIDs and calculate durations in real-time! 