import cv2
import os
import argparse
from pathlib import Path
import sys
# Add project root to sys.path to allow for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracker import PersonTracker
from performance_monitor import start_performance_monitoring, stop_performance_monitoring, print_performance_summary

# Add project root to sys.path to allow for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

def clear_entire_database():
    """Clear the entire database and all related files."""
    import shutil
    
    # print("--- CLEARING ENTIRE DATABASE ---")
    
    # Remove main database
    db_path = "db/database.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
        # print(f"Removed main database: {db_path}")
    
    # Remove index file
    index_path = "db/vector_db/fused.index"
    if os.path.exists(index_path):
        os.remove(index_path)
        # print(f"Removed index file: {index_path}")
    
    # print("--- ENTIRE DATABASE CLEARED ---")

def clear_output_dirs():
    """Clear output directories."""
    import shutil
    
    output_dirs = [
        "data/processed_output/tracked_persons",
        "data/processed_output/detected_faces", 
        "data/processed_output/gait_cycles"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            # print(f"Removed directory: {dir_path}")

def main(video_path: str, output_video_path: str, face_det_thresh: float):
    """
    Processes a video to track persons and saves the output.

    Args:
        video_path (str): Path to the input video file.
        output_video_path (str): Path to save the processed output video.
        face_det_thresh (float): The confidence threshold for face detection.
    """
    # Start performance monitoring
    start_performance_monitoring()
    
    # Clear databases and output directories before starting
    clear_entire_database()
    clear_output_dirs()

    # Initialize the person tracker
    tracker = PersonTracker(face_det_thresh=face_det_thresh)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties for the output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Create a resizable window before the loop starts
    cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)

    # print("Processing video... Press 'q' to stop.")
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Process the frame through the tracker
            processed_frame = tracker.process_frame(frame)

            # Write the processed frame to the output file
            out.write(processed_frame)

            # Display the frame
            cv2.imshow('Person Tracking', processed_frame)
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                # print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # print("Stopping video processing...")
                break
    finally:
        # Release everything when done
        # print("Finished processing.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        tracker.close()
        stop_performance_monitoring()
        print_performance_summary()

    # print(f"Finished processing. Output video saved to {output_video_path}")
    # print(f"Cropped person images saved in '{tracker.output_dir}'")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Track persons in a video.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input_videos/video test 3.mp4",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed_output/tracked_video_internet01.mp4",
        help="Path to save the output video file."
    )
    parser.add_argument(
        "--face-det-thresh", 
        type=float, 
        default=0.5,
        help="The confidence threshold for face detection."
    )
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(args.input, args.output, args.face_det_thresh)
