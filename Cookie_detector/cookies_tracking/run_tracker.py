import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import threading
import queue
import time
import torch
import torchvision.transforms as T
from torch.nn.functional import cosine_similarity

import torchreid

from config import MODEL_PATH
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.annotators.core import BoxAnnotator, LabelAnnotator
import database

DB_FILE = "cookie_tracking.db"
EXIT_TIMEOUT_SECONDS = 60 # Increased: A track is considered exited if not seen for this long
REID_MATCH_THRESHOLD = 0.80 # Cosine similarity threshold for a match
POSITION_MATCH_THRESHOLD = 0.90 # IoU threshold for a positional match

# --- Static Object Tracking Enhancements ---
import collections

# Add these global/static object tracking parameters
STATIC_STABILITY_FRAMES = 10  # Number of frames to check for stability
STATIC_MOVEMENT_THRESH = 10.0  # Max pixel movement to consider stationary
STATIC_LOST_BUFFER = 1800      # 30 seconds at 30fps
STATIC_DET_THRESH = 0.15      # Balanced detection threshold
STATIC_REID_THRESH = 0.85     # Slightly relaxed for performance
STATIC_POS_THRESH = 0.90      # Slightly relaxed for performance
FEATURE_HISTORY_LENGTH = 10   # Reduced for better performance

# Performance optimization parameters
FRAME_SKIP_REID = 2  # Run Re-ID every other frame for a balance of speed and accuracy
DB_UPDATE_INTERVAL = 30  # Update database every N frames
DASHBOARD_UPDATE_INTERVAL = 5  # Update dashboard data every 5 frames
MAX_QUEUE_SIZE = 5  # Smaller queue to reduce memory usage

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class ReIDTool:
    def __init__(self, model_name='osnet_x1_0', device='cuda'):
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,  # Not used for feature extraction
            pretrained=True
        ).to(device)
        self.model.eval()
        self.device = device
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, image_crop):
        return self.transform(image_crop).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def get_feature(self, image_crop):
        tensor_input = self._preprocess(image_crop)
        return self.model(tensor_input)

def frame_producer(cap, frame_queue, stop_event):
    """Reads frames from the video capture object and puts them into a queue."""
    i = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # Skip frames if queue is full to maintain real-time performance
        try:
            frame_queue.put((i, frame), timeout=0.1)
            i += 1
        except queue.Full:
            # Skip frame if queue is full
            i += 1
            continue
    frame_queue.put(None)  # Sentinel value to signal end of stream

def processor(frame_queue, result_queue, stop_event):
    """Processes frames: runs detection, tracking, and lifecycle management."""
    # Initialize database connection
    db_conn = database.create_connection(DB_FILE)
    if db_conn is not None:
        database.create_table(db_conn)
        database.create_cookie_sessions_table(db_conn)
    else:
        print("Error! Cannot create database connection.")

    model = YOLO(MODEL_PATH)
    byte_tracker = ByteTrack(lost_track_buffer=STATIC_LOST_BUFFER)
    reid_tool = ReIDTool()
    
    # Pre-create annotators to avoid recreation overhead
    box_annotator = BoxAnnotator()
    label_annotator = LabelAnnotator()
    
    gallery = {} 
    live_tracks = {} 
    history = {}
    next_global_id = 1
    
    # Performance tracking variables
    frame_count = 0
    db_batch_operations = []
    last_dashboard_data = None

    while not stop_event.is_set():
        try:
            data = frame_queue.get(timeout=0.5)  # Reduced timeout for responsiveness
            if data is None:
                break
            
            frame_index, frame = data
            frame_count += 1
            
            # Run detection and tracking
            results = model(frame, imgsz=480, conf=STATIC_DET_THRESH, verbose=False, half=True)[0]
            detections = Detections.from_ultralytics(results)
            tracked_objects = byte_tracker.update_with_detections(detections)

            current_tracker_ids = set()
            newly_appeared_detections = []
            
            for det in tracked_objects:
                bbox, _, _, class_id, tid, _ = det
                if tid is not None:
                    current_tracker_ids.add(tid)
                    # Update bbox of existing live tracks
                    if tid in live_tracks:
                        gid = live_tracks[tid]
                        gallery[gid]['bbox'] = bbox
                        gallery[gid]['last_seen_ts'] = time.time()
                        # --- Position history for stability ---
                        cx = (bbox[0]+bbox[2])/2
                        cy = (bbox[1]+bbox[3])/2
                        if 'positions' not in gallery[gid]:
                            gallery[gid]['positions'] = collections.deque(maxlen=STATIC_STABILITY_FRAMES)
                        gallery[gid]['positions'].append((cx, cy))
                    elif tid not in live_tracks:
                        newly_appeared_detections.append(det)

            # Find tracks that haven't been seen recently but aren't yet marked as exited
            now = time.time()
            lost_gids = [gid for gid, data in gallery.items() 
                        if now - data.get('last_seen_ts', 0) < EXIT_TIMEOUT_SECONDS 
                        and gid not in live_tracks.values()]
            
            # Find tracks that have been lost for too long and should be marked as exited
            definitely_lost_gids = [gid for gid, data in gallery.items() 
                                   if now - data.get('last_seen_ts', 0) >= EXIT_TIMEOUT_SECONDS 
                                   and gid in history 
                                   and history[gid].get('exit_time') is None]
            
            # Mark definitely lost tracks as exited (batch database updates)
            for gid in definitely_lost_gids:
                if gid in history:
                    exit_time = datetime.now()
                    history[gid]['exit_time'] = exit_time
                    history[gid]['duration'] = exit_time - history[gid]['entry_time']
                    
                    # Queue database update for batch processing
                    if db_conn is not None:
                        stationary = history[gid].get('stationary', False)
                        stability = history[gid].get('stability', 0.0)
                        db_batch_operations.append(('exit', gid, exit_time, stationary, stability))
            
            # Process new detections (with frame skipping for Re-ID)
            for det in newly_appeared_detections:
                bbox, _, confidence, class_id, tid, _ = det
                x1, y1, x2, y2 = map(int, bbox)
                if x1 >= x2 or y1 >= y2: continue
                
                best_match_gid = None
                max_similarity = -1
                cx = (bbox[0]+bbox[2])/2
                cy = (bbox[1]+bbox[3])/2
                
                # Simplified re-identification logic (only run Re-ID occasionally)
                if frame_count % FRAME_SKIP_REID == 0 and lost_gids:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        try:
                            new_feature = reid_tool.get_feature(crop)
                            
                            # Optimized re-identification - limit search scope
                            for gid in lost_gids[:5]:  # Only check top 5 recent lost tracks
                                class_match = (class_id is not None and 
                                             gallery[gid]['class_name'] == model.names[class_id])
                                if not class_match:
                                    continue
                                    
                                prev_bbox = gallery[gid]['bbox']
                                iou = get_iou(bbox, prev_bbox)
                                
                                # Quick distance check first
                                if iou < 0.3:  # Skip if too far apart
                                    continue
                                
                                feature_list = gallery[gid].get('features', [])
                                if feature_list:
                                    # Only check most recent feature for speed
                                    similarities = [float(cosine_similarity(new_feature, feature_list[-1]).item())]
                                    best_feat_sim = max(similarities) if similarities else 0.0
                                    
                                    time_since_last_seen = now - gallery[gid].get('last_seen_ts', now)
                                    sim_thresh = STATIC_REID_THRESH if time_since_last_seen < 30 else 0.75
                                    
                                    if best_feat_sim > sim_thresh and best_feat_sim > max_similarity:
                                        max_similarity = best_feat_sim
                                        best_match_gid = gid
                        except Exception as e:
                            # Skip Re-ID on error to maintain performance
                            pass
                
                if best_match_gid is not None:
                    live_tracks[tid] = best_match_gid
                    if best_match_gid in lost_gids:
                        lost_gids.remove(best_match_gid)
                    # Update feature history (limit size)
                    if frame_count % FRAME_SKIP_REID == 0:  # Only update features occasionally
                        gallery[best_match_gid]['features'].append(new_feature)
                        if len(gallery[best_match_gid]['features']) > FEATURE_HISTORY_LENGTH:
                            gallery[best_match_gid]['features'].pop(0)
                else:
                    # Create new track
                    live_tracks[tid] = next_global_id
                    class_name = model.names[class_id] if class_id is not None else "Unknown"
                    gallery[next_global_id] = {
                        'features': [], 
                        'class_name': class_name, 
                        'bbox': bbox, 
                        'last_seen_ts': now,
                        'positions': collections.deque(maxlen=STATIC_STABILITY_FRAMES)
                    }
                    gallery[next_global_id]['positions'].append((cx, cy))
                    entry_time = datetime.now()
                    history[next_global_id] = {
                        'entry_time': entry_time, 
                        'exit_time': None, 
                        'duration': None, 
                        'class_name': class_name, 
                        'stationary': False, 
                        'stability': 0.0
                    }
                    
                    # Queue database operation for batch processing
                    if db_conn is not None:
                        db_batch_operations.append(('entry', next_global_id, class_name, entry_time))
                    
                    next_global_id += 1

            # Position stability and stationary status update (optimized)
            for gid in list(gallery.keys()):  # Use list() to avoid dict changing during iteration
                pos_hist = gallery[gid].get('positions', None)
                if pos_hist and len(pos_hist) == STATIC_STABILITY_FRAMES:
                    # Optimized distance calculation
                    positions_array = np.array(pos_hist)
                    dists = np.linalg.norm(positions_array[1:] - positions_array[0], axis=1)
                    max_dist = float(np.max(dists)) if len(dists) > 0 else 0.0
                    
                    stability = 1.0 - min(max_dist / STATIC_MOVEMENT_THRESH, 1.0)
                    is_stationary = max_dist < STATIC_MOVEMENT_THRESH
                    
                    if gid in history:
                        history[gid]['stationary'] = is_stationary
                        history[gid]['stability'] = stability

            # Batch database updates (every N frames)
            if frame_count % DB_UPDATE_INTERVAL == 0 and db_batch_operations:
                try:
                    for operation in db_batch_operations:
                        if operation[0] == 'entry':
                            _, gid, class_name, entry_time = operation
                            database.insert_cookie_session(db_conn, gid, class_name, entry_time)
                        elif operation[0] == 'exit':
                            _, gid, exit_time, stationary, stability = operation
                            database.update_cookie_session_exit(db_conn, gid, exit_time, stationary, stability)
                    db_batch_operations.clear()
                except Exception as e:
                    print(f"Database error: {e}")

            # Prepare for annotation (simplified)
            labels = []
            for _, _, confidence, class_id, tracker_id, _ in tracked_objects:
                if tracker_id in live_tracks and class_id is not None:
                    global_id = live_tracks[tracker_id]
                    class_name = model.names[class_id]
                    # Simplified label
                    stat = history[global_id]['stationary'] if global_id in history else False
                    labels.append(f"{class_name}-{global_id}{' [S]' if stat else ''}")
                else:
                    labels.append("N/A")
            
            # Create annotated frame
            annotated_frame = box_annotator.annotate(scene=frame, detections=tracked_objects)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_objects, labels=labels)
            
            # Update dashboard data less frequently
            if frame_count % DASHBOARD_UPDATE_INTERVAL == 0:
                last_dashboard_data = (live_tracks.copy(), history.copy())
            
            # Use cached dashboard data
            dashboard_data = last_dashboard_data if last_dashboard_data else (live_tracks, history)
            
            try:
                result_queue.put((annotated_frame, dashboard_data[0], dashboard_data[1]), timeout=0.1)
            except queue.Full:
                # Skip frame if result queue is full
                continue
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            continue
    
    # Final database cleanup
    if db_batch_operations and db_conn is not None:
        try:
            for operation in db_batch_operations:
                if operation[0] == 'entry':
                    _, gid, class_name, entry_time = operation
                    database.insert_cookie_session(db_conn, gid, class_name, entry_time)
                elif operation[0] == 'exit':
                    _, gid, exit_time, stationary, stability = operation
                    database.update_cookie_session_exit(db_conn, gid, exit_time, stationary, stability)
        except Exception as e:
            print(f"Final database cleanup error: {e}")
    
    # Close database connection
    if db_conn is not None:
        db_conn.close()
    
    result_queue.put(None) # Sentinel

def draw_dashboard(live_tracks, history):
    dashboard_height, dashboard_width = 800, 900
    dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1

    # Column positions
    col_x = {'id': 10, 'cookie_id': 60, 'class': 180, 'entry': 300, 'exit': 420, 'duration': 540, 'stat': 650, 'stab': 750}
    
    # Headers
    y = 30
    cv2.putText(dashboard, "ID", (col_x['id'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Cookie ID", (col_x['cookie_id'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Class", (col_x['class'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Entry", (col_x['entry'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Exit", (col_x['exit'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Duration", (col_x['duration'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Static", (col_x['stat'], y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(dashboard, "Stab", (col_x['stab'], y), font, font_scale, (255, 255, 0), font_thickness)
    y += 10
    cv2.line(dashboard, (10, y), (dashboard_width - 10, y), (255, 255, 0), 1)
    y += 20
    
    # Live tracks (limit to prevent overflow)
    row_id = 1
    active_gids = set(live_tracks.values())
    for gid in sorted(list(active_gids))[:15]:  # Limit to 15 active tracks
        data = history.get(gid, {})
        class_name = data.get('class_name', 'N/A')
        entry_time = data.get('entry_time', datetime.now())
        duration = datetime.now() - entry_time
        entry_str = entry_time.strftime("%H:%M:%S")
        dur_str = str(duration).split('.')[0]
        stat = "Yes" if data.get('stationary', False) else "No"
        stab = f"{data.get('stability', 0.0):.2f}"
        
        cv2.putText(dashboard, str(row_id), (col_x['id'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, str(gid), (col_x['cookie_id'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, class_name, (col_x['class'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, entry_str, (col_x['entry'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, "--:--:--", (col_x['exit'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, dur_str, (col_x['duration'], y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(dashboard, stat, (col_x['stat'], y), font, font_scale, (0, 255, 0) if stat=="Yes" else (0,0,255), font_thickness)
        cv2.putText(dashboard, stab, (col_x['stab'], y), font, font_scale, (255, 255, 255), font_thickness)
        y += 25; row_id += 1
        if y > dashboard_height - 60: break
    
    # Exited tracks (limit to recent ones)
    exited_tracks = {gid: data for gid, data in history.items() if gid not in active_gids and data.get('exit_time')}
    sorted_history = sorted(exited_tracks.items(), key=lambda item: item[1]['exit_time'], reverse=True)[:10]  # Limit to 10 recent
    for gid, data in sorted_history:
        class_name = data.get('class_name', 'N/A')
        entry_str = data['entry_time'].strftime('%H:%M:%S')
        exit_str = data['exit_time'].strftime('%H:%M:%S')
        dur_str = str(data['duration']).split('.')[0]
        stat = "Yes" if data.get('stationary', False) else "No"
        stab = f"{data.get('stability', 0.0):.2f}"
        
        cv2.putText(dashboard, str(row_id), (col_x['id'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, str(gid), (col_x['cookie_id'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, class_name, (col_x['class'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, entry_str, (col_x['entry'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, exit_str, (col_x['exit'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, dur_str, (col_x['duration'], y), font, font_scale, (200, 200, 200), font_thickness)
        cv2.putText(dashboard, stat, (col_x['stat'], y), font, font_scale, (0, 255, 0) if stat=="Yes" else (0,0,255), font_thickness)
        cv2.putText(dashboard, stab, (col_x['stab'], y), font, font_scale, (200, 200, 200), font_thickness)
        y += 25; row_id += 1
        if y > dashboard_height - 20: break
        
    return dashboard

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_duration = 1.0 / fps if fps > 0 else 0.04  # Default to 25 FPS if not available

    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    stop_event = threading.Event()

    producer_thread = threading.Thread(target=frame_producer, args=(cap, frame_queue, stop_event))
    processor_thread = threading.Thread(target=processor, args=(frame_queue, result_queue, stop_event))

    producer_thread.start()
    processor_thread.start()

    cv2.namedWindow("Cookie Tracker", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tracking Dashboard", cv2.WINDOW_NORMAL)
    
    # Resize windows for better performance
    cv2.resizeWindow("Cookie Tracker", 960, 540)
    cv2.resizeWindow("Tracking Dashboard", 900, 600)

    frame_skip_count = 0
    dashboard_cache = None
    last_frame_time = time.time()
    
    while True:
        try:
            result = result_queue.get(timeout=1)
            if result is None:
                break
            
            annotated_frame, live_tracks, history = result
            
            # Update dashboard less frequently for performance
            frame_skip_count += 1
            if frame_skip_count % 3 == 0 or dashboard_cache is None:
                dashboard_cache = draw_dashboard(live_tracks, history)
            
            cv2.imshow("Cookie Tracker", annotated_frame)
            if dashboard_cache is not None:
                cv2.imshow("Tracking Dashboard", dashboard_cache)

            # --- FPS Synchronization to match video's original speed ---
            processing_time = time.time() - last_frame_time
            wait_duration = max(1, int((target_frame_duration - processing_time) * 1000))
            
            if cv2.waitKey(wait_duration) & 0xFF == ord("q"):
                stop_event.set()
                break
            
            last_frame_time = time.time()

        except queue.Empty:
            if not processor_thread.is_alive():
                break
            continue

    # Cleanup
    stop_event.set()
    producer_thread.join(timeout=2)
    processor_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video using YOLOv8 and ByteTrack.")
    parser.add_argument("video_path", help="Path to the video file.")
    args = parser.parse_args()
    main(args.video_path) 


