import numpy as np
import cv2
from collections import OrderedDict
import logging
from typing import List, Tuple, Optional
import time


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    Uses Kalman filter for motion prediction.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialize tracker with bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] format
        """
        # Define constant velocity model: [x, y, s, r, dx, dy, ds]
        # where x,y = center, s = scale (area), r = aspect ratio, dx,dy,ds = velocities
        self.kf = cv2.KalmanFilter(7, 4)
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix - we can only observe x, y, s, r
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.1
        
        # Measurement noise covariance  
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance matrix
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        
        # Initialize state with position and zero velocity
        pos = self.convert_bbox_to_z(bbox)
        # State is [x, y, s, r, dx, dy, ds] - add zero velocities
        initial_state = np.zeros((7, 1), dtype=np.float32)
        initial_state[:4, 0] = pos
        self.kf.statePre = initial_state.copy()
        self.kf.statePost = initial_state.copy()
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = 0.0

    def update(self, bbox, confidence=1.0):
        """
        Updates the state vector with observed bbox.
        """
        try:
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.confidence = confidence
            
            # Correct step with measurement
            measurement = self.convert_bbox_to_z(bbox).reshape(-1, 1)
            
            # Check measurement validity before correction
            if not all(np.isfinite(measurement.flatten())):
                logging.warning(f"Track {self.id}: Invalid measurement, skipping update")
                return
                
            self.kf.correct(measurement)
            
            # Check state validity after correction
            state = self.kf.statePost.flatten()
            if not all(np.isfinite(state)):
                # Reset to measurement-based state if correction failed
                z = self.convert_bbox_to_z(bbox)
                initial_state = np.zeros((7, 1), dtype=np.float32)
                initial_state[:4, 0] = z
                self.kf.statePost = initial_state.copy()
                logging.warning(f"Track {self.id}: Reset invalid state after correction")
                
        except Exception as e:
            logging.error(f"Track {self.id}: Update error - {e}")
            # Reset to a safe state based on current detection
            try:
                z = self.convert_bbox_to_z(bbox)
                initial_state = np.zeros((7, 1), dtype=np.float32)
                initial_state[:4, 0] = z
                self.kf.statePost = initial_state.copy()
                self.kf.statePre = initial_state.copy()
            except:
                pass  # Give up on this update

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        try:
            # Predict step
            self.kf.predict()
            
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            
            # Check for invalid state before converting
            state = self.kf.statePost.flatten()
            if not all(np.isfinite(state)):
                # Reset to a default state if invalid
                self.kf.statePost = np.array([[100], [100], [10000], [1], [0], [0], [0]], dtype=np.float32)
                logging.warning(f"Track {self.id}: Reset invalid Kalman state")
            
            self.history.append(self.convert_x_to_bbox(self.kf.statePost))
            return self.history[-1]
        except Exception as e:
            logging.error(f"Track {self.id}: Prediction error - {e}")
            # Return a safe default bbox
            return np.array([[100., 100., 200., 200.]])

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.statePost)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = max(bbox[2] - bbox[0], 1.0)  # Ensure minimum width
        h = max(bbox[3] - bbox[1], 1.0)  # Ensure minimum height
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # scale is just area
        r = w / float(h) if h > 0 else 1.0
        
        # Ensure no invalid values
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(s) and np.isfinite(r)):
            return np.array([100.0, 100.0, 10000.0, 1.0], dtype=np.float32)  # Default safe values
            
        return np.array([x, y, s, r], dtype=np.float32)

    @staticmethod 
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        # Ensure valid values
        if not (np.isfinite(x[2]) and np.isfinite(x[3]) and x[2] > 0 and x[3] > 0):
            # Return a default bbox if invalid
            if score is None:
                return np.array([100., 100., 200., 200.]).reshape((1, 4))
            else:
                return np.array([100., 100., 200., 200., 0.5]).reshape((1, 5))
        
        w = np.sqrt(max(x[2] * x[3], 1.0))  # Ensure positive value under sqrt
        h = x[2] / max(w, 1.0)  # Avoid division by zero
        
        # Ensure minimum dimensions
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def linear_assignment(cost_matrix):
    """
    Simple linear assignment using Hungarian algorithm
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))
    except ImportError:
        # Fallback to greedy assignment if scipy not available
        matches = []
        for i in range(min(cost_matrix.shape)):
            min_row = np.argmin(cost_matrix)
            row, col = np.unravel_index(min_row, cost_matrix.shape)
            matches.append([row, col])
            cost_matrix[row, :] = 1e6
            cost_matrix[:, col] = 1e6
        return np.array(matches)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_trackers, dtype=np.int32)


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker
    """
    
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Frame rate of the video
            track_thresh: Threshold for high confidence detections
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Threshold for matching tracks to detections
            aspect_ratio_thresh: Threshold for filtering out elongated bboxes
            min_box_area: Minimum area of bounding box to be considered
        """
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        
        # ByteTrack specific parameters
        self.low_thresh = max(track_thresh - 0.1, 0.1)
        
        KalmanBoxTracker.count = 0  # Reset counter
        
        logging.info(f"ByteTracker initialized with track_thresh={track_thresh}, match_thresh={match_thresh}")

    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, confidence]
        
        Returns:
            List of active tracks with format [x1, y1, x2, y2, track_id, confidence]
        """
        self.frame_id += 1
        
        # Filter detections
        if len(detections) > 0:
            detections = np.array(detections)
            
            # Filter by minimum area and aspect ratio
            valid_dets = []
            for det in detections:
                w = det[2] - det[0]
                h = det[3] - det[1]
                if w * h >= self.min_box_area and w / h <= self.aspect_ratio_thresh:
                    valid_dets.append(det)
            
            if len(valid_dets) > 0:
                detections = np.array(valid_dets)
            else:
                detections = np.empty((0, 5))
        else:
            detections = np.empty((0, 5))

        # Separate high and low confidence detections
        if len(detections) > 0:
            high_dets = detections[detections[:, 4] >= self.track_thresh]
            low_dets = detections[detections[:, 4] < self.track_thresh]
            low_dets = low_dets[low_dets[:, 4] >= self.low_thresh]
        else:
            high_dets = np.empty((0, 5))
            low_dets = np.empty((0, 5))

        # Predict all tracks
        for track in self.tracked_stracks:
            track.predict()

        # Get current track positions
        track_bboxes = np.array([track.get_state().flatten() for track in self.tracked_stracks])
        
        # First association with high confidence detections
        if len(high_dets) > 0 and len(track_bboxes) > 0:
            matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                high_dets[:, :4], track_bboxes, self.match_thresh)
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(high_dets)) if len(high_dets) > 0 else np.array([])
            unmatched_trks = np.arange(len(self.tracked_stracks))

        # Update matched tracks with high confidence detections
        for m in matches:
            self.tracked_stracks[m[1]].update(high_dets[m[0], :4], high_dets[m[0], 4])

        # Second association with low confidence detections for unmatched tracks
        if len(low_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_track_bboxes = track_bboxes[unmatched_trks]
            r_matches, r_unmatched_dets, r_unmatched_trks = associate_detections_to_trackers(
                low_dets[:, :4], unmatched_track_bboxes, 0.5)
            
            # Update matched tracks with low confidence detections
            for m in r_matches:
                track_idx = unmatched_trks[m[1]]
                self.tracked_stracks[track_idx].update(low_dets[m[0], :4], low_dets[m[0], 4])
            
            # Update unmatched tracks list - ensure r_unmatched_trks is integer type
            r_unmatched_trks = r_unmatched_trks.astype(np.int32)
            unmatched_trks = unmatched_trks[r_unmatched_trks]

        # Deal with unmatched detections (create new tracks)
        for i in unmatched_dets:
            if high_dets[i, 4] >= self.track_thresh:
                track = KalmanBoxTracker(high_dets[i, :4])
                track.update(high_dets[i, :4], high_dets[i, 4])
                self.tracked_stracks.append(track)

        # Mark unmatched tracks as lost
        for i in unmatched_trks:
            track = self.tracked_stracks[i]
            if track.time_since_update > self.track_buffer:
                self.lost_stracks.append(track)

        # Remove lost tracks
        self.tracked_stracks = [t for t in self.tracked_stracks if t.time_since_update <= self.track_buffer]

        # Remove old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks if self.frame_id - t.time_since_update <= self.track_buffer]

        # Prepare output
        output_stracks = []
        for track in self.tracked_stracks:
            if track.time_since_update <= 1 and track.hit_streak >= 1:
                try:
                    bbox = track.get_state().flatten()
                    # Validate bbox before adding to output
                    if len(bbox) >= 4 and all(np.isfinite(bbox[:4])):
                        output_stracks.append([bbox[0], bbox[1], bbox[2], bbox[3], track.id, track.confidence])
                except Exception as e:
                    logging.warning(f"Track {track.id}: Failed to get state - {e}")
                    continue

        return output_stracks

    def get_track_count(self):
        """Get total number of active tracks"""
        return len(self.tracked_stracks)

    def clear_tracks(self):
        """Clear all tracks"""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        KalmanBoxTracker.count = 0
        self.frame_id = 0
        logging.info("All tracks cleared") 