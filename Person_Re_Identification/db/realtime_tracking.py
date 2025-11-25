import datetime
import threading
import time
from typing import Dict, Optional
from db.tracking_database import PersonTrackingDatabase

class RealtimePersonTracker:
    """
    Real-time person tracking system that integrates with the tracking database.
    This class manages person entry/exit events and automatically records them.
    """
    
    def __init__(self, db_path="db/tracking_database.sqlite"):
        self.tracking_db = PersonTrackingDatabase(db_path)
        self.active_persons: Dict[int, Dict] = {}  # person_id -> tracking_info
        self.lock = threading.Lock()
        self.is_running = False
        self.tracking_thread = None
        
    def start_tracking(self):
        """Start the real-time tracking system."""
        if self.is_running:
            print("Tracking is already running.")
            return
        
        self.is_running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print("Real-time tracking started.")
    
    def stop_tracking(self):
        """Stop the real-time tracking system and close all active entries."""
        if not self.is_running:
            print("Tracking is not running.")
            return
        
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        
        # Close all active entries
        with self.lock:
            for person_id in list(self.active_persons.keys()):
                self._close_person_entry(person_id)
        
        print("Real-time tracking stopped.")
    
    def _tracking_loop(self):
        """Main tracking loop that monitors active persons."""
        while self.is_running:
            time.sleep(1)  # Check every second
    
    def person_detected(self, person_id: int, confidence: float = 1.0):
        """
        Called when a person is detected in the video frame.
        
        Args:
            person_id (int): The permanent ID of the person
            confidence (float): Detection confidence (0.0 to 1.0)
        """
        with self.lock:
            if person_id not in self.active_persons:
                # New person detected - record entry
                unique_id = self.tracking_db.record_person_entry(person_id)
                self.active_persons[person_id] = {
                    'unique_id': unique_id,
                    'entry_time': datetime.datetime.now(),
                    'last_seen': datetime.datetime.now(),
                    'confidence': confidence,
                    'frame_count': 1
                }
                print(f"🚶 Person {person_id} entered at {self.active_persons[person_id]['entry_time']}")
            else:
                # Person already active - update last seen time
                self.active_persons[person_id]['last_seen'] = datetime.datetime.now()
                self.active_persons[person_id]['frame_count'] += 1
                self.active_persons[person_id]['confidence'] = max(
                    self.active_persons[person_id]['confidence'], 
                    confidence
                )
    
    def person_lost(self, person_id: int, timeout_seconds: int = 5):
        """
        Called when a person is no longer detected in the video frame.
        
        Args:
            person_id (int): The permanent ID of the person
            timeout_seconds (int): Time to wait before considering person truly exited
        """
        with self.lock:
            if person_id in self.active_persons:
                # Mark as potentially lost
                self.active_persons[person_id]['lost_time'] = datetime.datetime.now()
                self.active_persons[person_id]['timeout_seconds'] = timeout_seconds
    
    def _close_person_entry(self, person_id: int):
        """Close the tracking entry for a person."""
        if person_id in self.active_persons:
            unique_id = self.active_persons[person_id]['unique_id']
            self.tracking_db.record_person_exit(person_id, unique_id=unique_id)
            del self.active_persons[person_id]
            print(f"🚶 Person {person_id} exited")
    
    def update_tracking(self, detected_persons: list, lost_persons: list = None):
        """
        Update tracking state based on current detections.
        
        Args:
            detected_persons (list): List of detected person IDs
            lost_persons (list, optional): List of person IDs that are no longer detected
        """
        if lost_persons is None:
            lost_persons = []
        
        with self.lock:
            # Process detected persons
            for person_id in detected_persons:
                self.person_detected(person_id)
            
            # Process lost persons
            for person_id in lost_persons:
                self.person_lost(person_id)
            
            # Check for timeouts and close entries
            current_time = datetime.datetime.now()
            to_close = []
            
            for person_id, info in self.active_persons.items():
                if 'lost_time' in info:
                    time_since_lost = (current_time - info['lost_time']).total_seconds()
                    if time_since_lost >= info.get('timeout_seconds', 5):
                        to_close.append(person_id)
            
            # Close timed-out entries
            for person_id in to_close:
                self._close_person_entry(person_id)
    
    def get_active_persons(self):
        """Get currently active persons."""
        with self.lock:
            return list(self.active_persons.keys())
    
    def get_tracking_summary(self):
        """Get current tracking summary."""
        return self.tracking_db.get_tracking_summary()
    
    def get_person_history(self, person_id: int):
        """Get tracking history for a specific person."""
        return self.tracking_db.get_person_tracking_history(person_id)
    
    def export_data(self, output_path: str = "tracking_data.csv"):
        """Export tracking data to CSV."""
        self.tracking_db.export_to_csv(output_path)


# Integration helper for existing tracking systems
class TrackingIntegration:
    """
    Helper class to integrate the real-time tracking with existing tracking systems.
    """
    
    def __init__(self, db_path="db/tracking_database.sqlite"):
        self.tracker = RealtimePersonTracker(db_path)
        self.tracker.start_tracking()
    
    def process_frame_detections(self, detections: list):
        """
        Process detections from a video frame.
        
        Args:
            detections (list): List of detection dictionaries with 'person_id' keys
        """
        detected_person_ids = [det['person_id'] for det in detections if 'person_id' in det]
        
        # Get currently active persons
        active_persons = set(self.tracker.get_active_persons())
        detected_persons = set(detected_person_ids)
        
        # Find lost persons
        lost_persons = list(active_persons - detected_persons)
        
        # Update tracking
        self.tracker.update_tracking(detected_person_ids, lost_persons)
    
    def get_current_status(self):
        """Get current tracking status."""
        active_persons = self.tracker.get_active_persons()
        summary = self.tracker.get_tracking_summary()
        
        return {
            'active_persons': active_persons,
            'total_entries': summary.get('total_entries', 0),
            'unique_persons': summary.get('unique_persons', 0),
            'completed_entries': summary.get('completed_entries', 0)
        }
    
    def cleanup(self):
        """Clean up and stop tracking."""
        self.tracker.stop_tracking()


# Example usage
if __name__ == '__main__':
    print("=== Real-time Tracking System Test ===")
    
    # Initialize tracking integration
    tracking_integration = TrackingIntegration()
    
    try:
        # Simulate video processing
        for frame_num in range(10):
            print(f"\n--- Frame {frame_num + 1} ---")
            
            # Simulate detections
            if frame_num < 3:
                # Person 1 appears
                detections = [{'person_id': 1, 'confidence': 0.95}]
            elif frame_num < 7:
                # Person 1 and 2 appear
                detections = [
                    {'person_id': 1, 'confidence': 0.92},
                    {'person_id': 2, 'confidence': 0.88}
                ]
            else:
                # Only person 2 remains
                detections = [{'person_id': 2, 'confidence': 0.90}]
            
            # Process detections
            tracking_integration.process_frame_detections(detections)
            
            # Get current status
            status = tracking_integration.get_current_status()
            print(f"Active persons: {status['active_persons']}")
            print(f"Total entries: {status['total_entries']}")
            
            time.sleep(1)  # Simulate frame processing time
        
        # Wait a bit for timeouts
        time.sleep(6)
        
        # Final status
        final_status = tracking_integration.get_current_status()
        print(f"\n=== Final Status ===")
        print(f"Active persons: {final_status['active_persons']}")
        print(f"Total entries: {final_status['total_entries']}")
        print(f"Unique persons: {final_status['unique_persons']}")
        print(f"Completed entries: {final_status['completed_entries']}")
        
        # Export data
        tracking_integration.tracker.export_data("test_tracking_data.csv")
        
    finally:
        # Cleanup
        tracking_integration.cleanup()
    
    print("\n=== Test completed ===") 