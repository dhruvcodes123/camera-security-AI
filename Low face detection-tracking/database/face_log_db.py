import sqlite3
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union


class FaceLogDatabase:
    """
    SQLite database for storing face tracking logs.
    
    Stores information about:
    - Face detection events
    - Person entry/exit times
    - Tracking statistics
    """
    
    def __init__(self, db_path: str = "face_logs.db"):
        """
        Initialize the face log database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.pending_operations = 0
        self.max_pending_operations = 10  # Commit after this many operations
        self.initialize_db()
        
    def initialize_db(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables
            self._create_tables()
            
            logging.info(f"Face log database initialized at {self.db_path}")
            
        except Exception as e:
            logging.error(f"Failed to initialize face log database: {e}")
            if self.conn:
                self.conn.close()
                
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Face tracking events table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_events (
            unique_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL,
            similarity_score REAL,
            frame_number INTEGER,
            track_id INTEGER,
            entry_time TEXT,
            exit_time TEXT
        )
        ''')
        
        # Person presence tracking table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_presence (
            unique_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            duration INTEGER,
            status TEXT DEFAULT 'active'
        )
        ''')
        
        # Session information table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_info (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            video_source TEXT,
            total_frames INTEGER,
            total_persons INTEGER,
            notes TEXT
        )
        ''')
        
        self.conn.commit()
        
    def start_new_session(self, video_source: str) -> int:
        """
        Start a new tracking session.
        
        Args:
            video_source: Path or identifier of the video source
            
        Returns:
            session_id: ID of the new session
        """
        try:
            current_time = datetime.now().isoformat()
            self.cursor.execute(
                "INSERT INTO session_info (start_time, video_source) VALUES (?, ?)",
                (current_time, video_source)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            logging.error(f"Failed to start new session: {e}")
            return -1
            
    def end_session(self, session_id: int, total_frames: int, total_persons: int):
        """
        End a tracking session and update statistics.
        
        Args:
            session_id: ID of the session to end
            total_frames: Total frames processed
            total_persons: Total unique persons detected
        """
        try:
            current_time = datetime.now().isoformat()
            self.cursor.execute(
                "UPDATE session_info SET end_time = ?, total_frames = ?, total_persons = ? WHERE session_id = ?",
                (current_time, total_frames, total_persons, session_id)
            )
            
            # Close any active person records
            self.cursor.execute(
                "UPDATE person_presence SET exit_time = ?, status = 'exited', duration = (strftime('%s', ?) - strftime('%s', entry_time)) WHERE exit_time IS NULL AND status = 'active'",
                (current_time, current_time)
            )
            
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to end session: {e}")
            
    def record_face_event(self, face_id: int, event_type: str, confidence: float = None,
                         similarity_score: float = None, frame_number: int = None,
                         track_id: int = None, entry_time: str = None, exit_time: str = None) -> int:
        """
        Record a face detection or tracking event.
        
        Args:
            face_id: Person ID of the face
            event_type: Type of event (NEW, REIDENTIFIED, TRACKED)
            confidence: Detection confidence
            similarity_score: Re-identification similarity score
            frame_number: Frame number where event occurred
            track_id: Track ID associated with the face
            entry_time: When the person entered the scene
            exit_time: When the person exited the scene
            
        Returns:
            unique_id: ID of the recorded event
        """
        try:
            current_time = datetime.now().isoformat()
            self.cursor.execute(
                "INSERT INTO face_events (face_id, event_type, timestamp, confidence, similarity_score, frame_number, track_id, entry_time, exit_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (face_id, event_type, current_time, confidence, similarity_score, frame_number, track_id, entry_time, exit_time)
            )
            self._increment_and_check_commit()
            return self.cursor.lastrowid
        except Exception as e:
            logging.error(f"Failed to record face event: {e}")
            return -1
            
    def _increment_and_check_commit(self):
        """Increment pending operations counter and commit if threshold reached"""
        self.pending_operations += 1
        if self.pending_operations >= self.max_pending_operations:
            self.conn.commit()
            self.pending_operations = 0
            
    def record_person_entry(self, face_id: int) -> int:
        """
        Record a person entering the scene.
        
        Args:
            face_id: Person ID of the face
            
        Returns:
            unique_id: ID of the entry record
        """
        try:
            current_time = datetime.now().isoformat()
            
            # Check if person is already active
            self.cursor.execute(
                "SELECT COUNT(*) FROM person_presence WHERE face_id = ? AND status = 'active'",
                (face_id,)
            )
            count = self.cursor.fetchone()[0]
            
            if count == 0:
                # Person is not active, create new entry
                self.cursor.execute(
                    "INSERT INTO person_presence (face_id, entry_time, status) VALUES (?, ?, 'active')",
                    (face_id, current_time)
                )
                self._increment_and_check_commit()
                return self.cursor.lastrowid
            else:
                # Person is already active, don't create duplicate
                return -1
                
        except Exception as e:
            logging.error(f"Failed to record person entry: {e}")
            return -1
            
    def record_person_exit(self, face_id: int) -> bool:
        """
        Record a person exiting the scene.
        
        Args:
            face_id: Person ID of the face
            
        Returns:
            success: Whether the exit was recorded successfully
        """
        try:
            current_time = datetime.now().isoformat()
            
            # Find the active entry for this person
            self.cursor.execute(
                "SELECT unique_id, entry_time FROM person_presence WHERE face_id = ? AND status = 'active'",
                (face_id,)
            )
            result = self.cursor.fetchone()
            
            if result:
                unique_id, entry_time = result
                # Calculate duration in seconds
                entry_dt = datetime.fromisoformat(entry_time)
                exit_dt = datetime.fromisoformat(current_time)
                duration = int((exit_dt - entry_dt).total_seconds())
                
                # Update the record
                self.cursor.execute(
                    "UPDATE person_presence SET exit_time = ?, status = 'exited', duration = ? WHERE unique_id = ?",
                    (current_time, duration, unique_id)
                )
                self._increment_and_check_commit()
                return True
            else:
                logging.warning(f"No active entry found for person {face_id}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to record person exit: {e}")
            return False
            
    def get_person_history(self, face_id: int) -> List[Dict]:
        """
        Get history of events for a specific person.
        
        Args:
            face_id: Person ID to query
            
        Returns:
            List of event dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT * FROM face_events WHERE face_id = ? ORDER BY timestamp",
                (face_id,)
            )
            columns = [col[0] for col in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get person history: {e}")
            return []
            
    def get_active_persons(self) -> List[Dict]:
        """
        Get list of currently active persons.
        
        Returns:
            List of active person dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT * FROM person_presence WHERE status = 'active'"
            )
            columns = [col[0] for col in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get active persons: {e}")
            return []
            
    def force_commit(self):
        """Force a commit of pending database operations"""
        if self.conn and self.pending_operations > 0:
            try:
                self.conn.commit()
                self.pending_operations = 0
            except Exception as e:
                logging.error(f"Failed to force commit: {e}")
            
    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                # Commit any pending changes
                if self.pending_operations > 0:
                    self.conn.commit()
                self.conn.close()
                logging.info("Face log database connection closed")
            except Exception as e:
                logging.error(f"Error closing database: {e}")
            
    def __del__(self):
        """Destructor to ensure database connection is closed."""
        self.close() 