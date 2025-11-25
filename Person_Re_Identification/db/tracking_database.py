import sqlite3
import uuid
import datetime
from pathlib import Path
import os

class PersonTrackingDatabase:
    def __init__(self, db_path="db/tracking_database.sqlite"):
        self.db_path = db_path
        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        self._create_tables()
    
    def _get_db_conn(self):
        return sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        """Create the tracking table with the specified schema."""
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_tracking (
                unique_id TEXT PRIMARY KEY,           -- UUID for each tracking session
                person_id INTEGER NOT NULL,           -- Permanent person ID (1, 2, 3...)
                entry_timestamp TIMESTAMP NOT NULL,   -- When person entered frame
                exit_timestamp TIMESTAMP,             -- When person disappeared from frame
                duration REAL,                        -- Time difference between entry and exit
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Create indexes for faster queries
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_id ON person_tracking(person_id)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entry_timestamp ON person_tracking(entry_timestamp)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exit_timestamp ON person_tracking(exit_timestamp)
            """)
            conn.commit()
    
    def generate_unique_id(self):
        """Generate a unique UUID for the tracking entry."""
        return str(uuid.uuid4())
    
    def record_person_entry(self, person_id, entry_timestamp=None):
        """
        Record when a person enters the video frame.
        
        Args:
            person_id (int): The permanent ID of the person
            entry_timestamp (datetime, optional): Entry timestamp. Defaults to current time.
        
        Returns:
            str: The unique_id of the created entry
        """
        if entry_timestamp is None:
            entry_timestamp = datetime.datetime.now()
        
        unique_id = self.generate_unique_id()
        
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO person_tracking 
            (unique_id, person_id, entry_timestamp)
            VALUES (?, ?, ?)
            """, (unique_id, person_id, entry_timestamp))
            conn.commit()
        
        # print(f"🚶 Person ID {person_id} entered at {entry_timestamp} (Unique ID: {unique_id})")
        return unique_id
    
    def record_person_exit(self, person_id, exit_timestamp=None, unique_id=None):
        """
        Record when a person exits the video frame.
        
        Args:
            person_id (int): The permanent ID of the person
            exit_timestamp (datetime, optional): Exit timestamp. Defaults to current time.
            unique_id (str, optional): Specific unique_id to update. If None, updates the most recent entry without exit_timestamp.
        
        Returns:
            bool: True if successfully updated, False otherwise
        """
        if exit_timestamp is None:
            exit_timestamp = datetime.datetime.now()
        
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            
            if unique_id:
                # Update specific entry
                cursor.execute("""
                UPDATE person_tracking 
                SET exit_timestamp = ?, duration = ?
                WHERE unique_id = ? AND person_id = ?
                """, (exit_timestamp, None, unique_id, person_id))
            else:
                # Update the most recent entry without exit_timestamp
                cursor.execute("""
                UPDATE person_tracking 
                SET exit_timestamp = ?, duration = ?
                WHERE person_id = ? AND exit_timestamp IS NULL
                ORDER BY entry_timestamp DESC
                LIMIT 1
                """, (exit_timestamp, None, person_id))
            
            if cursor.rowcount > 0:
                # Calculate duration
                cursor.execute("""
                UPDATE person_tracking 
                SET duration = (julianday(exit_timestamp) - julianday(entry_timestamp)) * 24 * 3600
                WHERE exit_timestamp = ? AND duration IS NULL
                """, (exit_timestamp,))
                
                conn.commit()
                # print(f"🚶 Person ID {person_id} exited at {exit_timestamp}")
                return True
            else:
                # print(f"No open entry found for Person ID {person_id}")
                return False
    

    
    def get_person_tracking_history(self, person_id):
        """
        Get all tracking entries for a specific person.
        
        Args:
            person_id (int): The person ID to query
        
        Returns:
            list: List of tracking records for the person
        """
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT unique_id, person_id, entry_timestamp, exit_timestamp, duration, created_at
            FROM person_tracking 
            WHERE person_id = ?
            ORDER BY entry_timestamp DESC
            """, (person_id,))
            
            return cursor.fetchall()
    
    def get_active_entries(self):
        """
        Get all entries that haven't been closed (no exit_timestamp).
        
        Returns:
            list: List of active tracking entries
        """
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT unique_id, person_id, entry_timestamp
            FROM person_tracking 
            WHERE exit_timestamp IS NULL
            ORDER BY entry_timestamp DESC
            """)
            
            return cursor.fetchall()
    
    def get_tracking_summary(self, person_id=None):
        """
        Get tracking summary statistics.
        
        Args:
            person_id (int, optional): Specific person ID. If None, returns overall summary.
        
        Returns:
            dict: Summary statistics
        """
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            
            if person_id:
                # Summary for specific person
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(exit_timestamp) as completed_entries,
                    AVG(duration) as avg_duration,
                    MIN(entry_timestamp) as first_seen,
                    MAX(entry_timestamp) as last_seen
                FROM person_tracking 
                WHERE person_id = ?
                """, (person_id,))
            else:
                # Overall summary
                cursor.execute("""
                SELECT 
                    COUNT(DISTINCT person_id) as unique_persons,
                    COUNT(*) as total_entries,
                    COUNT(exit_timestamp) as completed_entries,
                    AVG(duration) as avg_duration,
                    MIN(entry_timestamp) as first_entry,
                    MAX(entry_timestamp) as last_entry
                FROM person_tracking
                """)
            
            row = cursor.fetchone()
            if person_id:
                return {
                    'person_id': person_id,
                    'total_entries': row[0],
                    'completed_entries': row[1],
                    'avg_duration_seconds': row[2],
                    'first_seen': row[3],
                    'last_seen': row[4]
                }
            else:
                return {
                    'unique_persons': row[0],
                    'total_entries': row[1],
                    'completed_entries': row[2],
                    'avg_duration_seconds': row[3],
                    'first_entry': row[4],
                    'last_entry': row[5]
                }
    
    def clear_database(self):
        """Clear all tracking data."""
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM person_tracking")
            conn.commit()
        # print("All tracking data cleared.")
    
    def export_to_csv(self, output_path="tracking_data.csv"):
        """Export tracking data to CSV file."""
        import csv
        
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT unique_id, person_id, entry_timestamp, exit_timestamp, duration, created_at
            FROM person_tracking 
            ORDER BY entry_timestamp DESC
            """)
            
            rows = cursor.fetchall()
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['unique_id', 'person_id', 'entry_timestamp', 'exit_timestamp', 'duration', 'created_at'])
                writer.writerows(rows)
        
        # print(f"✅ Tracking data exported to {output_path}")


# Example usage and testing
if __name__ == '__main__':
    # Initialize database
    tracking_db = PersonTrackingDatabase()
    
    # Clear any existing data
    tracking_db.clear_database()
    
    # print("=== Testing Person Tracking Database ===")
    
    # Simulate person entry
    person_id = 1
    unique_id1 = tracking_db.record_person_entry(person_id)
    # print(f"   ✓ Person entry recorded with Unique ID: {unique_id1}")
    
    # print("\n2. Testing person exit...")
    # Simulate person exit
    import time
    time.sleep(2)  # Simulate time passing
    success = tracking_db.record_person_exit(person_id)
    # print(f"   ✓ Person exit recorded: {success}")
    
    # print("\n3. Testing another person entry...")
    # Simulate another entry
    time.sleep(1)
    unique_id2 = tracking_db.record_person_entry(person_id)
    # print(f"   ✓ Another person entry recorded with Unique ID: {unique_id2}")
    
    # print("\n4. Testing different person...")
    # Simulate a different person
    person_id_2 = 2
    unique_id3 = tracking_db.record_person_entry(person_id_2)
    # print(f"   ✓ Different person (ID 2) entry recorded with Unique ID: {unique_id3}")
    
    # print("\n5. Testing tracking history...")
    # Get tracking history
    history = tracking_db.get_person_tracking_history(person_id)
    # print(f"   📊 Tracking history for Person ID {person_id}:")
    # for entry in history:
    #     unique_id, person_id, entry_time, exit_time, duration, created = entry
    #     status = "🔄 COMPLETED" if exit_time else "🚶 ACTIVE"
    #     print(f"     {status}: {entry_time} | Exit: {exit_time} | Duration: {duration} seconds")
    
    # print("\n6. Testing summary statistics...")
    # Get summary
    summary = tracking_db.get_tracking_summary(person_id)
    # print(f"   📈 Summary for Person ID {person_id}:")
    # for key, value in summary.items():
    #     print(f"     {key}: {value}")
    
    # print("\n7. Testing overall summary...")
    # Get overall summary
    overall_summary = tracking_db.get_tracking_summary()
    # print(f"   📊 Overall Summary:")
    # for key, value in overall_summary.items():
    #     print(f"     {key}: {value}")
    
    # print("\n8. Testing active entries...")
    # Get active entries
    active_entries = tracking_db.get_active_entries()
    # print(f"   🚶 Active entries:")
    # for entry in active_entries:
    #     unique_id, person_id, entry_time = entry
    #     print(f"     Person ID {person_id}: {entry_id}: {entry_time}")
    
    # print("\n9. Testing CSV export...")
    # Export to CSV
    tracking_db.export_to_csv()
    # print(f"   ✓ Data exported to test_tracking_export.csv")
    
    # print("\n=== All tests passed! ===") 