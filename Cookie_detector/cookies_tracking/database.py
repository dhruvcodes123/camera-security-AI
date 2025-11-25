import sqlite3
from sqlite3 import Error
from datetime import datetime


def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn):
    """Create a table to store tracking data"""
    try:
        sql_create_tracks_table = """CREATE TABLE IF NOT EXISTS tracks (
                                        id integer PRIMARY KEY,
                                        frame integer NOT NULL,
                                        tracker_id integer NOT NULL,
                                        class_name text NOT NULL,
                                        confidence real NOT NULL,
                                        bbox_x1 real NOT NULL,
                                        bbox_y1 real NOT NULL,
                                        bbox_x2 real NOT NULL,
                                        bbox_y2 real NOT NULL,
                                        timestamp text NOT NULL
                                    );"""
        c = conn.cursor()
        c.execute(sql_create_tracks_table)
    except Error as e:
        print(e)


def create_cookie_sessions_table(conn):
    """Create a table to store cookie tracking sessions"""
    try:
        sql_create_sessions_table = """CREATE TABLE IF NOT EXISTS cookie_sessions (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        tracking_id integer NOT NULL UNIQUE,
                                        cookie_type text NOT NULL,
                                        entry_time text NOT NULL,
                                        exit_time text,
                                        duration_minutes real,
                                        is_stationary boolean DEFAULT 0,
                                        stability_score real DEFAULT 0.0,
                                        created_at text NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                        updated_at text NOT NULL DEFAULT CURRENT_TIMESTAMP
                                    );"""
        c = conn.cursor()
        c.execute(sql_create_sessions_table)
    except Error as e:
        print(e)


def insert_track(conn, track_data):
    """
    Create a new track entry
    :param conn:
    :param track_data:
    :return:
    """
    sql = ''' INSERT INTO tracks(frame,tracker_id,class_name,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,timestamp)
              VALUES(?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, track_data)
    conn.commit()
    return cur.lastrowid


def insert_many_tracks(conn, track_data_list):
    """
    Create new track entries for a list of track data
    :param conn:
    :param track_data_list:
    :return:
    """
    sql = ''' INSERT INTO tracks(frame,tracker_id,class_name,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,timestamp)
              VALUES(?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.executemany(sql, track_data_list)
    conn.commit()


def insert_cookie_session(conn, tracking_id, cookie_type, entry_time, is_stationary=False, stability_score=0.0):
    """
    Create a new cookie session entry
    :param conn: database connection
    :param tracking_id: unique tracking ID for the cookie
    :param cookie_type: type/class of the cookie
    :param entry_time: datetime when cookie was first detected
    :param is_stationary: whether the cookie is stationary
    :param stability_score: stability score of the cookie
    :return: session ID
    """
    sql = '''INSERT OR IGNORE INTO cookie_sessions(tracking_id, cookie_type, entry_time, is_stationary, stability_score)
             VALUES(?,?,?,?,?)'''
    cur = conn.cursor()
    entry_time_str = entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time
    cur.execute(sql, (tracking_id, cookie_type, entry_time_str, is_stationary, stability_score))
    conn.commit()
    return cur.lastrowid


def update_cookie_session_exit(conn, tracking_id, exit_time, is_stationary=False, stability_score=0.0):
    """
    Update cookie session with exit time and calculate duration
    :param conn: database connection
    :param tracking_id: unique tracking ID for the cookie
    :param exit_time: datetime when cookie exited
    :param is_stationary: whether the cookie was stationary
    :param stability_score: final stability score
    :return: number of rows updated
    """
    # First get the entry time to calculate duration
    cur = conn.cursor()
    cur.execute("SELECT entry_time FROM cookie_sessions WHERE tracking_id = ?", (tracking_id,))
    result = cur.fetchone()
    
    if result:
        entry_time_str = result[0]
        entry_time = datetime.fromisoformat(entry_time_str)
        exit_time_obj = exit_time if isinstance(exit_time, datetime) else datetime.fromisoformat(exit_time)
        duration = exit_time_obj - entry_time
        duration_minutes = duration.total_seconds() / 60.0
        
        exit_time_str = exit_time_obj.isoformat() if isinstance(exit_time_obj, datetime) else exit_time
        
        sql = '''UPDATE cookie_sessions 
                 SET exit_time = ?, duration_minutes = ?, is_stationary = ?, stability_score = ?, updated_at = CURRENT_TIMESTAMP
                 WHERE tracking_id = ?'''
        cur.execute(sql, (exit_time_str, duration_minutes, is_stationary, stability_score, tracking_id))
        conn.commit()
        return cur.rowcount
    return 0


def get_active_sessions(conn):
    """Get all active (not exited) cookie sessions"""
    cur = conn.cursor()
    cur.execute("SELECT * FROM cookie_sessions WHERE exit_time IS NULL")
    return cur.fetchall()


def get_all_sessions(conn):
    """Get all cookie sessions"""
    cur = conn.cursor()
    cur.execute("SELECT * FROM cookie_sessions ORDER BY entry_time DESC")
    return cur.fetchall()


def get_session_by_tracking_id(conn, tracking_id):
    """Get cookie session by tracking ID"""
    cur = conn.cursor()
    cur.execute("SELECT * FROM cookie_sessions WHERE tracking_id = ?", (tracking_id,))
    return cur.fetchone()
