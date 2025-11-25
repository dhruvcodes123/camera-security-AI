# Cookie Tracking Database

This project includes a SQLite database to store detailed cookie tracking session information.

## Database Structure

### `cookie_sessions` Table

The main table stores tracking sessions with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| `tracking_id` | INTEGER UNIQUE | Global tracking ID assigned to each cookie |
| `cookie_type` | TEXT | Type/class of the detected cookie |
| `entry_time` | TEXT (ISO format) | When the cookie was first detected |
| `exit_time` | TEXT (ISO format) | When the cookie exited (NULL for active sessions) |
| `duration_minutes` | REAL | Total duration in minutes (calculated automatically) |
| `is_stationary` | BOOLEAN | Whether the cookie was stationary (0/1) |
| `stability_score` | REAL | Stability score (0.0-1.0, higher = more stable) |
| `created_at` | TEXT | Database record creation timestamp |
| `updated_at` | TEXT | Last update timestamp |

## Usage

### Running the Tracker

The tracking system automatically creates and updates the database:

```bash
python run_tracker.py path/to/your/video.mp4
```

This will:
1. Create `cookie_tracking.db` if it doesn't exist
2. Insert new sessions when cookies are first detected
3. Update sessions with stationary status and stability scores during tracking
4. Mark sessions as exited with duration when cookies leave the scene

### Viewing Database Contents

Use the included database viewer script:

```bash
python view_database.py
```

Or specify a different database file:

```bash
python view_database.py my_custom_database.db
```

The viewer shows:
- **Statistics**: Total sessions, active vs completed, cookie type breakdown
- **Active Sessions**: Currently tracked cookies with live duration
- **All Sessions**: Complete history with entry/exit times and durations

### Database Files

- `cookie_tracking.db` - Main database created by the tracker
- `tracks.db` - Legacy frame-by-frame tracking data (if needed)

## Example Output

```
DATABASE STATISTICS
============================================================
Total Sessions:     15
Active Sessions:    3
Completed Sessions: 12

Cookie Type Breakdown:
  cookie: 12
  biscuit: 3

Stationary Objects: 8/15 (53.3%)
Average Duration:   2.4 minutes

ACTIVE COOKIE TRACKING SESSIONS
====================================================================================================
ID   Track ID Cookie Type  Entry Time          Duration        Static   Stability
----------------------------------------------------------------------------------------------------
1    45       cookie       2024-01-15 14:30:25 5.2 minutes     Yes      0.85
2    47       biscuit      2024-01-15 14:32:10 3.1 minutes     No       0.42
3    48       cookie       2024-01-15 14:33:45 1.8 minutes     Yes      0.91
```

## Database Management

### Manual Queries

You can also query the database directly using SQLite:

```bash
sqlite3 cookie_tracking.db
```

Example queries:
```sql
-- Get all sessions for a specific cookie type
SELECT * FROM cookie_sessions WHERE cookie_type = 'cookie';

-- Get average duration by cookie type
SELECT cookie_type, AVG(duration_minutes) as avg_duration 
FROM cookie_sessions 
WHERE duration_minutes IS NOT NULL 
GROUP BY cookie_type;

-- Get stationary vs moving cookies
SELECT is_stationary, COUNT(*) as count 
FROM cookie_sessions 
GROUP BY is_stationary;
```

### Backup and Export

To backup your tracking data:

```bash
# Create a backup
cp cookie_tracking.db cookie_tracking_backup_$(date +%Y%m%d).db

# Export to CSV (requires sqlite3 command-line tool)
sqlite3 -header -csv cookie_tracking.db "SELECT * FROM cookie_sessions;" > tracking_export.csv
```

## Integration

The database functionality is integrated into the main tracking pipeline:

1. **Entry Detection**: New cookies create database entries immediately
2. **Real-time Updates**: Stationary status and stability scores update during tracking
3. **Exit Detection**: Sessions are closed with duration calculation when cookies leave
4. **Thread Safety**: Database operations are handled within the processor thread

## Configuration

Key parameters in `run_tracker.py`:

- `DB_FILE = "cookie_tracking.db"` - Database filename
- `EXIT_TIMEOUT_SECONDS = 60` - Time before marking a track as exited
- `STATIC_STABILITY_FRAMES = 10` - Frames needed to calculate stability
- `STATIC_MOVEMENT_THRESH = 10.0` - Pixel threshold for stationary detection 