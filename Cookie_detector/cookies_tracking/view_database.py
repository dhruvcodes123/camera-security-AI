#!/usr/bin/env python3
"""
Script to view and analyze cookie tracking database contents
"""

import database
import sys
from datetime import datetime


def format_duration(duration_minutes):
    """Format duration in minutes to human readable format"""
    if duration_minutes is None:
        return "N/A"
    
    if duration_minutes < 1:
        return f"{duration_minutes * 60:.1f} seconds"
    elif duration_minutes < 60:
        return f"{duration_minutes:.1f} minutes"
    else:
        hours = int(duration_minutes // 60)
        minutes = duration_minutes % 60
        return f"{hours}h {minutes:.1f}m"


def view_all_sessions(conn):
    """Display all cookie tracking sessions"""
    sessions = database.get_all_sessions(conn)
    
    if not sessions:
        print("No cookie tracking sessions found in database.")
        return
    
    print("\n" + "="*120)
    print("COOKIE TRACKING SESSIONS")
    print("="*120)
    print(f"{'ID':<4} {'Track ID':<8} {'Cookie Type':<12} {'Entry Time':<19} {'Exit Time':<19} {'Duration':<15} {'Static':<8} {'Stability':<10}")
    print("-"*120)
    
    for session in sessions:
        id_, tracking_id, cookie_type, entry_time, exit_time, duration_minutes, is_stationary, stability_score, created_at, updated_at = session
        
        # Format times
        entry_formatted = datetime.fromisoformat(entry_time).strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A"
        exit_formatted = datetime.fromisoformat(exit_time).strftime("%Y-%m-%d %H:%M:%S") if exit_time else "ACTIVE"
        duration_formatted = format_duration(duration_minutes)
        static_status = "Yes" if is_stationary else "No"
        stability_formatted = f"{stability_score:.2f}" if stability_score is not None else "0.00"
        
        print(f"{id_:<4} {tracking_id:<8} {cookie_type:<12} {entry_formatted:<19} {exit_formatted:<19} {duration_formatted:<15} {static_status:<8} {stability_formatted:<10}")


def view_active_sessions(conn):
    """Display only active (not exited) cookie tracking sessions"""
    sessions = database.get_active_sessions(conn)
    
    if not sessions:
        print("No active cookie tracking sessions found.")
        return
    
    print("\n" + "="*100)
    print("ACTIVE COOKIE TRACKING SESSIONS")
    print("="*100)
    print(f"{'ID':<4} {'Track ID':<8} {'Cookie Type':<12} {'Entry Time':<19} {'Duration':<15} {'Static':<8} {'Stability':<10}")
    print("-"*100)
    
    current_time = datetime.now()
    
    for session in sessions:
        id_, tracking_id, cookie_type, entry_time, exit_time, duration_minutes, is_stationary, stability_score, created_at, updated_at = session
        
        # Calculate current duration for active sessions
        entry_dt = datetime.fromisoformat(entry_time)
        current_duration = (current_time - entry_dt).total_seconds() / 60.0
        
        entry_formatted = entry_dt.strftime("%Y-%m-%d %H:%M:%S")
        duration_formatted = format_duration(current_duration)
        static_status = "Yes" if is_stationary else "No"
        stability_formatted = f"{stability_score:.2f}" if stability_score is not None else "0.00"
        
        print(f"{id_:<4} {tracking_id:<8} {cookie_type:<12} {entry_formatted:<19} {duration_formatted:<15} {static_status:<8} {stability_formatted:<10}")


def show_statistics(conn):
    """Show database statistics"""
    # Get all sessions
    all_sessions = database.get_all_sessions(conn)
    active_sessions = database.get_active_sessions(conn)
    
    total_sessions = len(all_sessions)
    active_count = len(active_sessions)
    completed_count = total_sessions - active_count
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    print(f"Total Sessions:     {total_sessions}")
    print(f"Active Sessions:    {active_count}")
    print(f"Completed Sessions: {completed_count}")
    
    if all_sessions:
        # Cookie type breakdown
        cookie_types = {}
        stationary_count = 0
        total_duration = 0
        completed_sessions = 0
        
        for session in all_sessions:
            cookie_type = session[2]
            is_stationary = session[6]
            duration_minutes = session[5]
            
            cookie_types[cookie_type] = cookie_types.get(cookie_type, 0) + 1
            
            if is_stationary:
                stationary_count += 1
            
            if duration_minutes is not None:
                total_duration += duration_minutes
                completed_sessions += 1
        
        print(f"\nCookie Type Breakdown:")
        for cookie_type, count in cookie_types.items():
            print(f"  {cookie_type}: {count}")
        
        print(f"\nStationary Objects: {stationary_count}/{total_sessions} ({stationary_count/total_sessions*100:.1f}%)")
        
        if completed_sessions > 0:
            avg_duration = total_duration / completed_sessions
            print(f"Average Duration:   {format_duration(avg_duration)}")


def main():
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
    else:
        db_file = "cookie_tracking.db"
    
    # Create connection to database
    conn = database.create_connection(db_file)
    
    if conn is None:
        print(f"Error: Could not connect to database {db_file}")
        print("Make sure the database file exists and run the tracker first to create it.")
        return
    
    try:
        print(f"Viewing database: {db_file}")
        
        # Show statistics
        show_statistics(conn)
        
        # Show active sessions
        view_active_sessions(conn)
        
        # Show all sessions
        view_all_sessions(conn)
        
    except Exception as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main() 