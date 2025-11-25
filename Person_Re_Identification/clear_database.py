#!/usr/bin/env python3
"""
Database and Output Directory Cleanup Script

This script clears the entire database and all output directories before running the project.
This ensures a clean start for each run and prevents issues with stale data.
"""

import os
import shutil
import sqlite3
from pathlib import Path

def clear_entire_database():
    """Clear the entire database and all related files."""
    print("--- CLEARING ENTIRE DATABASE ---")
    
    # Remove main database
    db_path = "db/database.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed main database: {db_path}")
    else:
        print(f"ℹ️ Database file not found: {db_path}")
    
    # Remove all index files
    index_dir = Path("db/vector_db")
    if index_dir.exists():
        for index_file in index_dir.glob("*.index"):
            os.remove(index_file)
            print(f"✅ Removed index file: {index_file}")
    else:
        print(f"ℹ️ Index directory not found: {index_dir}")
    
    print("--- ENTIRE DATABASE CLEARED ---")

def clear_output_directories():
    """Clear all output directories."""
    print("--- CLEARING OUTPUT DIRECTORIES ---")
    
    output_dirs = [
        "data/processed_output/tracked_persons",
        "data/processed_output/detected_faces", 
        "data/processed_output/gait_cycles"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Removed directory: {dir_path}")
        else:
            print(f"ℹ️ Directory not found: {dir_path}")
    
    print("--- OUTPUT DIRECTORIES CLEARED ---")

def check_database_status():
    """Check the current status of the database."""
    print("--- DATABASE STATUS CHECK ---")
    
    # Check main database
    db_path = "db/database.sqlite"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📊 Database exists with {len(tables)} tables:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"   - {table[0]}: {count} rows")
            conn.close()
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    else:
        print("ℹ️ Database file does not exist")
    
    # Check index files
    index_dir = Path("db/vector_db")
    if index_dir.exists():
        index_files = list(index_dir.glob("*.index"))
        print(f"📊 Found {len(index_files)} index files:")
        for index_file in index_files:
            size = index_file.stat().st_size
            print(f"   - {index_file.name}: {size} bytes")
    else:
        print("ℹ️ Index directory does not exist")
    
    print("--- END DATABASE STATUS CHECK ---")

def main():
    """Main function to clear everything and verify."""
    print("🧹 DATABASE AND OUTPUT CLEANUP SCRIPT")
    print("=" * 50)
    
    # Check current status
    print("\n📋 Current Status:")
    check_database_status()
    
    # Clear everything
    print("\n🗑️ Clearing everything...")
    clear_entire_database()
    clear_output_directories()
    
    # Verify everything is cleared
    print("\n✅ Verification:")
    check_database_status()
    
    print("\n🎉 Cleanup completed! Database and output directories are now empty.")
    print("💡 IMPORTANT: When you run the project next time:")
    print("   - The first person will get a fused similarity score of 0.0 (no comparison made)")
    print("   - This indicates the vector database was truly empty")
    print("   - Subsequent persons will get proper similarity scores compared to existing persons")
    print("You can now run your project with a clean slate.")

if __name__ == "__main__":
    main() 