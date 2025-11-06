#!/usr/bin/env python3
"""
Repository Cleanup Script - Removes old/unused files and organizes outputs.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Files older than 7 days will be archived
ARCHIVE_DAYS = 7

def cleanup_old_outputs(outputs_dir: Path, archive_dir: Path):
    """Archive old output directories."""
    if not outputs_dir.exists():
        return
    
    archive_dir.mkdir(parents=True, exist_ok=True)
    cutoff_date = datetime.now() - timedelta(days=ARCHIVE_DAYS)
    
    archived_count = 0
    for item in outputs_dir.iterdir():
        if item.is_dir() and item.name != "archive" and item.name != "backups" and item.name != "debug" and item.name != "logs":
            # Check modification time
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            if mtime < cutoff_date:
                # Move to archive
                archive_path = archive_dir / item.name
                if archive_path.exists():
                    shutil.rmtree(archive_path)
                shutil.move(str(item), str(archive_path))
                archived_count += 1
                print(f"[ARCHIVED] {item.name}")
    
    return archived_count

def cleanup_old_logs(outputs_dir: Path):
    """Remove old log files (keep only last 7 days)."""
    logs_dir = outputs_dir / "logs"
    if not logs_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=ARCHIVE_DAYS)
    removed_count = 0
    
    for log_file in logs_dir.glob("*.log"):
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        if mtime < cutoff_date:
            log_file.unlink()
            removed_count += 1
            print(f"[REMOVED] {log_file.name}")
    
    return removed_count

def main():
    """Main cleanup function."""
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    archive_dir = outputs_dir / "archive"
    
    print("=" * 80)
    print("REPOSITORY CLEANUP")
    print("=" * 80)
    print()
    
    # Cleanup old outputs
    print("Cleaning up old outputs...")
    archived = cleanup_old_outputs(outputs_dir, archive_dir)
    print(f"  Archived {archived} old output directories")
    print()
    
    # Cleanup old logs
    print("Cleaning up old logs...")
    removed = cleanup_old_logs(outputs_dir)
    print(f"  Removed {removed} old log files")
    print()
    
    print("=" * 80)
    print("CLEANUP COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

