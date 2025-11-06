#!/usr/bin/env python3
"""
Cleanup Script - Remove old and superfluous files.

This script removes:
- Old output directories (keep only last 5)
- Temporary files
- Old log files (keep only last 10)
- Empty directories
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_old_outputs(outputs_dir: Path, keep_last: int = 5):
    """Remove old output directories, keeping only the last N."""
    if not outputs_dir.exists():
        return
    
    # Get all output directories
    output_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name != 'logs' and d.name != 'archive' and d.name != 'debug' and d.name != 'model_comparison' and d.name != 'iterative_tests']
    
    # Sort by modification time
    output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only last N
    to_remove = output_dirs[keep_last:]
    
    for dir_path in to_remove:
        try:
            print(f"Removing old output: {dir_path.name}")
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")

def cleanup_temp_files(base_dir: Path):
    """Remove temporary files."""
    temp_patterns = ['**/temp_*', '**/tmp_*', '**/*.tmp', '**/*.bak', '**/*~']
    
    for pattern in temp_patterns:
        for file_path in base_dir.glob(pattern):
            try:
                if file_path.is_file():
                    print(f"Removing temp file: {file_path}")
                    file_path.unlink()
                elif file_path.is_dir():
                    print(f"Removing temp dir: {file_path}")
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def cleanup_old_logs(logs_dir: Path, keep_last: int = 10):
    """Remove old log files, keeping only the last N."""
    if not logs_dir.exists():
        return
    
    log_files = [f for f in logs_dir.iterdir() if f.is_file() and f.suffix == '.log']
    
    # Sort by modification time
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only last N
    to_remove = log_files[keep_last:]
    
    for file_path in to_remove:
        try:
            print(f"Removing old log: {file_path.name}")
            file_path.unlink()
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def cleanup_empty_dirs(base_dir: Path):
    """Remove empty directories."""
    for dir_path in base_dir.rglob('*'):
        if dir_path.is_dir():
            try:
                if not any(dir_path.iterdir()):
                    print(f"Removing empty directory: {dir_path}")
                    dir_path.rmdir()
            except Exception:
                pass  # Directory might not be empty or already removed

def main():
    """Main cleanup function."""
    print("=== Starting Cleanup ===")
    
    base_dir = Path(".")
    outputs_dir = base_dir / "outputs"
    logs_dir = outputs_dir / "logs"
    
    # Cleanup old outputs
    print("\n1. Cleaning up old output directories...")
    cleanup_old_outputs(outputs_dir, keep_last=5)
    
    # Cleanup temp files
    print("\n2. Cleaning up temporary files...")
    cleanup_temp_files(base_dir)
    
    # Cleanup old logs
    print("\n3. Cleaning up old log files...")
    cleanup_old_logs(logs_dir, keep_last=10)
    
    # Cleanup empty directories
    print("\n4. Cleaning up empty directories...")
    cleanup_empty_dirs(base_dir)
    
    print("\n=== Cleanup Complete ===")

if __name__ == "__main__":
    main()

