#!/usr/bin/env python3
"""
Restore script for learning_db.json.

Restores a backup of the learning database.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def restore_learning_db(backup_path: Path, learning_db_path: Optional[Path] = None) -> bool:
    """
    Restore learning_db.json from a backup.
    
    Args:
        backup_path: Path to backup file
        learning_db_path: Path to restore to (default: project_root/learning_db.json)
        
    Returns:
        True if restore successful, False otherwise
    """
    if not backup_path.exists():
        print(f"[ERROR] Backup file not found: {backup_path}")
        return False
    
    if learning_db_path is None:
        project_root = Path(__file__).parent.parent
        learning_db_path = project_root / "learning_db.json"
    
    try:
        # Create backup of current learning_db if it exists
        if learning_db_path.exists():
            current_backup = learning_db_path.parent / f"learning_db_current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(learning_db_path, current_backup)
            print(f"[OK] Current learning_db backed up: {current_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, learning_db_path)
        print(f"[OK] Learning DB restored: {learning_db_path}")
        
        # Load and display summary
        with open(learning_db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  - Recent analyses: {len(data.get('recent_analyses', []))}")
        print(f"  - Successful patterns: {len(data.get('successful_patterns', {}))}")
        print(f"  - Common mistakes: {len(data.get('common_mistakes', []))}")
        print(f"  - Learned corrections: {len(data.get('learned_corrections', {}))}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Restore failed: {e}")
        return False


def list_backups(backup_dir: Optional[Path] = None) -> list:
    """
    List available backups.
    
    Args:
        backup_dir: Backup directory (default: outputs/backups)
        
    Returns:
        List of backup file paths
    """
    if backup_dir is None:
        backup_dir = Path("outputs") / "backups"
    
    if not backup_dir.exists():
        return []
    
    backups = sorted(
        backup_dir.glob("learning_db_backup_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    return backups


def main():
    """Main function."""
    import sys
    
    project_root = Path(__file__).parent.parent
    
    if len(sys.argv) > 1:
        backup_path = Path(sys.argv[1])
    else:
        # List available backups
        backups = list_backups()
        if not backups:
            print("No backups found. Run backup script first:")
            print("  python scripts/backup_learning_db.py")
            sys.exit(1)
        
        print("Available backups:")
        for i, backup in enumerate(backups[:10], 1):  # Show last 10
            timestamp = backup.stem.replace("learning_db_backup_", "")
            print(f"  {i}. {backup.name} ({timestamp})")
        
        choice = input("\nSelect backup number (or enter path): ")
        try:
            backup_idx = int(choice) - 1
            if 0 <= backup_idx < len(backups):
                backup_path = backups[backup_idx]
            else:
                print("Invalid selection.")
                sys.exit(1)
        except ValueError:
            backup_path = Path(choice)
    
    learning_db_path = project_root / "learning_db.json"
    
    # Ask for confirmation
    print(f"⚠ WARNING: This will restore {learning_db_path} from {backup_path}")
    if learning_db_path.exists():
        print("  Current learning_db will be backed up first.")
    response = input("  Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Restore cancelled.")
        sys.exit(0)
    
    success = restore_learning_db(backup_path, learning_db_path)
    
    if success:
        print("\n[OK] Restore successful!")
        sys.exit(0)
    else:
        print("\n[ERROR] Restore failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

