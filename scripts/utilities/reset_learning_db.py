#!/usr/bin/env python3
"""
Reset script for learning_db.json.

Backs up and resets the learning database to remove bad learned patterns.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def backup_learning_db(learning_db_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create a backup of learning_db.json.
    
    Args:
        learning_db_path: Path to learning_db.json
        backup_dir: Optional backup directory (default: outputs/backups)
        
    Returns:
        Path to backup file, or None if backup failed
    """
    if not learning_db_path.exists():
        print(f"Learning DB not found: {learning_db_path}. Nothing to backup.")
        return None
    
    if backup_dir is None:
        backup_dir = Path("outputs") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"learning_db_backup_{timestamp}.json"
    backup_path = backup_dir / backup_filename
    
    try:
        # Copy file
        shutil.copy2(learning_db_path, backup_path)
        print(f"[OK] Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        return None


def reset_learning_db(learning_db_path: Path, create_backup: bool = True) -> bool:
    """
    Reset learning_db.json to empty state.
    
    Args:
        learning_db_path: Path to learning_db.json
        create_backup: Whether to create a backup before resetting
        
    Returns:
        True if reset successful, False otherwise
    """
    # Create backup if requested
    if create_backup:
        backup_path = backup_learning_db(learning_db_path)
        if not backup_path:
            print("⚠ Warning: Backup failed, but continuing with reset...")
    
    # Create empty learning database structure
    empty_db = {
        'recent_analyses': [],
        'successful_patterns': {},
        'common_mistakes': [],
        'learned_corrections': {},
        'key_learnings': {
            'confidence_calibration': {
                'calibration_offset': 0.0
            }
        },
        'error_stats': {
            'critical_errors': []
        },
        'reset_timestamp': datetime.now().isoformat(),
        'reset_reason': 'Manual reset to remove bad learned patterns'
    }
    
    try:
        # Write empty database
        learning_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(learning_db_path, 'w', encoding='utf-8') as f:
            json.dump(empty_db, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Learning DB reset: {learning_db_path}")
        print(f"  - Reset timestamp: {empty_db['reset_timestamp']}")
        print(f"  - Reason: {empty_db['reset_reason']}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        return False


def main():
    """Main function."""
    import sys
    
    project_root = Path(__file__).parent.parent
    
    # Get learning_db path from config or use default
    learning_db_path = project_root / "learning_db.json"
    
    if len(sys.argv) > 1:
        learning_db_path = Path(sys.argv[1])
    
    # Ask for confirmation
    if learning_db_path.exists():
        print(f"[WARNING] This will reset {learning_db_path}")
        print("  A backup will be created automatically.")
        response = input("  Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Reset cancelled.")
            sys.exit(0)
    
    success = reset_learning_db(learning_db_path, create_backup=True)
    
    if success:
        print("\n[OK] Reset successful!")
        print("  You can restore from backup if needed:")
        print(f"  python scripts/restore_learning_db.py <backup_file>")
        sys.exit(0)
    else:
        print("\n[ERROR] Reset failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

