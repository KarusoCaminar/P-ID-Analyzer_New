#!/usr/bin/env python3
"""
Backup script for learning_db.json.

Creates a timestamped backup of the learning database before resetting.
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
        
        # Also save metadata
        metadata_path = backup_dir / f"learning_db_backup_{timestamp}_metadata.json"
        with open(learning_db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = {
            'backup_timestamp': datetime.now().isoformat(),
            'original_path': str(learning_db_path),
            'backup_path': str(backup_path),
            'file_size': learning_db_path.stat().st_size,
            'data_summary': {
                'recent_analyses': len(data.get('recent_analyses', [])),
                'successful_patterns': len(data.get('successful_patterns', {})),
                'common_mistakes': len(data.get('common_mistakes', [])),
                'learned_corrections': len(data.get('learned_corrections', {}))
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Metadata saved: {metadata_path}")
        return backup_path
        
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        return None


def main():
    """Main function."""
    import sys
    
    project_root = Path(__file__).parent.parent
    
    # Get learning_db path from config or use default
    learning_db_path = project_root / "learning_db.json"
    
    if len(sys.argv) > 1:
        learning_db_path = Path(sys.argv[1])
    
    backup_path = backup_learning_db(learning_db_path)
    
    if backup_path:
        print(f"\n[OK] Backup successful: {backup_path}")
        sys.exit(0)
    else:
        print("\n[ERROR] Backup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

