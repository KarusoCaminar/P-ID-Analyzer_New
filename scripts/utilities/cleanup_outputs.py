#!/usr/bin/env python3
"""
Cleanup und Reorganisierung des Output-Ordners

Ziel: Saubere, strukturierte Output-Organisation mit klarer Trennung zwischen:
- Produktionsläufen (runs/)
- Test-Läufen (tests/)
- Debug-Dateien (debug/)
- Logs (logs/)
- Archive (archive/)
- Backups (backups/)
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Projekt-Root
project_root = Path(__file__).parent.parent
outputs_dir = project_root / "outputs"

# Definierte Struktur
STRUCTURE = {
    "runs": "Produktionsläufe (mit Timestamp)",
    "tests": "Test-Läufe (phase1_tests, strategy_comparison, etc.)",
    "debug": "Debug-Dateien (LLM logs, prompts, responses)",
    "logs": "Haupt-Logs",
    "archive": "Alte Dateien (älter als 7 Tage)",
    "backups": "Backups (learning_db, etc.)"
}


def get_file_age_days(file_path: Path) -> float:
    """Berechne das Alter einer Datei in Tagen."""
    if not file_path.exists():
        return 0.0
    mtime = file_path.stat().st_mtime
    age_seconds = datetime.now().timestamp() - mtime
    return age_seconds / (24 * 60 * 60)


def should_archive(file_path: Path, max_age_days: int = 7) -> bool:
    """Prüfe ob eine Datei archiviert werden sollte."""
    return get_file_age_days(file_path) > max_age_days


def create_structure(base_dir: Path) -> None:
    """Erstelle die neue Output-Struktur."""
    for dir_name in STRUCTURE.keys():
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Struktur erstellt: {', '.join(STRUCTURE.keys())}")


def move_to_archive(file_path: Path, archive_dir: Path, reason: str = "") -> Path:
    """Verschiebe eine Datei/Ordner ins Archiv."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Erstelle Unterordner mit Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subdir = archive_dir / f"archive_{timestamp}"
    archive_subdir.mkdir(parents=True, exist_ok=True)
    
    # Verschiebe Datei/Ordner
    target = archive_subdir / file_path.name
    if file_path.exists():
        shutil.move(str(file_path), str(target))
        print(f"[ARCHIVE] Archiviert: {file_path.name} -> {target} ({reason})")
        return target
    return None


def organize_outputs() -> Dict[str, Any]:
    """Organisiere Outputs nach neuer Struktur."""
    stats = {
        "moved_to_runs": [],
        "moved_to_tests": [],
        "moved_to_debug": [],
        "moved_to_logs": [],
        "archived": [],
        "kept_in_backups": [],
        "errors": []
    }
    
    if not outputs_dir.exists():
        print(f"[ERROR] Outputs-Verzeichnis existiert nicht: {outputs_dir}")
        return stats
    
    # Erstelle neue Struktur
    create_structure(outputs_dir)
    
    runs_dir = outputs_dir / "runs"
    tests_dir = outputs_dir / "tests"
    debug_dir = outputs_dir / "debug"
    logs_dir = outputs_dir / "logs"
    archive_dir = outputs_dir / "archive"
    backups_dir = outputs_dir / "backups"
    
    # 1. Backups: Behalte alle Backups
    print("\n[BACKUPS] Organisiere Backups...")
    for item in outputs_dir.iterdir():
        if item.is_file() and "backup" in item.name.lower():
            target = backups_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                stats["kept_in_backups"].append(item.name)
                print(f"  [OK] Backup behalten: {item.name}")
    
    # 2. Logs: Verschiebe alle .log Dateien
    print("\n[LOGS] Organisiere Logs...")
    for item in outputs_dir.iterdir():
        if item.is_file() and item.suffix == ".log":
            target = logs_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                stats["moved_to_logs"].append(item.name)
                print(f"  [OK] Log verschoben: {item.name}")
    
    # 3. Debug-Dateien: Verschiebe debug/ und LLM-related Dateien
    print("\n[DEBUG] Organisiere Debug-Dateien...")
    debug_source = outputs_dir / "debug"
    if debug_source.exists() and debug_source.is_dir():
        # Verschiebe gesamten debug-Ordner
        for item in debug_source.iterdir():
            target = debug_dir / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(str(target))
                shutil.move(str(item), str(target))
                stats["moved_to_debug"].append(item.name)
                print(f"  [OK] Debug-Ordner verschoben: {item.name}")
            else:
                if not target.exists():
                    shutil.move(str(item), str(target))
                    stats["moved_to_debug"].append(item.name)
                    print(f"  [OK] Debug-Datei verschoben: {item.name}")
        # Lösche leeren debug-Ordner
        try:
            debug_source.rmdir()
        except OSError:
            pass
    
    # 4. Test-Ordner: Verschiebe alle Test-Ordner
    print("\n[TEST] Organisiere Test-Ordner...")
    test_keywords = ["phase1_tests", "strategy_comparison", "parameter_test", "strategy_calibration", 
                     "focused_amplification_test", "test_simple_and_uni", "model_comparison", "iterative_tests"]
    
    for item in outputs_dir.iterdir():
        if item.is_dir() and any(keyword in item.name.lower() for keyword in test_keywords):
            target = tests_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                stats["moved_to_tests"].append(item.name)
                print(f"  [OK] Test-Ordner verschoben: {item.name}")
    
    # 5. Produktionsläufe: Verschiebe alle _output_* Ordner
    print("\n[RUNS] Organisiere Produktionsläufe...")
    for item in outputs_dir.iterdir():
        if item.is_dir() and "_output_" in item.name:
            target = runs_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                stats["moved_to_runs"].append(item.name)
                print(f"  [OK] Run-Ordner verschoben: {item.name}")
    
    # 6. Archive: Verschiebe alte Dateien (älter als 7 Tage)
    print("\n[ARCHIVE] Archiviere alte Dateien...")
    max_age_days = 7
    for item in outputs_dir.iterdir():
        if item.is_file() or item.is_dir():
            if should_archive(item, max_age_days):
                try:
                    move_to_archive(item, archive_dir, f"älter als {max_age_days} Tage")
                    stats["archived"].append(item.name)
                except Exception as e:
                    stats["errors"].append(f"Fehler beim Archivieren von {item.name}: {e}")
                    print(f"  [ERROR] Fehler beim Archivieren: {item.name} - {e}")
    
    # 7. JSON-Dateien im Root: Verschiebe zu runs oder debug
    print("\n[JSON] Organisiere JSON-Dateien...")
    for item in outputs_dir.iterdir():
        if item.is_file() and item.suffix == ".json":
            if "test" in item.name.lower() or "result" in item.name.lower():
                target = tests_dir / item.name
            else:
                target = debug_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                print(f"  [OK] JSON verschoben: {item.name}")
    
    return stats


def print_summary(stats: Dict[str, Any]) -> None:
    """Zeige Zusammenfassung der Reorganisation."""
    print("\n" + "=" * 80)
    print("REORGANISATIONS-ZUSAMMENFASSUNG")
    print("=" * 80)
    
    print(f"\n[OK] Verschoben zu runs/: {len(stats['moved_to_runs'])}")
    for item in stats['moved_to_runs'][:5]:
        print(f"   - {item}")
    if len(stats['moved_to_runs']) > 5:
        print(f"   ... und {len(stats['moved_to_runs']) - 5} weitere")
    
    print(f"\n[OK] Verschoben zu tests/: {len(stats['moved_to_tests'])}")
    for item in stats['moved_to_tests'][:5]:
        print(f"   - {item}")
    if len(stats['moved_to_tests']) > 5:
        print(f"   ... und {len(stats['moved_to_tests']) - 5} weitere")
    
    print(f"\n[OK] Verschoben zu debug/: {len(stats['moved_to_debug'])}")
    for item in stats['moved_to_debug'][:5]:
        print(f"   - {item}")
    if len(stats['moved_to_debug']) > 5:
        print(f"   ... und {len(stats['moved_to_debug']) - 5} weitere")
    
    print(f"\n[OK] Verschoben zu logs/: {len(stats['moved_to_logs'])}")
    for item in stats['moved_to_logs'][:5]:
        print(f"   - {item}")
    if len(stats['moved_to_logs']) > 5:
        print(f"   ... und {len(stats['moved_to_logs']) - 5} weitere")
    
    print(f"\n[ARCHIVE] Archiviert: {len(stats['archived'])}")
    for item in stats['archived'][:5]:
        print(f"   - {item}")
    if len(stats['archived']) > 5:
        print(f"   ... und {len(stats['archived']) - 5} weitere")
    
    print(f"\n[BACKUP] Backups behalten: {len(stats['kept_in_backups'])}")
    
    if stats['errors']:
        print(f"\n[ERROR] Fehler: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"   - {error}")
    
    print("\n" + "=" * 80)
    print("[OK] Reorganisation abgeschlossen!")
    print("=" * 80)


def main():
    """Hauptfunktion."""
    print("=" * 80)
    print("OUTPUT-ORDNER AUFRAUMEN & REORGANISIEREN")
    print("=" * 80)
    print(f"\n[DIR] Output-Verzeichnis: {outputs_dir}")
    print(f"[TIME] Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not outputs_dir.exists():
        print(f"[ERROR] Outputs-Verzeichnis existiert nicht: {outputs_dir}")
        print("   Erstelle es jetzt...")
        outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorganisiere Outputs
    stats = organize_outputs()
    
    # Zeige Zusammenfassung
    print_summary(stats)
    
    # Speichere Statistik
    stats_file = outputs_dir / "cleanup_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Statistik gespeichert: {stats_file}")


if __name__ == "__main__":
    main()

