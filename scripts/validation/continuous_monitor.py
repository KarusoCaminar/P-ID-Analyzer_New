"""
Kontinuierlicher Monitor - Überwacht alle 2 Minuten und greift bei Problemen ein
"""

import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "outputs" / "overnight_optimization"

# Setup Logging
log_file = OUTPUT_DIR / "logs" / f"continuous_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPT = project_root / "scripts" / "validation" / "run_overnight_optimization.py"
DURATION = 8.0


def log_status(message, level='INFO'):
    """Loggt Status mit Timestamp."""
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def check_and_restart():
    """Prüft Status und startet bei Bedarf neu."""
    try:
        # Prüfe Process-File
        proc_file = OUTPUT_DIR / ".overnight_process.pid"
        process_running = False
        
        if proc_file.exists():
            try:
                with open(proc_file, 'r') as f:
                    proc_info = json.load(f)
                    pid = proc_info.get('pid')
                    
                    # Prüfe ob Prozess läuft (Windows)
                    result = subprocess.run(
                        ['tasklist', '/FI', f'PID eq {pid}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    process_running = str(pid) in result.stdout
            except:
                pass
        
        # Prüfe Log-Aktivität
        log_dir = OUTPUT_DIR / "logs"
        log_active = False
        if log_dir.exists():
            log_files = list(log_dir.glob("overnight_*.log"))
            if log_files:
                latest = max(log_files, key=lambda p: p.stat().st_mtime)
                age = time.time() - latest.stat().st_mtime
                log_active = age < 300  # Aktualisiert in letzten 5 Min
        
        if process_running or log_active:
            log_status(f"✓ System läuft (PID: {proc_info.get('pid') if proc_file.exists() else 'unknown'}, Log-Aktiv: {log_active})")
            return True
        else:
            log_status("⚠ System läuft NICHT - starte neu...")
            start_script()
            return False
            
    except Exception as e:
        log_status(f"✗ Fehler in Status-Check: {e}")
        return False


def start_script():
    """Startet das Hauptskript."""
    try:
        log_status("Starte run_overnight_optimization.py...")
        cmd = [sys.executable, str(SCRIPT), "--duration", str(DURATION)]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Speichere PID
        proc_file = OUTPUT_DIR / ".overnight_process.pid"
        with open(proc_file, 'w') as f:
            json.dump({'pid': process.pid, 'started_at': datetime.now().isoformat()}, f)
        
        log_status(f"✓ Skript gestartet (PID: {process.pid})")
        return True
    except Exception as e:
        log_status(f"✗ Fehler beim Starten: {e}")
        return False


def monitor_loop():
    """Haupt-Monitor-Loop."""
    log_status("=" * 70)
    log_status("KONTINUIERLICHER MONITOR GESTARTET")
    log_status("=" * 70)
    log_status(f"Überwache: {SCRIPT}")
    log_status(f"Check alle 2 Minuten")
    log_status("=" * 70)
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            log_status(f"\n--- Check #{check_count} ---")
            
            # Status prüfen und bei Bedarf neu starten
            check_and_restart()
            
            # Prüfe Fortschritt
            test_dir = OUTPUT_DIR / "test_results"
            if test_dir.exists():
                test_count = len(list(test_dir.rglob("*.json")))
                if test_count > 0:
                    log_status(f"Tests durchgeführt: {test_count}")
            
            # Warte 2 Minuten
            time.sleep(120)
            
    except KeyboardInterrupt:
        log_status("\n⚠ Monitor gestoppt")
    except Exception as e:
        log_status(f"✗ Fehler: {e}", exc_info=True)


if __name__ == "__main__":
    monitor_loop()

