"""
Auto Guardian - Kontinuierliche Überwachung des Overnight-Prozesses
Prüft alle 5 Minuten den Status und greift automatisch ein bei Problemen
"""

import sys
import time
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "outputs" / "overnight_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Setup Logging
log_file = OUTPUT_DIR / "logs" / f"guardian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
CHECK_INTERVAL = 300  # 5 Minuten
PROCESS_FILE = OUTPUT_DIR / ".overnight_process.pid"


def log(msg, level='info'):
    """Zentrale Logging-Funktion."""
    getattr(logger, level)(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def is_process_alive(pid):
    """Prüft ob Prozess mit PID läuft."""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return str(pid) in result.stdout
        else:
            import os
            os.kill(pid, 0)
            return True
    except:
        return False


def check_log_activity():
    """Prüft ob Logs aktiv aktualisiert werden."""
    log_dir = OUTPUT_DIR / "logs"
    if not log_dir.exists():
        return False, None
    
    log_files = list(log_dir.glob("overnight_*.log"))
    if not log_files:
        return False, None
    
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    age = time.time() - latest.stat().st_mtime
    is_active = age < 600  # Aktualisiert in letzten 10 Minuten
    
    # Lese letzte Zeilen für Status
    last_lines = []
    try:
        with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            last_lines = lines[-5:] if len(lines) > 5 else lines
    except:
        pass
    
    return is_active, ''.join(last_lines) if last_lines else None


def get_test_progress():
    """Ermittelt Fortschritt der Tests."""
    test_dir = OUTPUT_DIR / "test_results"
    if not test_dir.exists():
        return {'total': 0, 'successful': 0, 'failed': 0, 'strategies': {}}
    
    total = 0
    successful = 0
    failed = 0
    strategies = {}
    
    for json_file in test_dir.rglob("*.json"):
        total += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                strategy = data.get('strategy', 'unknown')
                if strategy not in strategies:
                    strategies[strategy] = {'total': 0, 'successful': 0, 'failed': 0}
                strategies[strategy]['total'] += 1
                
                if data.get('success', False):
                    successful += 1
                    strategies[strategy]['successful'] += 1
                else:
                    failed += 1
                    strategies[strategy]['failed'] += 1
        except:
            pass
    
    return {'total': total, 'successful': successful, 'failed': failed, 'strategies': strategies}


def start_overnight_script():
    """Startet das Overnight-Skript."""
    try:
        log("=" * 60, 'info')
        log("NEUSTART: Starte Overnight-Optimierungs-Skript", 'info')
        log("=" * 60, 'info')
        
        cmd = [sys.executable, str(SCRIPT), "--duration", str(DURATION)]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        # Speichere PID
        with open(PROCESS_FILE, 'w') as f:
            json.dump({
                'pid': process.pid,
                'started_at': datetime.now().isoformat(),
                'restart_count': load_restart_count() + 1
            }, f)
        
        log(f"✓ Skript gestartet (PID: {process.pid})", 'info')
        return process.pid
    except Exception as e:
        log(f"✗ Fehler beim Starten: {e}", 'error')
        return None


def load_restart_count():
    """Lädt Anzahl der Neustarts."""
    if PROCESS_FILE.exists():
        try:
            with open(PROCESS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('restart_count', 0)
        except:
            return 0
    return 0


def main_guardian():
    """Haupt-Guardian-Loop."""
    log("=" * 70, 'info')
    log("AUTO GUARDIAN GESTARTET", 'info')
    log("=" * 70, 'info')
    log(f"Überwache: {SCRIPT}", 'info')
    log(f"Check-Intervall: {CHECK_INTERVAL}s (5 Minuten)", 'info')
    log(f"Maximale Dauer: {DURATION} Stunden", 'info')
    log("=" * 70, 'info')
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION + 0.5)
    check_count = 0
    last_progress = {}
    
    try:
        while datetime.now() < end_time:
            check_count += 1
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds() / 3600
            remaining = (end_time - current_time).total_seconds() / 3600
            
            log(f"\n--- Check #{check_count} ---", 'info')
            log(f"⏱ Verstrichen: {elapsed:.2f}h / Verbleibend: {remaining:.2f}h", 'info')
            
            # Prüfe Process-File
            process_alive = False
            process_pid = None
            
            if PROCESS_FILE.exists():
                try:
                    with open(PROCESS_FILE, 'r') as f:
                        proc_info = json.load(f)
                        process_pid = proc_info.get('pid')
                        if process_pid:
                            process_alive = is_process_alive(process_pid)
                except:
                    pass
            
            # Prüfe Log-Aktivität
            log_active, last_log_lines = check_log_activity()
            
            if process_alive:
                log(f"✓ Prozess läuft (PID: {process_pid})", 'info')
            elif log_active:
                log("✓ Log-Aktivität erkannt - Skript läuft wahrscheinlich", 'info')
                process_alive = True
            else:
                log("⚠ Prozess läuft NICHT und keine Log-Aktivität!", 'warning')
                
                # Zeige letzte Log-Zeilen falls vorhanden
                if last_log_lines:
                    log("Letzte Log-Zeilen:", 'info')
                    for line in last_log_lines.strip().split('\n')[-3:]:
                        if line.strip():
                            log(f"  {line.strip()}", 'debug')
                
                # Starte neu wenn noch Zeit
                if remaining > 0.5:  # Mindestens 30 Minuten verbleibend
                    restart_count = load_restart_count()
                    if restart_count < 20:
                        log(f"🔄 Neustart #{restart_count + 1}...", 'info')
                        new_pid = start_overnight_script()
                        if new_pid:
                            process_pid = new_pid
                            process_alive = True
                            time.sleep(30)  # Warte nach Neustart
                            continue
                    else:
                        log(f"✗ Maximale Neustarts (20) erreicht!", 'error')
                else:
                    log("⏰ Zeit läuft ab - keine weiteren Neustarts", 'info')
            
            # Prüfe Fortschritt
            progress = get_test_progress()
            if progress != last_progress:
                log(f"📊 Fortschritt:", 'info')
                log(f"   Gesamt: {progress['total']} Tests", 'info')
                log(f"   ✓ Erfolgreich: {progress['successful']}", 'info')
                log(f"   ✗ Fehlgeschlagen: {progress['failed']}", 'info')
                if progress['strategies']:
                    for strategy, stats in progress['strategies'].items():
                        log(f"   {strategy}: {stats['successful']}/{stats['total']} erfolgreich", 'info')
                last_progress = progress
            
            # Prüfe Reports
            reports_dir = OUTPUT_DIR / "reports"
            if reports_dir.exists():
                report_count = len(list(reports_dir.glob("*.html"))) + len(list(reports_dir.glob("*.json")))
                if report_count > 0:
                    log(f"📄 Reports: {report_count}", 'info')
            
            # Warte bis nächster Check
            sleep_time = min(CHECK_INTERVAL, remaining * 3600)
            if sleep_time > 0:
                log(f"💤 Warte {int(sleep_time)}s bis nächster Check...", 'info')
                time.sleep(sleep_time)
            else:
                break
        
        # Finaler Status
        log("\n" + "=" * 70, 'info')
        log("GUARDIAN ABGESCHLOSSEN", 'info')
        log("=" * 70, 'info')
        final_progress = get_test_progress()
        log(f"Finaler Status:", 'info')
        log(f"  Tests: {final_progress['total']}", 'info')
        log(f"  Erfolgreich: {final_progress['successful']}", 'info')
        log(f"  Fehlgeschlagen: {final_progress['failed']}", 'info')
        log(f"  Neustarts: {load_restart_count()}", 'info')
        log("=" * 70, 'info')
        
    except KeyboardInterrupt:
        log("\n⚠ Guardian durch Benutzer gestoppt", 'warning')
    except Exception as e:
        log(f"✗ Kritischer Fehler: {e}", 'error', exc_info=True)


if __name__ == "__main__":
    main_guardian()

