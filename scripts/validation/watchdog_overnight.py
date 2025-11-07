"""
Watchdog für Overnight-Optimierung - Kontinuierliche Überwachung und automatische Fehlerbehebung
"""

import sys
import time
import subprocess
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup Logging
OUTPUT_DIR = project_root / "outputs" / "overnight_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / f"watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPT_PATH = project_root / "scripts" / "validation" / "run_overnight_optimization.py"
DURATION_HOURS = 8.0
CHECK_INTERVAL = 120  # 2 Minuten zwischen Checks
PROCESS_FILE = OUTPUT_DIR / ".overnight_process.pid"


def save_process_info(pid=None):
    """Speichert Prozess-Info in Datei."""
    info = {
        'pid': pid,
        'started_at': datetime.now().isoformat(),
        'script': str(SCRIPT_PATH)
    }
    with open(PROCESS_FILE, 'w') as f:
        json.dump(info, f)


def load_process_info():
    """Lädt Prozess-Info aus Datei."""
    if PROCESS_FILE.exists():
        try:
            with open(PROCESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def is_process_running(pid):
    """Prüft ob Prozess mit PID läuft."""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                  capture_output=True, text=True, timeout=5)
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except:
        return False


def check_log_activity():
    """Prüft ob Log-Dateien aktiv aktualisiert werden."""
    log_dir = OUTPUT_DIR / "logs"
    if not log_dir.exists():
        return False
    
    log_files = list(log_dir.glob("overnight_*.log"))
    if not log_files:
        return False
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    # Wenn Log in letzten 5 Minuten aktualisiert wurde
    age_seconds = time.time() - latest_log.stat().st_mtime
    return age_seconds < 300


def start_script():
    """Startet das Overnight-Skript."""
    logger.info("=" * 70)
    logger.info("STARTE OVERNIGHT-OPTIMIERUNGS-SKRIPT")
    logger.info("=" * 70)
    
    try:
        # Lade .env falls vorhanden
        env_file = project_root / ".env"
        env = os.environ.copy()
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                logger.info("✓ .env Datei geladen")
            except:
                pass
        
        # Starte Prozess
        cmd = [sys.executable, str(SCRIPT_PATH), "--duration", str(DURATION_HOURS)]
        
        # Starte in neuem Prozess (Windows)
        if sys.platform == 'win32':
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        save_process_info(process.pid)
        logger.info(f"✓ Skript gestartet mit PID: {process.pid}")
        logger.info(f"  Befehl: {' '.join(cmd)}")
        return process
        
    except Exception as e:
        logger.error(f"✗ Fehler beim Starten: {e}", exc_info=True)
        return None


def check_test_progress():
    """Prüft Fortschritt der Tests."""
    test_dir = OUTPUT_DIR / "test_results"
    if not test_dir.exists():
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    total = 0
    successful = 0
    failed = 0
    
    for json_file in test_dir.rglob("*.json"):
        total += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('success', False):
                    successful += 1
                else:
                    failed += 1
        except:
            pass
    
    return {'total': total, 'successful': successful, 'failed': failed}


def main_watchdog():
    """Haupt-Watchdog-Loop."""
    logger.info("=" * 70)
    logger.info("WATCHDOG FÜR OVERNIGHT-OPTIMIERUNG STARTET")
    logger.info("=" * 70)
    logger.info(f"Überwache: {SCRIPT_PATH}")
    logger.info(f"Dauer: {DURATION_HOURS} Stunden")
    logger.info(f"Check-Intervall: {CHECK_INTERVAL} Sekunden")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS + 0.5)  # +30 Min Puffer
    restart_count = 0
    max_restarts = 20
    
    # Prüfe ob bereits ein Prozess läuft
    proc_info = load_process_info()
    if proc_info and is_process_running(proc_info.get('pid')):
        logger.info(f"✓ Existierender Prozess gefunden (PID: {proc_info.get('pid')})")
        process_pid = proc_info.get('pid')
    elif check_log_activity():
        logger.info("✓ Log-Aktivität erkannt - Skript läuft wahrscheinlich")
        process_pid = None
    else:
        logger.info("⚠ Kein laufender Prozess gefunden - starte neu...")
        process = start_script()
        if process:
            process_pid = process.pid
        else:
            process_pid = None
    
    last_progress_check = {}
    
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds() / 3600
            remaining = (end_time - current_time).total_seconds() / 3600
            
            logger.info(f"\n[{current_time.strftime('%H:%M:%S')}] Status-Check")
            logger.info(f"  ⏱ Verstrichen: {elapsed:.2f}h / Verbleibend: {remaining:.2f}h")
            
            # Prüfe ob Prozess läuft
            process_alive = False
            if process_pid:
                process_alive = is_process_running(process_pid)
            
            # Alternative: Prüfe Log-Aktivität
            if not process_alive:
                process_alive = check_log_activity()
            
            if process_alive:
                logger.info("  ✓ Prozess läuft")
            else:
                logger.warning("  ✗ Prozess läuft NICHT!")
                
                # Prüfe ob wir noch Zeit haben
                if current_time < end_time - timedelta(minutes=30):
                    if restart_count < max_restarts:
                        restart_count += 1
                        logger.info(f"  🔄 Neustart #{restart_count}/{max_restarts}...")
                        process = start_script()
                        if process:
                            process_pid = process.pid
                            process_alive = True
                        else:
                            logger.error("  ✗ Neustart fehlgeschlagen!")
                            time.sleep(60)  # Warte länger bei Fehler
                            continue
                    else:
                        logger.error(f"  ✗ Maximale Neustarts ({max_restarts}) erreicht!")
                        break
                else:
                    logger.info("  ⏰ Zeit läuft ab - keine weiteren Neustarts")
            
            # Prüfe Fortschritt
            progress = check_test_progress()
            if progress != last_progress_check:
                logger.info(f"  📊 Fortschritt: {progress['total']} Tests")
                logger.info(f"     ✓ Erfolgreich: {progress['successful']}")
                logger.info(f"     ✗ Fehlgeschlagen: {progress['failed']}")
                last_progress_check = progress
            
            # Prüfe Reports
            reports_dir = OUTPUT_DIR / "reports"
            if reports_dir.exists():
                report_count = len(list(reports_dir.glob("*.html"))) + len(list(reports_dir.glob("*.json")))
                if report_count > 0:
                    logger.info(f"  📄 Reports: {report_count}")
            
            # Warte bis nächster Check
            sleep_time = min(CHECK_INTERVAL, remaining * 3600)
            if sleep_time > 0:
                logger.info(f"  💤 Warte {int(sleep_time)}s bis nächster Check...")
                time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("\n⚠ Watchdog durch Benutzer gestoppt")
            break
        except Exception as e:
            logger.error(f"⚠ Fehler in Watchdog-Loop: {e}", exc_info=True)
            time.sleep(30)
    
    # Finaler Status
    logger.info("\n" + "=" * 70)
    logger.info("WATCHDOG ABGESCHLOSSEN")
    logger.info("=" * 70)
    final_progress = check_test_progress()
    logger.info(f"Finaler Status:")
    logger.info(f"  Tests: {final_progress['total']}")
    logger.info(f"  Erfolgreich: {final_progress['successful']}")
    logger.info(f"  Fehlgeschlagen: {final_progress['failed']}")
    logger.info(f"  Neustarts: {restart_count}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main_watchdog()
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        sys.exit(1)

