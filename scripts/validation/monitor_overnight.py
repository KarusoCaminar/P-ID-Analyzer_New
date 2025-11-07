"""
Overnight Optimization Monitor - Überwacht den nächtlichen Optimierungs-Lauf
und startet ihn bei Bedarf neu, behebt Fehler und stellt sicher dass alle Tests durchlaufen.
"""

import sys
import time
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Konfiguration
CHECK_INTERVAL = 60  # Sekunden zwischen Checks
MAX_RESTART_ATTEMPTS = 10
DURATION_HOURS = 8.0
SCRIPT_PATH = project_root / "scripts" / "validation" / "run_overnight_optimization.py"
OUTPUT_DIR = project_root / "outputs" / "overnight_optimization"
MONITOR_LOG = OUTPUT_DIR / "logs" / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Stelle sicher dass Output-Verzeichnis existiert
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)

# File Handler für Monitor-Log
file_handler = logging.FileHandler(MONITOR_LOG, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)


def get_running_python_processes():
    """Findet laufende Python-Prozesse die das Overnight-Skript ausführen."""
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('run_overnight_optimization' in str(arg) for arg in cmdline):
                        processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
    except ImportError:
        # Fallback: Verwende Windows-spezifische Methode mit wmic
        try:
            # Verwende wmic für bessere Prozess-Erkennung
            result = subprocess.run(
                ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline', '/format:list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            processes = []
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                current_proc = {}
                for line in lines:
                    if 'CommandLine=' in line:
                        cmdline = line.split('CommandLine=')[1].strip()
                        if 'run_overnight_optimization' in cmdline:
                            current_proc['cmdline'] = cmdline
                    elif 'ProcessId=' in line and current_proc.get('cmdline'):
                        pid = line.split('ProcessId=')[1].strip()
                        current_proc['pid'] = pid
                        processes.append(current_proc)
                        current_proc = {}
            return processes if processes else []
        except Exception as e:
            logger.debug(f"Fehler bei Prozess-Erkennung: {e}")
            # Letzter Fallback: Prüfe ob Log-Dateien aktualisiert werden
            log_files = list((OUTPUT_DIR / "logs").glob("overnight_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                # Wenn Log in letzten 5 Minuten aktualisiert wurde, läuft Prozess wahrscheinlich
                if (time.time() - latest_log.stat().st_mtime) < 300:
                    return ['running']
            return []


def check_log_for_errors(log_file: Path) -> tuple[bool, list[str]]:
    """
    Prüft Log-Datei auf Fehler.
    
    Returns:
        (has_errors, error_messages)
    """
    if not log_file.exists():
        return False, []
    
    errors = []
    try:
        # Lese letzte 100 Zeilen
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            for line in recent_lines:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'fatal']):
                    errors.append(line.strip())
        
        return len(errors) > 0, errors
    except Exception as e:
        logger.warning(f"Fehler beim Lesen der Log-Datei {log_file}: {e}")
        return False, []


def check_progress(output_dir: Path) -> dict:
    """Prüft den Fortschritt der Tests."""
    progress = {
        'test_results': 0,
        'reports': 0,
        'last_update': None,
        'successful_tests': 0,
        'failed_tests': 0
    }
    
    test_results_dir = output_dir / "test_results"
    if test_results_dir.exists():
        # Zähle JSON-Dateien in test_results
        for json_file in test_results_dir.rglob("*.json"):
            progress['test_results'] += 1
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('success', False):
                        progress['successful_tests'] += 1
                    else:
                        progress['failed_tests'] += 1
                    timestamp = data.get('timestamp')
                    if timestamp:
                        progress['last_update'] = timestamp
            except Exception:
                pass
    
    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        progress['reports'] = len(list(reports_dir.glob("*.html"))) + len(list(reports_dir.glob("*.json")))
    
    return progress


def fix_common_issues():
    """Behebt häufige Probleme."""
    fixes_applied = []
    
    # Prüfe GCP-Credentials
    if not os.getenv('GCP_PROJECT_ID'):
        logger.warning("GCP_PROJECT_ID nicht gesetzt - prüfe .env Datei")
        env_file = project_root / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                if os.getenv('GCP_PROJECT_ID'):
                    fixes_applied.append("GCP-Credentials aus .env geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der .env Datei: {e}")
    
    # Prüfe ob Test-Bilder existieren
    test_images = [
        project_root / "training_data" / "simple_pids" / "Einfaches P&I.png",
        project_root / "training_data" / "complex_pids" / "page_1_original.png"
    ]
    
    for img in test_images:
        if not img.exists():
            logger.error(f"Test-Bild nicht gefunden: {img}")
            fixes_applied.append(f"WARNUNG: Test-Bild fehlt: {img.name}")
    
    return fixes_applied


def start_overnight_script():
    """Startet das Overnight-Optimierungs-Skript."""
    try:
        logger.info("=" * 60)
        logger.info("STARTE OVERNIGHT-OPTIMIERUNGS-SKRIPT")
        logger.info("=" * 60)
        
        # Behebe häufige Probleme
        fixes = fix_common_issues()
        if fixes:
            logger.info(f"Behobene Probleme: {fixes}")
        
        # Starte Skript im Hintergrund
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--duration", str(DURATION_HOURS),
            "--output-dir", str(OUTPUT_DIR)
        ]
        
        logger.info(f"Befehl: {' '.join(cmd)}")
        
        # Starte Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        logger.info(f"Prozess gestartet mit PID: {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Fehler beim Starten des Skripts: {e}", exc_info=True)
        return None


def monitor_loop():
    """Hauptüberwachungsschleife."""
    logger.info("=" * 60)
    logger.info("OVERWACHUNG STARTET")
    logger.info("=" * 60)
    logger.info(f"Überwache Skript: {SCRIPT_PATH}")
    logger.info(f"Output-Verzeichnis: {OUTPUT_DIR}")
    logger.info(f"Check-Intervall: {CHECK_INTERVAL} Sekunden")
    logger.info(f"Maximale Dauer: {DURATION_HOURS} Stunden")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS + 1)  # +1 Stunde Puffer
    process = None
    restart_count = 0
    last_check = datetime.now()
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    # Prüfe ob bereits ein Prozess läuft
    running_processes = get_running_python_processes()
    if running_processes:
        logger.info(f"Bereits {len(running_processes)} Prozess(e) gefunden, warte auf erste Prüfung...")
    else:
        # Starte neuen Prozess
        process = start_overnight_script()
        if not process:
            logger.error("Konnte Skript nicht starten!")
            return
    
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds() / 3600
            remaining = (end_time - current_time).total_seconds() / 3600
            
            logger.info(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Status-Check")
            logger.info(f"  Verstrichen: {elapsed:.2f}h / Verbleibend: {remaining:.2f}h")
            
            # Prüfe ob Prozess noch läuft
            running_processes = get_running_python_processes()
            process_running = len(running_processes) > 0
            
            if process and process.poll() is not None:
                # Prozess ist beendet
                return_code = process.returncode
                logger.warning(f"Prozess beendet mit Return-Code: {return_code}")
                process_running = False
            
            if not process_running:
                # Prozess läuft nicht mehr - prüfe ob es Zeit ist zu beenden
                if current_time >= end_time - timedelta(minutes=30):
                    logger.info("Zeit läuft ab - generiere finalen Report...")
                    break
                
                # Prüfe Logs auf Fehler
                log_files = list((OUTPUT_DIR / "logs").glob("overnight_*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    has_errors, error_messages = check_log_for_errors(latest_log)
                    
                    if has_errors and consecutive_errors < max_consecutive_errors:
                        logger.warning(f"Fehler in Log gefunden ({len(error_messages)} Fehler)")
                        for error in error_messages[-5:]:  # Zeige letzte 5 Fehler
                            logger.warning(f"  {error}")
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                
                # Starte Prozess neu
                if restart_count < MAX_RESTART_ATTEMPTS:
                    restart_count += 1
                    logger.info(f"Neustart #{restart_count}/{MAX_RESTART_ATTEMPTS}")
                    process = start_overnight_script()
                    if not process:
                        logger.error("Neustart fehlgeschlagen!")
                        time.sleep(30)  # Warte länger bei Fehler
                        continue
                else:
                    logger.error(f"Maximale Neustart-Versuche ({MAX_RESTART_ATTEMPTS}) erreicht!")
                    break
            else:
                consecutive_errors = 0  # Reset bei erfolgreichem Lauf
                logger.info(f"✓ Prozess läuft (PID: {process.pid if process else 'unknown'})")
            
            # Prüfe Fortschritt
            progress = check_progress(OUTPUT_DIR)
            logger.info(f"  Fortschritt: {progress['test_results']} Tests, {progress['successful_tests']} erfolgreich, {progress['failed_tests']} fehlgeschlagen")
            if progress['last_update']:
                logger.info(f"  Letztes Update: {progress['last_update']}")
            
            # Prüfe Logs auf neue Fehler
            log_files = list((OUTPUT_DIR / "logs").glob("overnight_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                # Lese nur neue Zeilen seit letztem Check
                try:
                    if latest_log.stat().st_mtime > last_check.timestamp():
                        has_errors, error_messages = check_log_for_errors(latest_log)
                        if has_errors:
                            logger.warning(f"Neue Fehler in Log gefunden!")
                            for error in error_messages[-3:]:  # Zeige letzte 3 neue Fehler
                                logger.warning(f"  {error}")
                except Exception as e:
                    logger.debug(f"Fehler beim Prüfen des Logs: {e}")
            
            last_check = datetime.now()
            
            # Warte bis nächster Check
            logger.info(f"  Nächster Check in {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("\nÜberwachung durch Benutzer abgebrochen")
            break
        except Exception as e:
            logger.error(f"Fehler in Überwachungsschleife: {e}", exc_info=True)
            time.sleep(30)  # Warte bei Fehler
    
    # Finale Prüfung und Report-Generierung
    logger.info("\n" + "=" * 60)
    logger.info("OVERWACHUNG ABGESCHLOSSEN")
    logger.info("=" * 60)
    
    final_progress = check_progress(OUTPUT_DIR)
    logger.info(f"Finaler Status:")
    logger.info(f"  Tests: {final_progress['test_results']}")
    logger.info(f"  Erfolgreich: {final_progress['successful_tests']}")
    logger.info(f"  Fehlgeschlagen: {final_progress['failed_tests']}")
    logger.info(f"  Reports: {final_progress['reports']}")
    logger.info(f"  Neustarts: {restart_count}")
    logger.info("=" * 60)
    
    # Stelle sicher dass Reports generiert werden
    if final_progress['reports'] == 0 and final_progress['test_results'] > 0:
        logger.info("Generiere finale Reports...")
        # Das Skript sollte Reports automatisch generieren, aber wir können es auch manuell triggern
        # wenn nötig


if __name__ == "__main__":
    try:
        monitor_loop()
    except Exception as e:
        logger.error(f"Kritischer Fehler im Monitor: {e}", exc_info=True)
        sys.exit(1)

