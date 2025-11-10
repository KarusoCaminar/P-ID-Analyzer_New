"""
Live Watchdog für laufende Tests - überwacht kontinuierlich auf Fehler.

Prüft:
- API-Fehler (Rate Limits, Timeouts)
- Hänger (keine Log-Updates)
- Exceptions und Errors
- Circuit Breaker Triggers
- Fortschritt (Phase, Progress)
"""

import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Projekt-Root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Find latest log file
output_base = project_root / "outputs" / "live_test"
log_files = list(output_base.glob("**/logs/*.log"))

if not log_files:
    print("[ERROR] No log files found. Start the test first!")
    sys.exit(1)

# Get latest log file
latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
print(f"[OK] Monitoring: {latest_log}")
print(f"[OK] Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
print()
print("=" * 80)
print("LIVE WATCHDOG - Monitoring for Errors")
print("=" * 80)
print()

# Track state
last_position = 0
last_update_time = time.time()
error_count = 0
warning_count = 0

# Patterns to watch for
error_patterns = [
    (re.compile(r'ERROR|Exception|Traceback|Failed|FAIL', re.IGNORECASE), "ERROR"),
    (re.compile(r'Rate limit|429|Resource Exhausted', re.IGNORECASE), "RATE_LIMIT"),
    (re.compile(r'Timeout|timed out', re.IGNORECASE), "TIMEOUT"),
    (re.compile(r'Circuit.*breaker|Circuit.*open', re.IGNORECASE), "CIRCUIT_BREAKER"),
    (re.compile(r'Connection.*error|Network.*error', re.IGNORECASE), "NETWORK_ERROR"),
    (re.compile(r'AttributeError|TypeError|KeyError|ValueError', re.IGNORECASE), "PYTHON_ERROR"),
]

warning_patterns = [
    (re.compile(r'WARNING|WARN', re.IGNORECASE), "WARNING"),
    (re.compile(r'Retry|retrying', re.IGNORECASE), "RETRY"),
    (re.compile(r'Fallback|falling back', re.IGNORECASE), "FALLBACK"),
]

# Progress tracking
progress_pattern = re.compile(r'Progress: (\d+)%', re.IGNORECASE)
phase_pattern = re.compile(r'Phase (\d+[a-z]?):', re.IGNORECASE)
current_phase = "Unknown"
current_progress = 0

# API tracking
api_request_pattern = re.compile(r'REQUEST.*model=([^\]]+)', re.IGNORECASE)
api_response_pattern = re.compile(r'RESPONSE.*length=(\d+)', re.IGNORECASE)
last_api_activity = time.time()

# Check interval (seconds)
CHECK_INTERVAL = 5
STALE_THRESHOLD = 120  # 2 minutes without updates = potential hang

def check_log_for_errors() -> List[Dict[str, Any]]:
    """Check log file for errors and warnings."""
    global last_position, error_count, warning_count, current_phase, current_progress, last_api_activity
    
    errors = []
    
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            # Seek to last position
            f.seek(last_position)
            
            # Read new lines
            new_lines = f.readlines()
            last_position = f.tell()
            
            # Check each new line
            for line in new_lines:
                # Check for errors
                for pattern, error_type in error_patterns:
                    if pattern.search(line):
                        errors.append({
                            'type': error_type,
                            'severity': 'ERROR',
                            'line': line.strip(),
                            'timestamp': datetime.now().isoformat()
                        })
                        error_count += 1
                
                # Check for warnings
                for pattern, warning_type in warning_patterns:
                    if pattern.search(line):
                        errors.append({
                            'type': warning_type,
                            'severity': 'WARNING',
                            'line': line.strip(),
                            'timestamp': datetime.now().isoformat()
                        })
                        warning_count += 1
                
                # Track progress
                progress_match = progress_pattern.search(line)
                if progress_match:
                    current_progress = int(progress_match.group(1))
                
                # Track phase
                phase_match = phase_pattern.search(line)
                if phase_match:
                    current_phase = phase_match.group(1)
                
                # Track API activity
                if api_request_pattern.search(line) or api_response_pattern.search(line):
                    last_api_activity = time.time()
            
            return errors
            
    except Exception as e:
        return [{
            'type': 'MONITOR_ERROR',
            'severity': 'ERROR',
            'line': f"Error reading log: {e}",
            'timestamp': datetime.now().isoformat()
        }]

def check_for_hang() -> Optional[Dict[str, Any]]:
    """Check if test has hung (no updates for too long)."""
    global last_update_time, last_api_activity
    
    current_time = time.time()
    time_since_update = current_time - last_update_time
    time_since_api = current_time - last_api_activity
    
    # Check if log file has been updated
    try:
        log_mtime = latest_log.stat().st_mtime
        time_since_log_update = current_time - log_mtime
        
        if time_since_log_update > STALE_THRESHOLD:
            return {
                'type': 'HANG_DETECTED',
                'severity': 'ERROR',
                'message': f"Log file not updated for {int(time_since_log_update)} seconds",
                'time_since_update': time_since_log_update,
                'time_since_api': time_since_api
            }
    except Exception:
        pass
    
    return None

def print_status():
    """Print current status."""
    global current_phase, current_progress, error_count, warning_count
    
    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Phase: {current_phase} | Progress: {current_progress}% | Errors: {error_count} | Warnings: {warning_count}", end='', flush=True)

def main():
    """Main monitoring loop."""
    global last_update_time
    
    print("Starting watchdog...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Check for errors
            errors = check_log_for_errors()
            
            # Print errors immediately
            for error in errors:
                print(f"\n[{error['severity']}] {error['type']}: {error['line'][:100]}")
            
            # Check for hang
            hang = check_for_hang()
            if hang:
                print(f"\n[ERROR] {hang['type']}: {hang['message']}")
                print(f"  Time since last update: {hang['time_since_update']:.1f}s")
                print(f"  Time since last API activity: {hang['time_since_api']:.1f}s")
            
            # Update last update time if log file was modified
            try:
                log_mtime = latest_log.stat().st_mtime
                if log_mtime > last_update_time:
                    last_update_time = log_mtime
            except Exception:
                pass
            
            # Print status
            print_status()
            
            # Sleep
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n[OK] Watchdog stopped by user")
        print(f"Summary: {error_count} errors, {warning_count} warnings")
    except Exception as e:
        print(f"\n[ERROR] Watchdog crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

