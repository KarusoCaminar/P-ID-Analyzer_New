"""
Monitor live test progress and analyze performance.

This script monitors the live test log file and provides:
- Real-time progress updates
- DSQ optimizer status
- Performance metrics
- Error detection
- Optimization suggestions
"""

import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

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

# Track metrics
metrics = {
    'api_calls': 0,
    'rate_limits': 0,
    'timeouts': 0,
    'errors': 0,
    'dsq_throttles': 0,
    'start_time': None,
    'last_update': time.time()
}

# Patterns to track
patterns = {
    'api_call': re.compile(r'LLM.*call|API.*request|call_llm', re.IGNORECASE),
    'rate_limit': re.compile(r'429|rate.*limit|resource.*exhausted|DSQ.*Rate.*Limit', re.IGNORECASE),
    'timeout': re.compile(r'timeout|timed.*out|deadline.*exceeded', re.IGNORECASE),
    'dsq_throttle': re.compile(r'DSQ.*Request.*Smoothing|Throttling.*request', re.IGNORECASE),
    'error': re.compile(r'ERROR|FAIL|Exception', re.IGNORECASE),
    'phase': re.compile(r'Phase\s+(\d+[a-z]?)|Starting.*phase', re.IGNORECASE),
    'dsq_status': re.compile(r'current.*rate.*(\d+\.?\d*).*RPM|success.*rate.*(\d+\.?\d*)%', re.IGNORECASE)
}

def analyze_line(line: str) -> Dict[str, Any]:
    """Analyze a log line and extract metrics."""
    result = {
        'api_call': bool(patterns['api_call'].search(line)),
        'rate_limit': bool(patterns['rate_limit'].search(line)),
        'timeout': bool(patterns['timeout'].search(line)),
        'dsq_throttle': bool(patterns['dsq_throttle'].search(line)),
        'error': bool(patterns['error'].search(line)),
        'phase': patterns['phase'].search(line),
        'dsq_status': patterns['dsq_status'].search(line)
    }
    return result

def print_status():
    """Print current status."""
    elapsed = time.time() - metrics['start_time'] if metrics['start_time'] else 0
    elapsed_min = elapsed / 60.0
    
    print("\n" + "=" * 80)
    print(f"LIVE TEST STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    print(f"Elapsed Time: {elapsed_min:.1f} minutes")
    print(f"API Calls: {metrics['api_calls']}")
    print(f"Rate Limits (429): {metrics['rate_limits']}")
    print(f"Timeouts: {metrics['timeouts']}")
    print(f"DSQ Throttles: {metrics['dsq_throttles']}")
    print(f"Errors: {metrics['errors']}")
    
    if metrics['api_calls'] > 0:
        rate = metrics['api_calls'] / elapsed_min if elapsed_min > 0 else 0
        print(f"API Call Rate: {rate:.1f} calls/minute")
    
    if metrics['rate_limits'] > 0:
        rate_limit_pct = (metrics['rate_limits'] / metrics['api_calls']) * 100 if metrics['api_calls'] > 0 else 0
        print(f"Rate Limit Rate: {rate_limit_pct:.1f}%")
        print(f"[WARNING] High rate limit rate - DSQ optimizer should reduce rate")
    
    print("=" * 80)

# Monitor log file
print("[OK] Starting live log monitoring...")
print("[OK] Press Ctrl+C to stop\n")

last_pos = 0
try:
    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
        # Read existing content
        content = f.read()
        last_pos = len(content)
        
        # Analyze existing content
        for line in content.split('\n'):
            analysis = analyze_line(line)
            if analysis['api_call']:
                metrics['api_calls'] += 1
            if analysis['rate_limit']:
                metrics['rate_limits'] += 1
            if analysis['timeout']:
                metrics['timeouts'] += 1
            if analysis['dsq_throttle']:
                metrics['dsq_throttles'] += 1
            if analysis['error']:
                metrics['errors'] += 1
            if analysis['phase'] and not metrics['start_time']:
                metrics['start_time'] = time.time()
        
        # Print initial status
        if metrics['api_calls'] > 0:
            print_status()
        
        # Monitor for new content
        while True:
            time.sleep(2)  # Check every 2 seconds
            
            # Read new content
            f.seek(last_pos)
            new_content = f.read()
            
            if new_content:
                # Analyze new lines
                for line in new_content.split('\n'):
                    if not line.strip():
                        continue
                    
                    analysis = analyze_line(line)
                    if analysis['api_call']:
                        metrics['api_calls'] += 1
                    if analysis['rate_limit']:
                        metrics['rate_limits'] += 1
                        print(f"\n[!] Rate Limit detected: {line[:100]}")
                    if analysis['timeout']:
                        metrics['timeouts'] += 1
                        print(f"\n[!] Timeout detected: {line[:100]}")
                    if analysis['dsq_throttle']:
                        metrics['dsq_throttles'] += 1
                    if analysis['error']:
                        metrics['errors'] += 1
                        print(f"\n[!] Error detected: {line[:100]}")
                    if analysis['dsq_status']:
                        match = analysis['dsq_status']
                        if match.group(1):
                            print(f"[DSQ] Current Rate: {match.group(1)} RPM")
                        if match.group(2):
                            print(f"[DSQ] Success Rate: {match.group(2)}%")
                
                last_pos = f.tell()
                
                # Print status every 30 seconds
                if time.time() - metrics['last_update'] > 30:
                    print_status()
                    metrics['last_update'] = time.time()
            
            # Check if file was truncated (new test started)
            current_size = latest_log.stat().st_size
            if current_size < last_pos:
                print("[INFO] Log file was truncated - new test started")
                last_pos = 0
                metrics = {
                    'api_calls': 0,
                    'rate_limits': 0,
                    'timeouts': 0,
                    'errors': 0,
                    'dsq_throttles': 0,
                    'start_time': time.time(),
                    'last_update': time.time()
                }

except KeyboardInterrupt:
    print("\n[OK] Monitoring stopped by user")
    print_status()
except FileNotFoundError:
    print(f"[ERROR] Log file not found: {latest_log}")
except Exception as e:
    print(f"[ERROR] Monitoring error: {e}")
    import traceback
    traceback.print_exc()

