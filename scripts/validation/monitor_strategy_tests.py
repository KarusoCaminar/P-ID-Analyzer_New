"""
Monitor Strategy Tests - Live monitoring of strategy test progress.

Shows:
1. Current test status
2. Latest log entries
3. Progress indicators
4. Test results summary
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_latest_test():
    """Find the latest test output directory."""
    output_base = project_root / "outputs" / "strategy_tests"
    if not output_base.exists():
        return None
    
    test_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, 
                      reverse=True)
    
    if not test_dirs:
        return None
    
    return test_dirs[0]


def monitor_tests():
    """Monitor test progress."""
    print("=" * 80)
    print("STRATEGY TEST MONITOR")
    print("=" * 80)
    print(f"Monitoring: {project_root / 'outputs' / 'strategy_tests'}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()
    
    last_log_size = {}
    
    try:
        while True:
            # Find latest test directory
            latest_test = find_latest_test()
            
            if latest_test:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Latest test: {latest_test.name}")
                
                # Check log file
                log_file = latest_test / "logs" / "test.log"
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    
                    if latest_test.name not in last_log_size:
                        last_log_size[latest_test.name] = 0
                    
                    if current_size > last_log_size[latest_test.name]:
                        # Read new log entries
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(last_log_size[latest_test.name])
                            new_lines = f.readlines()
                            
                            if new_lines:
                                print("\n--- New Log Entries ---")
                                for line in new_lines[-10:]:  # Last 10 lines
                                    print(line.rstrip())
                        
                        last_log_size[latest_test.name] = current_size
                
                # Check for result file
                result_file = latest_test / "data" / "test_result.json"
                if result_file.exists():
                    print(f"\n[OK] Test completed: {latest_test.name}")
                    print(f"Result file: {result_file}")
            
            # Check for summary file
            output_base = project_root / "outputs" / "strategy_tests"
            summary_files = list(output_base.glob("test_summary_*.json"))
            if summary_files:
                latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
                print(f"\n[OK] Summary file found: {latest_summary.name}")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Monitoring stopped")


if __name__ == "__main__":
    monitor_tests()

