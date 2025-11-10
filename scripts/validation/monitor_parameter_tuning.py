"""
Live Monitor for Parameter Tuning Test Run.

Monitors the parameter tuning process and displays real-time progress,
including test completion, best parameters found, and any errors.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def find_latest_output_dir() -> Optional[Path]:
    """Find the latest parameter tuning output directory."""
    output_base = project_root / "outputs" / "parameter_tuning"
    if not output_base.exists():
        return None
    
    dirs = sorted(output_base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None

def read_latest_log_lines(output_dir: Path, n_lines: int = 50) -> list[str]:
    """Read the latest N lines from the log file."""
    log_file = output_dir / "logs" / "parameter_tuning.log"
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n_lines:] if len(lines) > n_lines else lines
    except Exception as e:
        return [f"Error reading log file: {e}"]

def get_test_progress(output_dir: Path) -> Dict[str, Any]:
    """Get current test progress from results file."""
    results_file = output_dir / "data" / "parameter_tuning_results.json"
    if not results_file.exists():
        return {
            "status": "starting",
            "total_tests": 0,
            "completed_tests": 0,
            "failed_tests": 0,
            "progress_percent": 0.0,
            "best_connection_f1": 0.0,
            "best_parameters": None
        }
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        total = results.get('total_tests', 0)
        completed = results.get('completed_tests', 0)
        failed = results.get('failed_tests', 0)
        progress = (completed / total * 100) if total > 0 else 0.0
        
        return {
            "status": "running" if completed < total else "completed",
            "total_tests": total,
            "completed_tests": completed,
            "failed_tests": failed,
            "progress_percent": progress,
            "best_connection_f1": results.get('best_connection_f1', 0.0),
            "best_parameters": results.get('best_parameters', None),
            "current_test": results.get('current_test', None)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def format_progress_bar(percent: float, width: int = 40) -> str:
    """Create a progress bar string."""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:.1f}%"

def display_status(output_dir: Path):
    """Display current status of parameter tuning."""
    print("\n" + "=" * 80)
    print(f"PARAMETER TUNING LIVE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Output directory: {output_dir.name}")
    print()
    
    # Get progress
    progress = get_test_progress(output_dir)
    
    if progress.get("status") == "error":
        print(f"❌ Error: {progress.get('error')}")
        return
    
    if progress.get("status") == "starting":
        print("⏳ Test is starting...")
        print("\n=== Recent Logs ===")
        logs = read_latest_log_lines(output_dir, 20)
        for line in logs:
            print(line.rstrip())
        return
    
    # Display progress
    print("📊 TEST PROGRESS")
    print("-" * 80)
    print(f"Status: {'✅ Completed' if progress['status'] == 'completed' else '🔄 Running'}")
    print(f"Total tests: {progress['total_tests']}")
    print(f"Completed: {progress['completed_tests']}")
    print(f"Failed: {progress['failed_tests']}")
    print(f"Progress: {format_progress_bar(progress['progress_percent'])}")
    
    if progress.get('current_test'):
        print(f"\n🔬 Current test: {progress['current_test']}")
    
    # Display best results
    if progress.get('best_connection_f1', 0) > 0:
        print("\n🏆 BEST RESULTS SO FAR")
        print("-" * 80)
        print(f"Best Connection F1: {progress['best_connection_f1']:.4f}")
        if progress.get('best_parameters'):
            params = progress['best_parameters']
            print(f"Best parameters:")
            print(f"  - Factor: {params.get('adaptive_threshold_factor', 'N/A')}")
            print(f"  - Min: {params.get('adaptive_threshold_min', 'N/A')}")
            print(f"  - Max: {params.get('adaptive_threshold_max', 'N/A')}")
    
    # Display recent logs
    print("\n📝 RECENT LOGS (last 15 lines)")
    print("-" * 80)
    logs = read_latest_log_lines(output_dir, 15)
    for line in logs:
        # Highlight important messages
        line_stripped = line.rstrip()
        if "ERROR" in line_stripped or "FAIL" in line_stripped:
            print(f"❌ {line_stripped}")
        elif "WARNING" in line_stripped:
            print(f"⚠️  {line_stripped}")
        elif "Circuit breaker" in line_stripped:
            print(f"🔴 {line_stripped}")
        elif "RESULTS" in line_stripped or "Connection F1" in line_stripped:
            print(f"✅ {line_stripped}")
        elif "Analysis completed" in line_stripped:
            print(f"✅ {line_stripped}")
        else:
            print(line_stripped)
    
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring")

def main():
    """Main monitoring loop."""
    print("Starting Parameter Tuning Monitor...")
    print("Looking for latest output directory...")
    
    try:
        while True:
            output_dir = find_latest_output_dir()
            if output_dir:
                display_status(output_dir)
            else:
                print("⏳ Waiting for parameter tuning to start...")
                time.sleep(5)
                continue
            
            # Wait before next update
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error in monitor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

