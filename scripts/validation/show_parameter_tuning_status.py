"""
Display Parameter Tuning Status and Live Monitor.

Shows current status, progress, best parameters, and continuously monitors logs.
"""

import sys
import json
import time
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

def display_completed_status(output_dir: Path):
    """Display status of completed parameter tuning."""
    summary_file = output_dir / "data" / "parameter_tuning_summary.json"
    results_file = output_dir / "data" / "parameter_tuning_results.json"
    
    if not summary_file.exists():
        print("[ERROR] Summary file not found")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("\n" + "=" * 80)
    print("PARAMETER TUNING TEST - ERGEBNISSE")
    print("=" * 80)
    print(f"Output directory: {output_dir.name}")
    print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
    print()
    
    print("[STATS] TEST STATISTIKEN")
    print("-" * 80)
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"[OK] Successful: {summary.get('successful_tests', 0)}")
    print(f"[FAIL] Failed: {summary.get('failed_tests', 0)}")
    print()
    
    print("[BEST] BESTE PARAMETER")
    print("-" * 80)
    best_params = summary.get('best_parameters', {})
    print(f"  Factor: {best_params.get('adaptive_threshold_factor', 'N/A')}")
    print(f"  Min: {best_params.get('adaptive_threshold_min', 'N/A')}")
    print(f"  Max: {best_params.get('adaptive_threshold_max', 'N/A')}")
    print()
    
    print("[KPIs] BESTE KPIs")
    print("-" * 80)
    best_kpis = summary.get('best_kpis', {})
    print(f"  Connection F1: {best_kpis.get('connection_f1', 0.0):.4f}")
    print(f"  Element F1: {best_kpis.get('element_f1', 0.0):.4f}")
    print(f"  Quality Score: {best_kpis.get('quality_score', 0.0):.2f}")
    print(f"  Element Precision: {best_kpis.get('element_precision', 0.0):.4f}")
    print(f"  Element Recall: {best_kpis.get('element_recall', 0.0):.4f}")
    print()
    
    if best_kpis.get('connection_f1', 0.0) == 0.0:
        print("[WARNING] KRITISCHES PROBLEM ERKANNT")
        print("-" * 80)
        print("Connection F1 = 0.0 für ALLE Parameter-Kombinationen!")
        print("Das bedeutet:")
        print("  • Das Problem liegt NICHT in den Threshold-Parametern")
        print("  • Die Connection Detection funktioniert überhaupt nicht")
        print("  • Mögliche Ursachen:")
        print("    - CV Line Extraction findet keine Linien")
        print("    - Connection Matching funktioniert nicht")
        print("    - Ground Truth Connections sind falsch formatiert")
        print("    - Hybrid Validation blockiert alle Connections")
        print()
    
    print("[TOP5] TOP 5 ERGEBNISSE")
    print("-" * 80)
    top_5 = summary.get('top_5_results', [])[:5]
    for i, result in enumerate(top_5, 1):
        params = result.get('parameters', {})
        print(f"{i}. Factor={params.get('adaptive_threshold_factor')}, "
              f"Min={params.get('adaptive_threshold_min')}, "
              f"Max={params.get('adaptive_threshold_max')} -> "
              f"Connection F1: {result.get('connection_f1', 0.0):.4f}, "
              f"Element F1: {result.get('element_f1', 0.0):.4f}")
    print()
    
    print("=" * 80)

def monitor_logs_live(output_dir: Path, n_lines: int = 20):
    """Monitor logs in real-time."""
    log_file = output_dir / "logs" / "parameter_tuning.log"
    if not log_file.exists():
        print("[ERROR] Log file not found")
        return
    
    print("\n" + "=" * 80)
    print("LIVE LOG MONITOR - Drücke Ctrl+C zum Beenden")
    print("=" * 80)
    print(f"Monitoring: {log_file.name}")
    print("=" * 80)
    print()
    
    last_pos = 0
    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_pos:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_pos)
                        new_content = f.read()
                        if new_content:
                            print(new_content, end='', flush=True)
                        last_pos = current_size
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

def main():
    """Main function."""
    output_dir = find_latest_output_dir()
    if not output_dir:
        print("[ERROR] No parameter tuning output directory found")
        return
    
    # Display completed status
    display_completed_status(output_dir)
    
    # Ask user if they want to monitor logs
    print("\nMöchtest du die Logs live überwachen? (j/n): ", end='', flush=True)
    try:
        response = input().strip().lower()
        if response in ['j', 'ja', 'y', 'yes']:
            monitor_logs_live(output_dir)
    except (EOFError, KeyboardInterrupt):
        print("\n✅ Exiting")

if __name__ == "__main__":
    main()

