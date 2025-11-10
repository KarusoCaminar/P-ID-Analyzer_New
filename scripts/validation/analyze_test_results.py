"""
Analyze test results and compare with previous runs.

Provides:
- Quality Score comparison
- Connection F1 analysis
- Error analysis
- Recommendations
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Projekt-Root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_test_result(result_file: Path) -> Dict[str, Any]:
    """Analyze a single test result."""
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    analysis = {
        'strategy': result.get('strategy', 'unknown'),
        'duration_minutes': result.get('duration_minutes', 0.0),
        'success': result.get('success', False),
        'kpis': result.get('kpis', {}),
        'elements_count': len(result.get('result', {}).get('elements', [])),
        'connections_count': len(result.get('result', {}).get('connections', [])),
        'issues': []
    }
    
    # Check for issues
    kpis = analysis['kpis']
    
    # Quality Score
    quality_score = kpis.get('quality_score', 0.0)
    if quality_score < 50:
        analysis['issues'].append(f"Quality Score sehr niedrig: {quality_score:.2f} (< 50)")
    elif quality_score < 70:
        analysis['issues'].append(f"Quality Score niedrig: {quality_score:.2f} (< 70)")
    
    # Connection F1
    connection_f1 = kpis.get('connection_f1', 0.0)
    if connection_f1 < 0.1:
        analysis['issues'].append(f"Connection F1 kritisch niedrig: {connection_f1:.4f} (< 0.1)")
    elif connection_f1 < 0.5:
        analysis['issues'].append(f"Connection F1 niedrig: {connection_f1:.4f} (< 0.5)")
    
    # Element F1
    element_f1 = kpis.get('element_f1', 0.0)
    if element_f1 < 0.5:
        analysis['issues'].append(f"Element F1 niedrig: {element_f1:.4f} (< 0.5)")
    
    return analysis

def compare_results(current_result: Path, previous_result: Optional[Path] = None) -> Dict[str, Any]:
    """Compare current result with previous result."""
    current = analyze_test_result(current_result)
    
    comparison = {
        'current': current,
        'previous': None,
        'improvements': [],
        'regressions': [],
        'recommendations': []
    }
    
    if previous_result and previous_result.exists():
        previous = analyze_test_result(previous_result)
        comparison['previous'] = previous
        
        # Compare Quality Score
        current_score = current['kpis'].get('quality_score', 0.0)
        previous_score = previous['kpis'].get('quality_score', 0.0)
        score_diff = current_score - previous_score
        
        if score_diff > 5:
            comparison['improvements'].append(f"Quality Score verbessert: {previous_score:.2f} -> {current_score:.2f} (+{score_diff:.2f})")
        elif score_diff < -5:
            comparison['regressions'].append(f"Quality Score verschlechtert: {previous_score:.2f} -> {current_score:.2f} ({score_diff:.2f})")
        
        # Compare Connection F1
        current_conn_f1 = current['kpis'].get('connection_f1', 0.0)
        previous_conn_f1 = previous['kpis'].get('connection_f1', 0.0)
        conn_diff = current_conn_f1 - previous_conn_f1
        
        if conn_diff > 0.1:
            comparison['improvements'].append(f"Connection F1 verbessert: {previous_conn_f1:.4f} -> {current_conn_f1:.4f} (+{conn_diff:.4f})")
        elif conn_diff < -0.1:
            comparison['regressions'].append(f"Connection F1 verschlechtert: {previous_conn_f1:.4f} -> {current_conn_f1:.4f} ({conn_diff:.4f})")
        
        # Compare Element F1
        current_el_f1 = current['kpis'].get('element_f1', 0.0)
        previous_el_f1 = previous['kpis'].get('element_f1', 0.0)
        el_diff = current_el_f1 - previous_el_f1
        
        if el_diff > 0.1:
            comparison['improvements'].append(f"Element F1 verbessert: {previous_el_f1:.4f} -> {current_el_f1:.4f} (+{el_diff:.4f})")
        elif el_diff < -0.1:
            comparison['regressions'].append(f"Element F1 verschlechtert: {previous_el_f1:.4f} -> {current_el_f1:.4f} ({el_diff:.4f})")
    
    # Generate recommendations
    if current['kpis'].get('connection_f1', 0.0) < 0.1:
        comparison['recommendations'].append("Connection F1 ist kritisch niedrig - prüfe Fusion-Engine und ID-Matching")
    
    if current['kpis'].get('quality_score', 0.0) < 70:
        comparison['recommendations'].append("Quality Score ist niedrig - prüfe ob Phase 3 (Self-Correction) gelaufen ist")
    
    if len(current.get('issues', [])) > 3:
        comparison['recommendations'].append("Viele Probleme gefunden - prüfe Logs auf Fehler")
    
    return comparison

def find_latest_result(output_dir: Path) -> Optional[Path]:
    """Find latest test result."""
    result_files = list(output_dir.glob("**/test_result.json"))
    if not result_files:
        return None
    
    # Sort by modification time
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]

def find_previous_result(output_dir: Path, exclude: Path) -> Optional[Path]:
    """Find previous test result (excluding current)."""
    result_files = list(output_dir.glob("**/test_result.json"))
    if not result_files:
        return None
    
    # Filter out current result
    result_files = [p for p in result_files if p != exclude]
    if not result_files:
        return None
    
    # Sort by modification time
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]

def main():
    """Main analysis function."""
    output_base = project_root / "outputs" / "live_test"
    
    if not output_base.exists():
        print("[ERROR] No output directory found. Run test first!")
        sys.exit(1)
    
    # Find latest result
    latest_result = find_latest_result(output_base)
    if not latest_result:
        print("[ERROR] No test results found. Run test first!")
        sys.exit(1)
    
    print("=" * 80)
    print("TEST RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Latest Result: {latest_result}")
    print()
    
    # Analyze latest result
    current_analysis = analyze_test_result(latest_result)
    
    print("=== CURRENT TEST RESULTS ===")
    print(f"Strategy: {current_analysis['strategy']}")
    print(f"Duration: {current_analysis['duration_minutes']:.2f} minutes")
    print(f"Quality Score: {current_analysis['kpis'].get('quality_score', 0.0):.2f}")
    print(f"Element F1: {current_analysis['kpis'].get('element_f1', 0.0):.4f}")
    print(f"Connection F1: {current_analysis['kpis'].get('connection_f1', 0.0):.4f}")
    print(f"Elements Found: {current_analysis['elements_count']}")
    print(f"Connections Found: {current_analysis['connections_count']}")
    print()
    
    if current_analysis['issues']:
        print("=== ISSUES FOUND ===")
        for issue in current_analysis['issues']:
            print(f"  - {issue}")
        print()
    
    # Compare with previous result
    previous_result = find_previous_result(output_base, latest_result)
    if previous_result:
        print(f"=== COMPARISON WITH PREVIOUS RESULT ===")
        print(f"Previous Result: {previous_result}")
        print()
        
        comparison = compare_results(latest_result, previous_result)
        
        if comparison['improvements']:
            print("=== IMPROVEMENTS ===")
            for improvement in comparison['improvements']:
                print(f"  + {improvement}")
            print()
        
        if comparison['regressions']:
            print("=== REGRESSIONS ===")
            for regression in comparison['regressions']:
                print(f"  - {regression}")
            print()
    
    # Recommendations
    comparison = compare_results(latest_result, previous_result)
    if comparison['recommendations']:
        print("=== RECOMMENDATIONS ===")
        for recommendation in comparison['recommendations']:
            print(f"  -> {recommendation}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()

