"""
Analyze Best Test Results - Shows visualizations and detailed analysis.

This script analyzes the latest test results and shows:
1. Visualizations (debug maps, confidence maps, etc.)
2. Element/Connection counts
3. KPI analysis
4. Ground truth comparison
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json


def find_latest_test() -> Optional[Path]:
    """Find the latest test output directory."""
    output_base = project_root / "outputs" / "live_test"
    if not output_base.exists():
        return None
    
    test_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, 
                      reverse=True)
    
    if not test_dirs:
        return None
    
    return test_dirs[0]


def analyze_test_results(test_dir: Path):
    """Analyze test results and show summary."""
    print("=" * 80)
    print("TEST RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Test Directory: {test_dir}")
    print()
    
    # Load test result
    result_file = test_dir / "data" / "test_result.json"
    if not result_file.exists():
        print(f"[ERROR] Test result file not found: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        test_result = json.load(f)
    
    # Basic info
    print("BASIC INFO:")
    print(f"  Strategy: {test_result.get('strategy', 'unknown')}")
    print(f"  Image: {Path(test_result.get('image_path', '')).name}")
    print(f"  Duration: {test_result.get('duration_minutes', 0):.2f} minutes")
    print(f"  Timestamp: {test_result.get('timestamp', 'unknown')}")
    print()
    
    # Results
    result = test_result.get('result', {})
    elements = result.get('elements', [])
    connections = result.get('connections', [])
    
    print("ANALYSIS RESULTS:")
    print(f"  Elements Found: {len(elements)}")
    print(f"  Connections Found: {len(connections)}")
    print()
    
    # Show element types
    element_types = {}
    for el in elements:
        el_type = el.get('type', 'unknown')
        element_types[el_type] = element_types.get(el_type, 0) + 1
    
    print("ELEMENT TYPES:")
    for el_type, count in sorted(element_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {el_type}: {count}")
    print()
    
    # Ground truth
    gt_data = test_result.get('ground_truth', {})
    if gt_data:
        gt_elements = gt_data.get('elements', [])
        gt_connections = gt_data.get('connections', [])
        print("GROUND TRUTH:")
        print(f"  Elements: {len(gt_elements)}")
        print(f"  Connections: {len(gt_connections)}")
    else:
        print("GROUND TRUTH: NOT PROVIDED (KPIs will be 0)")
    print()
    
    # KPIs
    kpis = test_result.get('kpis', {})
    print("KPIs:")
    print(f"  Element F1: {kpis.get('element_f1', 0.0):.4f}")
    print(f"  Element Precision: {kpis.get('element_precision', 0.0):.4f}")
    print(f"  Element Recall: {kpis.get('element_recall', 0.0):.4f}")
    print(f"  Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
    print(f"  Quality Score: {kpis.get('quality_score', 0.0):.2f}")
    print()
    
    # Visualizations
    viz_dir = test_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        print(f"VISUALIZATIONS ({len(viz_files)} files):")
        for viz_file in sorted(viz_files):
            file_size = viz_file.stat().st_size / 1024  # KB
            print(f"  {viz_file.name} ({file_size:.1f} KB)")
    else:
        print("VISUALIZATIONS: Not found")
    print()
    
    # Show sample elements with bboxes
    print("SAMPLE ELEMENTS (with BBoxes):")
    for i, el in enumerate(elements[:5]):
        el_id = el.get('id', 'unknown')
        el_type = el.get('type', 'unknown')
        bbox = el.get('bbox', {})
        confidence = el.get('confidence', 0.0)
        print(f"  {i+1}. {el_id} ({el_type}) - Confidence: {confidence:.2f}")
        if bbox:
            print(f"      BBox: x={bbox.get('x', 0):.3f}, y={bbox.get('y', 0):.3f}, "
                  f"w={bbox.get('width', 0):.3f}, h={bbox.get('height', 0):.3f}")
    print()
    
    # Show sample connections
    print("SAMPLE CONNECTIONS:")
    for i, conn in enumerate(connections[:5]):
        from_id = conn.get('from_id', 'unknown')
        to_id = conn.get('to_id', 'unknown')
        confidence = conn.get('confidence', 0.0)
        kind = conn.get('kind', 'unknown')
        print(f"  {i+1}. {from_id} -> {to_id} ({kind}) - Confidence: {confidence:.2f}")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("To view visualizations, open:")
    print(f"  {viz_dir}")
    print()
    print("To view detailed results, open:")
    print(f"  {result_file}")


def main():
    """Main function."""
    test_dir = find_latest_test()
    
    if not test_dir:
        print("[ERROR] No test results found in outputs/live_test/")
        return
    
    analyze_test_results(test_dir)


if __name__ == "__main__":
    main()

