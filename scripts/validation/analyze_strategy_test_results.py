"""
Analyze Strategy Test Results - Comprehensive analysis of strategy test results.

Analyzes:
1. Quality Scores for all strategies
2. Element/Connection counts
3. Fusion effectiveness
4. Internal KPIs comparison
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_test_results() -> List[Path]:
    """Find all test result files."""
    output_base = project_root / "outputs" / "strategy_tests"
    if not output_base.exists():
        return []
    
    result_files = list(output_base.glob("*/*/data/test_result.json"))
    return sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)


def analyze_results():
    """Analyze all test results."""
    print("=" * 80)
    print("STRATEGY TEST RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Find all test results
    result_files = find_test_results()
    
    if not result_files:
        print("No test results found!")
        return
    
    print(f"Found {len(result_files)} test results")
    print()
    
    # Load all results
    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    if not results:
        print("No valid test results found!")
        return
    
    # Create summary table
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Group by strategy and image
    summary_data = []
    for result in results:
        strategy = result.get('strategy', 'unknown')
        image = result.get('image', 'unknown')
        success = result.get('success', False)
        
        if not success:
            summary_data.append({
                'Strategy': strategy,
                'Image': image,
                'Quality Score': 'FAILED',
                'Elements': 'N/A',
                'Connections': 'N/A',
                'Graph Density': 'N/A',
                'Duration (min)': 'N/A'
            })
            continue
        
        kpis = result.get('kpis', {})
        quality_score = kpis.get('quality_score', 0.0)
        elements = kpis.get('total_elements', 0)
        connections = kpis.get('total_connections', 0)
        graph_density = kpis.get('graph_density', 0.0)
        duration = result.get('duration_minutes', 0.0)
        
        summary_data.append({
            'Strategy': strategy,
            'Image': image,
            'Quality Score': f"{quality_score:.2f}",
            'Elements': elements,
            'Connections': connections,
            'Graph Density': f"{graph_density:.4f}",
            'Duration (min)': f"{duration:.2f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Print summary table
    print(df.to_string(index=False))
    print()
    
    # Fusion strategy analysis
    print("=" * 80)
    print("FUSION STRATEGY ANALYSIS (FOCUS)")
    print("=" * 80)
    print()
    
    fusion_results = [r for r in results if r.get('strategy') == 'hybrid_fusion' and r.get('success', False)]
    simple_results = [r for r in results if r.get('strategy') == 'simple_whole_image' and r.get('success', False)]
    
    if fusion_results and simple_results:
        # Compare fusion vs simple for each image
        for image_name in ['Simple PID', 'Uni-1']:
            fusion_result = next((r for r in fusion_results if r.get('image') == image_name), None)
            simple_result = next((r for r in simple_results if r.get('image') == image_name), None)
            
            if fusion_result and simple_result:
                fusion_kpis = fusion_result.get('kpis', {})
                simple_kpis = simple_result.get('kpis', {})
                
                fusion_score = fusion_kpis.get('quality_score', 0.0)
                simple_score = simple_kpis.get('quality_score', 0.0)
                
                fusion_elements = fusion_kpis.get('total_elements', 0)
                simple_elements = simple_kpis.get('total_elements', 0)
                
                fusion_connections = fusion_kpis.get('total_connections', 0)
                simple_connections = simple_kpis.get('total_connections', 0)
                
                print(f"{image_name}:")
                print(f"  Quality Score:")
                print(f"    Simple: {simple_score:.2f}")
                print(f"    Fusion: {fusion_score:.2f}")
                print(f"    Improvement: {fusion_score - simple_score:+.2f}")
                print()
                
                print(f"  Elements:")
                print(f"    Simple: {simple_elements}")
                print(f"    Fusion: {fusion_elements}")
                print(f"    Difference: {fusion_elements - simple_elements:+d}")
                print()
                
                print(f"  Connections:")
                print(f"    Simple: {simple_connections}")
                print(f"    Fusion: {fusion_connections}")
                print(f"    Difference: {fusion_connections - simple_connections:+d}")
                print()
                
                # Fusion effectiveness
                if fusion_score > simple_score:
                    print(f"  ✅ Fusion improves quality score by {fusion_score - simple_score:.2f} points")
                else:
                    print(f"  ❌ Fusion does not improve quality score (difference: {fusion_score - simple_score:.2f})")
                
                if fusion_elements > simple_elements:
                    print(f"  ✅ Fusion finds {fusion_elements - simple_elements} more elements")
                else:
                    print(f"  ⚠️  Fusion finds {fusion_elements - simple_elements} elements (difference: {fusion_elements - simple_elements:+d})")
                
                if fusion_connections > simple_connections:
                    print(f"  ✅ Fusion finds {fusion_connections - simple_connections} more connections")
                else:
                    print(f"  ⚠️  Fusion finds {fusion_connections - simple_connections} connections (difference: {fusion_connections - simple_connections:+d})")
                
                print()
                print("-" * 80)
                print()
    
    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    
    # Check if system works
    all_successful = all(r.get('success', False) for r in results)
    if all_successful:
        print("✅ All tests completed successfully")
    else:
        failed_count = sum(1 for r in results if not r.get('success', False))
        print(f"⚠️  {failed_count} test(s) failed")
    
    # Check if fusion works
    if fusion_results:
        fusion_scores = [r.get('kpis', {}).get('quality_score', 0.0) for r in fusion_results]
        avg_fusion_score = sum(fusion_scores) / len(fusion_scores) if fusion_scores else 0.0
        
        if avg_fusion_score > 50.0:
            print(f"✅ Fusion strategy works (avg quality score: {avg_fusion_score:.2f})")
        else:
            print(f"❌ Fusion strategy quality score is low (avg: {avg_fusion_score:.2f})")
    
    # Check if internal KPIs work
    kpi_scores = [r.get('kpis', {}).get('quality_score', 0.0) for r in results if r.get('success', False)]
    if kpi_scores:
        avg_kpi_score = sum(kpi_scores) / len(kpi_scores)
        print(f"✅ Internal KPIs work (avg quality score: {avg_kpi_score:.2f})")
    else:
        print("❌ Internal KPIs not calculated")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    analyze_results()

