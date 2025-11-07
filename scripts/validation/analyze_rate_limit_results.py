"""
Analyze rate limit test results and recommend maximum settings.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Projekt-Root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_results():
    """Analyze rate limit test results and recommend maximum settings."""
    results_dir = project_root / "outputs" / "rate_limit_test"
    
    if not results_dir.exists():
        print("[ERROR] Results directory not found!")
        return
    
    # Get latest results file
    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        print("[ERROR] No result files found!")
        return
    
    latest_file = result_files[0]
    print(f"Analyzing: {latest_file.name}")
    print(f"Last Modified: {latest_file.stat().st_mtime}")
    print()
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Filter successful results
    successful = [r for r in results if r.get('success', False) and r.get('available', True) != False]
    unavailable = [r for r in results if r.get('available', False)]
    
    print("=" * 80)
    print("RATE LIMIT TEST - ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Group by model and region
    from collections import defaultdict
    by_model_region = defaultdict(list)
    for r in successful:
        key = (r.get('model'), r.get('region'))
        by_model_region[key].append(r)
    
    print("RESULTS BY MODEL/REGION:")
    print("-" * 80)
    
    recommendations = {}
    
    for (model, region), test_results in by_model_region.items():
        print(f"\n{model} - {region}:")
        print("-" * 40)
        
        # Sort by workers
        test_results_sorted = sorted(test_results, key=lambda x: x.get('max_workers', 0))
        
        for r in test_results_sorted:
            workers = r.get('max_workers', 0)
            rpm = r.get('actual_rpm', 0)
            rate_limits = r.get('rate_limit_errors', 0)
            rate_limit_rate = r.get('rate_limit_rate', 0)
            success_rate = r.get('success_rate', 0)
            
            status = "[OK]" if rate_limit_rate < 0.1 else "[WARN]" if rate_limit_rate < 0.5 else "[FAIL]"
            
            print(f"  {workers:2d} Workers: {rpm:6.1f} RPM, {rate_limits:2d} Rate Limits, "
                  f"{rate_limit_rate*100:5.1f}% Rate, {success_rate*100:5.1f}% Success {status}")
        
        # Find best configuration (highest RPM with <10% rate limits)
        best = None
        for r in test_results_sorted:
            if r.get('rate_limit_rate', 1) < 0.1:  # <10% rate limits
                if best is None or r.get('actual_rpm', 0) > best.get('actual_rpm', 0):
                    best = r
        
        if best:
            recommendations[f"{model} - {region}"] = {
                'max_workers': best.get('max_workers', 15),
                'max_rpm': best.get('actual_rpm', 0),
                'rate_limit_rate': best.get('rate_limit_rate', 0),
                'success_rate': best.get('success_rate', 0)
            }
            print(f"\n  [BEST] {best.get('max_workers')} Workers: {best.get('actual_rpm', 0):.1f} RPM "
                  f"(Rate Limit Rate: {best.get('rate_limit_rate', 0)*100:.1f}%)")
    
    if unavailable:
        print("\n" + "=" * 80)
        print("UNAVAILABLE MODELS/REGIONS:")
        print("-" * 80)
        for r in unavailable:
            print(f"  {r.get('model')} - {r.get('region')}: {r.get('error', 'Not available')}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS - MAXIMUM SETTINGS")
    print("=" * 80)
    print()
    
    # Flash recommendations
    flash_us = recommendations.get("Google Gemini 2.5 Flash - us-central1")
    flash_eu = recommendations.get("Google Gemini 2.5 Flash - europe-west3")
    pro_us = recommendations.get("Google Gemini 2.5 Pro - us-central1")
    
    if flash_us:
        print("FOR FLASH MODELS (Speed Optimized):")
        print(f"  Recommended Max Workers: {flash_us['max_workers']}")
        print(f"  Recommended Initial RPM: {int(flash_us['max_rpm'] * 0.9)}  # 90% of max for safety")
        print(f"  Recommended Max RPM: {int(flash_us['max_rpm'] * 1.1)}  # 10% buffer")
        print(f"  Recommended Max Concurrent: {flash_us['max_workers']}")
        print()
    
    if pro_us:
        print("FOR PRO MODELS (Quality Optimized):")
        print(f"  Recommended Max Workers: {pro_us['max_workers']}")
        print(f"  Recommended Initial RPM: {int(pro_us['max_rpm'] * 0.9)}  # 90% of max for safety")
        print(f"  Recommended Max RPM: {int(pro_us['max_rpm'] * 1.1)}  # 10% buffer")
        print(f"  Recommended Max Concurrent: {pro_us['max_workers']}")
        print()
    
    print("=" * 80)
    print("CONFIG.YAML RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("# Flash-optimized settings (for speed):")
    if flash_us:
        print(f"llm_rate_limit_requests_per_minute: {int(flash_us['max_rpm'] * 0.9)}  # 90% of max ({flash_us['max_rpm']:.1f} RPM)")
        print(f"llm_max_concurrent_requests: {flash_us['max_workers']}")
        print(f"llm_executor_workers: {flash_us['max_workers']}")
        print()
    
    print("# Pro-optimized settings (for quality):")
    if pro_us:
        print(f"# Pro is much slower - use conservative settings")
        print(f"llm_rate_limit_requests_per_minute: {int(pro_us['max_rpm'] * 0.9)}  # 90% of max ({pro_us['max_rpm']:.1f} RPM)")
        print(f"llm_max_concurrent_requests: {pro_us['max_workers']}")
        print(f"llm_executor_workers: {pro_us['max_workers']}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_results()

