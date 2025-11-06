"""
Quick Test: Ground Truth Loading verbessert
"""

import json
from pathlib import Path
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.services.config_service import ConfigService

def test_truth_loading():
    """Test Ground Truth Loading mit verschiedenen Bildern."""
    
    print("\n" + "="*60)
    print("TEST: Ground Truth Loading")
    print("="*60)
    
    # Initialize coordinator
    config_service = ConfigService(Path("config.yaml"))
    coordinator = PipelineCoordinator(
        llm_client=None,  # Will be set by backend
        knowledge_manager=None,
        config_service=config_service,
        model_strategy={},
        active_logic_parameters={},
        progress_callback=None
    )
    
    # Test images
    test_images = [
        "training_data/organized_tests/simple_pids/Einfaches P&I.png",
        "training_data/organized_tests/complex_pids/page_1_original.png",
        "training_data/organized_tests/complex_pids/page_2_original.png",
        "training_data/organized_tests/complex_pids/pump-with-storage-tank.png",
    ]
    
    results = []
    for img_path in test_images:
        img_file = Path(img_path)
        if not img_file.exists():
            print(f"\n⚠  Bild nicht gefunden: {img_path}")
            continue
        
        print(f"\n📸 Test: {img_file.name}")
        truth_data = coordinator._load_truth_data(str(img_file))
        
        if truth_data:
            elements = len(truth_data.get('elements', []))
            connections = len(truth_data.get('connections', []))
            print(f"  ✅ Ground Truth geladen!")
            print(f"     - Elemente: {elements}")
            print(f"     - Connections: {connections}")
            results.append({
                'image': img_file.name,
                'loaded': True,
                'elements': elements,
                'connections': connections
            })
        else:
            print(f"  ❌ Kein Ground Truth gefunden")
            results.append({
                'image': img_file.name,
                'loaded': False
            })
    
    # Summary
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    loaded = sum(1 for r in results if r.get('loaded'))
    total = len(results)
    print(f"\nGround Truth geladen: {loaded}/{total} ({loaded/total*100:.0f}%)")
    
    if loaded == total:
        print("✅ Alle Ground Truth Dateien erfolgreich geladen!")
    elif loaded > 0:
        print(f"⚠️  {total-loaded} Dateien konnten nicht geladen werden")
    else:
        print("❌ Keine Ground Truth Dateien geladen!")
    
    return results

if __name__ == '__main__':
    test_truth_loading()

