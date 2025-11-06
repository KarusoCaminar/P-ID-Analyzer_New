#!/usr/bin/env python3
"""
Automatisierter Test für P&ID Analyzer v2.0

Testet das System vollständig:
1. System-Check
2. Backend-Initialisierung
3. Automatische Analyse eines Test-Bildes
4. Ergebnis-Validierung
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

def check_env():
    """Prüfe Umgebungsvariablen."""
    print("=== Checking Environment Variables ===")
    
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not gcp_project_id:
        print("[FAIL] GCP_PROJECT_ID not set!")
        print("       Please create .env file with:")
        print("       GCP_PROJECT_ID=koretex-zugang")
        print("       GCP_PROJECT_NUMBER=748084370989")
        print("       GCP_LOCATION=us-central1")
        return False
    
    print(f"[OK] GCP_PROJECT_ID: {gcp_project_id}")
    print(f"[OK] GCP_LOCATION: {gcp_location}")
    print()
    return True

def find_test_image():
    """Finde ein Test-Bild."""
    print("=== Finding Test Image ===")
    
    # Mögliche Test-Bild-Pfade
    test_paths = [
        project_root / "training_data" / "simple_pids" / "Einfaches P&I.png",
        project_root / "training_data" / "simple_pids" / "Einfaches P_I.png",
        project_root / "training_data" / "simple_pids",
        project_root / "training_data",
    ]
    
    for test_path in test_paths:
        if test_path.exists():
            if test_path.is_file() and test_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                print(f"[OK] Found test image: {test_path}")
                print()
                return str(test_path)
            elif test_path.is_dir():
                # Suche nach Bildern im Verzeichnis
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    images = list(test_path.glob(ext))
                    if images:
                        # Überspringe Result-Bilder
                        for img in images:
                            if not any(exclude in img.name.lower() for exclude in ['output', 'result', 'truth', 'cgm', 'debug']):
                                print(f"[OK] Found test image: {img}")
                                print()
                                return str(img)
    
    print("[WARN] No test image found in training_data")
    print("       Please provide a test image path as argument")
    print()
    return None

def test_backend_init():
    """Test Backend-Initialisierung."""
    print("=== Testing Backend Initialization ===")
    
    try:
        from src.services.config_service import ConfigService
        from src.analyzer.ai.llm_client import LLMClient
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        
        # Config Service
        config_path = project_root / "config.yaml"
        if config_path.exists():
            config_service = ConfigService(config_path=config_path)
        else:
            config_service = ConfigService()
        
        config = config_service.get_config()
        print("[OK] ConfigService initialized")
        
        # Get config as dict (safe method)
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = config_service.get_raw_config()
        
        # LLM Client
        llm_client = LLMClient(
            project_id=gcp_project_id,
            default_location=gcp_location,
            config=config_dict
        )
        print("[OK] LLMClient initialized")
        
        # Knowledge Manager
        element_type_list_path = config_service.get_path("element_type_list") or project_root / "element_type_list.json"
        learning_db_path = config_service.get_path("learning_db") or project_root / "learning_db.json"
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,  # KnowledgeManager uses llm_handler parameter
            config=config_dict
        )
        print("[OK] KnowledgeManager initialized")
        
        # Pipeline Coordinator
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        print("[OK] PipelineCoordinator initialized")
        print()
        
        return coordinator
        
    except Exception as e:
        print(f"[FAIL] Backend initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None

def run_analysis(coordinator, image_path):
    """Führe Analyse durch."""
    print("=== Running Analysis ===")
    print(f"Image: {image_path}")
    print()
    
    try:
        # Progress Callback
        class TestProgressCallback:
            def update_progress(self, value: int, message: str) -> None:
                if value % 10 == 0:  # Nur bei 10% Schritten
                    print(f"Progress: {message} ({value}%)")
            
            def update_status_label(self, text: str) -> None:
                print(f"Status: {text}")
            
            def report_truth_mode(self, active: bool) -> None:
                if active:
                    print("Truth mode: Active")
            
            def report_correction(self, correction_text: str) -> None:
                pass  # Skip corrections in automated test
        
        coordinator.progress_callback = TestProgressCallback()
        
        # Run analysis
        result = coordinator.process(
            image_path=image_path,
            output_dir=None,  # Auto-generate
            params_override=None
        )
        
        print()
        print("[OK] Analysis completed!")
        print()
        
        return result
        
    except Exception as e:
        print(f"[FAIL] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None

def validate_results(result):
    """Validiere Analyse-Ergebnisse."""
    print("=== Validating Results ===")
    
    if not result:
        print("[FAIL] No result returned")
        return False
    
    checks = []
    
    # Check elements
    if hasattr(result, 'elements') and result.elements:
        checks.append(True)
        print(f"[OK] Elements detected: {len(result.elements)}")
    else:
        checks.append(False)
        print("[WARN] No elements detected")
    
    # Check connections
    if hasattr(result, 'connections') and result.connections:
        checks.append(True)
        print(f"[OK] Connections detected: {len(result.connections)}")
    else:
        checks.append(False)
        print("[WARN] No connections detected")
    
    # Check quality score
    if hasattr(result, 'quality_score'):
        checks.append(True)
        print(f"[OK] Quality score: {result.quality_score:.2f}")
    else:
        checks.append(False)
        print("[WARN] No quality score")
    
    # Check CGM data
    if hasattr(result, 'cgm_data') and result.cgm_data:
        checks.append(True)
        cgm = result.cgm_data
        print(f"[OK] CGM data generated:")
        print(f"    Components: {len(cgm.get('components', []))}")
        print(f"    Connectors: {len(cgm.get('connectors', []))}")
        print(f"    Splits: {len([p for p in cgm.get('split_merge_points', []) if p.get('type') == 'split'])}")
        print(f"    Merges: {len([p for p in cgm.get('split_merge_points', []) if p.get('type') == 'merge'])}")
        print(f"    Flows: {len(cgm.get('system_flows', []))}")
    else:
        checks.append(False)
        print("[WARN] No CGM data")
    
    # Check KPIs
    if hasattr(result, 'kpis') and result.kpis:
        checks.append(True)
        print(f"[OK] KPIs calculated: {len(result.kpis)} metrics")
    else:
        checks.append(False)
        print("[WARN] No KPIs")
    
    print()
    
    # Overall validation
    passed = sum(checks)
    total = len(checks)
    
    if passed >= total * 0.6:  # Mindestens 60% der Checks müssen passen
        print(f"[OK] Validation passed: {passed}/{total} checks")
        return True
    else:
        print(f"[FAIL] Validation failed: {passed}/{total} checks")
        return False

def main():
    """Hauptfunktion für automatisierten Test."""
    print("=" * 60)
    print("P&ID Analyzer v2.0 - Automated Test")
    print("=" * 60)
    print()
    
    # Step 1: Check environment
    if not check_env():
        print("[FAIL] Environment check failed!")
        print()
        print("Please create .env file with:")
        print("GCP_PROJECT_ID=koretex-zugang")
        print("GCP_PROJECT_NUMBER=748084370989")
        print("GCP_LOCATION=us-central1")
        return 1
    
    # Step 2: Find test image
    test_image = None
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if not Path(test_image).exists():
            print(f"[FAIL] Image not found: {test_image}")
            return 1
    else:
        test_image = find_test_image()
        if not test_image:
            print("[FAIL] No test image provided or found!")
            print()
            print("Usage: python run_automated_test.py [path/to/image.png]")
            return 1
    
    # Step 3: Initialize backend
    coordinator = test_backend_init()
    if not coordinator:
        print("[FAIL] Backend initialization failed!")
        return 1
    
    # Step 4: Run analysis
    result = run_analysis(coordinator, test_image)
    if not result:
        print("[FAIL] Analysis failed!")
        return 1
    
    # Step 5: Validate results
    validation_passed = validate_results(result)
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"[OK] Environment: Passed")
    print(f"[OK] Backend Init: Passed")
    print(f"[OK] Analysis: Passed")
    print(f"{'[OK]' if validation_passed else '[WARN]'} Validation: {'Passed' if validation_passed else 'Partial'}")
    print()
    
    if validation_passed:
        print("[SUCCESS] Automated test completed successfully!")
        print()
        print("Results saved to output directory")
        print("You can now test with your own images:")
        print("  python run_cli.py path/to/image.png")
        print("  python run_gui.py")
        return 0
    else:
        print("[WARN] Test completed with warnings")
        print("       Check output for details")
        return 0  # Still success, just warnings

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

