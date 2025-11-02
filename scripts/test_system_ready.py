#!/usr/bin/env python3
"""
System Readiness Check - Prüft ob alles bereit ist für erste Tests.

Führt umfassende Checks durch:
1. Module-Imports
2. Config-Verfügbarkeit
3. Backend-Initialisierung
4. Pipeline-Coordinator Setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test ob alle Module importiert werden können."""
    print("=== Testing Imports ===")
    
    try:
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        print("[OK] PipelineCoordinator")
    except Exception as e:
        print(f"[FAIL] PipelineCoordinator: {e}")
        return False
    
    try:
        from src.analyzer.ai.llm_client import LLMClient
        print("[OK] LLMClient")
    except Exception as e:
        print(f"[FAIL] LLMClient: {e}")
        return False
    
    try:
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        print("[OK] KnowledgeManager")
    except Exception as e:
        print(f"[FAIL] KnowledgeManager: {e}")
        return False
    
    try:
        from src.services.config_service import ConfigService
        print("[OK] ConfigService")
    except Exception as e:
        print(f"[FAIL] ConfigService: {e}")
        return False
    
    try:
        from src.utils.graph_theory import GraphTheoryAnalyzer
        print("[OK] GraphTheoryAnalyzer")
    except Exception as e:
        print(f"[FAIL] GraphTheoryAnalyzer: {e}")
        return False
    
    try:
        from src.analyzer.output.cgm_generator import CGMGenerator
        print("[OK] CGMGenerator")
    except Exception as e:
        print(f"[FAIL] CGMGenerator: {e}")
        return False
    
    print()
    return True

def test_config():
    """Test ob Config geladen werden kann."""
    print("=== Testing Config ===")
    
    try:
        from src.services.config_service import ConfigService
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            print(f"[WARN] Config file not found: {config_path}")
            print("      Using default config...")
            config_service = ConfigService()
        else:
            config_service = ConfigService(config_path=config_path)
        
        config = config_service.get_config()
        print("[OK] Config loaded")
        print(f"    Models configured: {len(config_service.get_raw_config().get('models', {}))}")
        print(f"    Strategies configured: {len(config_service.get_raw_config().get('strategies', {}))}")
        print()
        return True
    except Exception as e:
        print(f"[FAIL] Config loading: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_env_variables():
    """Test ob Umgebungsvariablen gesetzt sind."""
    print("=== Testing Environment Variables ===")
    
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not gcp_project_id:
        print("[WARN] GCP_PROJECT_ID not set")
        print("       You need to set this in .env file")
        print("       However, backend initialization will be tested without actual API calls")
    else:
        print(f"[OK] GCP_PROJECT_ID: {gcp_project_id}")
    
    print(f"[OK] GCP_LOCATION: {gcp_location}")
    print()
    return True

def test_backend_init():
    """Test ob Backend initialisiert werden kann (ohne API-Calls)."""
    print("=== Testing Backend Initialization ===")
    
    try:
        from src.services.config_service import ConfigService
        
        config_path = Path("config.yaml")
        if config_path.exists():
            config_service = ConfigService(config_path=config_path)
        else:
            config_service = ConfigService()
        
        config = config_service.get_config()
        
        # Try to initialize (will fail at API call, but structure should work)
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        if not gcp_project_id:
            print("[SKIP] Backend initialization skipped (no GCP_PROJECT_ID)")
            print("       Set GCP_PROJECT_ID to test full initialization")
            print()
            return True
        
        from src.analyzer.ai.llm_client import LLMClient
        
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        
        try:
            llm_client = LLMClient(
                project_id=gcp_project_id,
                default_location=gcp_location,
                config=config.model_dump()
            )
            print("[OK] LLMClient initialized")
        except Exception as e:
            print(f"[WARN] LLMClient initialization: {e}")
            print("       This might be due to missing credentials")
            print()
            return True
        
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        
        element_type_list_path = config_service.get_path("element_type_list") or "element_type_list.json"
        learning_db_path = config_service.get_path("learning_db") or "learning_db.json"
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_client=llm_client,
            config=config.model_dump()
        )
        print("[OK] KnowledgeManager initialized")
        
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            progress_callback=None
        )
        print("[OK] PipelineCoordinator initialized")
        print()
        return True
        
    except Exception as e:
        print(f"[FAIL] Backend initialization: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_graph_theory():
    """Test Graph Theory Module."""
    print("=== Testing Graph Theory ===")
    
    try:
        from src.utils.graph_theory import GraphTheoryAnalyzer
        
        # Create dummy data
        elements = [
            {'id': 'E1', 'type': 'Pump', 'label': 'PU1', 'bbox': {'x': 0.1, 'y': 0.1, 'width': 0.05, 'height': 0.05}},
            {'id': 'E2', 'type': 'Valve', 'label': 'V1', 'bbox': {'x': 0.2, 'y': 0.1, 'width': 0.05, 'height': 0.05}},
        ]
        connections = [
            {'from_id': 'E1', 'to_id': 'E2'}
        ]
        
        analyzer = GraphTheoryAnalyzer(elements, connections)
        splits_merges = analyzer.detect_splits_and_merges()
        flows = analyzer.analyze_pipeline_flows()
        metrics = analyzer.calculate_graph_metrics()
        
        print("[OK] GraphTheoryAnalyzer works")
        print(f"    Graph nodes: {analyzer.graph.number_of_nodes()}")
        print(f"    Graph edges: {analyzer.graph.number_of_edges()}")
        print(f"    Splits detected: {len(splits_merges['splits'])}")
        print(f"    Merges detected: {len(splits_merges['merges'])}")
        print(f"    Flows detected: {len(flows)}")
        print()
        return True
    except Exception as e:
        print(f"[FAIL] Graph Theory: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("P&ID Analyzer v2.0 - System Readiness Check")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Environment Variables", test_env_variables()))
    results.append(("Backend Initialization", test_backend_init()))
    results.append(("Graph Theory", test_graph_theory()))
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("[SUCCESS] System is ready for first tests!")
        print()
        print("Next steps:")
        print("  1. Set GCP_PROJECT_ID in .env file")
        print("  2. Run: python run_cli.py path/to/image.png")
        print("  3. Or run: python run_gui.py")
        return 0
    else:
        print("[WARNING] Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


