"""
Test that all modules can be imported successfully.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    errors = []
    
    try:
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        print("[OK] PipelineCoordinator imported")
    except Exception as e:
        errors.append(f"PipelineCoordinator: {e}")
        print(f"[FAIL] PipelineCoordinator: {e}")
    
    try:
        from src.analyzer.analysis.swarm_analyzer import SwarmAnalyzer
        from src.analyzer.analysis.monolith_analyzer import MonolithAnalyzer
        from src.analyzer.analysis.fusion_engine import FusionEngine
        print("[OK] Analysis modules imported")
    except Exception as e:
        errors.append(f"Analysis modules: {e}")
        print(f"[FAIL] Analysis modules: {e}")
    
    try:
        from src.analyzer.learning import KnowledgeManager, SymbolLibrary, CorrectionLearner, PatternMatcher
        print("[OK] Learning modules imported")
    except Exception as e:
        errors.append(f"Learning modules: {e}")
        print(f"[FAIL] Learning modules: {e}")
    
    try:
        from src.services.config_service import ConfigService
        print("[OK] Services imported")
    except Exception as e:
        errors.append(f"Services: {e}")
        print(f"[FAIL] Services: {e}")
    
    try:
        from src.utils import image_utils, graph_utils, type_utils
        print("[OK] Utils imported")
    except Exception as e:
        errors.append(f"Utils: {e}")
        print(f"[FAIL] Utils: {e}")
    
    if errors:
        print(f"\n[FAIL] {len(errors)} import errors found")
        return False
    else:
        print("\n[OK] All modules imported successfully")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)


