"""
Test-Script zum Prüfen aller Imports
"""

import sys
from pathlib import Path

# Projekt-Root zum Python-Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Teste alle Module-Imports"""
    errors = []
    
    try:
        from src.analyzer.models.elements import Element, Connection, Port, BBox
        print("[OK] Models imported")
    except Exception as e:
        errors.append(f"Models: {e}")
        print(f"[FAIL] Models failed: {e}")
    
    try:
        from src.analyzer.models.pipeline import PipelineState, AnalysisResult
        print("[OK] Pipeline models imported")
    except Exception as e:
        errors.append(f"Pipeline models: {e}")
        print(f"[FAIL] Pipeline models failed: {e}")
    
    try:
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        print("[OK] Knowledge Manager imported")
    except Exception as e:
        errors.append(f"Knowledge Manager: {e}")
        print(f"[FAIL] Knowledge Manager failed: {e}")
    
    try:
        from src.analyzer.ai.llm_client import LLMClient
        print("[OK] LLM Client imported")
    except Exception as e:
        errors.append(f"LLM Client: {e}")
        print(f"[FAIL] LLM Client failed: {e}")
    
    try:
        from src.services.config_service import ConfigService
        print("[OK] Config Service imported")
    except Exception as e:
        errors.append(f"Config Service: {e}")
        print(f"[FAIL] Config Service failed: {e}")
    
    try:
        from src.services.cache_service import CacheService
        print("[OK] Cache Service imported")
    except Exception as e:
        errors.append(f"Cache Service: {e}")
        print(f"[FAIL] Cache Service failed: {e}")
    
    try:
        from src.services.logging_service import LoggingService
        print("[OK] Logging Service imported")
    except Exception as e:
        errors.append(f"Logging Service: {e}")
        print(f"[FAIL] Logging Service failed: {e}")
    
    try:
        from src.interfaces.processor import IProcessor
        from src.interfaces.analyzer import IAnalyzer
        from src.interfaces.exporter import IExporter
        print("[OK] Interfaces imported")
    except Exception as e:
        errors.append(f"Interfaces: {e}")
        print(f"[FAIL] Interfaces failed: {e}")
    
    if errors:
        print(f"\n{len(errors)} errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n[SUCCESS] All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

