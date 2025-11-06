#!/usr/bin/env python3
"""
Smoke-Test für GUI - Testet die neue Pipeline-Logik (Phase 0-4).

Ziel: Bestätigen, dass die Pipeline-Logik fehlerfrei von Anfang bis Ende durchläuft.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def smoke_test():
    """Smoke-Test: Testet die Pipeline-Logik mit SimplePID."""
    logger.info("=" * 60)
    logger.info("SMOKE-TEST: Pipeline-Logik (Phase 0-4)")
    logger.info("=" * 60)
    
    # Testbild
    test_image_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    
    if not test_image_path.exists():
        logger.error(f"Testbild nicht gefunden: {test_image_path}")
        return False
    
    logger.info(f"Testbild: {test_image_path}")
    
    # Output directory
    output_dir = project_root / "outputs" / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize services
        logger.info("Initialisiere Services...")
        config_service = ConfigService()
        config = config_service.get_raw_config() or {}
        
        # Get GCP project ID and location
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        project_id = config.get('gcp_project_id') or os.getenv('GCP_PROJECT_ID')
        location = config.get('gcp_location') or os.getenv('GCP_LOCATION', 'us-central1')
        
        if not project_id:
            logger.error("GCP_PROJECT_ID not found - cannot initialize LLMClient")
            return False
        
        # Initialize LLM Client (korrekte Signatur: project_id, default_location, config)
        llm_client = LLMClient(
            project_id=project_id,
            default_location=location,
            config=config
        )
        
        # Initialize Knowledge Manager
        element_type_list_path = config.get('paths', {}).get('element_type_list', 'element_type_list.json')
        learning_db_path = config.get('paths', {}).get('learning_db', 'learning_db.json')
        
        # Convert to Path objects
        if isinstance(element_type_list_path, str):
            element_type_list_path = Path(element_type_list_path)
        if isinstance(learning_db_path, str):
            learning_db_path = Path(learning_db_path)
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Initialize Active Learner
        from src.analyzer.learning.symbol_library import SymbolLibrary
        learning_db_path_obj = Path(learning_db_path) if isinstance(learning_db_path, str) else learning_db_path
        symbol_library = SymbolLibrary(llm_client, learning_db_path=learning_db_path_obj)
        
        active_learner = ActiveLearner(
            knowledge_manager=knowledge_manager,
            symbol_library=symbol_library,
            llm_client=llm_client,
            config=config
        )
        
        # Create pipeline coordinator
        logger.info("Erstelle Pipeline Coordinator...")
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        
        # Run analysis with simple_pid_strategy
        logger.info("Starte Analyse mit simple_pid_strategy...")
        logger.info("-" * 60)
        
        result = coordinator.process(
            image_path=str(test_image_path),
            output_dir=str(output_dir),
            params_override={
                'strategy': 'simple_pid_strategy',
                'use_phase0': True,  # Enable Phase 0 for automatic strategy selection
                'use_self_correction_loop': True,  # Enable Phase 3
                'use_post_processing': True  # Enable Phase 4
            }
        )
        
        logger.info("-" * 60)
        logger.info("=" * 60)
        logger.info("SMOKE-TEST: ERGEBNIS")
        logger.info("=" * 60)
        
        # Check results
        if result:
            elements_count = len(result.elements) if hasattr(result, 'elements') else 0
            connections_count = len(result.connections) if hasattr(result, 'connections') else 0
            quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
            
            logger.info(f"✓ Analyse erfolgreich abgeschlossen")
            logger.info(f"  - Elemente: {elements_count}")
            logger.info(f"  - Verbindungen: {connections_count}")
            logger.info(f"  - Quality Score: {quality_score:.2f}")
            
            if elements_count > 0:
                logger.info("✓ SMOKE-TEST BESTANDEN: Pipeline-Logik funktioniert!")
                return True
            else:
                logger.warning("⚠ SMOKE-TEST: Keine Elemente gefunden (Pipeline läuft, aber keine Elemente)")
                return False
        else:
            logger.error("✗ SMOKE-TEST FEHLGESCHLAGEN: Analyse zurückgegeben None")
            return False
            
    except Exception as e:
        logger.error(f"✗ SMOKE-TEST FEHLGESCHLAGEN: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)

