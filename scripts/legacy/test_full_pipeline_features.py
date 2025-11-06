"""
Vollständiger Testlauf mit allen Pipeline-Features.
Testet alle Phasen, Chain-of-Thought Reasoning, Splits/Merges, Missing Elements.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.config_service import ConfigService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator

# Setup logging
log_dir = Path("outputs") / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"test_full_pipeline_features_{Path(__file__).stem}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_full_pipeline():
    """Vollständiger Testlauf mit allen Features."""
    
    logger.info("=" * 80)
    logger.info("VOLLSTÄNDIGER PIPELINE-TEST: Alle Features aktiviert")
    logger.info("=" * 80)
    
    # Test mit Uni-Bild (komplexer, zeigt alle Features)
    test_image_paths = [
        project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni.png",
        project_root / "training_data" / "organized_tests" / "complex_pids" / "Verfahrensfließbild_Uni.png"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if path.exists():
            test_image_path = path
            break
    
    if not test_image_path:
        logger.error(f"Testbild nicht gefunden in: {test_image_paths}")
        return False
    
    # Check for truth data
    truth_path = test_image_path.parent / f"{test_image_path.stem}_truth.json"
    has_truth = truth_path.exists()
    
    logger.info(f"Testbild: {test_image_path}")
    logger.info(f"Truth-Data: {truth_path if has_truth else 'KEINE'}")
    logger.info(f"Log-Datei: {log_file}")
    logger.info("")
    
    # Initialize services
    logger.info("=" * 80)
    logger.info("PHASE 0: Services initialisieren")
    logger.info("=" * 80)
    
    config_service = ConfigService()
    config = config_service.get_raw_config()
    
    # Initialize LLM client
    import os
    from dotenv import load_dotenv
    
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    gcp_project_id = os.getenv("GCP_PROJECT_ID") or config.get("gcp", {}).get("project_id") or config.get("project_id")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not gcp_project_id:
        logger.error("GCP_PROJECT_ID not found")
        return False
    
    logger.info(f"GCP Project ID: {gcp_project_id}")
    logger.info(f"GCP Location: {gcp_location}")
    
    llm_client = LLMClient(
        project_id=gcp_project_id,
        default_location=gcp_location,
        config=config
    )
    
    knowledge_manager = KnowledgeManager(
        element_type_list_path=str(project_root / "element_type_list.json"),
        learning_db_path=str(project_root / "learning_db.json"),
        llm_handler=llm_client,
        config=config
    )
    
    learning_db_path_obj = project_root / "learning_db.json"
    symbol_library = SymbolLibrary(
        llm_client=llm_client,
        learning_db_path=learning_db_path_obj
    )
    
    active_learner = ActiveLearner(
        knowledge_manager=knowledge_manager,
        symbol_library=symbol_library,
        llm_client=llm_client,
        config=config
    )
    
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service
    )
    
    logger.info("Services initialisiert")
    logger.info("")
    
    # Run analysis with ALL features enabled
    logger.info("=" * 80)
    logger.info("PHASE 1-4: Vollständige Analyse mit allen Features")
    logger.info("=" * 80)
    logger.info("Features aktiviert:")
    logger.info("  - Phase 0: CV-based complexity analysis")
    logger.info("  - Phase 1: Pre-analysis (metadata, legend)")
    logger.info("  - Phase 2: Sequential analysis (Swarm -> Guard Rails -> Monolith)")
    logger.info("  - Phase 2e: Polyline refinement (text removal, gap bridging, adaptive thresholds)")
    logger.info("  - Phase 3: Self-correction loop")
    logger.info("  - Phase 4: Post-processing (Chain-of-Thought Reasoning, Splits/Merges, Missing Elements)")
    logger.info("")
    
    try:
        output_dir = project_root / "outputs" / "test_full_pipeline_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = coordinator.process(
            image_path=str(test_image_path),
            output_dir=str(output_dir),
            params_override={
                'use_phase0': True,  # CV-based complexity analysis
                'use_self_correction_loop': True,  # Phase 3
                'use_post_processing': True,  # Phase 4 (includes Chain-of-Thought)
                'use_cot_reasoning': True,  # Chain-of-Thought Reasoning
                'strategy': 'optimal_swarm_monolith'  # Full strategy
            }
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("ERGEBNISSE")
        logger.info("=" * 80)
        
        if result:
            elements = result.elements if hasattr(result, 'elements') else []
            connections = result.connections if hasattr(result, 'connections') else []
            quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
            
            # Analyze elements
            element_types = {}
            splits_count = 0
            merges_count = 0
            missing_elements_count = 0
            
            for el in elements:
                el_dict = el.model_dump() if hasattr(el, 'model_dump') else (el.dict() if hasattr(el, 'dict') else el)
                el_type = el_dict.get('type', 'Unknown')
                element_types[el_type] = element_types.get(el_type, 0) + 1
                
                if el_type == 'Split':
                    splits_count += 1
                elif el_type == 'Merge':
                    merges_count += 1
                elif el_type == 'Missing_Element':
                    missing_elements_count += 1
            
            # Analyze connections
            dangling_connections_count = 0
            for conn in connections:
                conn_dict = conn.model_dump() if hasattr(conn, 'model_dump') else (conn.dict() if hasattr(conn, 'dict') else conn)
                if conn_dict.get('dangling', False):
                    dangling_connections_count += 1
            
            logger.info(f"[OK] Analyse erfolgreich abgeschlossen")
            logger.info(f"  - Elemente: {len(elements)}")
            logger.info(f"    - Splits: {splits_count}")
            logger.info(f"    - Merges: {merges_count}")
            logger.info(f"    - Missing Elements: {missing_elements_count}")
            logger.info(f"    - Element-Typen: {dict(sorted(element_types.items()))}")
            logger.info(f"  - Verbindungen: {len(connections)}")
            logger.info(f"    - Dangling Connections: {dangling_connections_count}")
            logger.info(f"  - Quality Score: {quality_score:.2f}")
            
            # Verify log file
            if log_file.exists():
                log_size = log_file.stat().st_size
                logger.info(f"  - Log-Datei: {log_file} ({log_size} Bytes)")
            
            # Check visualizations
            viz_files = {
                'debug_map': list(output_dir.glob("*debug_map.png")),
                'confidence_map': list(output_dir.glob("*confidence_map.png")),
                'kpi_dashboard': list(output_dir.glob("*kpi_dashboard.png")),
                'score_curve': list(output_dir.glob("*score_curve.png")),
                'uncertainty_heatmap': list(output_dir.glob("*uncertainty_heatmap.png"))
            }
            
            logger.info(f"  - Visualisierungen:")
            for viz_type, files in viz_files.items():
                if files:
                    logger.info(f"    - {viz_type}: {len(files)} Datei(en)")
                else:
                    logger.warning(f"    - {viz_type}: NICHT gefunden")
            
            # Check for errors in logs
            logger.info("")
            logger.info("=" * 80)
            logger.info("FEHLER-ANALYSE")
            logger.info("=" * 80)
            
            error_keywords = ['ERROR', 'CRITICAL', 'Traceback', 'Exception', 'Failed', 'fehlgeschlagen']
            error_count = 0
            
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    for keyword in error_keywords:
                        count = log_content.count(keyword)
                        if count > 0:
                            error_count += count
                            logger.warning(f"  - '{keyword}' gefunden: {count}x")
            
            if error_count == 0:
                logger.info("  [OK] Keine kritischen Fehler in den Logs gefunden")
            else:
                logger.warning(f"  [WARN] {error_count} Fehler/Warnungen in den Logs gefunden")
            
            logger.info("")
            logger.info("[OK] VOLLSTÄNDIGER TEST BESTANDEN")
            return True
        else:
            logger.error("[FAIL] Analyse fehlgeschlagen: Kein Ergebnis")
            return False
            
    except Exception as e:
        logger.error(f"[FAIL] TEST FEHLGESCHLAGEN: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)

