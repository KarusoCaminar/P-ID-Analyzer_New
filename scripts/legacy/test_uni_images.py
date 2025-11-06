"""
Test script for Uni images (Verfahrensfließbild_Uni.png).
Tests the pipeline with Uni images and ensures log files are created.
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
log_file = log_dir / f"test_uni_images_{Path(__file__).stem}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_uni_images():
    """Test Uni images with truth data."""
    
    logger.info("=" * 60)
    logger.info("TEST: Uni-Bilder (Verfahrensfließbild_Uni)")
    logger.info("=" * 60)
    
    # Find Uni image
    uni_image_paths = [
        project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni.png",
        project_root / "training_data" / "organized_tests" / "complex_pids" / "Verfahrensfließbild_Uni.png"
    ]
    
    uni_image_path = None
    for path in uni_image_paths:
        if path.exists():
            uni_image_path = path
            break
    
    if not uni_image_path:
        logger.error(f"Uni image not found in: {uni_image_paths}")
        return False
    
    # Check for truth data
    truth_path = uni_image_path.parent / f"{uni_image_path.stem}_truth.json"
    has_truth = truth_path.exists()
    
    logger.info(f"Testbild: {uni_image_path}")
    logger.info(f"Truth-Data: {truth_path if has_truth else 'KEINE'}")
    logger.info(f"Log-Datei: {log_file}")
    logger.info(f"Log-Datei existiert: {log_file.exists()}")
    
    # Initialize services
    logger.info("Initialisiere Services...")
    
    config_service = ConfigService()
    config = config_service.get_raw_config()
    
    # Initialize LLM client
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not gcp_project_id:
        # Try to get from config
        gcp_project_id = config.get("gcp", {}).get("project_id") or config.get("project_id")
        if not gcp_project_id:
            logger.error("GCP_PROJECT_ID not found in environment or config")
            return False
    
    llm_client = LLMClient(
        project_id=gcp_project_id,
        default_location=gcp_location,
        config=config
    )
    
    # Initialize knowledge manager
    knowledge_manager = KnowledgeManager(
        element_type_list_path=str(project_root / "element_type_list.json"),
        learning_db_path=str(project_root / "learning_db.json"),
        llm_handler=llm_client,
        config=config
    )
    
    # Initialize symbol library and active learner
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
    
    # Create pipeline coordinator
    logger.info("Erstelle Pipeline Coordinator...")
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service
    )
    
    # Run analysis
    logger.info("Starte Analyse mit Uni-Bild...")
    logger.info("-" * 60)
    
    try:
        # Create output directory for logs/visualizations
        output_dir = project_root / "outputs" / "test_uni_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = coordinator.process(
            image_path=str(uni_image_path),
            output_dir=str(output_dir)
            # Will use truth data if available automatically
        )
        
        logger.info("-" * 60)
        logger.info("=" * 60)
        logger.info("TEST: ERGEBNIS")
        logger.info("=" * 60)
        
        if result:
            elements = result.elements if hasattr(result, 'elements') else []
            connections = result.connections if hasattr(result, 'connections') else []
            quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
            
            logger.info(f"[OK] Analyse erfolgreich abgeschlossen")
            logger.info(f"  - Elemente: {len(elements)}")
            logger.info(f"  - Verbindungen: {len(connections)}")
            logger.info(f"  - Quality Score: {quality_score:.2f}")
            
            # Verify log file was created
            if log_file.exists():
                log_size = log_file.stat().st_size
                logger.info(f"[OK] Log-Datei erstellt: {log_file}")
                logger.info(f"  - Groesse: {log_size} Bytes")
                logger.info(f"  - Pfad: {log_file.absolute()}")
            else:
                logger.warning(f"[WARN] Log-Datei wurde NICHT erstellt: {log_file}")
            
            # Check visualizations
            viz_dir = output_dir
            viz_files = {
                'debug_map': list(viz_dir.glob("*debug_map.png")),
                'confidence_map': list(viz_dir.glob("*confidence_map.png")),
                'kpi_dashboard': list(viz_dir.glob("*kpi_dashboard.png")),
                'score_curve': list(viz_dir.glob("*score_curve.png"))
            }
            
            logger.info(f"[OK] Visualisierungen:")
            for viz_type, files in viz_files.items():
                if files:
                    logger.info(f"  - {viz_type}: {len(files)} Datei(en)")
                else:
                    logger.warning(f"  - {viz_type}: NICHT gefunden")
            
            logger.info("[OK] TEST BESTANDEN")
            return True
        else:
            logger.error("[FAIL] Analyse fehlgeschlagen: Kein Ergebnis")
            return False
            
    except Exception as e:
        logger.error(f"[FAIL] TEST FEHLGESCHLAGEN: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_uni_images()
    sys.exit(0 if success else 1)

