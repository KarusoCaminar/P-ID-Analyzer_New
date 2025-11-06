"""
Test script for Core Phases only (Phase 0, 1, 2a Swarm, 2b Guard Rails, 2c Monolith, 2c Fusion).

Tests:
- Phase 0: Complexity Analysis (CV-based)
- Phase 1: Pre-analysis (Legend + Metadata extraction with Legend Critic)
- Phase 2a: Swarm Analysis (Element detection)
- Phase 2b: Guard Rails (Element cleaning)
- Phase 2c: Monolith Analysis (Connection detection)
- Phase 2c: Fusion (Confidence-based with legend authority)

Excludes:
- Phase 2d: Predictive Completion
- Phase 2e: Polyline Refinement
- Phase 3: Self-Correction Loop

Tests on Uni-Page 1 (page_1_original.png) with legend.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.services.config_service import ConfigService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.utils.test_harness import save_config_snapshot, save_test_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_core_phases_uni_page1():
    """Test Core Phases only on Uni-Page 1."""
    
    logger.info("=" * 80)
    logger.info("TEST: Core Phases Only (Phase 0, 1, 2a, 2b, 2c) - Uni-Page 1")
    logger.info("=" * 80)
    
    # Find Uni-Page 1 image
    uni_page1_path = project_root / "training_data" / "complex_pids" / "page_1_original.png"
    
    if not uni_page1_path.exists():
        logger.error(f"Uni-Page 1 image not found: {uni_page1_path}")
        return False
    
    logger.info(f"Test Image: {uni_page1_path}")
    logger.info(f"Image exists: {uni_page1_path.exists()}")
    
    # Load environment variables
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Initialize services
    logger.info("Initializing services...")
    
    config_service = ConfigService()
    config = config_service.get_raw_config()
    
    # Get GCP credentials
    gcp_project_id = os.getenv("GCP_PROJECT_ID") or config.get("gcp", {}).get("project_id") or config.get("project_id")
    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not gcp_project_id:
        logger.error("GCP_PROJECT_ID not found in environment or config")
        return False
    
    logger.info(f"GCP Project ID: {gcp_project_id}")
    logger.info(f"GCP Location: {gcp_location}")
    
    # Initialize LLM client
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
    symbol_library = SymbolLibrary(
        llm_client=llm_client,
        learning_db_path=str(project_root / "learning_db.json")
    )
    
    active_learner = ActiveLearner(
        knowledge_manager=knowledge_manager,
        symbol_library=symbol_library,
        llm_client=llm_client,
        config=config
    )
    
    # Create pipeline coordinator
    logger.info("Creating Pipeline Coordinator...")
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service
    )
    
    # CRITICAL: Ensure phases 2d, 2e, 3 are disabled
    logger.info("Configuring test: Disabling phases 2d, 2e, 3...")
    params_override = {
        'use_predictive_completion': False,  # Phase 2d
        'use_polyline_refinement': False,    # Phase 2e
        'use_self_correction_loop': False,   # Phase 3
        'use_active_learning': False         # Active Learning
    }
    
    # Get model strategy and logic parameters for test metadata
    model_strategy = config_service.get_model_strategy()
    logic_parameters = config_service.get_logic_parameters()
    
    # Create test output directory with timestamp
    test_name = "core_phases_uni_page1"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = project_root / "outputs" / f"{test_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save test harness artifacts
    logger.info("Saving test harness artifacts...")
    save_config_snapshot(config_service, str(output_dir))
    save_test_metadata(
        output_dir=str(output_dir),
        test_name=test_name,
        test_description="Test Core Phases Only (Phase 0, 1, 2a Swarm, 2b Guard Rails, 2c Monolith, 2c Fusion) on Uni-Page 1 with legend. Excludes phases 2d, 2e, 3.",
        model_strategy=model_strategy,
        logic_parameters={**logic_parameters, **params_override}
    )
    
    # Run analysis
    logger.info("Starting Core Phases Analysis...")
    logger.info("-" * 80)
    logger.info("Active Phases:")
    logger.info("  ✓ Phase 0: Complexity Analysis (CV-based)")
    logger.info("  ✓ Phase 1: Pre-analysis (Legend + Metadata + Legend Critic)")
    logger.info("  ✓ Phase 2a: Swarm Analysis (Element detection)")
    logger.info("  ✓ Phase 2b: Guard Rails (Element cleaning)")
    logger.info("  ✓ Phase 2c: Monolith Analysis (Connection detection)")
    logger.info("  ✓ Phase 2c: Fusion (Confidence-based with legend authority)")
    logger.info("  ✗ Phase 2d: Predictive Completion (DISABLED)")
    logger.info("  ✗ Phase 2e: Polyline Refinement (DISABLED)")
    logger.info("  ✗ Phase 3: Self-Correction Loop (DISABLED)")
    logger.info("  ✓ Phase 4: Post-Processing (KPIs, CGM, artifacts)")
    logger.info("-" * 80)
    
    try:
        result = coordinator.process(
            image_path=str(uni_page1_path),
            output_dir=str(output_dir),
            params_override=params_override
        )
        
        logger.info("-" * 80)
        logger.info("=" * 80)
        logger.info("TEST: RESULTS")
        logger.info("=" * 80)
        
        if result:
            elements = result.elements if hasattr(result, 'elements') else []
            connections = result.connections if hasattr(result, 'connections') else []
            quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
            
            logger.info(f"[OK] Analysis completed successfully")
            logger.info(f"  - Elements: {len(elements)}")
            logger.info(f"  - Connections: {len(connections)}")
            logger.info(f"  - Quality Score: {quality_score:.2f}")
            
            # Check for confidence scores
            elements_with_confidence = [el for el in elements if hasattr(el, 'confidence') or (isinstance(el, dict) and 'confidence' in el)]
            connections_with_confidence = [conn for conn in connections if hasattr(conn, 'confidence') or (isinstance(conn, dict) and 'confidence' in conn)]
            
            logger.info(f"  - Elements with confidence: {len(elements_with_confidence)}/{len(elements)}")
            logger.info(f"  - Connections with confidence: {len(connections_with_confidence)}/{len(connections)}")
            
            # Check test harness structure
            logger.info("\n" + "=" * 80)
            logger.info("TEST HARNESS STRUCTURE CHECK")
            logger.info("=" * 80)
            
            test_harness_files = [
                "config_snapshot.yaml",
                "prompts_snapshot.json",
                "test_metadata.md"
            ]
            
            intermediate_results = [
                "output_phase_2a_swarm.json",
                "output_phase_2b_guardrails.json",
                "output_phase_2c_monolith.json",
                "output_phase_2c_fusion.json"
            ]
            
            all_files_exist = True
            for file_name in test_harness_files:
                file_path = output_dir / file_name
                exists = file_path.exists()
                logger.info(f"  {'✓' if exists else '✗'} {file_name}: {exists}")
                if not exists:
                    all_files_exist = False
            
            logger.info("\nIntermediate Results:")
            for file_name in intermediate_results:
                file_path = output_dir / file_name
                exists = file_path.exists()
                logger.info(f"  {'✓' if exists else '✗'} {file_name}: {exists}")
                if not exists:
                    all_files_exist = False
            
            # Check for legend data
            legend_data = coordinator._analysis_results.get('legend_data', {})
            if legend_data:
                symbol_map = legend_data.get('symbol_map', {})
                line_map = legend_data.get('line_map', {})
                legend_confidence = legend_data.get('legend_confidence', 0.0)
                is_plausible = legend_data.get('is_plausible', False)
                
                logger.info("\nLegend Data:")
                logger.info(f"  - Symbols in legend: {len(symbol_map)}")
                logger.info(f"  - Lines in legend: {len(line_map)}")
                logger.info(f"  - Legend confidence: {legend_confidence:.2f}")
                logger.info(f"  - Is plausible: {is_plausible}")
            
            if all_files_exist:
                logger.info("\n[OK] Test harness structure is complete")
            else:
                logger.warning("\n[WARN] Some test harness files are missing")
            
            logger.info(f"\n[OK] Output directory: {output_dir}")
            logger.info(f"[OK] TEST PASSED")
            return True
        else:
            logger.error("[FAIL] Analysis failed: No result")
            return False
            
    except Exception as e:
        logger.error(f"[FAIL] TEST FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_core_phases_uni_page1()
    sys.exit(0 if success else 1)

