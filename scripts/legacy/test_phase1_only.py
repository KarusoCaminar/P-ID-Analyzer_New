"""
Test script for Phase 1 only (Legend and Metadata extraction).

This script tests the step-by-step Phase 1 logic:
1. CV: Has diagram legend? (yes/no)
2. If yes: Extract legend with Pro-Model
3. CV: Has diagram text stamp? (yes/no)
4. If yes: Extract metadata with Pro-Model
5. Limit bounding boxes for legend and text stamp
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.active_learner import ActiveLearner
import logging

# Setup logging
LoggingService.setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Test Phase 1 only."""
    logger.info("=== Phase 1 Test Run ===")
    
    # Find test image
    test_image_path = project_root / "training_data" / "complex_pids" / "page_1_original.png"
    if not test_image_path.exists():
        # Try alternative path
        test_image_path = project_root / "training_data" / "organized_tests" / "complex_pids" / "page_1_original.png"
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return
    
    logger.info(f"Using test image: {test_image_path}")
    
    # Initialize services
    import os
    from dotenv import load_dotenv
    
    # Load .env file if it exists
    load_dotenv()
    
    config_service = ConfigService()
    config = config_service.get_config()
    
    # Get config as dict (safe method)
    if hasattr(config, 'model_dump'):
        config_dict = config.model_dump()
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = config_service.get_raw_config()
    
    # Get GCP credentials
    gcp_project_id = config_dict.get('gcp_project_id') or os.getenv('GCP_PROJECT_ID')
    gcp_location = config_dict.get('gcp_location') or os.getenv('GCP_LOCATION', 'us-central1')
    
    if not gcp_project_id:
        logger.error("GCP_PROJECT_ID not found. Set in .env file or config.yaml")
        return
    
    # Initialize LLM client
    llm_client = LLMClient(
        project_id=gcp_project_id,
        default_location=gcp_location,
        config=config_dict
    )
    
    # Initialize knowledge manager
    from src.analyzer.learning.knowledge_manager import KnowledgeManager
    element_type_list_path = config_service.get_path("element_type_list") or project_root / "element_type_list.json"
    learning_db_path = config_service.get_path("learning_db") or project_root / "learning_db.json"
    
    knowledge_manager = KnowledgeManager(
        element_type_list_path=str(element_type_list_path),
        learning_db_path=str(learning_db_path),
        llm_handler=llm_client,
        config=config_dict
    )
    
    # Initialize symbol library
    from src.analyzer.learning.symbol_library import SymbolLibrary
    symbol_library = SymbolLibrary(
        llm_client=llm_client,
        learning_db_path=learning_db_path
    )
    
    # Initialize active learner
    active_learner = ActiveLearner(
        knowledge_manager=knowledge_manager,
        symbol_library=symbol_library,
        llm_client=llm_client,
        config=config_dict
    )
    
    # Initialize pipeline coordinator
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service
    )
    
    # Create output directory
    output_dir = project_root / "outputs" / "test_phase1_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Phase 1 test run...")
    
    # Run Phase 1 only
    try:
        # Initialize run
        coordinator._initialize_run(str(test_image_path), {})
        
        # Run Phase 1
        coordinator._run_phase_1_pre_analysis(str(test_image_path))
        
        # Get results
        legend_data = coordinator._analysis_results.get('legend_data', {})
        metadata = coordinator._analysis_results.get('metadata', {})
        excluded_zones = coordinator._excluded_zones
        
        logger.info("=== Phase 1 Results ===")
        logger.info(f"Legend symbols: {len(legend_data.get('symbol_map', {}))}")
        logger.info(f"Legend line rules: {len(legend_data.get('line_map', {}))}")
        logger.info(f"Metadata: {metadata}")
        logger.info(f"Excluded zones: {len(excluded_zones)}")
        
        # Save results
        import json
        results = {
            'legend_data': legend_data,
            'metadata': metadata,
            'excluded_zones': excluded_zones
        }
        
        results_path = output_dir / "phase1_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_path}")
        logger.info("=== Phase 1 Test Complete ===")
        
    except Exception as e:
        logger.error(f"Error during Phase 1 test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

