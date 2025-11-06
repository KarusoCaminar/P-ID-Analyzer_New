"""
Test Harness Script - Run tests with structured output folders.

This script creates a structured test environment:
- experiment_YYYY-MM-DD/
  - test_01_phase1_legend_only/
  - test_02_baseline_monolith_all/
  - test_03_baseline_swarm_only/
  - test_04_baseline_specialist_chain/
  - test_05a_chain_with_predictive_2d/
  - test_05b_chain_with_polyline_2e/
  - test_05c_chain_with_selfcorrect_3/

Each test folder contains:
- config_snapshot.yaml
- prompts_snapshot.json
- test_metadata.md
- output_phase_*.json (intermediate results)
- logs/
- debug/
- results.json, kpis.json, etc.
"""

import sys
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
import logging

# Setup logging
LoggingService.setup_logging()
logger = logging.getLogger(__name__)


def create_test_output_dir(experiment_name: str, test_name: str) -> Path:
    """
    Create structured test output directory.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'experiment_2025-11-06')
        test_name: Name of the test (e.g., 'test_04_baseline_specialist_chain')
        
    Returns:
        Path to test output directory
    """
    experiment_dir = project_root / "outputs" / experiment_name
    test_dir = experiment_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def run_test(
    test_name: str,
    test_description: str,
    image_path: str,
    experiment_name: str,
    params_override: dict
) -> dict:
    """
    Run a single test with test harness.
    
    Args:
        test_name: Name of the test
        test_description: Description of the test
        image_path: Path to test image
        experiment_name: Name of the experiment
        params_override: Parameter overrides for this test
        
    Returns:
        Test result dictionary
    """
    logger.info(f"=== Running Test: {test_name} ===")
    logger.info(f"Description: {test_description}")
    
    # Create test output directory
    test_output_dir = create_test_output_dir(experiment_name, test_name)
    logger.info(f"Test output directory: {test_output_dir}")
    
    # Load .env file if it exists
    load_dotenv()
    
    # Initialize services
    config_service = ConfigService()
    config = config_service.get_config()
    
    # Get config as dict
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
        return {"error": "GCP_PROJECT_ID not found"}
    
    # Initialize LLM client
    llm_client = LLMClient(
        project_id=gcp_project_id,
        default_location=gcp_location,
        config=config_dict
    )
    
    # Initialize knowledge manager
    element_type_list_path = config_service.get_path("element_type_list") or project_root / "element_type_list.json"
    learning_db_path = config_service.get_path("learning_db") or project_root / "learning_db.json"
    
    knowledge_manager = KnowledgeManager(
        element_type_list_path=str(element_type_list_path),
        learning_db_path=str(learning_db_path),
        llm_handler=llm_client,
        config=config_dict
    )
    
    # Initialize symbol library
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
    
    # Add test_name and test_description to params_override
    params_override['test_name'] = test_name
    params_override['test_description'] = test_description
    
    # Run analysis
    try:
        result = coordinator.process(
            image_path=image_path,
            output_dir=str(test_output_dir),
            params_override=params_override
        )
        
        logger.info(f"Test {test_name} completed successfully")
        return {
            "test_name": test_name,
            "success": True,
            "output_dir": str(test_output_dir),
            "result": result
        }
    except Exception as e:
        logger.error(f"Test {test_name} failed: {e}", exc_info=True)
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "output_dir": str(test_output_dir)
        }


def main():
    """Main function to run test harness."""
    logger.info("=== Test Harness ===")
    
    # Find test image
    test_image_path = project_root / "training_data" / "complex_pids" / "page_1_original.png"
    if not test_image_path.exists():
        test_image_path = project_root / "training_data" / "organized_tests" / "complex_pids" / "page_1_original.png"
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return
    
    logger.info(f"Using test image: {test_image_path}")
    
    # Create experiment name
    experiment_name = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"
    logger.info(f"Experiment: {experiment_name}")
    
    # Define tests
    tests = [
        {
            "name": "test_01_phase1_legend_only",
            "description": "Test Phase 1 only: Legend and metadata extraction",
            "params": {
                "use_swarm_analysis": False,
                "use_monolith_analysis": False,
                "use_self_correction_loop": False,
                "use_predictive_completion": False,
                "use_polyline_refinement": False
            }
        },
        {
            "name": "test_04_baseline_specialist_chain",
            "description": "Baseline: Specialist chain (Swarm → Guard Rails → Monolith → Fusion)",
            "params": {
                "use_swarm_analysis": True,
                "use_monolith_analysis": True,
                "use_self_correction_loop": False,
                "use_predictive_completion": False,
                "use_polyline_refinement": False
            }
        },
        {
            "name": "test_05a_chain_with_predictive_2d",
            "description": "Specialist chain + Predictive completion (Phase 2d)",
            "params": {
                "use_swarm_analysis": True,
                "use_monolith_analysis": True,
                "use_self_correction_loop": False,
                "use_predictive_completion": True,
                "use_polyline_refinement": False
            }
        },
        {
            "name": "test_05b_chain_with_polyline_2e",
            "description": "Specialist chain + Polyline refinement (Phase 2e)",
            "params": {
                "use_swarm_analysis": True,
                "use_monolith_analysis": True,
                "use_self_correction_loop": False,
                "use_predictive_completion": False,
                "use_polyline_refinement": True
            }
        },
        {
            "name": "test_05c_chain_with_selfcorrect_3",
            "description": "Specialist chain + Self-correction loop (Phase 3)",
            "params": {
                "use_swarm_analysis": True,
                "use_monolith_analysis": True,
                "use_self_correction_loop": True,
                "use_predictive_completion": False,
                "use_polyline_refinement": False
            }
        }
    ]
    
    # Run tests
    results = []
    for test in tests:
        result = run_test(
            test_name=test["name"],
            test_description=test["description"],
            image_path=str(test_image_path),
            experiment_name=experiment_name,
            params_override=test["params"]
        )
        results.append(result)
    
    # Summary
    logger.info("=== Test Harness Summary ===")
    for result in results:
        status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
        logger.info(f"{status}: {result.get('test_name')}")
        if result.get("output_dir"):
            logger.info(f"  Output: {result.get('output_dir')}")
        if result.get("error"):
            logger.error(f"  Error: {result.get('error')}")


if __name__ == "__main__":
    main()

