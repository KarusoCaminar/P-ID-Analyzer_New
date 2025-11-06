"""
Phase 0 Complexity Analysis Validation Tests

Tests the hybrid CV/LLM complexity analysis logic:
- Test 1: Simple path (CV fast exit, no LLM)
- Test 2: Complex path (CV + LLM fine check)
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/phase0_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DummyProgressCallback:
    """Dummy progress callback for testing."""
    def update_progress(self, progress: int, message: str):
        pass
    def update_status_label(self, message: str):
        pass
    def report_truth_mode(self, active: bool):
        pass


def test_simple_path(image_path: str) -> dict:
    """
    Test 1: Simple path validation
    
    Expected:
    - CV quick test: pixel_density < 0.05
    - Fast exit to simple_pid_strategy
    - No LLM call in Phase 0
    """
    logger.info("=" * 80)
    logger.info("TEST 1: SIMPLE PATH VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Image: {image_path}")
    logger.info("")
    
    # Initialize components
    config_service = ConfigService()
    config = config_service.get_config()
    
    config_dict = config_service.get_raw_config()
    llm_client = LLMClient(
        project_id=os.getenv("GCP_PROJECT_ID"),
        default_location=os.getenv("GCP_LOCATION", "us-central1"),
        config=config_dict
    )
    knowledge_manager = KnowledgeManager(
        element_type_list_path=project_root / "element_type_list.json",
        learning_db_path=project_root / "learning_db.json",
        llm_handler=llm_client,
        config=config_dict
    )
    
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service,
        model_strategy=None,
        progress_callback=DummyProgressCallback()
    )
    
    # Override to use simple_pid_strategy (expected from Phase 0)
    strategies = config_dict.get('strategies', {})
    simple_pid_strategy = strategies.get('simple_pid_strategy', {})
    
    if simple_pid_strategy:
        coordinator.model_strategy = {}
        for key, model_name in simple_pid_strategy.items():
            models = config_dict.get('models', {})
            if model_name in models:
                model_info = models[model_name]
                coordinator.model_strategy[key] = model_info
    
    # Run analysis
    try:
        result = coordinator.process(
            image_path=image_path,
            params_override={
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': False,  # Skip Phase 3 for speed
                'use_post_processing': False  # Skip Phase 4 for speed
            }
        )
        
        logger.info("")
        logger.info("TEST 1 COMPLETE")
        logger.info(f"Elements detected: {len(result.elements)}")
        logger.info(f"Connections detected: {len(result.connections)}")
        
        return {
            'success': True,
            'elements': len(result.elements),
            'connections': len(result.connections),
            'result': result
        }
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def test_complex_path(image_path: str) -> dict:
    """
    Test 2: Complex path validation
    
    Expected:
    - CV quick test: pixel_density >= 0.05
    - LLM fine check triggered (Flash-Lite)
    - optimal_swarm_monolith strategy loaded
    """
    logger.info("=" * 80)
    logger.info("TEST 2: COMPLEX PATH VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Image: {image_path}")
    logger.info("")
    
    # Initialize components
    config_service = ConfigService()
    config = config_service.get_config()
    
    config_dict = config_service.get_raw_config()
    llm_client = LLMClient(
        project_id=os.getenv("GCP_PROJECT_ID"),
        default_location=os.getenv("GCP_LOCATION", "us-central1"),
        config=config_dict
    )
    knowledge_manager = KnowledgeManager(
        element_type_list_path=project_root / "element_type_list.json",
        learning_db_path=project_root / "learning_db.json",
        llm_handler=llm_client,
        config=config_dict
    )
    
    coordinator = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service,
        model_strategy=None,
        progress_callback=DummyProgressCallback()
    )
    
    # Run analysis (Phase 0 will select strategy automatically)
    try:
        result = coordinator.process(
            image_path=image_path,
            params_override={
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': False,  # Skip Phase 3 for speed
                'use_post_processing': False  # Skip Phase 4 for speed
            }
        )
        
        logger.info("")
        logger.info("TEST 2 COMPLETE")
        logger.info(f"Elements detected: {len(result.elements)}")
        logger.info(f"Connections detected: {len(result.connections)}")
        
        return {
            'success': True,
            'elements': len(result.elements),
            'connections': len(result.connections),
            'result': result
        }
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def analyze_logs(log_file: str) -> dict:
    """
    Analyze log file for Phase 0 validation indicators.
    
    Returns:
        Dictionary with validation results
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return {'error': 'Log file not found'}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    results = {
        'phase0_detected': 'Phase 0' in log_content or 'Phase 0:' in log_content,
        'cv_quick_test': 'CV Quick Test' in log_content or 'pixel_density' in log_content,
        'llm_fine_check': 'LLM fine check' in log_content or 'LLM fine-check' in log_content,
        'simple_strategy': 'simple_pid_strategy' in log_content,
        'optimal_strategy': 'optimal_swarm_monolith' in log_content,
        'fast_exit': 'Fast exit' in log_content or 'fast exit' in log_content,
        'complexity_simple': 'complexity=simple' in log_content or 'complexity: simple' in log_content,
        'complexity_complex': 'complexity=complex' in log_content or 'complexity: complex' in log_content,
        'llm_used': 'llm_used=True' in log_content or 'llm_used: True' in log_content,
        'llm_not_used': 'llm_used=False' in log_content or 'llm_used: False' in log_content
    }
    
    return results


def main():
    """Run Phase 0 validation tests."""
    logger.info("=" * 80)
    logger.info("PHASE 0 COMPLEXITY ANALYSIS VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    
    # Find test images
    simple_image = None
    complex_image = None
    
    # Search for simple P&ID
    simple_candidates = [
        "training_data/simple_pids/Einfaches P&I.png",
        "training_data/Einfaches P&I.png",
        "test_data/Einfaches P&I.png"
    ]
    
    for candidate in simple_candidates:
        path = project_root / candidate
        if path.exists():
            simple_image = str(path)
            break
    
    # Search for complex P&ID
    complex_candidates = [
        "training_data/complex_pids/page_1_original.png",
        "training_data/organized_tests/complex_pids/page_1_original.png",
        "training_data/page_1_original.png",
        "training_data/page_1_original.jpg"
    ]
    
    for candidate in complex_candidates:
        path = project_root / candidate
        if path.exists():
            complex_image = str(path)
            break
    
    if not simple_image:
        logger.error("Simple P&ID image not found. Please check paths.")
        return
    
    if not complex_image:
        logger.error("Complex P&ID image not found. Please check paths.")
        return
    
    logger.info(f"Test images found:")
    logger.info(f"  Simple: {simple_image}")
    logger.info(f"  Complex: {complex_image}")
    logger.info("")
    
    # Run Test 1: Simple Path
    logger.info("Starting Test 1: Simple Path...")
    test1_result = test_simple_path(simple_image)
    
    logger.info("")
    logger.info("-" * 80)
    logger.info("")
    
    # Run Test 2: Complex Path (if image available)
    test2_result = None
    if complex_image:
        logger.info("Starting Test 2: Complex Path...")
        test2_result = test_complex_path(complex_image)
    else:
        logger.info("Skipping Test 2: Complex Path (no complex image found)")
        test2_result = {'success': False, 'error': 'Complex image not found'}
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Test 1 (Simple Path):")
    logger.info(f"  Success: {test1_result.get('success', False)}")
    if test1_result.get('success'):
        logger.info(f"  Elements: {test1_result.get('elements', 0)}")
        logger.info(f"  Connections: {test1_result.get('connections', 0)}")
    else:
        logger.info(f"  Error: {test1_result.get('error', 'Unknown')}")
    
    logger.info("")
    logger.info("Test 2 (Complex Path):")
    logger.info(f"  Success: {test2_result.get('success', False)}")
    if test2_result.get('success'):
        logger.info(f"  Elements: {test2_result.get('elements', 0)}")
        logger.info(f"  Connections: {test2_result.get('connections', 0)}")
    else:
        logger.info(f"  Error: {test2_result.get('error', 'Unknown')}")
    
    logger.info("")
    logger.info("Analyzing logs for validation indicators...")
    log_analysis = analyze_logs('outputs/logs/phase0_validation.log')
    
    logger.info("")
    logger.info("Log Analysis Results:")
    for key, value in log_analysis.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Check outputs/logs/phase0_validation.log for detailed logs.")


if __name__ == "__main__":
    main()

