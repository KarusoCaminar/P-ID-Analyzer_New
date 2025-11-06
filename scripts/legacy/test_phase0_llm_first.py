"""
Phase 0 LLM-First Validation Tests

Tests the inverted LLM-First complexity analysis logic:
- Test 1: Simple P&ID (Einfaches P&I.png) → should use simple_pid_strategy
- Test 2: Complex P&ID (page_1_original.jpg) → should use optimal_swarm_monolith
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
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

# Create output directory for this test run
output_base = project_root / "outputs" / "phase1_tests" / "phase0_llm_first_validation"
output_base.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = output_base / f"phase0_llm_first_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DummyProgressCallback:
    """Dummy progress callback for testing."""
    def update_progress(self, progress: int, message: str):
        logger.info(f"[Progress {progress}%] {message}")
    
    def update_status_label(self, message: str):
        logger.info(f"[Status] {message}")
    
    def report_truth_mode(self, active: bool):
        logger.info(f"[Truth Mode] {'ACTIVE' if active else 'INACTIVE'}")


def test_phase0_llm_first(image_path: str, expected_strategy: str, test_name: str) -> dict:
    """
    Test Phase 0 LLM-First complexity analysis.
    
    Args:
        image_path: Path to test image
        expected_strategy: Expected strategy name
        test_name: Name of the test
        
    Returns:
        Dictionary with test results
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"TEST: {test_name}")
    logger.info("=" * 80)
    logger.info(f"Image: {image_path}")
    logger.info(f"Expected Strategy: {expected_strategy}")
    logger.info("")
    
    try:
        # Initialize components
        config_service = ConfigService()
        config = config_service.get_config()
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
        
        # Initialize LLM client
        project_id = os.getenv('GCP_PROJECT_ID')
        default_location = os.getenv('GCP_LOCATION', 'us-central1')
        
        llm_client = LLMClient(
            project_id=project_id,
            default_location=default_location,
            config=config_dict
        )
        
        # Initialize knowledge manager
        knowledge_manager = KnowledgeManager(
            element_type_list_path=project_root / "element_type_list.json",
            learning_db_path=project_root / "learning_db.json",
            llm_handler=llm_client,
            config=config_dict
        )
        
        # Initialize pipeline coordinator
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            progress_callback=DummyProgressCallback()
        )
        
        # CRITICAL FIX 1.2: Reset circuit breaker before each test
        if hasattr(llm_client, 'retry_handler') and hasattr(llm_client.retry_handler, 'circuit_breaker'):
            llm_client.retry_handler.circuit_breaker.reset()
            logger.info("Circuit breaker reset for new test")
        
        # Create output directory for this test (clean invalid characters for Windows)
        import re
        safe_name = re.sub(r'[<>:"/\\|?*&→]', '_', test_name.lower())
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('_')
        test_output_dir = output_base / safe_name
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {test_output_dir}")
        logger.info("")
        
        # Run Phase 0 complexity analysis
        logger.info("Running Phase 0: Complexity Analysis (LLM-First)...")
        phase0_result = coordinator._run_phase_0_complexity_analysis(str(image_path))
        
        if not phase0_result:
            return {
                'test_name': test_name,
                'success': False,
                'error': 'Phase 0 returned None',
                'expected_strategy': expected_strategy,
                'actual_strategy': None
            }
        
        # Extract results
        actual_complexity = phase0_result.get('complexity', 'unknown')
        actual_strategy = phase0_result.get('strategy', 'unknown')
        llm_used = phase0_result.get('llm_used', False)
        reasoning = phase0_result.get('reasoning', 'N/A')
        
        # Check if strategy matches
        strategy_match = actual_strategy == expected_strategy
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PHASE 0 RESULTS: {test_name}")
        logger.info("=" * 80)
        logger.info(f"Complexity: {actual_complexity}")
        logger.info(f"Strategy: {actual_strategy}")
        logger.info(f"Expected Strategy: {expected_strategy}")
        logger.info(f"Strategy Match: {'✅ PASS' if strategy_match else '❌ FAIL'}")
        logger.info(f"LLM Used: {llm_used}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info("=" * 80)
        logger.info("")
        
        # Save results
        result_file = test_output_dir / "phase0_result.json"
        import json
        from src.utils.json_encoder import json_dump_safe
        
        result_data = {
            'test_name': test_name,
            'image_path': str(image_path),
            'expected_strategy': expected_strategy,
            'actual_complexity': actual_complexity,
            'actual_strategy': actual_strategy,
            'strategy_match': strategy_match,
            'llm_used': llm_used,
            'reasoning': reasoning,
            'phase0_result': phase0_result,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json_dump_safe(result_data, f, indent=2)
        
        logger.info(f"Results saved to: {result_file}")
        
        return {
            'test_name': test_name,
            'success': strategy_match,
            'expected_strategy': expected_strategy,
            'actual_strategy': actual_strategy,
            'actual_complexity': actual_complexity,
            'strategy_match': strategy_match,
            'llm_used': llm_used,
            'reasoning': reasoning,
            'result_file': str(result_file)
        }
        
    except Exception as e:
        logger.error(f"Test {test_name} failed: {e}", exc_info=True)
        return {
            'test_name': test_name,
            'success': False,
            'error': str(e),
            'expected_strategy': expected_strategy,
            'actual_strategy': None
        }


def main():
    """Run Phase 0 LLM-First validation tests."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 0 LLM-FIRST VALIDATION TESTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Find test images
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    
    # Try multiple paths for complex P&ID
    complex_pid_paths = [
        project_root / "training_data" / "complex_pids" / "page_1_original.png",
        project_root / "training_data" / "uni_bilder" / "page_1_original.jpg",
        project_root / "training_data" / "uni_bilder" / "page_1_original.png",
        project_root / "training_data" / "uni_bilder" / "page_1.jpg",
        project_root / "training_data" / "uni_bilder" / "page_1.png",
        project_root / "training_data" / "page_1_original.jpg",
        project_root / "training_data" / "page_1_original.png",
    ]
    
    complex_pid_path = None
    for path in complex_pid_paths:
        if path.exists():
            complex_pid_path = path
            logger.info(f"Found complex P&ID: {complex_pid_path}")
            break
    
    tests = []
    
    # Test 1: Simple P&ID
    if simple_pid_path.exists():
        logger.info(f"Found simple P&ID: {simple_pid_path}")
        result1 = test_phase0_llm_first(
            image_path=str(simple_pid_path),
            expected_strategy='simple_pid_strategy',
            test_name='Test 1: Simple P&ID (LLM-First → simple_pid_strategy)'
        )
        tests.append(result1)
    else:
        logger.error(f"Simple P&ID not found: {simple_pid_path}")
    
    logger.info("")
    
    # Test 2: Complex P&ID
    if complex_pid_path and complex_pid_path.exists():
        logger.info(f"Found complex P&ID: {complex_pid_path}")
        result2 = test_phase0_llm_first(
            image_path=str(complex_pid_path),
            expected_strategy='optimal_swarm_monolith',
            test_name='Test 2: Complex P&ID (LLM-First → optimal_swarm_monolith)'
        )
        tests.append(result2)
    else:
        logger.error(f"Complex P&ID not found: {complex_pid_path}")
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for t in tests if t.get('success', False))
    total = len(tests)
    
    for test in tests:
        status = "✅ PASS" if test.get('success', False) else "❌ FAIL"
        logger.info(f"{status}: {test.get('test_name', 'Unknown')}")
        if not test.get('success', False):
            logger.info(f"  Error: {test.get('error', 'Unknown error')}")
        else:
            logger.info(f"  Strategy: {test.get('actual_strategy', 'unknown')}")
            logger.info(f"  LLM Used: {test.get('llm_used', False)}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    # Save summary
    summary_file = output_base / "test_summary.json"
    import json
    from src.utils.json_encoder import json_dump_safe
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total,
        'passed_tests': passed,
        'failed_tests': total - passed,
        'tests': tests
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json_dump_safe(summary_data, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("")
    
    return summary_data


if __name__ == "__main__":
    main()

