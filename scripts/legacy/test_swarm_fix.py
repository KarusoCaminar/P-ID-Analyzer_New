"""
Test Swarm Fix - Simple P&ID with Swarm Specialist

Tests the Swarm-Fix implementation for finding SamplePoint-S and ISA-Supply:
- Strategy: simple_pid_strategy
- Swarm: ACTIVATED (Flash-Lite, specialist prompt)
- Monolith: ACTIVATED (Pro, baseline)
- Fusion: ACTIVATED (with Monolith prioritization)
- Expected: 10/10 elements (95%+ F1)
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
output_base = project_root / "outputs" / "phase1_tests" / "swarm_fix_test"
output_base.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = output_base / f"swarm_fix_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def main():
    """Run Swarm Fix test."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("SWARM FIX TEST - Simple P&ID")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Strategy: simple_pid_strategy")
    logger.info("  - Swarm: ACTIVATED (Flash-Lite, specialist prompt)")
    logger.info("  - Monolith: ACTIVATED (Pro, baseline)")
    logger.info("  - Fusion: ACTIVATED (with Monolith prioritization)")
    logger.info("  - Expected: 10/10 elements (95%+ F1)")
    logger.info("")
    
    # Find test image
    image_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    truth_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    
    if not image_path.exists():
        logger.error(f"Test image not found: {image_path}")
        return
    
    if not truth_path.exists():
        logger.warning(f"Truth data not found: {truth_path}")
        truth_path = None
    
    logger.info(f"Test image: {image_path}")
    if truth_path:
        logger.info(f"Truth data: {truth_path}")
    logger.info("")
    
    try:
        # Initialize components
        config_service = ConfigService()
        config = config_service.get_config()
        config_dict = config_service.get_raw_config()
        
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
        
        # Reset circuit breaker
        if hasattr(llm_client, 'retry_handler') and hasattr(llm_client.retry_handler, 'circuit_breaker'):
            llm_client.retry_handler.circuit_breaker.reset()
            logger.info("Circuit breaker reset")
        
        # Create output directory
        test_output_dir = output_base / "test_run"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {test_output_dir}")
        logger.info("")
        
        # Run analysis with simple_pid_strategy
        logger.info("Starting analysis with simple_pid_strategy...")
        logger.info("")
        
        result = coordinator.process(
            image_path=str(image_path),
            output_dir=str(test_output_dir),
            params_override={
                'strategy': 'simple_pid_strategy',
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'use_post_processing': True,
                'use_active_learning': False
            }
        )
        
        # Extract results
        if hasattr(result, 'elements'):
            elements = result.elements
            connections = result.connections
            quality_score = getattr(result, 'quality_score', 0.0)
        elif isinstance(result, dict):
            elements = result.get('elements', [])
            connections = result.get('connections', [])
            quality_score = result.get('quality_score', 0.0)
        else:
            elements = []
            connections = []
            quality_score = 0.0
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Elements detected: {len(elements)}")
        logger.info(f"Connections detected: {len(connections)}")
        logger.info(f"Quality Score: {quality_score:.2f}")
        logger.info("")
        
        # Validate against truth if available
        if truth_path:
            import json
            from src.utils.json_encoder import json_dump_safe
            
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_data = json.load(f)
            
            truth_elements = truth_data.get('elements', [])
            truth_connections = truth_data.get('connections', [])
            
            logger.info(f"Truth data: {len(truth_elements)} elements, {len(truth_connections)} connections")
            logger.info("")
            
            # Simple validation
            element_ids = {el.get('id') if isinstance(el, dict) else el.id for el in elements}
            truth_element_ids = {el.get('id') if isinstance(el, dict) else el.id for el in truth_elements}
            
            matched = len(element_ids & truth_element_ids)
            missed = len(truth_element_ids - element_ids)
            hallucinated = len(element_ids - truth_element_ids)
            
            precision = matched / len(element_ids) if element_ids else 0.0
            recall = matched / len(truth_element_ids) if truth_element_ids else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            logger.info("Element Validation:")
            logger.info(f"  - Matched: {matched}/{len(truth_element_ids)}")
            logger.info(f"  - Missed: {missed}")
            logger.info(f"  - Hallucinated: {hallucinated}")
            logger.info(f"  - Precision: {precision:.2%}")
            logger.info(f"  - Recall: {recall:.2%}")
            logger.info(f"  - F1: {f1:.2%}")
            logger.info("")
            
            # Save results
            result_file = test_output_dir / "test_result.json"
            result_data = {
                'test_name': 'Swarm Fix Test',
                'image_path': str(image_path),
                'truth_path': str(truth_path),
                'results': {
                    'elements_count': len(elements),
                    'connections_count': len(connections),
                    'quality_score': quality_score
                },
                'validation': {
                    'elements': {
                        'matched': matched,
                        'missed': missed,
                        'hallucinated': hallucinated,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json_dump_safe(result_data, f, indent=2)
            
            logger.info(f"Results saved to: {result_file}")
            
            # Check if goal achieved
            if f1 >= 0.95:
                logger.info("")
                logger.info("✅ GOAL ACHIEVED: F1 >= 95%")
            else:
                logger.info("")
                logger.info(f"❌ GOAL NOT ACHIEVED: F1 = {f1:.2%} (target: 95%+)")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

