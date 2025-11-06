#!/usr/bin/env python3
"""Quick test with single image and timeout."""

import sys
import os
import time
import signal
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded timeout")

def main():
    """Quick test with timeout."""
    timeout_seconds = 300  # 5 minutes max
    
    try:
        from src.services.config_service import ConfigService
        from src.analyzer.ai.llm_client import LLMClient
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        
        logger.info("Initializing backend...")
        
        gcp_project_id = os.getenv('GCP_PROJECT_ID')
        gcp_location = os.getenv('GCP_LOCATION', 'us-central1')
        
        if not gcp_project_id:
            logger.warning("GCP_PROJECT_ID not set - API calls will fail")
        
        config_path = project_root / "config.yaml"
        config_service = ConfigService(config_path=config_path if config_path.exists() else None)
        config = config_service.get_config()
        
        llm_client = LLMClient(
            project_id=gcp_project_id or "dummy",
            default_location=gcp_location,
            config=config.model_dump()
        )
        
        # Reset circuit breaker
        if hasattr(llm_client, 'retry_handler') and hasattr(llm_client.retry_handler, 'circuit_breaker'):
            llm_client.retry_handler.circuit_breaker.reset()
        
        element_type_list_path = config_service.get_path("element_type_list") or project_root / "element_type_list.json"
        learning_db_path = config_service.get_path("learning_db") or project_root / "learning_db.json"
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config.model_dump()
        )
        
        pipeline_coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        
        logger.info("Backend initialized")
        
        # Find first test image - prioritize simple_pids
        test_image = None
        
        # First try simple_pids
        simple_pids_dir = project_root / "training_data" / "simple_pids"
        if simple_pids_dir.exists():
            logger.info(f"Searching in simple_pids: {simple_pids_dir}")
            images = list(simple_pids_dir.rglob("*.png")) + list(simple_pids_dir.rglob("*.jpg")) + list(simple_pids_dir.rglob("*.jpeg"))
            images = [img for img in images if not any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp', 'correction', 'symbol'])]
            if images:
                test_image = images[0]
                logger.info(f"Found simple test image: {test_image.name}")
        
        # Fallback to organized_tests
        if not test_image:
            test_dirs = [
                project_root / "training_data" / "organized_tests",
                project_root / "training_data"
            ]
            
            for test_dir in test_dirs:
                if test_dir.exists():
                    # Use rglob for recursive search
                    images = list(test_dir.rglob("*.png")) + list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg"))
                    # Filter out truth files and temp files
                    images = [img for img in images if not any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp', 'correction', 'symbol'])]
                    if images:
                        test_image = images[0]
                        logger.info(f"Found test image: {test_image.name}")
                        logger.info(f"Path: {test_image}")
                        break
        
        if not test_image:
            logger.error("No test images found!")
            return
        
        logger.info(f"Processing image: {test_image.name}...")
        
        # Progress callback
        class ProgressCallback:
            def update_progress(self, value: int, message: str):
                logger.info(f"  [{value}%] {message}")
            
            def update_status_label(self, text: str):
                logger.info(f"  Status: {text}")
            
            def report_truth_mode(self, active: bool):
                pass
            
            def report_correction(self, correction_text: str):
                logger.info(f"  Correction: {correction_text[:100]}")
        
        pipeline_coordinator.progress_callback = ProgressCallback()
        
        start_time = time.time()
        logger.info("Starting analysis pipeline...")
        
        result = pipeline_coordinator.process(
            image_path=str(test_image),
            output_dir=None
        )
        
        duration = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analysis completed in {duration:.1f} seconds")
        logger.info(f"{'='*60}")
        
        if result:
            quality_score = getattr(result, 'quality_score', 0.0)
            elements_count = len(getattr(result, 'elements', []))
            connections_count = len(getattr(result, 'connections', []))
            
            logger.info(f"Quality Score: {quality_score:.2f}")
            logger.info(f"Elements: {elements_count}")
            logger.info(f"Connections: {connections_count}")
            
            if quality_score > 0:
                logger.info("✅ SUCCESS: Pipeline working!")
            else:
                logger.warning("⚠️  Quality score is 0 - check for errors")
        else:
            logger.error("❌ FAILED: No result returned")
        
    except TimeoutError:
        logger.error(f"❌ Test exceeded timeout of {timeout_seconds} seconds")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

