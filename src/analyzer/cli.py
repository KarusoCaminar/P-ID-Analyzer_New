"""
Command-line interface for P&ID Analyzer.

Refactored to use the new PipelineCoordinator.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator, ProgressCallback
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.utils.env_loader import load_env_automatically

# Load .env file automatically (centralized approach)
load_env_automatically()

logger = logging.getLogger(__name__)


class CliProgressCallback(ProgressCallback):
    """Progress callback for CLI output."""
    
    def __init__(self):
        self.last_value = 0
    
    def update_progress(self, value: int, message: str) -> None:
        """Update progress for CLI."""
        if value > self.last_value:
            logger.info(f"Progress: {message} ({value}%)")
            self.last_value = value
    
    def update_status_label(self, text: str) -> None:
        """Update status for CLI."""
        logger.info(f"Status: {text}")
    
    def report_truth_mode(self, active: bool) -> None:
        """Report truth mode status."""
        if active:
            logger.info("Truth mode: Active")
    
    def report_correction(self, correction_text: str) -> None:
        """Report correction information."""
        logger.info(f"Correction: {correction_text}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="P&ID Analyzer - Analyze Piping and Instrumentation Diagrams")
    parser.add_argument("image_path", type=str, help="Path to P&ID image file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Check environment variables
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    # If GCP_PROJECT_ID is missing, fall back to a local dummy LLM client so the pipeline
    # can run for offline testing and development without exiting.
    use_dummy_llm = False
    if not gcp_project_id:
        logger.warning(
            "GCP_PROJECT_ID environment variable not set. Falling back to Dummy LLM client for local runs."
        )
        use_dummy_llm = True

    gcp_location = os.getenv("GCP_LOCATION", "us-central1")
    
    # Initialize services
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        config_service = ConfigService(config_path=config_path)
        config = config_service.get_config()
        
        # Setup logging service
        log_dir = Path("outputs") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"analysis_{Path(args.image_path).stem}.log"
        LoggingService.setup_logging(
            log_level=log_level,
            log_file=log_file
        )
        
        # Get config as dict (safe method)
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = config_service.get_raw_config()
        
        # Initialize LLM client (use a dummy offline client if GCP not configured)
        if use_dummy_llm:
            # Import DummyLLMClient lazily to avoid importing vertexai or requiring credentials
            from src.analyzer.ai.llm_client import DummyLLMClient
            llm_client = DummyLLMClient(project_id="local", default_location=gcp_location, config=config_dict)
        else:
            # CRITICAL FIX: Ensure gcp_project_id is not None before passing to LLMClient
            if not gcp_project_id:
                logger.error("GCP_PROJECT_ID is required but not set. Cannot initialize LLMClient.")
                sys.exit(1)
            from src.analyzer.ai.llm_client import LLMClient
            llm_client = LLMClient(
                project_id=gcp_project_id,
                default_location=gcp_location,
                config=config_dict
            )
        
        # Initialize knowledge manager
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(config_service.get_path("element_type_list") or "element_type_list.json"),
            learning_db_path=str(config_service.get_path("learning_db") or "learning_db.json"),
            llm_handler=llm_client,  # KnowledgeManager uses llm_handler parameter
            config=config_dict
        )
        
        # Check image path
        image_path = Path(args.image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            sys.exit(1)
        
        logger.info(f"Starting analysis of: {image_path}")
        logger.info(f"Using model strategy: default")
        
        # CRITICAL FIX: Define coordinator outside try block for finally access
        coordinator = None
        try:
            # Create pipeline coordinator
            coordinator = PipelineCoordinator(
                llm_client=llm_client,
                knowledge_manager=knowledge_manager,
                config_service=config_service,
                progress_callback=CliProgressCallback()
            )
            
            # Run analysis
            result = coordinator.process(
                image_path=str(image_path),
                output_dir=args.output_dir,
                params_override=None
            )
            
            # Print results summary
            logger.info("=" * 60)
            logger.info("Analysis Complete!")
            logger.info("=" * 60)
            logger.info(f"Image: {result.image_name}")
            logger.info(f"Elements detected: {len(result.elements)}")
            logger.info(f"Connections detected: {len(result.connections)}")
            logger.info(f"Quality score: {result.quality_score:.2f}")
            
            if result.kpis:
                logger.info("KPIs:")
                for key, value in result.kpis.items():
                    logger.info(f"  {key}: {value}")
            
            logger.info(f"Results saved to: {args.output_dir or 'outputs'}")
            logger.info("=" * 60)
        except Exception as e:
            logger.critical(f"CLI-Fehler: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # CRITICAL FIX: Shutdown ThreadPoolExecutor to prevent resource leak
            if coordinator and hasattr(coordinator, 'llm_client') and coordinator.llm_client:
                logger.info("Fahre LLMClient ThreadPools herunter...")
                try:
                    if hasattr(coordinator.llm_client, 'close'):
                        coordinator.llm_client.close()
                        logger.info("Shutdown abgeschlossen.")
                    elif hasattr(coordinator.llm_client, 'timeout_executor'):
                        coordinator.llm_client.timeout_executor.shutdown(wait=True, cancel_futures=False)
                        logger.info("Shutdown abgeschlossen.")
                except Exception as e:
                    logger.warning(f"Fehler beim Shutdown: {e}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

