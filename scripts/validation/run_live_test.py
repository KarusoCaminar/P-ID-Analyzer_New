"""
Live Test Runner - Runs a full pipeline test with live log monitoring.

This script runs a complete pipeline test on a test image and displays
live logs in the console while monitoring for errors.
"""

import sys
import json
import os
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_encoder import json_dump_safe
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator
from scripts.utils.live_log_monitor import LiveLogMonitor

# Load .env file automatically
try:
    from src.utils.env_loader import load_env_automatically
    if load_env_automatically():
        print(f"[OK] .env Datei automatisch geladen")
    else:
        print(f"[WARNING] .env Datei nicht gefunden")
except (ImportError, Exception) as e:
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[OK] .env Datei geladen: {env_file}")
    except ImportError:
        pass

# Configuration
TEST_IMAGE_SIMPLE = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
TEST_IMAGE_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original.png"
TEST_GROUND_TRUTH_SIMPLE = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
TEST_GROUND_TRUTH_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original_truth_cgm.json"

STRATEGY = "simple_whole_image"
OUTPUT_BASE_DIR = project_root / "outputs" / "live_test"


class LiveTestRunner:
    """Test runner with live log monitoring."""
    
    def __init__(self, test_image: Path, test_truth: Optional[Path] = None, use_complex: bool = False):
        """
        Initialize test runner.
        
        Args:
            test_image: Path to test image
            test_truth: Optional path to ground truth file
            use_complex: If True, use complex image (page_1_original)
        """
        self.test_image = test_image
        self.test_truth = test_truth
        self.use_complex = use_complex
        
        # CRITICAL: Use OutputStructureManager for structured output
        from src.utils.output_structure_manager import OutputStructureManager
        self.output_manager = OutputStructureManager(OUTPUT_BASE_DIR.parent, "live_test")
        self.output_dir = self.output_manager.get_output_dir()
        
        # Log file → logs/ subdirectory
        self.log_file = self.output_manager.get_log_file("test.log")
        
        # Create README explaining structure
        self.output_manager.create_readme()
        
        # Setup logging to file and console
        LoggingService.setup_logging(
            log_level=logging.INFO,
            log_file=str(self.log_file)
        )
        self.logger = logging.getLogger(__name__)
        
        # Live log monitor
        self.log_monitor: Optional[LiveLogMonitor] = None
        
        # Services
        self.config_service = None
        self.llm_client = None
        self.knowledge_manager = None
        self.coordinator = None
        
    def setup_services(self):
        """Initialize all services."""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING SERVICES")
        self.logger.info("=" * 80)
        
        try:
            # Config service
            self.logger.info("[1/4] Loading configuration...")
            self.config_service = ConfigService()
            config = self.config_service.get_raw_config()
            self.logger.info("[OK] Configuration loaded")
            
            # GCP credentials
            project_id = os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "us-central1")
            
            if not project_id:
                raise ValueError("GCP_PROJECT_ID nicht gesetzt!")
            
            # LLM Client
            self.logger.info("[2/4] Initializing LLM Client...")
            self.llm_client = LLMClient(project_id, location, config)
            self.logger.info("[OK] LLM Client initialized")
            
            # Knowledge Manager
            self.logger.info("[3/4] Initializing Knowledge Manager...")
            element_type_list = self.config_service.get_path('element_type_list')
            learning_db = self.config_service.get_path('learning_db')
            
            start_km = time.time()
            self.knowledge_manager = KnowledgeManager(
                element_type_list_path=str(element_type_list),
                learning_db_path=str(learning_db),
                llm_handler=self.llm_client,
                config=config
            )
            km_time = time.time() - start_km
            self.logger.info(f"[OK] Knowledge Manager initialized in {km_time:.2f} seconds")
            
            # Pipeline Coordinator
            self.logger.info("[4/4] Initializing Pipeline Coordinator...")
            self.coordinator = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service
            )
            self.logger.info("[OK] Pipeline Coordinator initialized")
            
            self.logger.info("=" * 80)
            self.logger.info("ALL SERVICES INITIALIZED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing services: {e}", exc_info=True)
            raise
    
    def start_live_monitoring(self):
        """Start live log monitoring."""
        self.logger.info("Starting live log monitoring...")
        self.log_monitor = LiveLogMonitor(self.log_file)
        self.log_monitor.start()
        self.logger.info("[OK] Live log monitoring started")
    
    def stop_live_monitoring(self):
        """Stop live log monitoring."""
        if self.log_monitor:
            self.log_monitor.flush()
            self.log_monitor.stop()
            self.log_monitor = None
    
    def load_ground_truth(self) -> Optional[dict]:
        """Load ground truth data."""
        if not self.test_truth or not self.test_truth.exists():
            self.logger.warning(f"Ground truth not found: {self.test_truth}")
            return None
        
        try:
            with open(self.test_truth, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            self.logger.info(f"[OK] Ground truth loaded: {len(gt_data.get('elements', []))} elements")
            return gt_data
        except Exception as e:
            self.logger.error(f"Error loading ground truth: {e}", exc_info=True)
            return None
    
    def run_test(self):
        """Run the complete test."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING TEST RUN")
        self.logger.info("=" * 80)
        self.logger.info(f"Test Image: {self.test_image}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Strategy: {STRATEGY}")
        self.logger.info("=" * 80)
        
        # Verify image exists
        if not self.test_image.exists():
            raise FileNotFoundError(f"Test image not found: {self.test_image}")
        
        # Load strategy config
        config = self.config_service.get_raw_config()
        strategies = config.get('strategies', {})
        strategy_config = strategies.get(STRATEGY, {})
        
        if not strategy_config:
            raise ValueError(f"Strategy '{STRATEGY}' not found!")
        
        # Prepare parameters
        params_override = {
            **strategy_config,
            'test_name': f"{STRATEGY}_live_test",
            'test_description': f"Live test: {STRATEGY} on {self.test_image.name}"
        }
        
        # Load ground truth
        gt_data = self.load_ground_truth()
        
        # Run analysis
        start_time = time.time()
        self.logger.info(f"\nStarting analysis at {datetime.now().strftime('%H:%M:%S')}...")
        
        try:
            # CRITICAL: Use structured output directory
            result = self.coordinator.process(
                image_path=str(self.test_image),
                output_dir=self.output_manager.get_output_dir_str(),
                params_override=params_override
            )
            
            end_time = time.time()
            duration = (end_time - start_time) / 60.0
            
            self.logger.info(f"\n[OK] Analysis completed in {duration:.2f} minutes")
            
            # Convert result
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result if isinstance(result, dict) else {
                    'elements': getattr(result, 'elements', []),
                    'connections': getattr(result, 'connections', [])
                }
            
            self.logger.info(f"  Found Elements: {len(result_dict.get('elements', []))}")
            self.logger.info(f"  Found Connections: {len(result_dict.get('connections', []))}")
            
            # Calculate KPIs
            kpis = {}
            if gt_data:
                self.logger.info("\nCalculating KPIs...")
                kpi_calc = KPICalculator()
                kpis = kpi_calc.calculate_comprehensive_kpis(
                    analysis_data=result_dict,
                    truth_data=gt_data
                )
                
                self.logger.info("\n" + "=" * 80)
                self.logger.info("RESULTS")
                self.logger.info("=" * 80)
                self.logger.info(f"Element F1: {kpis.get('element_f1', 0.0):.4f}")
                self.logger.info(f"Element Precision: {kpis.get('element_precision', 0.0):.4f}")
                self.logger.info(f"Element Recall: {kpis.get('element_recall', 0.0):.4f}")
                self.logger.info(f"Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
                self.logger.info(f"Quality Score: {kpis.get('quality_score', 0.0):.2f}")
                self.logger.info("=" * 80)
            
            # CRITICAL: Save results to data/ subdirectory
            result_file = self.output_manager.get_data_path("test_result.json")
            test_result = {
                'strategy': STRATEGY,
                'image_path': str(self.test_image),
                'duration_minutes': duration,
                'timestamp': datetime.now().isoformat(),
                'result': result_dict,
                'kpis': kpis,
                'success': True
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json_dump_safe(test_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n[OK] Results saved: {result_file}")
            self.logger.info("=" * 80)
            self.logger.info("TEST COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"\n[FAIL] TEST FAILED: {e}", exc_info=True)
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_live_monitoring()
        if self.llm_client and hasattr(self.llm_client, 'close'):
            try:
                self.llm_client.close()
            except:
                pass


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run live test with full pipeline")
    parser.add_argument(
        '--image',
        type=str,
        choices=['simple', 'complex', 'page1'],
        default='simple',
        help='Test image to use (simple, complex, or page1)'
    )
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Disable live log monitoring'
    )
    
    args = parser.parse_args()
    
    # Select test image
    if args.image == 'simple':
        test_image = TEST_IMAGE_SIMPLE
        test_truth = TEST_GROUND_TRUTH_SIMPLE
        use_complex = False
    else:  # complex or page1
        test_image = TEST_IMAGE_COMPLEX
        test_truth = TEST_GROUND_TRUTH_COMPLEX
        use_complex = True
    
    # Create test runner
    runner = LiveTestRunner(test_image, test_truth, use_complex)
    
    try:
        # Setup services
        runner.setup_services()
        
        # Start live monitoring (if enabled)
        if not args.no_monitor:
            runner.start_live_monitoring()
        
        # Run test
        runner.run_test()
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrupted by user")
        runner.logger.warning("Test interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Test failed: {e}")
        runner.logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()

