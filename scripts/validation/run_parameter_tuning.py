"""
Parameter Tuning Script - Optimizes adaptive_threshold parameters for Connection F1-Score.

This script systematically tests different adaptive_threshold_factor values to find
the optimal parameters for maximum Connection F1-Score.

Features:
- Grid search over adaptive_threshold_factor values
- Live log monitoring
- Structured output (OutputStructureManager)
- KPI tracking and comparison
- Best parameter selection based on Connection F1-Score
"""

import sys
import json
import os
import logging
import time
import copy
import random
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from queue import Queue

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
from src.utils.output_structure_manager import OutputStructureManager

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

# Test Configuration
TEST_IMAGE_SIMPLE = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
TEST_IMAGE_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original.png"
TEST_GROUND_TRUTH_SIMPLE = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
TEST_GROUND_TRUTH_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original_truth_cgm.json"

# Parameter ranges to test (from PARAMETER_TUNING_GUIDE.md)
# REDUCED RANGE for faster testing - can be expanded after finding optimal range
ADAPTIVE_THRESHOLD_FACTORS = [0.01, 0.02, 0.03, 0.05]  # Reduced from 6 to 4 values
ADAPTIVE_THRESHOLD_MINS = [20, 25, 30]  # Reduced from 5 to 3 values
ADAPTIVE_THRESHOLD_MAXS = [125, 150, 200]  # Reduced from 5 to 3 values
# Total combinations: 4 × 3 × 3 = 36 (instead of 150)

# Strategy to use for parameter tuning
STRATEGY = "simple_whole_image"  # Best for simple P&IDs, fast and accurate

# Output directory
OUTPUT_BASE_DIR = project_root / "outputs" / "parameter_tuning"

# Worker Pool Configuration
# CRITICAL: Concurrency limiting to avoid rate limits
# Start with 5 workers, increase to 10 after quota increase
MAX_WORKERS = 5  # Max parallel API requests (adjust based on your quota: 60 req/min = 5 workers, 300 req/min = 10-15 workers)


class ParameterTuningRunner:
    """Parameter tuning runner with live log monitoring."""
    
    def __init__(self, test_image: Path, test_truth: Optional[Path] = None, use_simple: bool = True):
        """
        Initialize parameter tuning runner.
        
        Args:
            test_image: Path to test image
            test_truth: Optional path to ground truth file
            use_simple: If True, use simple image (faster for tuning)
        """
        self.test_image = test_image
        self.test_truth = test_truth
        self.use_simple = use_simple
        
        # Create output structure
        self.output_manager = OutputStructureManager(OUTPUT_BASE_DIR.parent, "parameter_tuning")
        self.output_dir = self.output_manager.get_output_dir()
        
        # Main log file
        self.log_file = self.output_manager.get_log_file("parameter_tuning.log")
        
        # Create README
        self.output_manager.create_readme()
        
        # Setup logging
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
        
        # Results storage (thread-safe)
        self.results = []
        self.results_lock = threading.Lock()
        
        # Worker pool configuration
        self.max_workers = MAX_WORKERS
    
    def setup_services(self):
        """Initialize all services."""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING SERVICES FOR PARAMETER TUNING")
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
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading ground truth from {self.test_truth}: {e}", exc_info=True)
            return None
    
    def run_test_with_parameters(
        self,
        adaptive_threshold_factor: float,
        adaptive_threshold_min: int,
        adaptive_threshold_max: int,
        test_number: int = 0,
        total_tests: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single test with specific parameters.
        
        Args:
            adaptive_threshold_factor: Factor for adaptive threshold (0.01-0.10)
            adaptive_threshold_min: Minimum threshold in pixels
            adaptive_threshold_max: Maximum threshold in pixels
            
        Returns:
            Dictionary with test results and KPIs
        """
        self.logger.info("=" * 80)
        self.logger.info(f"TESTING PARAMETERS:")
        self.logger.info(f"  adaptive_threshold_factor: {adaptive_threshold_factor}")
        self.logger.info(f"  adaptive_threshold_min: {adaptive_threshold_min}")
        self.logger.info(f"  adaptive_threshold_max: {adaptive_threshold_max}")
        self.logger.info("=" * 80)
        
        # Update coordinator's active_logic_parameters
        # LineExtractor reads from logic_parameters -> line_extraction in its config
        # The coordinator passes active_logic_parameters to LineExtractor via its config
        if 'line_extraction' not in self.coordinator.active_logic_parameters:
            self.coordinator.active_logic_parameters['line_extraction'] = {}
        
        self.coordinator.active_logic_parameters['line_extraction'].update({
            'adaptive_threshold_factor': adaptive_threshold_factor,
            'adaptive_threshold_min': adaptive_threshold_min,
            'adaptive_threshold_max': adaptive_threshold_max
        })
        
        # Load strategy config
        config = self.config_service.get_raw_config()
        strategies = config.get('strategies', {})
        strategy_config = strategies.get(STRATEGY, {})
        
        if not strategy_config:
            raise ValueError(f"Strategy '{STRATEGY}' not found!")
        
        # Prepare parameters
        # CRITICAL: For parameter tuning, disable expensive features to speed up tests
        params_override = {
            **strategy_config,
            'test_name': f"param_tune_{adaptive_threshold_factor}_{adaptive_threshold_min}_{adaptive_threshold_max}",
            'test_description': f"Parameter tuning: factor={adaptive_threshold_factor}, min={adaptive_threshold_min}, max={adaptive_threshold_max}",
            # Disable expensive features for faster parameter tuning
            'use_self_correction_loop': False,  # CRITICAL: Disable self-correction (saves ~50 minutes per test)
            'use_polyline_refinement': False,   # Disable polyline refinement (not needed for threshold testing)
            'use_predictive_completion': False, # Disable predictive completion (not needed for threshold testing)
            'use_visual_feedback': False        # Disable visual feedback (not needed for threshold testing)
        }
        
        # Load ground truth
        gt_data = self.load_ground_truth()
        
        # Run analysis
        start_time = time.time()
        self.logger.info(f"\nStarting analysis at {datetime.now().strftime('%H:%M:%S')}...")
        
        try:
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
                self.logger.info(f"Connection Precision: {kpis.get('connection_precision', 0.0):.4f}")
                self.logger.info(f"Connection Recall: {kpis.get('connection_recall', 0.0):.4f}")
                self.logger.info(f"Quality Score: {kpis.get('quality_score', 0.0):.2f}")
                self.logger.info("=" * 80)
            
            # Create result entry
            test_result = {
                'parameters': {
                    'adaptive_threshold_factor': adaptive_threshold_factor,
                    'adaptive_threshold_min': adaptive_threshold_min,
                    'adaptive_threshold_max': adaptive_threshold_max
                },
                'strategy': STRATEGY,
                'image_path': str(self.test_image),
                'duration_minutes': duration,
                'timestamp': datetime.now().isoformat(),
                'result': result_dict,
                'kpis': kpis,
                'success': True
            }
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"\n[FAIL] TEST FAILED: {e}", exc_info=True)
            return {
                'parameters': {
                    'adaptive_threshold_factor': adaptive_threshold_factor,
                    'adaptive_threshold_min': adaptive_threshold_min,
                    'adaptive_threshold_max': adaptive_threshold_max
                },
                'strategy': STRATEGY,
                'error': str(e),
                'success': False
            }
    
    def run_parameter_tuning(self):
        """
        Run parameter tuning with grid search using Worker Pool.
        
        CRITICAL: Uses ThreadPoolExecutor with concurrency limiting to avoid rate limits.
        - MAX_WORKERS parallel workers (default: 5)
        - Exponential backoff handled by LLMClient
        - Thread-safe results storage
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PARAMETER TUNING - WORKER POOL MODE")
        self.logger.info("=" * 80)
        self.logger.info(f"Test Image: {self.test_image}")
        self.logger.info(f"Strategy: {STRATEGY}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Max Workers: {self.max_workers} (parallel API requests)")
        self.logger.info("=" * 80)
        
        # Check if image exists
        if not self.test_image.exists():
            self.logger.error(f"Test-Bild nicht gefunden: {self.test_image}")
            self.stop_live_monitoring()
            sys.exit(1)
        
        # Grid search over parameters - create test jobs
        total_tests = len(ADAPTIVE_THRESHOLD_FACTORS) * len(ADAPTIVE_THRESHOLD_MINS) * len(ADAPTIVE_THRESHOLD_MAXS)
        self.logger.info(f"\nTotal parameter combinations to test: {total_tests}")
        self.logger.info(f"Worker Pool: {self.max_workers} parallel workers")
        self.logger.info("Exponential backoff: Enabled (handled by LLMClient)")
        self.logger.info("This may take a while...\n")
        
        # Create test jobs queue
        test_jobs = []
        test_count = 0
        for factor in ADAPTIVE_THRESHOLD_FACTORS:
            for min_val in ADAPTIVE_THRESHOLD_MINS:
                for max_val in ADAPTIVE_THRESHOLD_MAXS:
                    test_count += 1
                    test_jobs.append({
                        'test_number': test_count,
                        'factor': factor,
                        'min_val': min_val,
                        'max_val': max_val
                    })
        
        # Thread-safe shared state
        best_result_lock = threading.Lock()
        best_result = None
        best_connection_f1 = -1.0
        completed_tests = 0
        completed_lock = threading.Lock()
        
        def process_test_job(job: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single test job (called by worker thread)."""
            nonlocal best_result, best_connection_f1, completed_tests
            
            test_num = job['test_number']
            factor = job['factor']
            min_val = job['min_val']
            max_val = job['max_val']
            
            try:
                # Run test with exponential backoff (handled by LLMClient)
                result = self.run_test_with_parameters(
                    factor, min_val, max_val, test_num, total_tests
                )
                
                # Thread-safe results storage
                with self.results_lock:
                    self.results.append(result)
                
                # Check if this is the best result (thread-safe)
                if result.get('success') and result.get('kpis'):
                    connection_f1 = result['kpis'].get('connection_f1', 0.0)
                    with best_result_lock:
                        if connection_f1 > best_connection_f1:
                            best_connection_f1 = connection_f1
                            best_result = result
                            self.logger.info(f"\n⭐ NEW BEST RESULT! Connection F1: {connection_f1:.4f}")
                            self.logger.info(f"   Parameters: factor={factor}, min={min_val}, max={max_val}")
                
                # Update progress (thread-safe)
                with completed_lock:
                    completed_tests += 1
                    progress = (completed_tests / total_tests) * 100
                    self.logger.info(f"\n[PROGRESS] {completed_tests}/{total_tests} tests completed ({progress:.1f}%)")
                
                # Save intermediate results (thread-safe via lock)
                self.save_results()
                
                return result
                
            except Exception as e:
                self.logger.error(f"\n[FAIL] TEST {test_num} FAILED: {e}", exc_info=True)
                error_result = {
                    'parameters': {
                        'adaptive_threshold_factor': factor,
                        'adaptive_threshold_min': min_val,
                        'adaptive_threshold_max': max_val
                    },
                    'strategy': STRATEGY,
                    'error': str(e),
                    'success': False,
                    'test_number': test_num
                }
                with self.results_lock:
                    self.results.append(error_result)
                with completed_lock:
                    completed_tests += 1
                self.save_results()
                return error_result
        
        # Execute tests with Worker Pool (ThreadPoolExecutor)
        start_time = time.time()
        self.logger.info(f"\n[WORKER POOL] Starting {self.max_workers} workers...")
        self.logger.info(f"[WORKER POOL] Processing {total_tests} tests in parallel (max {self.max_workers} concurrent)\n")
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs to executor
                future_to_job = {
                    executor.submit(process_test_job, job): job
                    for job in test_jobs
                }
                
                # Process completed futures
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        # Result already processed in process_test_job
                    except Exception as e:
                        self.logger.error(f"\n[FAIL] Job {job['test_number']} raised exception: {e}", exc_info=True)
            
            end_time = time.time()
            total_duration = (end_time - start_time) / 60.0
            self.logger.info(f"\n[WORKER POOL] All tests completed in {total_duration:.2f} minutes")
            self.logger.info(f"[WORKER POOL] Average time per test: {total_duration / total_tests:.2f} minutes")
            
        except KeyboardInterrupt:
            self.logger.warning("\n[WORKER POOL] Interrupted by user - stopping workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        
        # Get best result (thread-safe) - must be inside try block scope
        final_best_result = None
        with best_result_lock:
            final_best_result = best_result
        
        # Final summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PARAMETER TUNING COMPLETE")
        self.logger.info("=" * 80)
        
        if final_best_result:
            self.logger.info(f"\n[BEST] BEST PARAMETERS:")
            self.logger.info(f"   adaptive_threshold_factor: {final_best_result['parameters']['adaptive_threshold_factor']}")
            self.logger.info(f"   adaptive_threshold_min: {final_best_result['parameters']['adaptive_threshold_min']}")
            self.logger.info(f"   adaptive_threshold_max: {final_best_result['parameters']['adaptive_threshold_max']}")
            self.logger.info(f"\n[BEST] BEST RESULTS:")
            kpis = final_best_result.get('kpis', {})
            self.logger.info(f"   Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
            self.logger.info(f"   Element F1: {kpis.get('element_f1', 0.0):.4f}")
            self.logger.info(f"   Quality Score: {kpis.get('quality_score', 0.0):.2f}")
        else:
            self.logger.warning("No successful test results found!")
        
        # Save final results
        self.save_results()
        
        # Save summary
        self.save_summary(final_best_result)
        
        self.stop_live_monitoring()
    
    def save_results(self):
        """Save results to JSON file (thread-safe)."""
        results_file = self.output_manager.get_data_path("parameter_tuning_results.json")
        
        # Thread-safe: Copy results list while holding lock
        with self.results_lock:
            results_copy = copy.deepcopy(self.results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json_dump_safe({
                'timestamp': datetime.now().isoformat(),
                'strategy': STRATEGY,
                'test_image': str(self.test_image),
                'total_tests': len(results_copy),
                'completed_tests': sum(1 for r in results_copy if 'success' in r),
                'failed_tests': sum(1 for r in results_copy if not r.get('success', True)),
                'results': results_copy
            }, f, indent=2, ensure_ascii=False)
        
        # Don't log every save (too verbose with worker pool)
        # self.logger.info(f"Results saved to: {results_file}")
    
    def save_summary(self, best_result: Optional[Dict[str, Any]]):
        """Save summary report (thread-safe)."""
        summary_file = self.output_manager.get_data_path("parameter_tuning_summary.json")
        
        # Thread-safe: Copy results list while holding lock
        with self.results_lock:
            results_copy = copy.deepcopy(self.results)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'strategy': STRATEGY,
            'test_image': str(self.test_image),
            'total_tests': len(results_copy),
            'successful_tests': sum(1 for r in results_copy if r.get('success')),
            'failed_tests': sum(1 for r in results_copy if not r.get('success')),
            'worker_pool_config': {
                'max_workers': self.max_workers,
                'mode': 'parallel'
            }
        }
        
        if best_result:
            summary['best_parameters'] = best_result['parameters']
            summary['best_kpis'] = best_result.get('kpis', {})
        
        # Sort results by Connection F1-Score (use results_copy, not self.results)
        successful_results = [r for r in results_copy if r.get('success') and r.get('kpis')]
        successful_results.sort(key=lambda x: x.get('kpis', {}).get('connection_f1', 0.0), reverse=True)
        
        summary['top_5_results'] = [
            {
                'parameters': r['parameters'],
                'connection_f1': r.get('kpis', {}).get('connection_f1', 0.0),
                'element_f1': r.get('kpis', {}).get('element_f1', 0.0),
                'quality_score': r.get('kpis', {}).get('quality_score', 0.0)
            }
            for r in successful_results[:5]
        ]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json_dump_safe(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Choose test image (simple is faster for parameter tuning)
    USE_SIMPLE = True  # Set to False to use complex image
    
    if USE_SIMPLE:
        test_image = TEST_IMAGE_SIMPLE
        test_truth = TEST_GROUND_TRUTH_SIMPLE
    else:
        test_image = TEST_IMAGE_COMPLEX
        test_truth = TEST_GROUND_TRUTH_COMPLEX
    
    runner = ParameterTuningRunner(test_image, test_truth, use_simple=USE_SIMPLE)
    
    print("=" * 80)
    print("PARAMETER TUNING - WORKER POOL MODE")
    print("=" * 80)
    total_combos = len(ADAPTIVE_THRESHOLD_FACTORS) * len(ADAPTIVE_THRESHOLD_MINS) * len(ADAPTIVE_THRESHOLD_MAXS)
    print(f"Total parameter combinations: {total_combos}")
    print(f"Worker Pool: {MAX_WORKERS} parallel workers")
    print(f"Estimated time: {total_combos * 5 / MAX_WORKERS:.1f}-{total_combos * 10 / MAX_WORKERS:.1f} minutes")
    print(f"  (Sequential: 3-6 hours, Parallel: {total_combos * 5 / MAX_WORKERS:.1f}-{total_combos * 10 / MAX_WORKERS:.1f} minutes)")
    print(f"Self-Correction: DISABLED (faster, no circuit breaker issues)")
    print(f"Exponential Backoff: ENABLED (handled by LLMClient)")
    print(f"Rate Limiting: {MAX_WORKERS} concurrent requests max")
    print("=" * 80)
    print()
    
    runner.setup_services()
    runner.start_live_monitoring()
    
    try:
        runner.run_parameter_tuning()
    except KeyboardInterrupt:
        runner.logger.info("\nParameter tuning interrupted by user")
        runner.stop_live_monitoring()
    except Exception as e:
        runner.logger.error(f"Fatal error: {e}", exc_info=True)
        runner.stop_live_monitoring()
        sys.exit(1)

