#!/usr/bin/env python3
"""
Automatisiertes Testcamp - Iteratives Testen mit Log-Auswertung.

Analysiert Testbilder automatisch, wertet Logs aus und iteriert bis gute Ergebnisse erreicht werden.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AutomatedTestCamp:
    """Automatisiertes Testcamp mit iterativem Testen."""
    
    def __init__(self):
        """Initialize test camp."""
        self.pipeline_coordinator = None
        self.config_service = None
        self.test_images: List[Path] = []
        self.results: List[Dict[str, Any]] = []
        self.iteration_count = 0
        self.max_iterations = 10
        self.target_quality_score = 80.0
        
        # Setup backend
        self._initialize_backend()
        
        # Find test images - start with simple_pids, then add uni page_2
        self._find_test_images(test_only_simple=False, test_uni_images=True, max_uni_images=2)  # Test simple_pids + uni page_2
    
    def _initialize_backend(self):
        """Initialize backend components."""
        try:
            from src.services.config_service import ConfigService
            from src.analyzer.ai.llm_client import LLMClient
            from src.analyzer.learning.knowledge_manager import KnowledgeManager
            from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
            
            gcp_project_id = os.getenv('GCP_PROJECT_ID')
            gcp_location = os.getenv('GCP_LOCATION', 'us-central1')
            
            if not gcp_project_id:
                raise ValueError("GCP_PROJECT_ID not set in .env file")
            
            # Config Service
            config_path = project_root / "config.yaml"
            self.config_service = ConfigService(config_path=config_path if config_path.exists() else None)
            config = self.config_service.get_config()
            
            # LLM Client
            self.llm_client = LLMClient(
                project_id=gcp_project_id,
                default_location=gcp_location,
                config=config.model_dump()
            )
            
            # Knowledge Manager
            element_type_list_path = self.config_service.get_path("element_type_list") or project_root / "element_type_list.json"
            learning_db_path = self.config_service.get_path("learning_db") or project_root / "learning_db.json"
            
            self.knowledge_manager = KnowledgeManager(
                element_type_list_path=str(element_type_list_path),
                learning_db_path=str(learning_db_path),
                llm_handler=self.llm_client,
                config=config.model_dump()
            )
            
            # Pipeline Coordinator
            self.pipeline_coordinator = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service
            )
            
            # Reset circuit breaker to ensure clean state for testing
            if hasattr(self.llm_client, 'retry_handler') and hasattr(self.llm_client.retry_handler, 'circuit_breaker'):
                self.llm_client.retry_handler.circuit_breaker.reset()
                logger.info("Circuit breaker reset to CLOSED state for testing")
            
            logger.info("Backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}", exc_info=True)
            raise
    
    def _find_test_images(self, test_only_simple: bool = True, test_uni_images: bool = False, max_uni_images: int = 4):
        """Find test images - prioritize simple_pids and uni page_2 specifically."""
        self.test_images = []  # Reset images list
        
        # First priority: simple_pids (simple test images)
        simple_pids_dir = project_root / "training_data" / "simple_pids"
        
        if simple_pids_dir.exists():
            logger.info(f"Searching for simple test images in: {simple_pids_dir}")
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                for img in simple_pids_dir.rglob(ext):
                    # Skip truth files, output files, and temp files
                    if any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp', 'correction', 'symbol']):
                        continue
                    if img not in self.test_images:
                        self.test_images.append(img)
                        logger.info(f"Found test image: {img.name}")
        
        # If test_only_simple is True, stop here
        if test_only_simple:
            if not self.test_images:
                logger.error("No simple_pids test images found!")
            else:
                logger.info(f"Testing only simple_pids: {len(self.test_images)} images")
            return
        
        # Add Uni page_2 specifically if test_uni_images is True
        if test_uni_images:
            # Try organized_tests/complex_pids first
            uni_dir = project_root / "training_data" / "organized_tests" / "complex_pids"
            if not uni_dir.exists():
                # Fallback to complex_pids
                uni_dir = project_root / "training_data" / "complex_pids"
            
            if uni_dir.exists():
                logger.info(f"Searching for Uni page_2 image in: {uni_dir}")
                # Specifically test page_2 (max_uni_images=2)
                target_page = max_uni_images if max_uni_images and max_uni_images > 0 else 2
                uni_path = uni_dir / f"page_{target_page}_original.png"
                if uni_path.exists() and uni_path not in self.test_images:
                    self.test_images.append(uni_path)
                    logger.info(f"Found Uni page_{target_page} image: {uni_path.name}")
                elif not uni_path.exists():
                    logger.warning(f"Uni page_{target_page} image not found at: {uni_path}")
        
        if not self.test_images:
            logger.error("No test images found!")
        else:
            logger.info(f"Total test images found: {len(self.test_images)}")
            for img in self.test_images:
                logger.info(f"  - {img.name}")
    
    def run_iterative_testing(self, max_iterations: int = 10, target_score: float = 80.0):
        """Run iterative testing until target score is reached or no improvement."""
        self.max_iterations = max_iterations
        self.target_quality_score = target_score
        
        logger.info(f"Starting iterative testing with {len(self.test_images)} images")
        logger.info(f"Target quality score: {target_score}, Max iterations: {max_iterations}")
        logger.info(f"Timeout per image: 3 minutes (180 seconds)")
        logger.info(f"NOTE: Will iterate until target score is reached or no improvement")
        
        best_score = 0.0
        iteration = 0
        consecutive_no_improvement = 0
        max_no_improvement = 2  # Stop if 2 consecutive iterations show no improvement
        
        # First run with simple_pids only
        logger.info("=" * 60)
        logger.info("Phase 1: Testing with simple_pids only")
        logger.info("=" * 60)
        
        while iteration < max_iterations:
            iteration += 1
            self.iteration_count = iteration
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")
            
            # Run tests on all images
            iteration_results = self._run_test_iteration(iteration)
            
            # Calculate average score
            scores = [r.get('quality_score', 0.0) for r in iteration_results if r]
            if scores:
                avg_score = sum(scores) / len(scores)
                best_score = max(best_score, avg_score)
                
                logger.info(f"Iteration {iteration} - Average Score: {avg_score:.2f}")
                logger.info(f"Best Score So Far: {best_score:.2f}")
                
                # Check for improvement
                if avg_score > best_score:
                    improvement = avg_score - best_score
                    logger.info(f"Score improved by {improvement:.2f} points!")
                    consecutive_no_improvement = 0
                    best_score = avg_score
                else:
                    consecutive_no_improvement += 1
                    logger.info(f"No improvement for {consecutive_no_improvement} consecutive iteration(s)")
                    
                    # Stop if no improvement for max_no_improvement iterations
                    if consecutive_no_improvement >= max_no_improvement:
                        logger.info(f"No improvement for {max_no_improvement} consecutive iterations. Stopping.")
                        break
                
                # Check if target reached
                if avg_score >= target_score:
                    logger.info(f"Target score reached! Average: {avg_score:.2f} >= Target: {target_score}")
                    # Target reached, break
                    break
            else:
                logger.warning(f"No valid results in iteration {iteration}")
            
            # Save intermediate results
            self._save_results(iteration)
            
            # Brief pause between iterations
            if iteration < max_iterations:
                time.sleep(2)
        
        # Final summary
        self._print_summary(best_score)
        
        return {
            'iterations': iteration,
            'best_score': best_score,
            'target_reached': best_score >= target_score,
            'results': self.results
        }
    
    def _run_test_iteration(self, iteration: int) -> List[Dict[str, Any]]:
        """Run a single test iteration."""
        iteration_results = []
        
        for img_idx, image_path in enumerate(self.test_images):
            logger.info(f"\nProcessing image {img_idx + 1}/{len(self.test_images)}: {image_path.name}")
            
            try:
                # Reset circuit breaker before each image
                if hasattr(self.llm_client, 'retry_handler') and hasattr(self.llm_client.retry_handler, 'circuit_breaker'):
                    self.llm_client.retry_handler.circuit_breaker.reset()
                    logger.debug(f"Circuit breaker reset for {image_path.name}")
                
                # Progress callback
                class TestProgressCallback:
                    def update_progress(self, value: int, message: str):
                        if value % 25 == 0:
                            logger.info(f"  Progress: {message} ({value}%)")
                    
                    def update_status_label(self, text: str):
                        pass
                    
                    def report_truth_mode(self, active: bool):
                        pass
                    
                    def report_correction(self, correction_text: str):
                        pass
                
                self.pipeline_coordinator.progress_callback = TestProgressCallback()
                
                # Run analysis with timeout (3 minutes = 180 seconds)
                start_time = time.time()
                timeout_seconds = 180  # 3 minutes timeout per image
                
                try:
                    # Use threading to implement timeout
                    import threading
                    result_container = {'result': None, 'exception': None}
                    
                    def run_analysis():
                        try:
                            result_container['result'] = self.pipeline_coordinator.process(
                                image_path=str(image_path),
                                output_dir=None
                            )
                        except Exception as e:
                            result_container['exception'] = e
                    
                    analysis_thread = threading.Thread(target=run_analysis, daemon=True)
                    analysis_thread.start()
                    analysis_thread.join(timeout=timeout_seconds)
                    
                    if analysis_thread.is_alive():
                        logger.error(f"Analysis timeout after {timeout_seconds} seconds for {image_path.name}")
                        logger.error("Skipping this image and continuing with next iteration")
                        iteration_results.append(None)
                        continue
                    
                    if result_container['exception']:
                        raise result_container['exception']
                    
                    result = result_container['result']
                    duration = time.time() - start_time
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                    logger.warning("Continuing with next image...")
                    iteration_results.append(None)
                    continue
                
                # Extract results (AnalysisResult is a Pydantic BaseModel with direct attributes)
                quality_score = 0.0
                elements_count = 0
                connections_count = 0
                result_data = None
                
                try:
                    quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
                    elements_count = len(result.elements) if hasattr(result, 'elements') and result.elements else 0
                    connections_count = len(result.connections) if hasattr(result, 'connections') and result.connections else 0
                    
                    # Handle KPIs - can be Dict or Pydantic Model
                    kpis_data = {}
                    if hasattr(result, 'kpis') and result.kpis:
                        if hasattr(result.kpis, 'model_dump'):
                            kpis_data = result.kpis.model_dump()
                        elif isinstance(result.kpis, dict):
                            kpis_data = result.kpis
                    
                    result_data = {
                        'iteration': iteration,
                        'image': image_path.name,
                        'image_path': str(image_path),
                        'quality_score': quality_score,
                        'elements_count': elements_count,
                        'connections_count': connections_count,
                        'duration_seconds': duration,
                        'timestamp': datetime.now().isoformat(),
                        'kpis': kpis_data
                    }
                except Exception as e:
                    logger.error(f"  Error extracting results: {e}", exc_info=True)
                    # Try to get basic info even if extraction fails
                    try:
                        quality_score = getattr(result, 'quality_score', 0.0)
                        elements_count = len(getattr(result, 'elements', []))
                        connections_count = len(getattr(result, 'connections', []))
                        result_data = {
                            'iteration': iteration,
                            'image': image_path.name,
                            'image_path': str(image_path),
                            'quality_score': quality_score,
                            'elements_count': elements_count,
                            'connections_count': connections_count,
                            'duration_seconds': duration,
                            'timestamp': datetime.now().isoformat(),
                            'kpis': {}
                        }
                    except Exception as e2:
                        logger.error(f"  Failed to extract basic results: {e2}", exc_info=True)
                        result_data = None
                
                if result_data:
                    iteration_results.append(result_data)
                    self.results.append(result_data)
                    logger.info(f"  Score: {quality_score:.2f}, Elements: {elements_count}, Connections: {connections_count}, Duration: {duration:.1f}s")
                else:
                    logger.warning(f"  No valid result data extracted for {image_path.name}")
                    iteration_results.append(None)
                
            except Exception as e:
                logger.error(f"  Error processing {image_path.name}: {e}", exc_info=True)
                iteration_results.append(None)
        
        return iteration_results
    
    def _save_results(self, iteration: int):
        """Save test results to JSON file."""
        results_file = project_root / "testcamp_results.json"
        
        results_data = {
            'test_camp_info': {
                'iterations_completed': iteration,
                'max_iterations': self.max_iterations,
                'target_score': self.target_quality_score,
                'test_images': [str(img) for img in self.test_images]
            },
            'results': self.results,
            'summary': self._calculate_summary()
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.results:
            return {}
        
        scores = [r.get('quality_score', 0.0) for r in self.results if r]
        elements = [r.get('elements_count', 0) for r in self.results if r]
        connections = [r.get('connections_count', 0) for r in self.results if r]
        durations = [r.get('duration_seconds', 0.0) for r in self.results if r]
        
        return {
            'total_runs': len(self.results),
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'best_score': max(scores) if scores else 0.0,
            'worst_score': min(scores) if scores else 0.0,
            'average_elements': sum(elements) / len(elements) if elements else 0.0,
            'average_connections': sum(connections) / len(connections) if connections else 0.0,
            'average_duration': sum(durations) / len(durations) if durations else 0.0,
            'total_duration': sum(durations) if durations else 0.0
        }
    
    def _print_summary(self, best_score: float):
        """Print final summary."""
        summary = self._calculate_summary()
        
        print("\n" + "=" * 60)
        print("Testcamp Summary")
        print("=" * 60)
        print(f"Total Iterations: {self.iteration_count}")
        print(f"Total Runs: {summary.get('total_runs', 0)}")
        print(f"Best Score: {best_score:.2f}")
        print(f"Average Score: {summary.get('average_score', 0.0):.2f}")
        print(f"Average Elements: {summary.get('average_elements', 0.0):.1f}")
        print(f"Average Connections: {summary.get('average_connections', 0.0):.1f}")
        print(f"Total Duration: {summary.get('total_duration', 0.0):.1f}s")
        print(f"Target Reached: {'Yes' if best_score >= self.target_quality_score else 'No'}")
        print("=" * 60)
        
        # Results saved to file
        results_file = project_root / "testcamp_results.json"
        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main function."""
    print("=" * 60)
    print("Automatisiertes Testcamp")
    print("=" * 60)
    print()
    
    try:
        # Parse arguments
        max_iterations = 10
        target_score = 80.0
        
        if len(sys.argv) > 1:
            max_iterations = int(sys.argv[1])
        if len(sys.argv) > 2:
            target_score = float(sys.argv[2])
        
        # Create and run test camp
        testcamp = AutomatedTestCamp()
        
        if not testcamp.test_images:
            print("ERROR: No test images found!")
            print("Please ensure test images exist in training_data/")
            return 1
        
        print(f"Found {len(testcamp.test_images)} test images:")
        for img in testcamp.test_images:
            print(f"  - {img}")
        print()
        
        # Run iterative testing
        results = testcamp.run_iterative_testing(
            max_iterations=max_iterations,
            target_score=target_score
        )
        
        return 0 if results['target_reached'] else 1
        
    except KeyboardInterrupt:
        print("\nTestcamp interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error in testcamp: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

