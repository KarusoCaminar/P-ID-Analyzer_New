"""
API Rate Limit Test - Testet maximale API Call Rate.

Dieses Skript testet, wie viele API Calls pro Minute möglich sind,
bevor Rate Limits (429) auftreten. Es hilft dabei, die optimale
Rate für DSQ zu finden.
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Projekt-Root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.ai.dsq_optimizer import get_dsq_optimizer, reset_dsq_optimizer

# Load .env
try:
    from src.utils.env_loader import load_env_automatically
    load_env_automatically()
except:
    from dotenv import load_dotenv
    load_dotenv()

# Setup logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Configuration
TEST_REGIONS = [
    "us-central1",      # Aktuell konfiguriert
    "europe-west3",     # Frankfurt (näher)
    "europe-west4",     # Netherlands (Alternative)
]
TEST_MODELS = [
    "Google Gemini 2.5 Flash",  # Schnelleres Modell für Rate-Test
    "Google Gemini 2.5 Pro",    # Pro-Modell prüfen
]

# Rate Test Configuration
MAX_WORKERS_RANGE = [5, 10, 15, 20]  # Test verschiedene Worker-Anzahlen
TARGET_REQUESTS = 100  # Anzahl Requests für Test
TEST_DURATION_SECONDS = 60  # Test-Dauer in Sekunden


class RateLimitTester:
    """Test API rate limits for different configurations."""
    
    def __init__(self, region: str, model_name: str):
        """Initialize rate limit tester."""
        self.region = region
        self.model_name = model_name
        self.config_service = ConfigService()
        self.config = self.config_service.get_raw_config()
        self.llm_client = None
        self.results = []
    
    def setup_llm_client(self):
        """Setup LLM client for region."""
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("GCP_PROJECT_ID not set!")
        
        logger.info(f"Initializing LLM Client for region: {self.region}")
        self.llm_client = LLMClient(
            project_id=project_id,
            default_location=self.region,
            config=self.config
        )
        logger.info(f"[OK] LLM Client initialized for {self.region}")
    
    def test_model_availability(self) -> bool:
        """Test if model is available in region."""
        try:
            # Try to get model info
            model_config = self.config.get('models', {}).get(self.model_name, {})
            if not model_config:
                logger.error(f"Model {self.model_name} not found in config")
                return False
            
            # Try a simple API call
            test_prompt = "Return JSON: {\"test\": true}"
            result = self.llm_client.call_llm(
                model_info=model_config,
                system_prompt="You are a test assistant.",
                user_prompt=test_prompt,
                use_cache=False,  # Don't cache test calls
                timeout=30
            )
            
            if result:
                logger.info(f"[OK] Model {self.model_name} is available in {self.region}")
                return True
            else:
                logger.warning(f"[WARNING] Model {self.model_name} returned None in {self.region}")
                return False
                
        except Exception as e:
            logger.error(f"[FAIL] Model {self.model_name} not available in {self.region}: {e}")
            return False
    
    def test_rate_limit(self, max_workers: int, target_rpm: int) -> Dict[str, Any]:
        """
        Test rate limit for given configuration.
        
        Args:
            max_workers: Number of parallel workers
            target_rpm: Target requests per minute
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"RATE LIMIT TEST: {self.region} - {self.model_name}")
        logger.info(f"Max Workers: {max_workers}, Target RPM: {target_rpm}")
        logger.info(f"{'=' * 80}\n")
        
        # Reset DSQ optimizer for clean test
        reset_dsq_optimizer()
        dsq_optimizer = get_dsq_optimizer(
            initial_requests_per_minute=target_rpm,
            max_requests_per_minute=target_rpm * 2
        )
        
        # Get model config
        model_config = self.config.get('models', {}).get(self.model_name, {})
        if not model_config:
            return {
                'success': False,
                'error': f"Model {self.model_name} not found"
            }
        
        # Test parameters
        test_prompt = '{"elements": [], "connections": []}'
        system_prompt = "You are a test assistant. Return JSON only."
        
        # Track metrics
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        rate_limit_errors = 0
        other_errors = 0
        request_times = []
        
        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single API request."""
            request_start = time.time()
            try:
                result = self.llm_client.call_llm(
                    model_info=model_config,
                    system_prompt=system_prompt,
                    user_prompt=test_prompt,
                    use_cache=False,  # Don't cache test calls
                    timeout=30
                )
                request_duration = time.time() - request_start
                
                return {
                    'request_id': request_id,
                    'success': result is not None,
                    'duration': request_duration,
                    'error': None,
                    'rate_limit': False
                }
            except Exception as e:
                request_duration = time.time() - request_start
                error_msg = str(e).lower()
                is_rate_limit = any(term in error_msg for term in [
                    '429', 'rate limit', 'resource exhausted', 'quota'
                ])
                
                return {
                    'request_id': request_id,
                    'success': False,
                    'duration': request_duration,
                    'error': str(e),
                    'rate_limit': is_rate_limit
                }
        
        # Execute test with ThreadPoolExecutor
        logger.info(f"Starting rate test with {max_workers} workers...")
        test_start = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit requests
                futures = {
                    executor.submit(make_request, i): i
                    for i in range(TARGET_REQUESTS)
                }
                
                # Collect results
                for future in as_completed(futures):
                    request_id = futures[future]
                    try:
                        result = future.result()
                        request_times.append(result['duration'])
                        
                        if result['success']:
                            completed_requests += 1
                        else:
                            failed_requests += 1
                            if result['rate_limit']:
                                rate_limit_errors += 1
                            else:
                                other_errors += 1
                        
                        # Log progress
                        if (completed_requests + failed_requests) % 10 == 0:
                            elapsed = time.time() - test_start
                            current_rpm = (completed_requests + failed_requests) / (elapsed / 60.0)
                            logger.info(
                                f"Progress: {completed_requests + failed_requests}/{TARGET_REQUESTS} "
                                f"(RPM: {current_rpm:.1f}, Rate Limits: {rate_limit_errors})"
                            )
                        
                        # Stop if too many rate limits
                        if rate_limit_errors > 10:
                            logger.warning(f"Too many rate limits ({rate_limit_errors}) - stopping test")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")
                        failed_requests += 1
                        other_errors += 1
        
        except KeyboardInterrupt:
            logger.warning("Test interrupted by user")
        
        test_duration = time.time() - test_start
        actual_rpm = (completed_requests + failed_requests) / (test_duration / 60.0) if test_duration > 0 else 0
        
        # Get DSQ optimizer status
        dsq_status = dsq_optimizer.get_status()
        
        # Calculate results
        result = {
            'region': self.region,
            'model': self.model_name,
            'max_workers': max_workers,
            'target_rpm': target_rpm,
            'test_duration': test_duration,
            'total_requests': completed_requests + failed_requests,
            'completed_requests': completed_requests,
            'failed_requests': failed_requests,
            'rate_limit_errors': rate_limit_errors,
            'other_errors': other_errors,
            'actual_rpm': actual_rpm,
            'success_rate': completed_requests / (completed_requests + failed_requests) if (completed_requests + failed_requests) > 0 else 0,
            'rate_limit_rate': rate_limit_errors / (completed_requests + failed_requests) if (completed_requests + failed_requests) > 0 else 0,
            'avg_request_duration': sum(request_times) / len(request_times) if request_times else 0,
            'dsq_status': dsq_status,
            'success': rate_limit_errors < 5  # Test successful if < 5 rate limits
        }
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TEST RESULTS:")
        logger.info(f"  Completed: {completed_requests}/{completed_requests + failed_requests}")
        logger.info(f"  Rate Limits: {rate_limit_errors}")
        logger.info(f"  Actual RPM: {actual_rpm:.1f}")
        logger.info(f"  Success Rate: {result['success_rate']:.2%}")
        logger.info(f"  Rate Limit Rate: {result['rate_limit_rate']:.2%}")
        logger.info(f"{'=' * 80}\n")
        
        return result


def main():
    """Main function to run rate limit tests."""
    print("=" * 80)
    print("API RATE LIMIT TEST")
    print("=" * 80)
    print("This script tests the maximum API call rate for different regions and configurations.")
    print("It will help you find the optimal rate for DSQ optimization.")
    print("=" * 80)
    print()
    
    # Get current region from environment
    current_region = os.getenv("GCP_LOCATION", "us-central1")
    print(f"Current Region (from .env): {current_region}")
    print()
    
    # Test results
    all_results = []
    
    # Test each region and model combination
    for region in TEST_REGIONS:
        for model_name in TEST_MODELS:
            try:
                tester = RateLimitTester(region, model_name)
                tester.setup_llm_client()
                
                # Test model availability
                if not tester.test_model_availability():
                    logger.warning(f"Skipping {model_name} in {region} - not available")
                    continue
                
                # Test different worker configurations
                for max_workers in MAX_WORKERS_RANGE:
                    # Calculate target RPM based on workers
                    # Assume ~1 request per second per worker = 60 RPM per worker
                    target_rpm = max_workers * 60
                    
                    # Test rate limit
                    result = tester.test_rate_limit(max_workers, target_rpm)
                    all_results.append(result)
                    
                    # Wait between tests to avoid interference
                    time.sleep(5)
                    
                    # Stop if we found a good configuration
                    if result['success'] and result['rate_limit_rate'] < 0.05:
                        logger.info(f"[OK] Found good configuration: {region} - {max_workers} workers - {target_rpm} RPM")
            
            except Exception as e:
                logger.error(f"Error testing {region} - {model_name}: {e}", exc_info=True)
                continue
    
    # Save results
    output_dir = project_root / "outputs" / "rate_limit_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"rate_limit_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_config': {
                'regions': TEST_REGIONS,
                'models': TEST_MODELS,
                'max_workers_range': MAX_WORKERS_RANGE,
                'target_requests': TARGET_REQUESTS
            },
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[OK] Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    # Find best configuration
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x.get('actual_rpm', 0))
        print(f"\n[BEST] Best Configuration:")
        print(f"  Region: {best_result['region']}")
        print(f"  Model: {best_result['model']}")
        print(f"  Max Workers: {best_result['max_workers']}")
        print(f"  Actual RPM: {best_result['actual_rpm']:.1f}")
        print(f"  Rate Limit Rate: {best_result['rate_limit_rate']:.2%}")
        print(f"  Success Rate: {best_result['success_rate']:.2%}")
    else:
        print("\n[WARNING] No successful configurations found!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

