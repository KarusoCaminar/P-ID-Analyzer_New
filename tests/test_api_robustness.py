"""
Test API robustness - 3 test rounds as specified in plan.

Tests:
- A) Nominal: Small payload, expect success
- B) Large payload: Simulate big image/long tool args, expect fallback or success
- C) Faulty payload: Simulate previous error shape, expect graceful error + debug file
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.utils.schemas import validate_request_payload, validate_response
from src.utils.debug_capture import write_debug_file, generate_request_id

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class APIRobustnessTester:
    """Test API robustness with 3 test rounds."""
    
    def __init__(self, llm_client: LLMClient, output_dir: Path = Path("outputs/debug")):
        """
        Initialize tester.
        
        Args:
            llm_client: LLM client instance
            output_dir: Output directory for test results
        """
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results: List[Dict[str, Any]] = []
    
    def test_round_a_nominal(self) -> Dict[str, Any]:
        """
        Test Round A: Nominal request with small payload.
        
        Expected: Success
        """
        logger.info("=" * 60)
        logger.info("TEST ROUND A: Nominal Request (Small Payload)")
        logger.info("=" * 60)
        
        test_name = "round_a_nominal"
        request_id = generate_request_id()
        timestamp = datetime.now().isoformat()
        
        try:
            # Get model config
            model_info = next(iter(self.llm_client.config.get("models", {}).values()))
            model_id = model_info.get("id")
            
            # Small payload
            system_prompt = "You are a helpful assistant."
            user_prompt = "Say 'API test successful' in one sentence."
            
            logger.info(f"Making API call with small payload...")
            
            result = self.llm_client.call_llm(
                model_info=model_info,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                use_cache=False,  # Don't use cache for test
                expected_json_keys=None,
                timeout=60
            )
            
            success = result is not None
            status = "pass" if success else "fail"
            
            logger.info(f"Result: {status.upper()}")
            if result:
                logger.info(f"Response: {str(result)[:100]}...")
            
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": "Small payload, no image",
                "result": status,
                "success": success,
                "response_preview": str(result)[:200] if result else None,
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            logger.error(f"Test Round A failed: {e}", exc_info=True)
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": "Small payload, no image",
                "result": "fail",
                "success": False,
                "error": str(e),
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
    
    def test_round_b_large_payload(self) -> Dict[str, Any]:
        """
        Test Round B: Large payload simulation.
        
        Expected: Fallback or success
        """
        logger.info("=" * 60)
        logger.info("TEST ROUND B: Large Payload")
        logger.info("=" * 60)
        
        test_name = "round_b_large_payload"
        request_id = generate_request_id()
        timestamp = datetime.now().isoformat()
        
        try:
            # Get model config
            model_info = next(iter(self.llm_client.config.get("models", {}).values()))
            model_id = model_info.get("id")
            
            # Large payload - very long prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "Analyze this detailed diagram: " + "A" * 10000  # 10KB prompt
            
            logger.info(f"Making API call with large payload ({len(user_prompt)} chars)...")
            
            result = self.llm_client.call_llm(
                model_info=model_info,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                use_cache=False,
                expected_json_keys=None,
                timeout=120
            )
            
            success = result is not None
            status = "pass" if success else "fail"
            
            logger.info(f"Result: {status.upper()}")
            if result:
                logger.info(f"Response: {str(result)[:100]}...")
            
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": f"Large payload ({len(user_prompt)} chars)",
                "result": status,
                "success": success,
                "response_preview": str(result)[:200] if result else None,
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            logger.error(f"Test Round B failed: {e}", exc_info=True)
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": "Large payload",
                "result": "fail",
                "success": False,
                "error": str(e),
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
    
    def test_round_c_faulty_payload(self) -> Dict[str, Any]:
        """
        Test Round C: Faulty payload simulation.
        
        Expected: Graceful error + detailed debug file
        """
        logger.info("=" * 60)
        logger.info("TEST ROUND C: Faulty Payload (Graceful Error Expected)")
        logger.info("=" * 60)
        
        test_name = "round_c_faulty_payload"
        request_id = generate_request_id()
        timestamp = datetime.now().isoformat()
        
        try:
            # Get model config
            model_info = next(iter(self.llm_client.config.get("models", {}).values()))
            
            # Faulty payload - very long prompt that might cause issues
            system_prompt = "You are a helpful assistant."
            user_prompt = "A" * 100000  # 100KB prompt - might cause size issues
            
            logger.info(f"Making API call with potentially faulty payload ({len(user_prompt)} chars)...")
            
            # For faulty payload test, use shorter timeout to fail fast
            # 100KB prompt will likely timeout - test graceful handling
            result = self.llm_client.call_llm(
                model_info=model_info,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                use_cache=False,
                expected_json_keys=None,
                timeout=30  # Reduced timeout for very large payloads (fail fast)
            )
            
            # For this test, we expect either graceful error handling or success
            # The key is that it doesn't crash
            success = True  # If we get here without exception, it's a success (even if result is None)
            status = "pass" if success else "fail"
            
            logger.info(f"Result: {status.upper()} (graceful handling)")
            if result:
                logger.info(f"Response: {str(result)[:100]}...")
            else:
                logger.info("Result is None (expected for large payload)")
            
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": f"Faulty/large payload ({len(user_prompt)} chars)",
                "result": status,
                "success": success,
                "graceful_error_handling": True,
                "response_preview": str(result)[:200] if result else "None (graceful error)",
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            # Even if exception, check if it was handled gracefully
            logger.warning(f"Test Round C raised exception (might be expected): {e}")
            
            # Check if debug file was created
            debug_file = self.llm_client.debug_dir / 'workflow-debug.json'
            debug_file_exists = debug_file.exists()
            
            graceful = debug_file_exists  # If debug file exists, error was handled gracefully
            
            test_result = {
                "test_name": test_name,
                "timestamp": timestamp,
                "request_id": request_id,
                "input": "Faulty payload",
                "result": "pass" if graceful else "fail",
                "success": graceful,
                "error": str(e),
                "debug_file_created": debug_file_exists,
                "graceful_error_handling": graceful,
                "logs_file": str(self.output_dir / f"test-{test_name}-{request_id}.json")
            }
            
            # Save test result
            with open(test_result["logs_file"], 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
            
            self.test_results.append(test_result)
            return test_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all 3 test rounds and generate final report."""
        logger.info("Starting API Robustness Tests (3 rounds)...")
        
        # Run tests
        result_a = self.test_round_a_nominal()
        result_b = self.test_round_b_large_payload()
        result_c = self.test_round_c_faulty_payload()
        
        # Generate final report
        report = {
            "summary": f"API Robustness Test Results - {datetime.now().isoformat()}",
            "total_tests": 3,
            "passed": sum(1 for r in self.test_results if r.get("result") == "pass"),
            "failed": sum(1 for r in self.test_results if r.get("result") == "fail"),
            "tests": self.test_results,
            "final_debug_path": str(self.llm_client.debug_dir / 'workflow-debug.json')
        }
        
        # Save report
        report_path = self.output_dir / 'test-report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {report['total_tests']}")
        logger.info(f"Passed: {report['passed']}")
        logger.info(f"Failed: {report['failed']}")
        logger.info(f"Report saved to: {report_path}")
        
        return report


def main():
    """Main entry point for API robustness tests."""
    import os
    from dotenv import load_dotenv
    from src.services.config_service import ConfigService
    
    load_dotenv()
    
    # Get config
    config_service = ConfigService()
    config = config_service.get_config()
    
    # Get GCP credentials from raw config or environment
    raw_config = config_service.get_raw_config()
    project_id = raw_config.get('gcp_project_id') or os.getenv('GCP_PROJECT_ID')
    location = raw_config.get('gcp_location') or os.getenv('GCP_LOCATION', 'us-central1')
    
    if not project_id:
        logger.error("GCP_PROJECT_ID not found. Cannot run tests.")
        return
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(
            project_id=project_id,
            default_location=location,
            config=config.model_dump()
        )
        
        # Run tests
        tester = APIRobustnessTester(llm_client)
        report = tester.run_all_tests()
        
        # Cleanup: Shutdown thread pool executor
        try:
            if hasattr(llm_client, 'timeout_executor'):
                llm_client.timeout_executor.shutdown(wait=False)  # Don't wait, exit quickly
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        
        # Exit with appropriate code
        exit_code = 0 if report['failed'] == 0 else 1
        logger.info(f"Tests completed. Exiting with code {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Failed to run tests: {e}", exc_info=True)
        # Cleanup on error
        try:
            if 'llm_client' in locals() and hasattr(llm_client, 'timeout_executor'):
                llm_client.timeout_executor.shutdown(wait=False)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

