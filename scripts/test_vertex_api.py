#!/usr/bin/env python3
"""Quick test for Vertex AI API connection."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test Vertex AI API connection."""
    logger.info("="*60)
    logger.info("Vertex AI API Connection Test")
    logger.info("="*60)
    
    # 1. Check GCP_PROJECT_ID
    gcp_project_id = os.getenv('GCP_PROJECT_ID')
    gcp_location = os.getenv('GCP_LOCATION', 'us-central1')
    
    logger.info(f"\n1. Environment Variables:")
    logger.info(f"   GCP_PROJECT_ID: {gcp_project_id if gcp_project_id else '❌ NOT SET'}")
    logger.info(f"   GCP_LOCATION: {gcp_location}")
    
    if not gcp_project_id:
        logger.error("\n❌ ERROR: GCP_PROJECT_ID not set in .env file!")
        return
    
    # 2. Try to initialize LLM Client
    try:
        from src.services.config_service import ConfigService
        from src.analyzer.ai.llm_client import LLMClient
        
        logger.info(f"\n2. Initializing LLM Client...")
        config_path = project_root / "config.yaml"
        config_service = ConfigService(config_path=config_path if config_path.exists() else None)
        config = config_service.get_config()
        
        llm_client = LLMClient(
            project_id=gcp_project_id,
            default_location=gcp_location,
            config=config.model_dump()
        )
        
        logger.info("   ✅ LLM Client initialized")
        
        # 3. Check Circuit Breaker
        logger.info(f"\n3. Circuit Breaker Status:")
        cb = llm_client.retry_handler.circuit_breaker
        logger.info(f"   State: {cb.get_state()}")
        logger.info(f"   Failure Count: {cb.failure_count}")
        logger.info(f"   Threshold: {cb.failure_threshold}")
        logger.info(f"   Recovery Timeout: {cb.recovery_timeout}s")
        
        if cb.get_state() != "closed":
            logger.warning(f"   ⚠️  Circuit Breaker is {cb.get_state()} - resetting...")
            cb.reset()
            logger.info(f"   ✅ Circuit Breaker reset to CLOSED")
        
        # 4. Check available models
        logger.info(f"\n4. Available Gemini Models:")
        if llm_client.gemini_clients:
            for model_id in llm_client.gemini_clients.keys():
                logger.info(f"   ✅ {model_id}")
        else:
            logger.warning("   ⚠️  No models loaded!")
        
        # 5. Test API call
        logger.info(f"\n5. Testing API Call...")
        test_model_id = 'gemini-2.5-flash'
        
        if test_model_id not in llm_client.gemini_clients:
            logger.error(f"   ❌ Model {test_model_id} not available")
            logger.info(f"   Available models: {list(llm_client.gemini_clients.keys())}")
            return
        
        model = llm_client.gemini_clients[test_model_id]
        logger.info(f"   Using model: {test_model_id}")
        
        try:
            # Simple test call
            logger.info("   Making test API call...")
            response = model.generate_content("Say 'API test successful' in one sentence.")
            
            if hasattr(response, 'text'):
                logger.info(f"   ✅ API Response: {response.text[:100]}")
                logger.info(f"\n✅ SUCCESS: Vertex AI API is working!")
                
                # Record success
                llm_client.retry_handler.record_success()
                logger.info(f"   Circuit Breaker state after success: {cb.get_state()}")
                
            else:
                logger.error(f"   ❌ Response has no 'text' attribute: {response}")
                
        except Exception as e:
            logger.error(f"   ❌ API Call Failed:")
            logger.error(f"      Type: {type(e).__name__}")
            logger.error(f"      Message: {str(e)}")
            
            # Record failure to see what happens
            llm_client.retry_handler.record_failure(e)
            logger.info(f"   Circuit Breaker state after failure: {cb.get_state()}")
            logger.info(f"   Failure count: {cb.failure_count}")
            
            # Check error details
            import traceback
            logger.debug(f"\nFull traceback:\n{traceback.format_exc()}")
            
            return
        
    except ImportError as e:
        logger.error(f"\n❌ Import Error: {e}")
        logger.error("   Make sure all dependencies are installed")
        return
    except Exception as e:
        logger.error(f"\n❌ Unexpected Error: {e}", exc_info=True)
        return
    
    logger.info(f"\n" + "="*60)
    logger.info("Test Complete")
    logger.info("="*60)

if __name__ == "__main__":
    main()



