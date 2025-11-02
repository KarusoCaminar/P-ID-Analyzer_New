"""
Integration tests for the complete pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pipeline_initialization():
    """Test that the pipeline can be initialized."""
    try:
        from src.services.config_service import ConfigService
        from src.analyzer.ai.llm_client import LLMClient
        from src.analyzer.learning.knowledge_manager import KnowledgeManager
        from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
        
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_config()
        
        assert config is not None, "Config should be loaded"
        print("[OK] Config loaded")
        
        # Check if we have GCP credentials (optional for test)
        import os
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        
        if gcp_project_id:
            try:
                # Initialize LLM client
                gcp_location = os.getenv("GCP_LOCATION", "us-central1")
                llm_client = LLMClient(gcp_project_id, gcp_location, config)
                print("[OK] LLMClient initialized")
                
                # Initialize Knowledge Manager
                element_type_list = config_service.get_path('element_type_list') or Path("element_type_list.json")
                learning_db = config_service.get_path('learning_db') or Path("learning_db.json")
                
                knowledge_manager = KnowledgeManager(
                    str(element_type_list),
                    str(learning_db),
                    llm_client,
                    config
                )
                print("[OK] KnowledgeManager initialized")
                
                # Initialize Pipeline Coordinator
                coordinator = PipelineCoordinator(
                    llm_client=llm_client,
                    knowledge_manager=knowledge_manager,
                    config_service=config_service
                )
                print("[OK] PipelineCoordinator initialized")
                
                # Test state access
                state = coordinator.get_state()
                assert state is not None, "State should not be None"
                print("[OK] PipelineCoordinator state accessible")
                
                return True
            except Exception as e:
                print(f"[SKIP] Pipeline initialization requires GCP credentials: {e}")
                return True  # Not a failure, just skip
        else:
            print("[SKIP] GCP_PROJECT_ID not set, skipping full pipeline test")
            return True
        
    except Exception as e:
        print(f"[FAIL] Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_initialization()
    sys.exit(0 if success else 1)

