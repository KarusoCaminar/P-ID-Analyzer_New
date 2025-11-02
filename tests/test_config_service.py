"""
Test ConfigService functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_service():
    """Test ConfigService basic functionality."""
    try:
        from src.services.config_service import ConfigService
        
        config_service = ConfigService()
        config = config_service.get_config()
        
        assert config is not None, "Config should not be None"
        print("[OK] ConfigService initialized and config loaded")
        
        # Test path access
        path = config_service.get_path('element_type_list')
        assert path is not None, "Path should not be None"
        print("[OK] ConfigService path access works")
        
        # Test logic parameters
        logic_params = config_service.get_logic_parameters()
        assert isinstance(logic_params, dict), "Logic parameters should be dict"
        print("[OK] ConfigService logic parameters access works")
        
        return True
    except Exception as e:
        print(f"[FAIL] ConfigService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_service()
    sys.exit(0 if success else 1)


