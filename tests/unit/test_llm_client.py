"""
Unit tests for LLMClient with mocking (no real API calls).
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import TimeoutError as FutureTimeoutError

from src.analyzer.ai.llm_client import LLMClient


@pytest.fixture
def mock_config():
    """Create mock config."""
    return {
        'paths': {
            'llm_cache_dir': '.test_cache',
            'debug_dir': 'outputs/debug'
        },
        'logic_parameters': {
            'llm_disk_cache_size_gb': 0.1,
            'llm_memory_cache_size': 10,
            'llm_default_timeout': 60,
            'llm_max_retries': 3,
            'circuit_breaker_failure_threshold': 5,
            'circuit_breaker_recovery_timeout': 60
        },
        'models': {
            'test_model': {
                'id': 'gemini-2.5-flash',
                'access_method': 'gemini'
            }
        }
    }


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLLMClient:
    """Tests for LLMClient."""
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    def test_llm_client_initialization(self, mock_vertexai, mock_config, temp_cache_dir):
        """Test LLMClient initialization."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        with patch('src.analyzer.ai.llm_client.GenerativeModel'):
            client = LLMClient(
                project_id="test-project",
                default_location="us-central1",
                config=mock_config
            )
            
            assert client.project_id == "test-project"
            assert client.default_location == "us-central1"
            assert client.disk_cache is not None
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    def test_generate_cache_key(self, mock_vertexai, mock_config, temp_cache_dir):
        """Test cache key generation."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        with patch('src.analyzer.ai.llm_client.GenerativeModel'):
            client = LLMClient(
                project_id="test-project",
                default_location="us-central1",
                config=mock_config
            )
            
            model_info = {'id': 'test-model'}
            # Test without image_path
            key1 = client._generate_cache_key(model_info, "prompt1", "user1", None)
            key2 = client._generate_cache_key(model_info, "prompt1", "user1", None)
            key3 = client._generate_cache_key(model_info, "prompt2", "user1", None)
            
            # Same inputs should produce same key
            assert key1 == key2
            
            # Different inputs should produce different keys
            assert key1 != key3
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    def test_sanitize_payload(self, mock_vertexai, mock_config, temp_cache_dir):
        """Test payload sanitization."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        with patch('src.analyzer.ai.llm_client.GenerativeModel'):
            client = LLMClient(
                project_id="test-project",
                default_location="us-central1",
                config=mock_config
            )
            
            # Test with date object
            from datetime import datetime
            payload = {
                'date': datetime.now(),
                'normal': 'value'
            }
            
            sanitized = client._sanitize_payload(payload)
            
            assert 'date' in sanitized
            assert isinstance(sanitized['date'], str)  # Should be converted to string
            assert sanitized['normal'] == 'value'
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    def test_validate_payload_size(self, mock_vertexai, mock_config, temp_cache_dir):
        """Test payload size validation."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        with patch('src.analyzer.ai.llm_client.GenerativeModel'):
            client = LLMClient(
                project_id="test-project",
                default_location="us-central1",
                config=mock_config
            )
            
            # Small payload
            small_payload = {'key': 'value'}
            valid, error = client._validate_payload_size(small_payload, max_size_mb=4.0)
            assert valid == True
            assert error is None
            
            # Very large payload (mock)
            large_payload = {'data': 'x' * 10_000_000}  # 10MB
            valid, error = client._validate_payload_size(large_payload, max_size_mb=4.0)
            assert valid == False
            assert error is not None
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    @patch('src.analyzer.ai.llm_client.GenerativeModel')
    def test_call_llm_cache_hit(self, mock_model_class, mock_vertexai, mock_config, temp_cache_dir):
        """Test LLM call with cache hit."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        # Mock GenerativeModel
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        client = LLMClient(
            project_id="test-project",
            default_location="us-central1",
            config=mock_config
        )
        
        model_info = {'id': 'gemini-2.5-flash', 'generation_config': {}}
        
        # Set cached response (don't use image_path for cache key)
        cache_key = client._generate_cache_key(model_info, "system", "user", None)
        client.disk_cache[cache_key] = {'elements': [], 'connections': []}
        
        # Call LLM (should hit cache)
        result = client.call_llm(
            model_info=model_info,
            system_prompt="system",
            user_prompt="user",
            image_path=None,
            use_cache=True
        )
        
        # Should return cached result (no API call)
        assert result == {'elements': [], 'connections': []}
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    @patch('src.analyzer.ai.llm_client.GenerativeModel')
    def test_call_llm_with_timeout_error(self, mock_model_class, mock_vertexai, mock_config, temp_cache_dir):
        """Test LLM call with timeout error."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        # Mock GenerativeModel
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        client = LLMClient(
            project_id="test-project",
            default_location="us-central1",
            config=mock_config
        )
        
        model_info = {'id': 'gemini-2.5-flash', 'generation_config': {}}
        
        # Mock timeout error
        def mock_call_with_timeout(*args, **kwargs):
            raise FutureTimeoutError("Request timeout")
        
        client._call_with_timeout = mock_call_with_timeout
        
        # Call LLM (should retry and eventually fail)
        result = client.call_llm(
            model_info=model_info,
            system_prompt="system",
            user_prompt="user",
            image_path=None,
            use_cache=False,
            timeout=5
        )
        
        # Should return None after all retries fail
        assert result is None
    
    @patch('src.analyzer.ai.llm_client.vertexai')
    @patch('src.analyzer.ai.llm_client.GenerativeModel')
    def test_call_llm_circuit_breaker_open(self, mock_model_class, mock_vertexai, mock_config, temp_cache_dir):
        """Test LLM call when circuit breaker is open."""
        mock_config['paths']['llm_cache_dir'] = str(temp_cache_dir)
        
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        client = LLMClient(
            project_id="test-project",
            default_location="us-central1",
            config=mock_config
        )
        
        # Open circuit breaker
        for _ in range(5):
            client.retry_handler.record_failure(Exception("Error"))
        
        assert client.retry_handler.circuit_breaker.get_state() == "open"
        
        model_info = {'id': 'gemini-2.5-flash', 'generation_config': {}}
        
        # Call LLM (should return None immediately)
        result = client.call_llm(
            model_info=model_info,
            system_prompt="system",
            user_prompt="user",
            image_path=None,
            use_cache=False
        )
        
        # Should return None or cached result, not make API call
        assert result is None or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

