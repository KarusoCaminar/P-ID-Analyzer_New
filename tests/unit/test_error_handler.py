"""
Unit tests for error handling (ErrorClassifier, CircuitBreaker, IntelligentRetryHandler).
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import time

from src.analyzer.ai.error_handler import (
    ErrorClassifier,
    CircuitBreaker,
    IntelligentRetryHandler,
    ErrorType
)


class TestErrorClassifier:
    """Tests for ErrorClassifier."""
    
    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        classifier = ErrorClassifier()
        
        # Mock rate limit error
        error = Exception("429 Too Many Requests")
        error_info = classifier.classify_error(error)
        
        assert error_info.error_type == ErrorType.RATE_LIMIT
        assert error_info.retryable == True
    
    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        classifier = ErrorClassifier()
        
        # Mock timeout error
        error = TimeoutError("Request timeout")
        error_info = classifier.classify_error(error)
        
        assert error_info.error_type == ErrorType.TIMEOUT
        assert error_info.retryable == True
    
    def test_classify_network_error(self):
        """Test classification of network errors."""
        classifier = ErrorClassifier()
        
        # Mock network error
        error = ConnectionError("Network unreachable")
        error_info = classifier.classify_error(error)
        
        assert error_info.error_type == ErrorType.NETWORK
        assert error_info.retryable == True
    
    def test_classify_auth_error(self):
        """Test classification of auth errors."""
        classifier = ErrorClassifier()
        
        # Mock auth error (should be classified as PERMANENT)
        error = Exception("401 Unauthorized")
        error_info = classifier.classify_error(error)
        
        # Auth errors should be non-retryable
        assert error_info.retryable == False
    
    def test_classify_serialization_error(self):
        """Test classification of serialization errors."""
        classifier = ErrorClassifier()
        
        # Mock serialization error
        error = Exception("Serialization error in aiserver.v1.StreamUnifiedChatRequestWithTools")
        error_info = classifier.classify_error(error)
        
        assert error_info.error_type == ErrorType.SERIALIZATION
        assert error_info.retryable == False


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_circuit_breaker_initial_state(self):
        """Test that circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        assert cb.get_state() == "closed"
        assert cb.failure_count == 0
        assert cb.can_proceed() == True
    
    def test_circuit_breaker_open_on_threshold(self):
        """Test that circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        # Record failures up to threshold (no parameter needed)
        for _ in range(3):
            cb.record_failure()
        
        assert cb.get_state() == "open"
        assert cb.can_proceed() == False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)  # Short timeout for testing
        
        # Open circuit breaker
        for _ in range(2):
            cb.record_failure()
        assert cb.get_state() == "open"
        assert cb.can_proceed() == False
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should transition to half-open when can_proceed() is called
        assert cb.can_proceed() == True
        assert cb.get_state() == "half_open"
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        
        # Open circuit breaker
        for _ in range(2):
            cb.record_failure()
        assert cb.get_state() == "open"
        
        # Reset
        cb.reset()
        assert cb.get_state() == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_save_load_state(self):
        """Test saving and loading circuit breaker state."""
        temp_dir = Path(tempfile.mkdtemp())
        state_path = temp_dir / "circuit-state.json"
        
        try:
            cb1 = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
            
            # Record some failures
            cb1.record_failure()
            cb1.record_failure()
            
            # Save state
            cb1.save_state(state_path)
            
            # Create new circuit breaker and load state
            cb2 = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
            cb2.load_state(state_path)
            
            assert cb2.failure_count == cb1.failure_count
            assert cb2.get_state() == cb1.get_state()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestIntelligentRetryHandler:
    """Tests for IntelligentRetryHandler."""
    
    def test_retry_handler_should_retry_retryable_error(self):
        """Test that retryable errors should be retried."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        handler = IntelligentRetryHandler(cb)
        
        # Network error is retryable
        error = ConnectionError("Network error")
        should_retry, backoff = handler.should_retry(error, attempt=0, max_attempts=3)
        
        assert should_retry == True
        assert backoff > 0
    
    def test_retry_handler_should_not_retry_non_retryable_error(self):
        """Test that non-retryable errors should not be retried."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        handler = IntelligentRetryHandler(cb)
        
        # Auth error is not retryable
        error = Exception("401 Unauthorized")
        should_retry, backoff = handler.should_retry(error, attempt=0, max_attempts=3)
        
        assert should_retry == False
    
    def test_retry_handler_max_retries_exceeded(self):
        """Test that retries stop after max_retries."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        handler = IntelligentRetryHandler(cb)
        
        error = ConnectionError("Network error")
        should_retry, backoff = handler.should_retry(error, attempt=3, max_attempts=3)
        
        assert should_retry == False  # Should not retry after max attempts
    
    def test_retry_handler_exponential_backoff(self):
        """Test exponential backoff calculation."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        handler = IntelligentRetryHandler(cb)
        
        error = ConnectionError("Network error")
        
        # First attempt
        should_retry1, backoff1 = handler.should_retry(error, attempt=0, max_attempts=3)
        
        # Second attempt
        should_retry2, backoff2 = handler.should_retry(error, attempt=1, max_attempts=3)
        
        assert should_retry1 == True
        assert should_retry2 == True
        # Backoff should increase (with jitter, so approximate)
        assert backoff2 >= backoff1 * 0.5  # Jitter makes it approximate
    
    def test_retry_handler_record_success_resets_failure_count(self):
        """Test that recording success resets failure count."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        handler = IntelligentRetryHandler(cb)
        
        # Record some failures
        handler.record_failure(Exception("Error"))
        handler.record_failure(Exception("Error"))
        
        assert cb.failure_count > 0
        
        # Record success
        handler.record_success()
        
        assert cb.failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

