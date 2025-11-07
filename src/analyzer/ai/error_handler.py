"""
Intelligent error handler for LLM API calls with error classification, 
circuit breaker pattern, and API call minimization.

Features:
- Error classification (temporary vs. permanent)
- Circuit breaker pattern
- Intelligent retry logic
- API call minimization
- Fallback strategies
"""

import logging
import time
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Error type classification."""
    TEMPORARY = "temporary"  # Retryable errors
    PERMANENT = "permanent"  # Non-retryable errors
    RATE_LIMIT = "rate_limit"  # Rate limit errors
    AUTH_ERROR = "auth_error"  # Authentication errors
    TIMEOUT = "timeout"  # Timeout errors
    NETWORK = "network"  # Network errors
    SERIALIZATION = "serialization"  # Serialization errors
    CONNECT = "connect"  # Connection errors


@dataclass
class ErrorInfo:
    """Error information with classification."""
    error_type: ErrorType
    error_message: str
    retryable: bool
    backoff_seconds: float
    max_retries: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern for API calls.
    
    Prevents cascading failures by stopping requests when
    error rate exceeds threshold.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
    
    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    # Transition to half-open
                    self.state = "half_open"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    return True
            return False
        
        if self.state == "half_open":
            if self.half_open_calls < self.half_open_max_calls:
                return True
            return False
        
        return False
    
    def record_success(self):
        """Record successful call."""
        if self.state == "half_open":
            # Transition to closed
            self.state = "closed"
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        elif self.state == "closed":
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "half_open":
            # Transition back to open
            self.state = "open"
            self.half_open_calls = 0
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            # Transition to open
            self.state = "open"
            logger.warning(f"Circuit breaker: CLOSED -> OPEN (failures: {self.failure_count})")
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        logger.info("Circuit breaker manually reset to CLOSED state")
    
    def save_state(self, file_path: Path) -> None:
        """
        Save circuit breaker state to file.
        
        Args:
            file_path: Path to save state file
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "half_open_calls": self.half_open_calls,
                "timestamp": datetime.now().isoformat()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save circuit breaker state: {e}", exc_info=True)
    
    def load_state(self, file_path: Path) -> None:
        """
        Load circuit breaker state from file.
        
        Args:
            file_path: Path to load state file from
        """
        try:
            if not file_path.exists():
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.state = state.get("state", "closed")
            self.failure_count = state.get("failure_count", 0)
            if state.get("last_failure_time"):
                self.last_failure_time = datetime.fromisoformat(state["last_failure_time"])
            self.half_open_calls = state.get("half_open_calls", 0)
            
            logger.info(f"Loaded circuit breaker state: {self.state}")
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}", exc_info=True)


class ErrorClassifier:
    """Classify errors and determine retry strategy."""
    
    @staticmethod
    def classify_error(error: Exception) -> ErrorInfo:
        """
        Classify error and return error info.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorInfo with classification and retry strategy
        """
        error_message = str(error).lower()
        error_type_str = str(type(error).__name__).lower()
        
        # Rate limit errors (temporary, but need longer backoff)
        # CRITICAL: Exponential backoff for rate limits (60s → 120s → 240s)
        if any(term in error_message for term in ["rate limit", "429", "quota", "quota exceeded", "resource_exhausted"]):
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT,
                error_message=str(error),
                retryable=True,
                backoff_seconds=120.0,  # Increased from 60s to 120s for rate limits (exponential: 120s → 240s → 480s)
                max_retries=5
            )
        
        # Timeout errors (temporary, exponential backoff)
        # CRITICAL: Exponential backoff for timeouts (10s → 20s → 40s)
        if any(term in error_message for term in ["timeout", "timed out", "deadline exceeded"]):
            return ErrorInfo(
                error_type=ErrorType.TIMEOUT,
                error_message=str(error),
                retryable=True,
                backoff_seconds=10.0,  # Increased from 5s to 10s for timeouts (exponential: 10s → 20s → 40s)
                max_retries=5  # Increased from 3 to 5 retries for timeouts
            )
        
        # Network errors (temporary, exponential backoff)
        if any(term in error_message for term in [
            "503", "502", "504", "connection", "unavailable", 
            "end of tcp stream", "network", "socket"
        ]):
            return ErrorInfo(
                error_type=ErrorType.NETWORK,
                error_message=str(error),
                retryable=True,
                backoff_seconds=2.0,
                max_retries=3
            )
        
        # Authentication errors (permanent, don't retry)
        if any(term in error_message for term in [
            "401", "403", "unauthorized", "forbidden", "authentication",
            "permission denied", "invalid credentials"
        ]):
            return ErrorInfo(
                error_type=ErrorType.AUTH_ERROR,
                error_message=str(error),
                retryable=False,
                backoff_seconds=0.0,
                max_retries=0
            )
        
        # Invalid request errors (permanent, don't retry)
        if any(term in error_message for term in [
            "400", "invalid", "bad request", "malformed", "syntax error"
        ]):
            return ErrorInfo(
                error_type=ErrorType.PERMANENT,
                error_message=str(error),
                retryable=False,
                backoff_seconds=0.0,
                max_retries=0
            )
        
        # Serialization errors (permanent, don't retry - use fallback instead)
        if any(term in error_message for term in [
            "serialization", "serialize", "not json serializable",
            "cannot serialize", "unserializable", "json encode"
        ]):
            return ErrorInfo(
                error_type=ErrorType.SERIALIZATION,
                error_message=str(error),
                retryable=False,  # Don't retry, use fallback instead
                backoff_seconds=0.0,
                max_retries=0
            )
        
        # Connection errors (temporary, exponential backoff)
        if any(term in error_message for term in [
            "connecterror", "connect error", "connection refused",
            "connection reset", "connection timeout", "failed to connect"
        ]):
            return ErrorInfo(
                error_type=ErrorType.CONNECT,
                error_message=str(error),
                retryable=True,
                backoff_seconds=2.0,
                max_retries=3
            )
        
        # JSON parsing errors (temporary, might be fixable)
        if "json" in error_message or "parse" in error_message:
            return ErrorInfo(
                error_type=ErrorType.TEMPORARY,
                error_message=str(error),
                retryable=True,
                backoff_seconds=1.0,
                max_retries=2  # Fewer retries for parsing errors
            )
        
        # Default: treat as temporary with conservative retry
        return ErrorInfo(
            error_type=ErrorType.TEMPORARY,
            error_message=str(error),
            retryable=True,
            backoff_seconds=2.0,
            max_retries=2  # Conservative default
        )


class IntelligentRetryHandler:
    """
    Intelligent retry handler with error classification and API minimization.
    """
    
    def __init__(self, circuit_breaker: Optional[CircuitBreaker] = None):
        """
        Initialize retry handler.
        
        Args:
            circuit_breaker: Optional circuit breaker instance
        """
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.error_classifier = ErrorClassifier()
        self.retry_history: List[Dict[str, Any]] = []
    
    def should_retry(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int
    ) -> tuple[bool, float]:
        """
        Determine if should retry and backoff time.
        
        Args:
            error: Exception that occurred
            attempt: Current attempt number (0-based)
            max_attempts: Maximum number of attempts
            
        Returns:
            Tuple of (should_retry, backoff_seconds)
        """
        # Check circuit breaker first (minimizes API calls)
        if not self.circuit_breaker.can_proceed():
            logger.warning(
                f"Circuit breaker is {self.circuit_breaker.get_state()}. "
                "Skipping retry to minimize API calls."
            )
            return False, 0.0
        
        # Check if we've exceeded max attempts
        if attempt >= max_attempts:
            logger.debug(f"Max attempts ({max_attempts}) reached. No retry.")
            return False, 0.0
        
        # Classify error
        error_info = self.error_classifier.classify_error(error)
        
        # Record error
        self.retry_history.append({
            'attempt': attempt,
            'error_type': error_info.error_type.value,
            'error_message': error_info.error_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if error is retryable
        if not error_info.retryable:
            logger.debug(
                f"Non-retryable error ({error_info.error_type.value}): "
                f"{error_info.error_message}. No retry."
            )
            return False, 0.0
        
        # Check max retries for this error type
        if attempt >= error_info.max_retries:
            logger.debug(
                f"Max retries ({error_info.max_retries}) for error type "
                f"{error_info.error_type.value} reached. No retry."
            )
            return False, 0.0
        
        # Calculate exponential backoff with jitter
        import random
        
        # CRITICAL: For DSQ (Dynamic Shared Quota), use adaptive backoff
        # DSQ Insight: 429 errors mean shared pool is temporarily overloaded
        # Longer backoffs are better than aggressive retries
        if error_info.error_type == ErrorType.RATE_LIMIT:
            # Use DSQ-optimized backoff for rate limits
            from src.analyzer.ai.dsq_optimizer import get_dsq_optimizer
            dsq_optimizer = get_dsq_optimizer()
            backoff_seconds = dsq_optimizer.calculate_backoff_for_429(
                attempt=attempt,
                base_backoff=error_info.backoff_seconds
            )
            # Record rate limit for adaptive rate limiting
            dsq_optimizer.record_rate_limit()
        else:
            # Standard exponential backoff for other errors
            base_backoff = error_info.backoff_seconds
            exponential_backoff = base_backoff * (2 ** attempt)
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, exponential_backoff * 0.1)
            backoff_seconds = exponential_backoff + jitter
            
            # Cap backoff at reasonable maximum (60s for non-rate-limit errors)
            backoff_seconds = min(backoff_seconds, 60.0)
        
        logger.info(
            f"Retryable error ({error_info.error_type.value}). "
            f"Will retry after {backoff_seconds:.2f}s (attempt {attempt + 1}/{max_attempts})"
        )
        
        return True, backoff_seconds
    
    def record_success(self):
        """Record successful call."""
        self.circuit_breaker.record_success()
    
    def record_failure(self, error: Exception):
        """Record failed call."""
        self.circuit_breaker.record_failure()
        error_info = self.error_classifier.classify_error(error)
        logger.debug(f"Recorded failure: {error_info.error_type.value}")
    
    def get_retry_history(self) -> List[Dict[str, Any]]:
        """Get retry history."""
        return self.retry_history.copy()
    
    def clear_history(self):
        """Clear retry history."""
        self.retry_history.clear()

