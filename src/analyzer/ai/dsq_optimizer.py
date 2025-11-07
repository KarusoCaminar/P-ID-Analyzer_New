"""
Dynamic Shared Quota (DSQ) Optimizer for Vertex AI.

Based on: https://cloud.google.com/vertex-ai/generative-ai/docs/dynamic-shared-quota

Key Insights:
1. DSQ has NO fixed per-customer limits - resources are shared dynamically
2. 429 errors mean shared pool is temporarily overloaded, NOT quota exceeded
3. DSQ prioritizes steady traffic over burst traffic
4. Exponential backoff with jitter is critical for DSQ

This module implements:
- Adaptive rate limiting (adjusts based on success rate)
- Request smoothing (evenly distributes requests over time)
- Intelligent 429 handling (longer backoffs, adaptive recovery)
- Traffic shaping (prevents burst patterns that DSQ penalizes)
"""

import time
import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Track request metrics for adaptive rate limiting."""
    success_count: int = 0
    failure_count: int = 0
    rate_limit_count: int = 0
    last_request_time: Optional[float] = None
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=60))  # Last 60 requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate current requests per minute."""
        if len(self.recent_requests) < 2:
            return 0.0
        
        time_span = self.recent_requests[-1] - self.recent_requests[0]
        if time_span == 0:
            return 0.0
        
        return (len(self.recent_requests) - 1) / (time_span / 60.0)
    
    def record_request(self, success: bool, is_rate_limit: bool = False):
        """Record a request."""
        current_time = time.time()
        self.recent_requests.append(current_time)
        self.last_request_time = current_time
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if is_rate_limit:
                self.rate_limit_count += 1


class DSQOptimizer:
    """
    Optimizer for Dynamic Shared Quota (DSQ).
    
    Implements adaptive rate limiting and request smoothing based on DSQ behavior.
    """
    
    def __init__(
        self,
        initial_requests_per_minute: int = 60,
        min_requests_per_minute: int = 10,
        max_requests_per_minute: int = 300,
        smoothing_window_seconds: int = 60
    ):
        """
        Initialize DSQ Optimizer.
        
        Args:
            initial_requests_per_minute: Starting request rate
            min_requests_per_minute: Minimum safe rate
            max_requests_per_minute: Maximum allowed rate
            smoothing_window_seconds: Time window for request smoothing
        """
        self.initial_rpm = initial_requests_per_minute
        self.min_rpm = min_requests_per_minute
        self.max_rpm = max_requests_per_minute
        self.smoothing_window = smoothing_window_seconds
        
        # Current rate (adaptive)
        self.current_rpm = initial_requests_per_minute
        
        # Request metrics
        self.metrics = RequestMetrics()
        
        # Rate limit tracking
        self.rate_limit_events: deque = deque(maxlen=10)
        self.last_rate_limit_time: Optional[float] = None
        
        # Request smoothing
        self.request_times: deque = deque()
        self.min_delay_between_requests: float = 60.0 / initial_requests_per_minute  # seconds
        
        logger.info(f"DSQ Optimizer initialized: {initial_requests_per_minute} RPM initial")
    
    def calculate_backoff_for_429(self, attempt: int, base_backoff: float = 2.0) -> float:
        """
        Calculate adaptive backoff for 429 errors based on DSQ behavior.
        
        DSQ Insight: 429 errors mean shared pool is temporarily overloaded.
        Longer backoffs are better than aggressive retries.
        
        Args:
            attempt: Retry attempt number (0-based)
            base_backoff: Base backoff in seconds
            
        Returns:
            Backoff time in seconds
        """
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 64s...
        exponential = base_backoff * (2 ** attempt)
        
        # Add jitter (10-20% random variation) to prevent thundering herd
        jitter_multiplier = random.uniform(0.9, 1.1)
        backoff = exponential * jitter_multiplier
        
        # Adaptive adjustment based on recent rate limit frequency
        if self.last_rate_limit_time:
            time_since_last = time.time() - self.last_rate_limit_time
            # If rate limits are frequent (< 30s apart), increase backoff
            if time_since_last < 30:
                backoff *= 1.5  # Increase backoff by 50%
                logger.info(f"Frequent rate limits detected - increasing backoff to {backoff:.1f}s")
        
        # Cap at reasonable maximum (120s for DSQ - system needs time to recover)
        backoff = min(backoff, 120.0)
        
        # Minimum backoff (even first retry should wait a bit)
        backoff = max(backoff, 1.0)
        
        return backoff
    
    def should_throttle(self) -> tuple[bool, float]:
        """
        Determine if request should be throttled for request smoothing.
        
        DSQ Insight: Steady traffic is prioritized over burst traffic.
        We should evenly distribute requests over time.
        
        Returns:
            Tuple of (should_throttle, delay_seconds)
        """
        current_time = time.time()
        
        # Clean old request times (outside smoothing window)
        while self.request_times and (current_time - self.request_times[0]) > self.smoothing_window:
            self.request_times.popleft()
        
        # Calculate desired delay between requests
        target_requests_per_minute = self.current_rpm
        if target_requests_per_minute <= 0:
            return False, 0.0
        
        desired_delay = 60.0 / target_requests_per_minute
        
        # If we have recent requests, check if we're sending too fast
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            
            if time_since_last < desired_delay:
                # We're sending too fast - throttle
                throttle_delay = desired_delay - time_since_last
                
                # Add small random jitter to prevent synchronized requests
                jitter = random.uniform(0, desired_delay * 0.1)
                throttle_delay += jitter
                
                return True, throttle_delay
        
        return False, 0.0
    
    def record_success(self):
        """Record successful request."""
        self.metrics.record_request(success=True)
        current_time = time.time()
        self.request_times.append(current_time)
        
        # Gradually increase rate if success rate is high
        if self.metrics.success_rate > 0.95 and self.current_rpm < self.max_rpm:
            # Increase by 10% (conservative)
            self.current_rpm = min(self.current_rpm * 1.1, self.max_rpm)
            logger.debug(f"Success rate high ({self.metrics.success_rate:.2%}) - increasing rate to {self.current_rpm:.1f} RPM")
    
    def record_rate_limit(self):
        """Record 429 rate limit error."""
        current_time = time.time()
        self.last_rate_limit_time = current_time
        self.rate_limit_events.append(current_time)
        self.metrics.record_request(success=False, is_rate_limit=True)
        
        # Aggressively reduce rate on rate limit
        # DSQ Insight: If we get 429, we're sending too fast for current pool capacity
        self.current_rpm = max(self.current_rpm * 0.7, self.min_rpm)  # Reduce by 30%
        logger.warning(
            f"Rate limit detected - reducing rate to {self.current_rpm:.1f} RPM "
            f"(success rate: {self.metrics.success_rate:.2%})"
        )
    
    def record_failure(self, is_rate_limit: bool = False):
        """Record failed request."""
        if is_rate_limit:
            self.record_rate_limit()
        else:
            self.metrics.record_request(success=False, is_rate_limit=False)
            
            # Slightly reduce rate on other failures
            if self.metrics.success_rate < 0.8:
                self.current_rpm = max(self.current_rpm * 0.9, self.min_rpm)
                logger.debug(f"Failure rate high - reducing rate to {self.current_rpm:.1f} RPM")
    
    def get_adaptive_delay(self) -> float:
        """
        Get adaptive delay between requests based on current rate.
        
        Returns:
            Delay in seconds
        """
        should_throttle, delay = self.should_throttle()
        if should_throttle:
            return delay
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current optimizer status."""
        return {
            'current_rpm': self.current_rpm,
            'success_rate': self.metrics.success_rate,
            'requests_per_minute': self.metrics.requests_per_minute,
            'rate_limit_count': self.metrics.rate_limit_count,
            'total_requests': self.metrics.success_count + self.metrics.failure_count,
            'rate_limit_events_last_minute': sum(
                1 for event_time in self.rate_limit_events
                if time.time() - event_time < 60
            )
        }


# Global DSQ Optimizer instance (shared across all LLM calls)
_dsq_optimizer: Optional[DSQOptimizer] = None


def get_dsq_optimizer(
    initial_requests_per_minute: int = 60,
    min_requests_per_minute: int = 10,
    max_requests_per_minute: int = 300
) -> DSQOptimizer:
    """
    Get or create global DSQ Optimizer instance.
    
    Args:
        initial_requests_per_minute: Initial request rate
        min_requests_per_minute: Minimum safe rate
        max_requests_per_minute: Maximum allowed rate
        
    Returns:
        DSQOptimizer instance
    """
    global _dsq_optimizer
    
    if _dsq_optimizer is None:
        _dsq_optimizer = DSQOptimizer(
            initial_requests_per_minute=initial_requests_per_minute,
            min_requests_per_minute=min_requests_per_minute,
            max_requests_per_minute=max_requests_per_minute
        )
    
    return _dsq_optimizer


def reset_dsq_optimizer():
    """Reset global DSQ Optimizer (for testing)."""
    global _dsq_optimizer
    _dsq_optimizer = None

