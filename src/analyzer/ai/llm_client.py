"""
LLM Client - Refactored from llm_handler.py.

Handles interactions with LLM providers (Google Vertex AI, etc.)
with improved error handling, caching, and retry logic.
"""

import os
import json
import time
import logging
import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from io import BytesIO
from datetime import datetime

import diskcache
import vertexai
from vertexai.generative_models import HarmCategory, HarmBlockThreshold, GenerativeModel, Part
from PIL import Image

from .error_handler import IntelligentRetryHandler, CircuitBreaker, ErrorType
from src.utils.schemas import validate_request_payload, validate_response
from src.utils.response_validator import is_raw_response_valid
from src.utils.debug_capture import (
    generate_request_id, capture_request, capture_response, write_debug_file
)

logger = logging.getLogger(__name__)
llm_logger = logging.getLogger('llm_calls')  # Dedicated logger for LLM calls


class LLMClient:
    """
    Client for interacting with LLM providers (currently Google Vertex AI).
    
    Features:
    - Multi-provider support (prepared for OpenAI, Claude, etc.)
    - Disk caching for API responses
    - Retry logic with exponential backoff
    - Request batching
    - Timeout handling
    """
    
    def __init__(
        self,
        project_id: str,
        default_location: str,
        config: Dict[str, Any]
    ):
        self.project_id = project_id
        self.default_location = default_location
        self.config = config
        
        # Initialize Vertex AI SDK
        vertexai.init(project=self.project_id, location=self.default_location)
        
        self.gemini_clients: Dict[str, GenerativeModel] = {}
        self._initialized_locations: set[str] = set()
        
        # Load models
        self._load_models()
        
        # Setup multi-level cache (memory + disk)
        cache_dir_name = config.get('paths', {}).get('llm_cache_dir', '.pni_analyzer_cache')
        cache_path = Path(str(cache_dir_name))
        cache_size_gb = config.get('logic_parameters', {}).get('llm_disk_cache_size_gb', 2)
        memory_cache_size = config.get('logic_parameters', {}).get('llm_memory_cache_size', 100)
        
        # Use multi-level cache from cache service
        from src.services.cache_service import MultiLevelCache
        self.disk_cache = MultiLevelCache(
            cache_dir=cache_path,
            memory_size=memory_cache_size,
            disk_size_gb=cache_size_gb,
            memory_ttl_hours=24
        )
        logger.info(f"Multi-level cache initialized at: {cache_path} (Memory={memory_cache_size}, Disk={cache_size_gb}GB)")
        
        # Timeout executor
        max_workers_timeout = config.get('logic_parameters', {}).get('llm_timeout_executor_workers', 1)
        self.timeout_executor = ThreadPoolExecutor(max_workers=max_workers_timeout)
        
        # Intelligent error handling
        circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('logic_parameters', {}).get('circuit_breaker_failure_threshold', 5),
            recovery_timeout=config.get('logic_parameters', {}).get('circuit_breaker_recovery_timeout', 60)
        )
        self.retry_handler = IntelligentRetryHandler(circuit_breaker)
        
        # Debug directory
        debug_dir = config.get('paths', {}).get('debug_dir', 'outputs/debug')
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Load circuit breaker state if exists
        circuit_state_path = self.debug_dir / 'circuit-state.json'
        # CRITICAL FIX: Don't load state from file - always start with clean state
        # The state will be loaded per-request if needed, but we want fresh start
        # circuit_breaker.load_state(circuit_state_path)  # DISABLED: Always start fresh
        logger.info("Circuit breaker initialized with clean state (file loading disabled for fresh start)")
    
    def close(self) -> None:
        """Close client and cleanup resources."""
        # CRITICAL FIX: Shutdown ThreadPoolExecutor to prevent thread leaks
        if hasattr(self, 'timeout_executor') and self.timeout_executor:
            logger.debug("Shutting down ThreadPoolExecutor...")
            self.timeout_executor.shutdown(wait=True)
            logger.debug("ThreadPoolExecutor shut down")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        # CRITICAL FIX: Ensure ThreadPoolExecutor is shut down on deletion
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def _load_models(self) -> None:
        """Load all configured models."""
        for model_name_display, model_info in self.config.get("models", {}).items():
            model_id = model_info.get("id")
            if not model_id:
                logger.warning(f"Model '{model_name_display}' in config has no ID. Skipping.")
                continue
            
            if model_info.get('access_method') == 'gemini':
                try:
                    if model_id not in self.gemini_clients:
                        self.gemini_clients[model_id] = GenerativeModel(model_id)
                        logger.info(f"Successfully loaded Gemini model: {model_id}")
                except Exception as e:
                    logger.error(f"Error loading Gemini model {model_id}: {e}", exc_info=True)
    
    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize payload for JSON serialization.
        
        Converts non-serializable objects to safe types:
        - Dates → ISO format strings
        - Functions → removed (keep identifiers only)
        - Circular refs → detected and resolved
        - Enums → numbers or strings
        
        Args:
            payload: Payload to sanitize
            
        Returns:
            Sanitized payload
        """
        def _sanitize(obj: Any, visited: Optional[set] = None) -> Any:
            if visited is None:
                visited = set()
            
            # Check for circular references
            obj_id = id(obj)
            if obj_id in visited:
                return "<circular_reference>"
            visited.add(obj_id)
            
            try:
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        # Skip functions and callables
                        if callable(v) and not isinstance(v, type):
                            continue
                        result[k] = _sanitize(v, visited)
                    return result
                elif isinstance(obj, list):
                    return [_sanitize(item, visited) for item in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    # Try to convert objects with __dict__ to dict
                    try:
                        return _sanitize(obj.__dict__, visited)
                    except Exception:
                        return str(obj)
                else:
                    # Fallback to string representation
                    return str(obj)
            finally:
                visited.discard(obj_id)
        
        return _sanitize(payload)
    
    def _validate_payload_size(self, payload: Dict[str, Any], max_size_mb: float = 4.0) -> tuple[bool, Optional[str]]:
        """
        Validate payload size.
        
        Args:
            payload: Payload to validate
            max_size_mb: Maximum size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            payload_json = json.dumps(payload)
            size_bytes = len(payload_json.encode('utf-8'))
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > max_size_mb:
                return False, f"Payload size {size_mb:.2f}MB exceeds maximum {max_size_mb}MB"
            
            return True, None
        except Exception as e:
            return False, f"Failed to validate payload size: {str(e)}"
    
    def call_llm(
        self,
        model_info: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        use_cache: bool = True,
        expected_json_keys: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> Optional[Union[Dict[str, Any], str]]:
        """
        Call LLM with image and prompt.
        
        Args:
            model_info: Model configuration dict
            system_prompt: System prompt
            user_prompt: User prompt
            image_path: Optional path to image
            use_cache: Whether to use cache
            expected_json_keys: Expected JSON keys for validation
            timeout: Optional timeout in seconds
            
        Returns:
            LLM response (dict if JSON, else str), or None on error
        """
        # Generate cache key
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(model_info, system_prompt, user_prompt, image_path)
            # CRITICAL FIX: Use explicit get() method instead of container semantics
            cached_value = self.disk_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for LLM call")
                return cached_value
        
        # Prepare request
        model_id = model_info.get('id')
        if model_id not in self.gemini_clients:
            logger.error(f"Model {model_id} not loaded")
            return None
        
        model = self.gemini_clients[model_id]
        # CRITICAL FIX: Copy generation_config before modifying to avoid mutating shared config
        generation_config = dict(model_info.get('generation_config', {}))
        
        # OPTIMIZATION: Ensure response_mime_type is set for structured output
        if 'response_mime_type' not in generation_config:
            # Default to JSON for structured output if not specified
            generation_config['response_mime_type'] = "application/json"
        
        # OPTIMIZATION: Build response_schema if response_mime_type is JSON
        # (Note: response_schema is optional and can be added later if needed)
        # For now, we rely on prompts to enforce structure
        
        # Generate request ID for tracking
        request_id = generate_request_id()
        
        # ENHANCED LOGGING: Log request details
        llm_logger.info(
            f"REQUEST [model={model_id}] [prompt_length={len(user_prompt)}] "
            f"[system_prompt_length={len(system_prompt)}] [has_image={image_path is not None}]",
            extra={'request_id': request_id}
        )
        
        # Log prompt preview (first 500 chars) - Handle Unicode encoding errors
        prompt_preview = user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt
        # Replace Unicode arrows with ASCII equivalent to avoid encoding errors
        prompt_preview_safe = prompt_preview.replace('→', '->').replace('←', '<-').replace('↔', '<->')
        try:
            llm_logger.debug(
                f"PROMPT_PREVIEW: {prompt_preview_safe}",
                extra={'request_id': request_id}
            )
        except UnicodeEncodeError:
            # Fallback: encode to ASCII with errors='replace'
            prompt_preview_ascii = prompt_preview_safe.encode('ascii', errors='replace').decode('ascii')
            llm_logger.debug(
                f"PROMPT_PREVIEW: {prompt_preview_ascii}",
                extra={'request_id': request_id}
            )
        
        # Log full prompt to separate file for debugging
        llm_log_save_prompts = self.config.get('logic_parameters', {}).get('llm_log_save_prompts', True)
        if llm_log_save_prompts and self.debug_dir:
            prompt_log_file = self.debug_dir / f"prompt-{request_id}.txt"
            try:
                with open(prompt_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== REQUEST ID: {request_id} ===\n")
                    f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n\n")
                    f.write(f"=== MODEL: {model_id} ===\n\n")
                    f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
                    f.write(f"=== USER PROMPT ===\n{user_prompt}\n")
                    if image_path:
                        f.write(f"\n=== IMAGE PATH ===\n{image_path}\n")
                llm_logger.debug(f"Full prompt saved to: {prompt_log_file}", extra={'request_id': request_id})
            except Exception as e:
                llm_logger.warning(f"Failed to save prompt file: {e}", extra={'request_id': request_id})
        
        # Prepare content
        # CRITICAL FIX: parts must be List[Union[str, Part]] not list[str]
        parts: List[Union[str, Part]] = [user_prompt]
        if image_path and os.path.exists(image_path):
            parts.append(self._image_to_part(image_path))
        
        # Build payload for validation
        # CRITICAL FIX: Don't convert Part objects to strings for size validation
        # Keep Part objects as-is for SDK, but create a separate sanitized version for validation
        payload = {
            "model": model_id,
            "parts": parts,  # Keep Part objects as-is for SDK
            "generation_config": generation_config
        }
        
        # Create sanitized payload for size validation (convert Parts to size estimates)
        # CRITICAL FIX: Estimate Part size instead of converting to string
        def estimate_part_size(part):
            """Estimate size of Part object for validation."""
            if isinstance(part, str):
                return len(part.encode('utf-8'))
            elif isinstance(part, Part):
                # Estimate: Part objects are typically images, estimate ~100KB average
                # This is a rough estimate for validation purposes
                return 100 * 1024  # 100KB estimate
            else:
                return len(str(part).encode('utf-8'))
        
        # Build sanitized payload for validation (with size estimates)
        validation_payload = {
            "model": model_id,
            "parts": [estimate_part_size(part) for part in parts],  # Use size estimates
            "generation_config": generation_config
        }
        
        # Sanitize payload (for logging/debugging only, not for SDK)
        try:
            sanitized_payload = self._sanitize_payload(payload)
        except Exception as e:
            logger.warning(f"Failed to sanitize payload: {e}")
            sanitized_payload = payload
        
        # Validate payload
        payload_validation_ok = True
        payload_validation_errors = []
        
        # Size validation (use validation_payload with size estimates)
        size_valid, size_error = self._validate_payload_size(validation_payload)
        if not size_valid:
            payload_validation_ok = False
            payload_validation_errors.append(size_error)
        
        # Schema validation (optional - may not always apply)
        try:
            # Note: SDK handles serialization, but we validate structure
            # Try to validate payload structure if possible
            schema_valid, schema_errors = validate_request_payload(sanitized_payload)
            if not schema_valid:
                payload_validation_ok = False
                payload_validation_errors.extend(schema_errors or [])
                logger.warning(f"Payload schema validation failed: {schema_errors}")
        except Exception as e:
            logger.debug(f"Schema validation warning: {e}")
        
        # Capture request
        try:
            capture_request(request_id, sanitized_payload, output_dir=self.debug_dir)
        except Exception as e:
            logger.debug(f"Failed to capture request: {e}")
        
        # Get timeout from config or parameter
        # CRITICAL FIX: Ensure timeout is int, not float
        base_timeout_raw = timeout or self.config.get('logic_parameters', {}).get('llm_default_timeout', 240)
        base_timeout = int(base_timeout_raw) if base_timeout_raw is not None else 240
        max_retries = self.config.get('logic_parameters', {}).get('llm_max_retries', 3)
        
        # CRITICAL FIX: Adjust timeout based on payload size
        # Large payloads need MORE time, not less! (Previous logic was backwards)
        total_prompt_length = len(system_prompt) + len(user_prompt)
        if total_prompt_length > 100000:  # >100KB
            # Very large payloads need significantly more time
            timeout_seconds = int(base_timeout * 1.8)  # 1.8x timeout for very large payloads
            logger.info(f"Very large payload ({total_prompt_length} chars) - using increased timeout: {timeout_seconds}s (base: {base_timeout}s)")
        elif total_prompt_length > 50000:  # >50KB
            # Large payloads need more time
            timeout_seconds = int(base_timeout * 1.4)  # 1.4x timeout for large payloads
            logger.info(f"Large payload ({total_prompt_length} chars) - using increased timeout: {timeout_seconds}s (base: {base_timeout}s)")
        else:
            timeout_seconds = base_timeout
        
        # Intelligent retry logic with error classification
        last_error = None
        api_error_data = None
        response_summary = {"success": False, "status_code": None, "tokens": None}
        
        for attempt in range(max_retries):
            try:
                # CRITICAL: DSQ Request Smoothing - throttle requests for steady traffic
                # DSQ Insight: Steady traffic is prioritized over burst traffic
                from src.analyzer.ai.dsq_optimizer import get_dsq_optimizer
                dsq_optimizer = get_dsq_optimizer(
                    initial_requests_per_minute=self.config.get('logic_parameters', {}).get('llm_rate_limit_requests_per_minute', 60),
                    max_requests_per_minute=self.config.get('logic_parameters', {}).get('llm_rate_limit_requests_per_minute', 300)
                )
                
                # Apply request smoothing (throttle if sending too fast)
                throttle_delay = dsq_optimizer.get_adaptive_delay()
                if throttle_delay > 0:
                    logger.debug(f"DSQ Request Smoothing: Throttling request by {throttle_delay:.2f}s (current rate: {dsq_optimizer.current_rpm:.1f} RPM)")
                    time.sleep(throttle_delay)
                
                # Check circuit breaker before making call (minimizes API calls)
                if not self.retry_handler.circuit_breaker.can_proceed():
                    logger.warning(
                        f"Circuit breaker is {self.retry_handler.circuit_breaker.get_state()}. "
                        "Skipping API call to minimize failures."
                    )
                    # Return cached result if available (fallback)
                    # CRITICAL FIX: Use explicit get() method instead of container semantics
                    if cache_key:
                        cached_value = self.disk_cache.get(cache_key)
                        if cached_value is not None:
                            logger.info("Returning cached result due to circuit breaker")
                            return cached_value
                    return None
                
                response = self._call_with_timeout(
                    model,
                    parts,
                    generation_config,
                    timeout_seconds
                )
                
                # Capture response
                try:
                    capture_response(request_id, response, output_dir=self.debug_dir)
                except Exception as e:
                    logger.debug(f"Failed to capture response: {e}")
                
                # Pattern 2: Defensive JSON validation before parsing
                # This prevents ValidationError crashes from corrupt or empty responses
                if not is_raw_response_valid(response, expected_keys=expected_json_keys):
                    logger.error("LLM response failed validation, discarding.")
                    llm_logger.error(
                        f"RESPONSE_VALIDATION_FAILED: Response structure invalid",
                        extra={'request_id': request_id}
                    )
                    # Record failure for circuit breaker
                    self.retry_handler.record_failure(ValueError("Invalid response structure"))
                    # Try retry if attempts remaining
                    if attempt < max_retries - 1:
                        should_retry, backoff_seconds = self.retry_handler.should_retry(
                            ValueError("Invalid response structure"), attempt, max_retries
                        )
                        if should_retry:
                            time.sleep(backoff_seconds)
                            continue
                    # All retries exhausted or non-retryable
                    return None
                
                # Parse response (now safe - validation passed)
                parsed_response = self._parse_response(response, expected_json_keys)
                
                # ENHANCED LOGGING: Log response details
                if response:
                    # Extract response text
                    # CRITICAL FIX: Check if response is Vertex AI response object before accessing .text
                    response_text = None
                    if not isinstance(response, (str, dict)) and hasattr(response, 'text'):
                        response_text = response.text
                    elif isinstance(response, dict):
                        response_text = json.dumps(response, indent=2, ensure_ascii=False)
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                    
                    # Log response summary
                    response_length = len(response_text) if response_text else 0
                    llm_logger.info(
                        f"RESPONSE [length={response_length}] [type={type(response).__name__}]",
                        extra={'request_id': request_id}
                    )
                    
                    # Log full response (preview in log, full in file)
                    llm_log_save_responses = self.config.get('logic_parameters', {}).get('llm_log_save_responses', True)
                    llm_log_max_length = self.config.get('logic_parameters', {}).get('llm_log_max_response_length', 10000)
                    
                    if response_text:
                        # Log preview
                        response_preview = response_text[:llm_log_max_length] + "..." if len(response_text) > llm_log_max_length else response_text
                        llm_logger.debug(
                            f"RESPONSE_PREVIEW: {response_preview}",
                            extra={'request_id': request_id}
                        )
                        
                        # Save full response to separate file
                        if llm_log_save_responses and self.debug_dir:
                            response_log_file = self.debug_dir / f"response-{request_id}.txt"
                            try:
                                with open(response_log_file, 'w', encoding='utf-8') as f:
                                    f.write(f"=== REQUEST ID: {request_id} ===\n")
                                    f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n\n")
                                    f.write(f"=== RAW RESPONSE ===\n{response_text}\n")
                                    if hasattr(response, '__dict__'):
                                        f.write(f"\n=== RESPONSE ATTRIBUTES ===\n")
                                        try:
                                            f.write(json.dumps(vars(response), indent=2, default=str))
                                        except Exception:
                                            f.write(str(vars(response)))
                                llm_logger.debug(f"Full response saved to: {response_log_file}", extra={'request_id': request_id})
                            except Exception as e:
                                llm_logger.warning(f"Failed to save response file: {e}", extra={'request_id': request_id})
                    
                    # Log parsed response structure
                    if isinstance(parsed_response, dict):
                        keys = list(parsed_response.keys())
                        llm_logger.info(
                            f"RESPONSE_STRUCTURE [keys={keys}] [element_count={len(parsed_response.get('elements', []))}] "
                            f"[connection_count={len(parsed_response.get('connections', []))}]",
                            extra={'request_id': request_id}
                        )
                    elif isinstance(response, dict):
                        keys = list(response.keys())
                        llm_logger.info(
                            f"RESPONSE_STRUCTURE [keys={keys}] [element_count={len(response.get('elements', []))}] "
                            f"[connection_count={len(response.get('connections', []))}]",
                            extra={'request_id': request_id}
                        )
                    elif hasattr(response, 'text'):
                        # Try to parse as JSON
                        try:
                            parsed = json.loads(response.text)
                            if isinstance(parsed, dict):
                                keys = list(parsed.keys())
                                llm_logger.info(
                                    f"RESPONSE_STRUCTURE [keys={keys}] [element_count={len(parsed.get('elements', []))}] "
                                    f"[connection_count={len(parsed.get('connections', []))}]",
                                    extra={'request_id': request_id}
                                )
                        except json.JSONDecodeError:
                            llm_logger.warning(
                                f"RESPONSE_NOT_JSON: Could not parse response as JSON",
                                extra={'request_id': request_id}
                            )
                else:
                    llm_logger.error(
                        f"RESPONSE_EMPTY: No response received",
                        extra={'request_id': request_id}
                    )
                
                # Update response summary
                # CRITICAL FIX: Check if response is Vertex AI response object before accessing attributes
                tokens = None
                if not isinstance(response, (str, dict)) and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    # UsageMetadata is a proto object, access attributes directly
                    tokens = getattr(response.usage_metadata, 'total_token_count', None)
                
                response_summary = {
                    "success": True,
                    "status_code": 200,
                    "tokens": tokens
                }
                
                # Cache result (even on success, helps with future failures)
                # CRITICAL FIX: Use explicit set() method instead of container semantics
                if use_cache and cache_key:
                    self.disk_cache.set(cache_key, parsed_response)
                
                # Record success for circuit breaker
                self.retry_handler.record_success()
                
                # Save circuit breaker state
                try:
                    circuit_state_path = self.debug_dir / 'circuit-state.json'
                    self.retry_handler.circuit_breaker.save_state(circuit_state_path)
                except Exception as e:
                    logger.debug(f"Failed to save circuit breaker state: {e}")
                
                # Write debug file for successful call
                try:
                    write_debug_file(
                        request_id=request_id,
                        call_type="sync",
                        payload_validation={"ok": payload_validation_ok, "errors": payload_validation_errors},
                        api_error=None,
                        response_summary=response_summary,
                        attempts=attempt + 1,
                        suggested_patch=None,
                        output_path=self.debug_dir / 'workflow-debug.json'
                    )
                except Exception as e:
                    logger.debug(f"Failed to write debug file: {e}")
                
                return parsed_response
                
            except FutureTimeoutError as e:
                last_error = e
                logger.warning(f"LLM call timed out (attempt {attempt + 1}/{max_retries})")
                
                # Capture error response
                try:
                    capture_response(request_id, None, error=e, output_dir=self.debug_dir)
                except Exception:
                    pass
                
                # Record failure
                self.retry_handler.record_failure(e)
                
                # Update error data
                api_error_data = {
                    "message": str(e),
                    "stack": traceback.format_exc(),
                    "code": "TIMEOUT"
                }
                
                # Intelligent retry decision
                should_retry, backoff_seconds = self.retry_handler.should_retry(e, attempt, max_retries)
                
                if should_retry:
                    # For timeout errors, don't increase timeout (pointless)
                    # Instead, keep same timeout or even reduce slightly
                    if "timeout" in str(e).lower():
                        logger.warning("Timeout error - keeping same timeout for retry (increasing timeout won't help)")
                        # Don't increase timeout, it's already too long
                    else:
                        # CRITICAL FIX: Ensure timeout_seconds is int
                        timeout_seconds = int(min(timeout_seconds * 1.5, 600))  # Cap timeout at 10 minutes
                    time.sleep(backoff_seconds)
                    continue
                else:
                    break
            
            except ValueError as e:
                # Pattern 4: Specific exception handling for ValueError (content blocking, invalid input)
                # Vertex AI SDK raises ValueError for content blocking
                last_error = e
                error_message = str(e).lower()
                
                # Capture error response
                try:
                    capture_response(request_id, None, error=e, output_dir=self.debug_dir)
                except Exception:
                    pass
                
                # Check if this is a content blocking error
                if any(term in error_message for term in ["blocked", "safety", "harm", "content policy"]):
                    logger.error(f"LLM call blocked by content policy (attempt {attempt + 1}/{max_retries}): {e}")
                    llm_logger.error(
                        f"CONTENT_BLOCKED [message={str(e)}] [attempt={attempt + 1}/{max_retries}]",
                        extra={'request_id': request_id}
                    )
                    api_error_data = {
                        "message": str(e),
                        "stack": traceback.format_exc(),
                        "code": "CONTENT_BLOCKED"
                    }
                    # Content blocking is permanent - don't retry
                    self.retry_handler.record_failure(e)
                    break
                else:
                    # Other ValueError (invalid input, etc.) - might be retryable
                    logger.error(f"LLM call ValueError (attempt {attempt + 1}/{max_retries}): {e}")
                    llm_logger.error(
                        f"VALUE_ERROR [message={str(e)}] [attempt={attempt + 1}/{max_retries}]",
                        extra={'request_id': request_id}
                    )
                    api_error_data = {
                        "message": str(e),
                        "stack": traceback.format_exc(),
                        "code": "VALUE_ERROR"
                    }
                    self.retry_handler.record_failure(e)
                    should_retry, backoff_seconds = self.retry_handler.should_retry(e, attempt, max_retries)
                    if should_retry:
                        time.sleep(backoff_seconds)
                        continue
                    else:
                        break
            
            except (ConnectionError, OSError) as e:
                # Pattern 4: Specific exception handling for network/connection errors
                last_error = e
                logger.warning(f"LLM call network/connection error (attempt {attempt + 1}/{max_retries}): {e}")
                llm_logger.warning(
                    f"NETWORK_ERROR [type={type(e).__name__}] [message={str(e)}] [attempt={attempt + 1}/{max_retries}]",
                    extra={'request_id': request_id}
                )
                
                # Capture error response
                try:
                    capture_response(request_id, None, error=e, output_dir=self.debug_dir)
                except Exception:
                    pass
                
                # Record failure
                self.retry_handler.record_failure(e)
                
                # Update error data
                api_error_data = {
                    "message": str(e),
                    "stack": traceback.format_exc(),
                    "code": "NETWORK_ERROR"
                }
                
                # Intelligent retry decision
                should_retry, backoff_seconds = self.retry_handler.should_retry(e, attempt, max_retries)
                if should_retry:
                    time.sleep(backoff_seconds)
                    continue
                else:
                    break
                    
            except Exception as e:
                # Pattern 4: Generic exception as last resort (catch-all)
                # This should only catch unexpected errors not covered by specific handlers above
                last_error = e
                error_type = type(e).__name__
                logger.error(f"LLM call unexpected error (attempt {attempt + 1}/{max_retries}): {error_type}: {e}")
                llm_logger.error(
                    f"ERROR [type={error_type}] [message={str(e)}] [attempt={attempt + 1}/{max_retries}]",
                    extra={'request_id': request_id}, exc_info=True
                )
                
                # Capture error response
                try:
                    capture_response(request_id, None, error=e, output_dir=self.debug_dir)
                except Exception:
                    pass
                
                # Record failure
                self.retry_handler.record_failure(e)
                
                # Update error data
                api_error_data = {
                    "message": str(e),
                    "stack": traceback.format_exc(),
                    "code": type(e).__name__
                }
                
                # Check if serialization error - use fallback
                error_info = self.retry_handler.error_classifier.classify_error(e)
                if error_info.error_type == ErrorType.SERIALIZATION:
                    logger.warning("Serialization error detected - using fallback (non-streaming)")
                    # Try fallback: non-streaming call
                    try:
                        fallback_response = self._call_with_fallback(
                            model, parts, generation_config, timeout_seconds
                        )
                        if fallback_response:
                            parsed_response = self._parse_response(fallback_response, expected_json_keys)
                            # CRITICAL FIX: Use explicit set() method instead of container semantics
                            if use_cache and cache_key:
                                self.disk_cache.set(cache_key, parsed_response)
                            self.retry_handler.record_success()
                            
                            # Write debug file
                            try:
                                write_debug_file(
                                    request_id=request_id,
                                    call_type="sync",
                                    payload_validation={"ok": payload_validation_ok, "errors": payload_validation_errors},
                                    api_error=api_error_data,
                                    response_summary={"success": True, "status_code": 200, "tokens": None},
                                    attempts=attempt + 1,
                                    suggested_patch={"type": "fallback_to_non_streaming", "applied": True},
                                    output_path=self.debug_dir / 'workflow-debug.json'
                                )
                            except Exception:
                                pass
                            
                            return parsed_response
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                
                # Intelligent retry decision with error classification
                should_retry, backoff_seconds = self.retry_handler.should_retry(e, attempt, max_retries)
                
                if should_retry:
                    time.sleep(backoff_seconds)
                    continue
                else:
                    # For non-retryable errors, return cached result if available (fallback)
                    # CRITICAL FIX: Use explicit get() method instead of container semantics
                    if cache_key:
                        cached_value = self.disk_cache.get(cache_key)
                        if cached_value is not None:
                            logger.info("Returning cached result due to non-retryable error")
                            return cached_value
                    break
        
        # All retries failed - try to return cached result as fallback
        # CRITICAL FIX: Use explicit get() method instead of container semantics
        if cache_key:
            cached_value = self.disk_cache.get(cache_key)
            if cached_value is not None:
                logger.warning(
                    f"All retry attempts failed. Returning cached result as fallback. "
                    f"Last error: {last_error}"
                )
                return cached_value
        
        # Save circuit breaker state
        try:
            circuit_state_path = self.debug_dir / 'circuit-state.json'
            self.retry_handler.circuit_breaker.save_state(circuit_state_path)
        except Exception as e:
            logger.debug(f"Failed to save circuit breaker state: {e}")
        
        # Write debug file for failed call
        try:
            write_debug_file(
                request_id=request_id,
                call_type="sync",
                payload_validation={"ok": payload_validation_ok, "errors": payload_validation_errors},
                api_error=api_error_data,
                response_summary=response_summary,
                attempts=max_retries,
                suggested_patch=None,
                output_path=self.debug_dir / 'workflow-debug.json'
            )
        except Exception as e:
            logger.debug(f"Failed to write debug file: {e}")
        
        logger.error(f"All retry attempts failed for LLM call. Last error: {last_error}")
        return None
    
    def _call_with_fallback(
        self,
        model: GenerativeModel,
        parts: List[Union[str, Part]],
        generation_config: Dict[str, Any],
        timeout: int
    ) -> Any:
        """
        Fallback method for API calls when streaming fails.
        
        Tries non-streaming call with same content (potentially truncated).
        
        Args:
            model: Generative model
            parts: Content parts
            generation_config: Generation config
            timeout: Timeout in seconds
            
        Returns:
            Response or None on error
        """
        try:
            # For SDK-based calls, fallback is same as normal call
            # This is placeholder for future streaming implementation
            def _call():
                return model.generate_content(
                    parts,
                    generation_config=generation_config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            
            future = self.timeout_executor.submit(_call)
            return future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Fallback call failed: {e}")
            return None
    
    def _call_with_timeout(
        self,
        model: GenerativeModel,
        parts: List[Union[str, Part]],
        generation_config: Dict[str, Any],
        timeout: int
    ) -> Any:  # Vertex AI response type
        """Call model with timeout."""
        def _call():
            return model.generate_content(
                parts,
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
        
        future = self.timeout_executor.submit(_call)
        return future.result(timeout=timeout)
    
    def _image_to_part(self, image_path: str) -> Part:
        """Convert image path to Vertex AI Part."""
        # CRITICAL FIX: Add try/except around file read to handle missing files gracefully
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image path does not exist: {image_path}")
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return Part.from_data(image_bytes, mime_type="image/png")
        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError
        except Exception as e:
            logger.error(f"Error reading image file {image_path}: {e}", exc_info=True)
            raise
    
    def _generate_cache_key(
        self,
        model_info: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str]
    ) -> str:
        """Generate cache key for request."""
        key_parts = [
            model_info.get('id', ''),
            system_prompt,
            user_prompt
        ]
        if image_path:
            # CRITICAL FIX: Add try/except around file read to handle missing files gracefully
            try:
                if not os.path.exists(image_path):
                    logger.warning(f"Image path does not exist for cache key: {image_path}, using path as hash")
                    # Use path as fallback hash
                    image_hash = hashlib.md5(image_path.encode()).hexdigest()
                else:
                    with open(image_path, "rb") as f:
                        image_hash = hashlib.md5(f.read()).hexdigest()
                key_parts.append(image_hash)
            except Exception as e:
                logger.warning(f"Error reading image file for cache key {image_path}: {e}, using path as hash")
                # Use path as fallback hash
                image_hash = hashlib.md5(image_path.encode()).hexdigest()
                key_parts.append(image_hash)
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _parse_response(
        self,
        response: Any,
        expected_json_keys: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Parse LLM response.
        
        CRITICAL FIX: Accepts dict, list, or string responses.
        Professional companies accept various response formats and convert them.
        """
        # Helper function to convert list to dict
        def convert_list_to_dict(parsed_list: List[Any], expected_keys: Optional[List[str]]) -> Dict[str, Any]:
            """Convert list response to dict based on expected keys."""
            if expected_keys:
                if "elements" in expected_keys:
                    logger.info(f"Converting list to dict with 'elements' key (length={len(parsed_list)})")
                    return {"elements": parsed_list}
                elif "connections" in expected_keys:
                    logger.info(f"Converting list to dict with 'connections' key (length={len(parsed_list)})")
                    return {"connections": parsed_list}
                else:
                    logger.info(f"Converting list to dict with 'data' key (length={len(parsed_list)})")
                    return {"data": parsed_list}
            else:
                logger.info(f"Converting list to dict with 'data' key (length={len(parsed_list)})")
                return {"data": parsed_list}
        
        # CRITICAL FIX: Accept dict/str/list responses before checking .text attribute
        if isinstance(response, dict):
            # Already a dict - validate expected keys if provided
            if expected_json_keys:
                missing = [k for k in expected_json_keys if k not in response]
                if missing:
                    logger.warning(f"Expected JSON keys missing: {missing}")
            return response
        
        if isinstance(response, list):
            # CRITICAL FIX: Handle list responses (convert to dict)
            logger.info(f"LLM returned list instead of dict - converting... (length={len(response)})")
            return convert_list_to_dict(response, expected_json_keys)
        
        if isinstance(response, str):
            # Already a string - try to parse as JSON
            text = response.strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if expected_json_keys:
                        missing = [k for k in expected_json_keys if k not in parsed]
                        if missing:
                            logger.warning(f"Expected JSON keys missing: {missing}")
                    return parsed
                elif isinstance(parsed, list):
                    # CRITICAL FIX: Handle parsed list from string
                    logger.info(f"Parsed JSON string is a list - converting to dict... (length={len(parsed)})")
                    return convert_list_to_dict(parsed, expected_json_keys)
                return text
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, list):
                            return convert_list_to_dict(parsed, expected_json_keys)
                        return parsed
                    except json.JSONDecodeError:
                        pass
                return text
        
        # Check for .text attribute (Vertex AI response object)
        if not hasattr(response, 'text'):
            logger.error(f"LLM response has no 'text' attribute and is not dict/str/list. Type: {type(response)}")
            return {}
        
        text = response.text.strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Validate expected keys
                if expected_json_keys:
                    missing = [k for k in expected_json_keys if k not in parsed]
                    if missing:
                        logger.warning(f"Expected JSON keys missing: {missing}")
                return parsed
            elif isinstance(parsed, list):
                # CRITICAL FIX: Handle parsed list from response.text
                logger.info(f"Parsed response.text is a list - converting to dict... (length={len(parsed)})")
                return convert_list_to_dict(parsed, expected_json_keys)
            return text
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    if isinstance(parsed, list):
                        return convert_list_to_dict(parsed, expected_json_keys)
                    return parsed
                except json.JSONDecodeError:
                    pass
            return text
    
    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> Optional[List[float]]:
        """
        Get image embedding using Vertex AI MultiModal Embedding Model.
        
        Args:
            image_input: Path to image (str) or PIL.Image object
            
        Returns:
            Embedding vector or None on error
        """
        try:
            from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
            
            # Initialize model
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            
            # Convert input to Vertex Image
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    logger.error(f"Image path does not exist: {image_input}")
                    return None
                # Read file as bytes
                with open(image_input, 'rb') as f:
                    image_bytes = f.read()
                vertex_image = VertexImage(image_bytes=image_bytes)
            elif isinstance(image_input, Image.Image):
                # Convert PIL to bytes
                buffered = BytesIO()
                image_input.save(buffered, format="PNG")
                vertex_image = VertexImage(image_bytes=buffered.getvalue())
            else:
                logger.error(f"Invalid image input type: {type(image_input)}")
                return None
            
            # Get embedding
            embeddings = model.get_embeddings(image=vertex_image)
            if embeddings and hasattr(embeddings, 'image_embedding') and embeddings.image_embedding:
                return list(embeddings.image_embedding)
            return None
        except Exception as e:
            logger.error(f"Error getting image embedding: {e}", exc_info=True)
            return None


class DummyLLMClient:
    """
    Lightweight dummy LLM client used for local/offline testing when real LLM
    provider credentials are not available. This implements a minimal subset of
    the LLMClient interface used by the pipeline so analysis can run without
    external API calls.
    """

    def __init__(self, project_id: str, default_location: str, config: Dict[str, Any]):
        self.project_id = project_id
        self.default_location = default_location
        self.config = config or {}
        self.debug_dir = Path(self.config.get('paths', {}).get('debug_dir', 'outputs/debug'))
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def call_llm(self, *args, **kwargs) -> Optional[Union[Dict[str, Any], str]]:
        """Return a conservative, empty-but-valid analysis scaffold so the
        rest of the pipeline can continue during offline runs.
        """
        # Minimal compatible structure the pipeline expects
        response = {
            "elements": [],
            "connections": [],
            "quality_score": 0.0,
            "kpis": {}
        }
        return response

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> Optional[List[float]]:
        # Return a fixed small embedding to allow similarity steps to run
        return [0.0] * 16

