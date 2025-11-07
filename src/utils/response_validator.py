"""
Response Validator - Defensive JSON validation for LLM responses.

Provides validation functions to check LLM responses before parsing,
preventing ValidationError crashes from corrupt or empty responses.

Pattern 2: Defensive JSON-Validierung
This is the direct solution for the "NoneType Time Bomb" error (C3).
It catches corrupt or empty LLM responses before they reach Pydantic models.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def is_raw_response_valid(
    raw_response: Any,
    expected_keys: Optional[List[str]] = None,
    required_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate raw LLM response before parsing.
    
    Pattern 2: Defensive "Gate" that checks if LLM response has expected structure
    ("elements", "connections", "ports") before attempting to parse it.
    
    This prevents ValidationError crashes from corrupt or empty responses.
    
    Args:
        raw_response: Raw LLM response (dict, str, or response object)
        expected_keys: Optional list of expected top-level keys (e.g., ["elements", "connections"])
        required_keys: Optional list of required keys (must be present)
        
    Returns:
        True if response is valid, False otherwise
    """
    if raw_response is None:
        logger.warning("LLM response is None - validation failed")
        return False
    
    # Handle string responses (try to parse as JSON)
    if isinstance(raw_response, str):
        try:
            import json
            raw_response = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.warning("LLM response is string but not valid JSON - validation failed")
            return False
    
    # Handle response objects with .text attribute (Vertex AI)
    if hasattr(raw_response, 'text') and not isinstance(raw_response, dict):
        try:
            import json
            raw_response = json.loads(raw_response.text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("LLM response object has .text but not valid JSON - validation failed")
            return False
    
    # CRITICAL FIX: Accept lists and strings (parser will convert them)
    # Professional companies accept various response formats (dict, list, string)
    if isinstance(raw_response, list):
        logger.info(f"LLM response is a list (length={len(raw_response)}) - will convert to dict during parsing")
        return True  # Accept list, parser will convert to dict
    
    # Must be a dictionary at this point (or string, which will be parsed)
    if not isinstance(raw_response, (dict, str)):
        logger.warning(f"LLM response is not a dictionary or string (type: {type(raw_response).__name__}) - validation failed")
        return False
    
    # Check for empty dictionary
    if not raw_response:
        logger.warning("LLM response is empty dictionary - validation failed")
        return False
    
    # Check required keys (must be present)
    if required_keys:
        missing_keys = [key for key in required_keys if key not in raw_response]
        if missing_keys:
            logger.warning(f"LLM response missing required keys: {missing_keys} - validation failed")
            return False
    
    # Check expected keys (at least one should be present)
    if expected_keys:
        found_keys = [key for key in expected_keys if key in raw_response]
        if not found_keys:
            logger.warning(f"LLM response has none of expected keys: {expected_keys} - validation failed")
            return False
        
        # Validate that expected keys contain valid data (not empty lists/dicts)
        for key in found_keys:
            value = raw_response.get(key)
            if value is None:
                logger.warning(f"LLM response key '{key}' is None - validation failed")
                return False
            # For lists, check if empty (might be acceptable, but log warning)
            if isinstance(value, list) and len(value) == 0:
                logger.debug(f"LLM response key '{key}' is empty list (acceptable but unusual)")
    
    # Basic structure validation passed
    logger.debug("LLM response passed validation")
    return True


def validate_elements_response(response: Dict[str, Any]) -> bool:
    """
    Validate response containing elements.
    
    Args:
        response: Response dictionary
        
    Returns:
        True if valid, False otherwise
    """
    return is_raw_response_valid(
        response,
        expected_keys=["elements", "connections"],
        required_keys=[]  # Elements might be empty, that's OK
    )


def validate_connections_response(response: Dict[str, Any]) -> bool:
    """
    Validate response containing connections.
    
    Args:
        response: Response dictionary
        
    Returns:
        True if valid, False otherwise
    """
    return is_raw_response_valid(
        response,
        expected_keys=["connections"],
        required_keys=[]  # Connections might be empty, that's OK
    )


def validate_full_analysis_response(response: Dict[str, Any]) -> bool:
    """
    Validate full analysis response with elements and connections.
    
    Args:
        response: Response dictionary
        
    Returns:
        True if valid, False otherwise
    """
    return is_raw_response_valid(
        response,
        expected_keys=["elements", "connections"],
        required_keys=[]  # Both might be empty, that's OK
    )

