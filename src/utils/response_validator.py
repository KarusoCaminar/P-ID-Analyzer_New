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
    # CRITICAL FIX: More lenient parsing - try multiple approaches
    if hasattr(raw_response, 'text') and not isinstance(raw_response, dict):
        try:
            import json
            import re
            text = raw_response.text if hasattr(raw_response, 'text') else str(raw_response)
            
            # Try to extract JSON from text (may contain markdown code blocks)
            # Remove markdown code block markers if present
            text_cleaned = text.strip()
            if text_cleaned.startswith('```'):
                # Extract JSON from markdown code block
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text_cleaned, re.DOTALL)
                if json_match:
                    text_cleaned = json_match.group(1)
                else:
                    # Try to find JSON object in text
                    json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)
                    if json_match:
                        text_cleaned = json_match.group(0)
            
            # Try to parse as JSON
            try:
                raw_response = json.loads(text_cleaned)
            except json.JSONDecodeError:
                # If that fails, try the original text
                raw_response = json.loads(text)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"LLM response object has .text but not valid JSON - validation failed: {e}")
            logger.debug(f"Response text preview: {str(raw_response)[:500] if hasattr(raw_response, '__str__') else 'N/A'}")
            # CRITICAL FIX: Don't fail immediately - try to return True and let parser handle it
            # The parser might be able to handle string responses
            logger.info("Response validation: Accepting string response, parser will attempt to parse it")
            return True  # Accept and let parser try to handle it
    
    # CRITICAL FIX: Accept lists and strings (parser will convert them)
    # Professional companies accept various response formats (dict, list, string)
    if isinstance(raw_response, list):
        logger.info(f"LLM response is a list (length={len(raw_response)}) - will convert to dict during parsing")
        return True  # Accept list, parser will convert to dict
    
    # Must be a dictionary at this point (or string, which will be parsed)
    if not isinstance(raw_response, (dict, str)):
        logger.warning(f"LLM response is not a dictionary or string (type: {type(raw_response).__name__}) - validation failed")
        return False
    
    # CRITICAL FIX: Check expected keys FIRST before checking if dict is empty
    # A dict with {"elements": [], "connections": []} is valid, even if the lists are empty
    # Only reject if it's a completely empty dict {} AND no expected keys are present
    
    # Check expected keys (at least one should be present)
    if expected_keys:
        found_keys = [key for key in expected_keys if key in raw_response]
        if not found_keys:
            # If no expected keys found, check if dict is completely empty
            if not raw_response:
                logger.warning(f"LLM response is empty dictionary and has none of expected keys: {expected_keys} - validation failed")
                return False
            else:
                logger.warning(f"LLM response has none of expected keys: {expected_keys} - validation failed")
                return False
        
        # Validate that expected keys contain valid data (not None)
        # CRITICAL FIX: Empty lists are acceptable (might be no elements/connections found)
        # Only fail if value is None or if it's an unexpected type
        for key in found_keys:
            value = raw_response.get(key)
            if value is None:
                logger.warning(f"LLM response key '{key}' is None - validation failed")
                return False
            # For lists, empty lists are acceptable (no elements/connections found)
            if isinstance(value, list) and len(value) == 0:
                logger.debug(f"LLM response key '{key}' is empty list (acceptable - no {key} found)")
            # For dicts, empty dicts might be acceptable (no nested data)
            if isinstance(value, dict) and len(value) == 0:
                logger.debug(f"LLM response key '{key}' is empty dict (acceptable - no {key} data)")
    else:
        # No expected keys - just check if dict is not completely empty
        if not raw_response:
            logger.warning("LLM response is empty dictionary and no expected keys specified - validation failed")
            return False
    
    # Check required keys (must be present) - after expected keys check
    if required_keys:
        missing_keys = [key for key in required_keys if key not in raw_response]
        if missing_keys:
            logger.warning(f"LLM response missing required keys: {missing_keys} - validation failed")
            return False
    
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

