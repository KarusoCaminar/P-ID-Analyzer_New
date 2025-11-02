"""
JSON Schema definitions for API request/response validation.

Provides strict schema validation for Gemini API payloads and responses.
"""

import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Request payload schema for Gemini API
GEMINI_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "contents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "parts": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "inline_data": {
                                            "type": "object",
                                            "properties": {
                                                "mime_type": {"type": "string"},
                                                "data": {"type": "string"}
                                            },
                                            "required": ["mime_type", "data"]
                                        }
                                    },
                                    "required": ["inline_data"]
                                }
                            ]
                        }
                    }
                },
                "required": ["parts"]
            }
        },
        "generation_config": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "top_p": {"type": "number"},
                "top_k": {"type": "integer"},
                "max_output_tokens": {"type": "integer"}
            }
        },
        "safety_settings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "threshold": {"type": "string"}
                },
                "required": ["category", "threshold"]
            }
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function_declarations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "properties": {"type": "object"},
                                        "required": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["type"]
                                }
                            },
                            "required": ["name"]
                        }
                    }
                }
            }
        }
    }
}


# Response schema for Gemini API
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "properties": {
                            "parts": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "functionCall": {"type": "object"}
                                            }
                                        }
                                    ]
                                }
                            },
                            "role": {"type": "string"}
                        },
                        "required": ["parts"]
                    },
                    "finishReason": {"type": "string"},
                    "safetyRatings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "probability": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "usageMetadata": {
            "type": "object",
            "properties": {
                "promptTokenCount": {"type": "integer"},
                "candidatesTokenCount": {"type": "integer"},
                "totalTokenCount": {"type": "integer"}
            }
        }
    }
}


# Tool metadata schema
TOOL_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "properties": {"type": "object"},
                "required": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["type"]
        }
    },
    "required": ["name"]
}


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[List[str]]]:
    """
    Validate data against schema using jsonschema.
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, errors_list)
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except ImportError:
        # Fallback to basic validation if jsonschema not available
        logger.warning("jsonschema not available, using basic validation")
        return _basic_validate(data, schema), None
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Schema validation error: {str(e)}"]


def _basic_validate(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Basic validation fallback if jsonschema is not available.
    
    Args:
        data: Data to validate
        schema: Schema to validate against
        
    Returns:
        True if basic checks pass
    """
    # Very basic type checking
    if not isinstance(data, dict):
        return False
    
    # Check required fields if specified
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            return False
    
    return True


def validate_request_payload(payload: Dict[str, Any]) -> tuple[bool, Optional[List[str]]]:
    """
    Validate Gemini request payload against schema.
    
    Args:
        payload: Request payload to validate
        
    Returns:
        Tuple of (is_valid, errors_list)
    """
    return validate_schema(payload, GEMINI_REQUEST_SCHEMA)


def validate_response(response: Dict[str, Any]) -> tuple[bool, Optional[List[str]]]:
    """
    Validate Gemini response against schema.
    
    Args:
        response: Response to validate
        
    Returns:
        Tuple of (is_valid, errors_list)
    """
    return validate_schema(response, GEMINI_RESPONSE_SCHEMA)


def validate_tool_metadata(tool: Dict[str, Any]) -> tuple[bool, Optional[List[str]]]:
    """
    Validate tool metadata against schema.
    
    Args:
        tool: Tool metadata to validate
        
    Returns:
        Tuple of (is_valid, errors_list)
    """
    return validate_schema(tool, TOOL_METADATA_SCHEMA)

