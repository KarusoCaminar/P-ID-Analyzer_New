"""
Debug capture utility for API requests and responses.

Captures request/response data for debugging and writes debug files.
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


def sanitize_for_json(obj: Any) -> Any:
    """
    Sanitize object for JSON serialization.
    
    Removes non-serializable objects and converts to safe types.
    
    Args:
        obj: Object to sanitize
        
    Returns:
        Sanitized object
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Try to convert objects with __dict__ to dict
        try:
            return sanitize_for_json(obj.__dict__)
        except Exception:
            return str(obj)
    else:
        # Fallback to string representation
        return str(obj)


def capture_request(
    request_id: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    output_dir: Path = Path("outputs/debug")
) -> Path:
    """
    Capture and save request data.
    
    Args:
        request_id: Unique request ID
        payload: Request payload
        headers: Request headers (optional)
        output_dir: Output directory for debug files
        
    Returns:
        Path to saved request file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    request_data = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "payload": sanitize_for_json(payload),
        "headers": sanitize_for_json(headers) if headers else None
    }
    
    file_path = output_dir / f"request-{request_id}.json"
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Captured request to {file_path}")
    except Exception as e:
        logger.error(f"Failed to capture request: {e}", exc_info=True)
    
    return file_path


def capture_response(
    request_id: str,
    response: Any,
    status_code: Optional[int] = None,
    error: Optional[Exception] = None,
    output_dir: Path = Path("outputs/debug")
) -> Path:
    """
    Capture and save response data.
    
    Args:
        request_id: Unique request ID (matching request)
        response: Response object or data
        status_code: HTTP status code (if available)
        error: Exception if request failed
        output_dir: Output directory for debug files
        
    Returns:
        Path to saved response file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to extract response data
    response_data = None
    if hasattr(response, 'text'):
        response_data = response.text
    elif hasattr(response, '__dict__'):
        try:
            response_data = sanitize_for_json(response.__dict__)
        except Exception:
            response_data = str(response)
    else:
        response_data = sanitize_for_json(response)
    
    error_data = None
    if error:
        error_data = {
            "type": type(error).__name__,
            "message": str(error),
            "args": [str(arg) for arg in error.args] if hasattr(error, 'args') else None
        }
    
    response_capture = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "status_code": status_code,
        "response": response_data,
        "error": error_data
    }
    
    file_path = output_dir / f"response-{request_id}.json"
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response_capture, f, indent=2, ensure_ascii=False)
        logger.debug(f"Captured response to {file_path}")
    except Exception as e:
        logger.error(f"Failed to capture response: {e}", exc_info=True)
    
    return file_path


def write_debug_file(
    request_id: str,
    call_type: str,
    payload_validation: Dict[str, Any],
    api_error: Optional[Dict[str, Any]],
    response_summary: Dict[str, Any],
    attempts: int,
    suggested_patch: Optional[Dict[str, Any]] = None,
    output_path: Path = Path("outputs/debug/workflow-debug.json")
) -> Path:
    """
    Write workflow debug file with complete request/response information.
    
    Args:
        request_id: Unique request ID
        call_type: "stream" or "sync"
        payload_validation: Validation results {"ok": bool, "errors": []}
        api_error: Error information {"message": str, "stack": str, "code": str} or None
        response_summary: Response summary {"success": bool, "status_code": int, "tokens": int}
        attempts: Number of attempts made
        suggested_patch: Suggested patch (optional)
        output_path: Path to debug file
        
    Returns:
        Path to written debug file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "requestId": request_id,
        "call_type": call_type,
        "payload_validation": payload_validation,
        "api_error": api_error,
        "response_summary": response_summary,
        "attempts": attempts,
        "suggested_patch": suggested_patch
    }
    
    # Atomic write: write to temp file first, then rename
    temp_path = output_path.with_suffix('.tmp')
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_path.replace(output_path)
        
        logger.debug(f"Written debug file to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write debug file: {e}", exc_info=True)
        # Clean up temp file if exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
    
    return output_path

