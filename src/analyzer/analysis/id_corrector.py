"""
ID Corrector - LLM-based ID extraction and correction.

This module provides a simple, effective solution for correcting element IDs
by asking the LLM to extract the correct IDs directly from the image text.
Instead of complex matching logic, we simply send the image with current IDs
to the LLM and ask it to correct them based on what it sees in the image.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class IDCorrector:
    """
    LLM-based ID corrector that extracts correct IDs from image text.
    
    Instead of complex matching logic, this sends the image + current IDs
    to the LLM and asks it to extract the correct IDs from the image text.
    """
    
    def __init__(
        self,
        llm_client: Any,
        config_service: Any
    ):
        """
        Initialize ID Corrector.
        
        Args:
            llm_client: LLM client for API calls
            config_service: Configuration service
        """
        self.llm_client = llm_client
        self.config_service = config_service
        self.config = config_service.get_raw_config()
        
        # Get model strategy
        models_cfg = self.config.get('models', {})
        model_strategy = self.config.get('model_strategy', {})
        self.model_info = model_strategy.get('correction_model') or model_strategy.get('meta_model')
        if not self.model_info:
            # Fallback to Pro model
            self.model_info = models_cfg.get('Google Gemini 2.5 Pro', {})
        
        if not self.model_info:
            logger.warning("No correction model available. ID correction will be skipped.")
            self.model_info = {}
    
    def correct_ids(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Correct element IDs by asking LLM to extract correct IDs from image text.
        
        This is a simple, effective approach:
        1. Send image + current elements/connections to LLM
        2. Ask LLM to extract correct IDs from image text
        3. Return corrected elements and connections
        
        Args:
            image_path: Path to P&ID image
            elements: List of elements with potentially incorrect IDs
            connections: List of connections with potentially incorrect IDs
            
        Returns:
            Dictionary with corrected 'elements' and 'connections'
        """
        if not self.model_info:
            logger.warning("No correction model available. Skipping ID correction.")
            return {
                'elements': elements,
                'connections': connections
            }
        
        logger.info("=== Starting LLM-based ID Correction ===")
        logger.info(f"Correcting {len(elements)} elements and {len(connections)} connections")
        
        try:
            # Build prompt for ID correction
            prompt = self._build_id_correction_prompt(elements, connections)
            
            # Call LLM with image
            response = self.llm_client.call_llm(
                model_info=self.model_info,
                system_prompt="You are a P&ID ID extraction specialist. Your task is to extract the CORRECT element IDs from the image text.",
                user_prompt=prompt,
                image_path=image_path
            )
            
            # Parse response
            corrected_data = self._parse_response(response)
            
            if corrected_data:
                corrected_elements = corrected_data.get('elements', elements)
                corrected_connections = corrected_data.get('connections', connections)
                
                logger.info(f"ID correction complete: {len(corrected_elements)} elements, {len(corrected_connections)} connections")
                
                # Log ID changes
                self._log_id_changes(elements, corrected_elements)
                
                return {
                    'elements': corrected_elements,
                    'connections': corrected_connections
                }
            else:
                logger.warning("ID correction returned empty result. Using original IDs.")
                return {
                    'elements': elements,
                    'connections': connections
                }
                
        except Exception as e:
            logger.error(f"Error in ID correction: {e}", exc_info=True)
            return {
                'elements': elements,
                'connections': connections
            }
    
    def _build_id_correction_prompt(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for ID correction.
        
        Args:
            elements: Current elements with potentially incorrect IDs
            connections: Current connections with potentially incorrect IDs
            
        Returns:
            Prompt string for LLM
        """
        # Format current data as JSON
        elements_json = json.dumps(elements, indent=2, ensure_ascii=False)
        connections_json = json.dumps(connections, indent=2, ensure_ascii=False)
        
        prompt = f"""**ROLE:** You are a P&ID ID extraction specialist.

**TASK:** Extract the CORRECT element IDs from the image text and correct the provided elements and connections.

**CURRENT DATA (may have incorrect IDs):**
Elements:
{elements_json}

Connections:
{connections_json}

**CRITICAL INSTRUCTIONS:**
1. Look at the image and find the ACTUAL text labels next to each element symbol
2. Extract the CORRECT P&ID Tag Names (IDs) from the image text (e.g., "P-201", "Fv-3-3040", "MV3121A", "PU3121")
3. For each element in the list, correct the "id" field to match what you see in the image
4. Update all connection "from_id" and "to_id" fields to use the corrected IDs
5. DO NOT change element types, bboxes, or other fields - ONLY correct the IDs
6. If you cannot find an ID in the image, keep the original ID but add a note

**OUTPUT FORMAT:**
Return a JSON object with:
- "elements": List of elements with CORRECTED IDs (all other fields unchanged)
- "connections": List of connections with CORRECTED IDs (from_id, to_id updated)

**EXAMPLE:**
If current element has id: "PU321" but image shows "PU3121", correct it to "PU3121"
If current connection has from_id: "CHP1" -> to_id: "PU321", but image shows "CHP1" -> "PU3121", correct it to "CHP1" -> "PU3121"

**RETURN ONLY VALID JSON, NO ADDITIONAL TEXT:**
```json
{{
  "elements": [...],
  "connections": [...]
}}
```"""
        
        return prompt
    
    def _parse_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract corrected elements and connections.
        
        Args:
            response: LLM response
            
        Returns:
            Dictionary with 'elements' and 'connections' or None
        """
        try:
            # Handle different response types
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                # Already parsed
                if 'elements' in response or 'connections' in response:
                    return response
                response_text = json.dumps(response)
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return None
            
            # Try to extract JSON from response
            # LLM might wrap JSON in markdown code blocks
            if '```json' in response_text:
                # Extract JSON from markdown code block
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif '```' in response_text:
                # Extract JSON from generic code block
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            
            # Parse JSON
            corrected_data = json.loads(response_text)
            
            # Validate structure
            if not isinstance(corrected_data, dict):
                logger.warning("Response is not a dictionary")
                return None
            
            if 'elements' not in corrected_data and 'connections' not in corrected_data:
                logger.warning("Response does not contain 'elements' or 'connections'")
                return None
            
            return corrected_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}", exc_info=True)
            return None
    
    def _log_id_changes(
        self,
        original_elements: List[Dict[str, Any]],
        corrected_elements: List[Dict[str, Any]]
    ) -> None:
        """
        Log ID changes for debugging.
        
        Args:
            original_elements: Original elements
            corrected_elements: Corrected elements
        """
        # Build mapping of original to corrected IDs
        original_ids = {el.get('id'): el for el in original_elements}
        corrected_ids = {el.get('id'): el for el in corrected_elements}
        
        changes = []
        for orig_id, orig_el in original_ids.items():
            # Find matching element (by bbox or type)
            for corr_id, corr_el in corrected_ids.items():
                # Simple matching: same bbox or same type + similar position
                orig_bbox = orig_el.get('bbox', {})
                corr_bbox = corr_el.get('bbox', {})
                
                if (orig_bbox.get('x') == corr_bbox.get('x') and
                    orig_bbox.get('y') == corr_bbox.get('y')):
                    # Same position - check if ID changed
                    if orig_id != corr_id:
                        changes.append(f"{orig_id} -> {corr_id}")
                    break
        
        if changes:
            logger.info(f"ID corrections: {len(changes)} IDs changed")
            for change in changes[:10]:  # Log first 10 changes
                logger.info(f"  {change}")
            if len(changes) > 10:
                logger.info(f"  ... and {len(changes) - 10} more")
        else:
            logger.info("No ID changes detected")

