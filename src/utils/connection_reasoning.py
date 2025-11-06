"""
Connection Reasoning with Chain-of-Thought (CoT) Logic.

Uses LLM for intelligent connection validation and completion based on:
- P&ID domain knowledge (valves before sensors, pumps before mixers)
- Spatial relationships (proximity, flow direction)
- Component semantics (pumps are sources, sinks are destinations)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)


def reason_connections_with_cot(
    elements: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    llm_client: Any,
    image_path: str,
    model_info: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Use Chain-of-Thought reasoning to validate and complete connections.
    
    Args:
        elements: List of detected elements
        connections: List of existing connections
        llm_client: LLM client for reasoning
        image_path: Path to image for context
        model_info: Model configuration
        
    Returns:
        (validated_connections, suggested_connections, result_metadata)
        where result_metadata contains: splits, merges, missing_elements, dangling_connections
    """
    logger.info("Chain-of-Thought Reasoning: Validating and completing connections...")
    
    # Prepare elements data for LLM
    elements_for_llm = []
    for el in elements:
        el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
        elements_for_llm.append({
            'id': el_dict.get('id', ''),
            'label': el_dict.get('label', ''),
            'type': el_dict.get('type', ''),
            'bbox': el_dict.get('bbox', {}),
            'confidence': el_dict.get('confidence', 0.5)
        })
    
    # Prepare connections data for LLM
    connections_for_llm = []
    for conn in connections:
        conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
        connections_for_llm.append({
            'from_id': conn_dict.get('from_id', ''),
            'to_id': conn_dict.get('to_id', ''),
            'confidence': conn_dict.get('confidence', 0.5)
        })
    
    # OPTIMIZED: Kürzerer Prompt für Token-Limit
    # Reduziere Elemente auf wichtige Felder (keine BBox, kein Confidence)
    elements_summary = []
    for el in elements_for_llm[:100]:  # Erhöht auf 100 für bessere Vollständigkeit
        elements_summary.append({
            'id': el.get('id', ''),
            'type': el.get('type', ''),
            'label': el.get('label', '')[:40]  # Kürze Label auf 40 Zeichen
        })
    
    # Build element ID set for missing element detection
    element_ids_set = {el.get('id', '') for el in elements_summary}
    
    connections_summary = []
    for conn in connections_for_llm[:100]:  # Erhöht auf 100 für bessere Vollständigkeit
        from_id = conn.get('from_id', '')
        to_id = conn.get('to_id', '')
        connections_summary.append({
            'from_id': from_id,
            'to_id': to_id,
            'confidence': conn.get('confidence', 0.5)
        })
    
    # Build missing elements list from connections (pre-detection)
    missing_elements_detected = []
    for conn in connections_summary:
        from_id = conn.get('from_id', '')
        to_id = conn.get('to_id', '')
        if from_id and from_id not in element_ids_set:
            if not any(m.get('label') == from_id for m in missing_elements_detected):
                missing_elements_detected.append({'label': from_id, 'id': f"Missing_{from_id}"})
        if to_id and to_id not in element_ids_set:
            if not any(m.get('label') == to_id for m in missing_elements_detected):
                missing_elements_detected.append({'label': to_id, 'id': f"Missing_{to_id}"})
    
    prompt = f"""Analyse P&ID Verbindungen mit Chain-of-Thought.

**ELEMENTE ({len(elements_summary)}):**
{json.dumps(elements_summary, ensure_ascii=False)}

**VERBINDUNGEN ({len(connections_summary)}):**
{json.dumps(connections_summary, ensure_ascii=False)}

**HINWEIS:** Analysiere alle Verbindungen. Wenn 'from_id' oder 'to_id' NICHT in ELEMENTE-Liste → Missing Element.

**CHAIN-OF-THOUGHT:**

**1. Domain Knowledge:**
- Pumpen=SOURCES, Valves=zwischen Pumpen/Sensoren, Flow Sensors=nach Valves, Mixer=nach Sensoren, Sinks=Endpunkte
- Pattern: CHP1/CHP2 -> MV3121A/MV3121B -> PU3121/PU3131 -> Mixer

**2. Räumliche Analyse:**
- Nahe Elemente=wahrscheinlich verbunden
- Flow: oben->unten, links->rechts

**3. Validierung:**
- Pump->Valve->Sensor->Mixer: OK
- Pump->Mixer (ohne Valve): Fehlt Valve
- Sensor->Pump: Falsche Richtung
- ISA-Supply->Valve: OK (Steuerlinie)
- ISA-Supply->Pump: Fehler (Steuerlinie≠Prozesslinie)

**4. Fehlende Verbindungen:**
- 2 Pumpen → beide zu Mixer
- Pump->Valve->Sensor → auch Sensor->Mixer

**5. Splits/Merges:**
- **Split-Punkte:** Wenn ein Element mehrere Ausgänge hat (out_degree > 1), markiere als Split
  - Beispiel: W2 -> W3 und W2 -> W4 = Split bei W2
  - Markiere Split-Position: Durchschnitt der Positionen von W2, W3, W4
- **Merge-Punkte:** Wenn ein Element mehrere Eingänge hat (in_degree > 1), markiere als Merge
  - Beispiel: Mixer-M-08 mit Eingängen von FT-10 und FT-11 = Merge bei M-08
  - Markiere Merge-Position: Durchschnitt der Positionen der Eingangs-Elemente und M-08
- **WICHTIG:** Füge Split/Merge-Elemente zur Element-Liste hinzu mit:
  - type: "Split" oder "Merge"
  - connected_from/connected_to: Liste der verbundenen Element-IDs
  - position: Berechnete Position (Baryzentrum)

**Schritt 6: Fehlende Elemente identifizieren und markieren**
- Wenn Verbindungen zu nicht erkannten Elementen führen (z.B. "K1", "B1", "B2", "B3/B4"), markiere diese:
  - Erstelle "Missing Element" Marker mit:
    - id: "Missing_K1", "Missing_B1", etc.
    - type: "Missing_Element"
    - label: Name des fehlenden Elements (z.B. "K1", "B1")
    - confidence: 0.3 (niedrig, da nicht erkannt)
    - missing_reason: "Connection points to unrecognized element"
    - connected_from/connected_to: Liste der Verbindungen, die hierher führen/wegführen
- Markiere Verbindungen, die "ins Leere" führen:
  - Füge "dangling_connection" Flag hinzu
  - Füge "missing_target" hinzu mit ID des fehlenden Elements

**Schritt 7: Falsche Verbindungen entfernen + Steuerlinien-Validierung**
- Verbindungen die gegen Domain Knowledge verstoßen
- Verbindungen mit nicht-existierenden Elementen
- **CRITICAL FIX 2.2: Steuerlinien-Halluzinationen entfernen:**
  - Entferne Verbindungen von ISA-Supply zu Pump/Flow Sensor/Mixer (ISA-Supply ist Steuerlinie, nicht Prozesslinie)
  - Entferne Verbindungen von ISA-Supply zu Reactor (ISA-Supply ist Steuerlinie, nicht Prozesslinie)
  - Behalte nur Verbindungen von ISA-Supply zu Valves (Steuerlinien-Semantik)

**RETURN FORMAT (STRICT JSON - NO MARKDOWN):**
{{
  "validated_connections": [{{"from_id": "P-201", "to_id": "M-08", "confidence": 0.9}}],
  "suggested_connections": [{{"from_id": "P-504", "to_id": "M-08", "confidence": 0.8}}],
  "removed_connections": [{{"from_id": "FT-10", "to_id": "P-201"}}],
  "splits": [{{"id": "Split_at_W2", "type": "Split", "position": {{"x": 0.5, "y": 0.5}}, "connected_from": ["W2"], "connected_to": ["W3", "W4"]}}],
  "merges": [{{"id": "Merge_at_M-08", "type": "Merge", "position": {{"x": 0.5, "y": 0.5}}, "connected_from": ["FT-10", "FT-11"], "connected_to": ["M-08"]}}],
  "missing_elements": [{{"id": "Missing_K1", "type": "Missing_Element", "label": "K1", "confidence": 0.3}}],
  "dangling_connections": [{{"from_id": "Konzentration", "to_id": "Missing_K1", "missing_target": "K1"}}]
}}

**WICHTIG:**
- Nur JSON zurückgeben (kein Markdown ```json, kein Text vorher/nachher)
- Confidence >0.7 für Vorschläge
- Max 2000 Tokens Response
- Missing Elements: Prüfe alle Verbindungen, wenn from_id/to_id NICHT in ELEMENTE → Missing Element
"""
    
    try:
        # OPTIMIZED: Kürzerer System-Prompt für Token-Limit
        system_prompt = "P&ID Expert. Analyse Verbindungen mit Chain-of-Thought. Return ONLY valid JSON."
        
        # OPTIMIZED: Set max_tokens in model_info if available
        model_info_optimized = model_info.copy() if model_info else {}
        if 'generation_config' not in model_info_optimized:
            model_info_optimized['generation_config'] = {}
        if 'max_output_tokens' not in model_info_optimized['generation_config']:
            model_info_optimized['generation_config']['max_output_tokens'] = 2000  # Limit response tokens
        
        response = llm_client.call_llm(
            model_info_optimized,
            system_prompt=system_prompt,
            user_prompt=prompt,
            image_path=image_path
        )
        
        # Handle None response (fallback)
        if response is None:
            logger.warning("Chain-of-Thought Reasoning: LLM returned None, using fallback")
            empty_metadata = {
                'splits': [],
                'merges': [],
                'missing_elements': [],
                'dangling_connections': []
            }
            return connections, [], empty_metadata
        
        # CRITICAL FIX: Robust JSON extraction using regex (handles text before/after JSON)
        if isinstance(response, str):
            try:
                import re
                response_str = response.strip()
                
                # Remove markdown code blocks if present
                if '```json' in response_str:
                    start = response_str.find('```json') + 7
                    end = response_str.find('```', start)
                    if end > start:
                        response_str = response_str[start:end].strip()
                elif '```' in response_str:
                    start = response_str.find('```') + 3
                    end = response_str.find('```', start)
                    if end > start:
                        response_str = response_str[start:end].strip()
                
                # ROBUST JSON EXTRACTION: Find complete JSON object even with text before/after
                # Method 1: Try to find JSON object boundaries by counting braces (most robust)
                if '{' in response_str:
                    start = response_str.find('{')
                    # Find matching closing brace by counting nested braces
                    brace_count = 0
                    end = len(response_str)
                    for i in range(start, len(response_str)):
                        char = response_str[i]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if brace_count == 0:  # Found complete JSON object
                        json_str = response_str[start:end]
                        try:
                            response = json.loads(json_str)
                            logger.info("Chain-of-Thought Reasoning: Successfully extracted JSON using brace counting")
                        except json.JSONDecodeError:
                            # If brace counting found invalid JSON, try regex fallback
                            raise
                    else:
                        # Incomplete JSON (unmatched braces) - try regex as fallback
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'
                        json_match = re.search(json_pattern, response_str, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            response = json.loads(json_str)
                            logger.info("Chain-of-Thought Reasoning: Successfully extracted JSON using regex fallback")
                        else:
                            raise json.JSONDecodeError("No valid JSON object found in response", response_str, 0)
                else:
                    raise json.JSONDecodeError("No JSON object found in response", response_str, 0)
            except json.JSONDecodeError as e:
                logger.warning(f"Chain-of-Thought Reasoning: Could not parse JSON from string response: {e}")
                logger.debug(f"Response preview: {response[:200] if len(response) > 200 else response}")
                empty_metadata = {
                    'splits': [],
                    'merges': [],
                    'missing_elements': [],
                    'dangling_connections': []
                }
                return connections, [], empty_metadata
        
        if isinstance(response, dict):
            validated_connections = response.get('validated_connections', connections_for_llm)
            suggested_connections = response.get('suggested_connections', [])
            removed_connections = response.get('removed_connections', [])
            splits = response.get('splits', [])
            merges = response.get('merges', [])
            missing_elements = response.get('missing_elements', [])
            dangling_connections = response.get('dangling_connections', [])
            # CRITICAL FIX: Removed chain_of_thought from response to save tokens
            # The CoT instructions in the prompt are still followed by the model, but
            # the model no longer needs to write its reasoning into the JSON response
            
            logger.info(f"Chain-of-Thought Reasoning: {len(validated_connections)} validierte Verbindungen, "
                       f"{len(suggested_connections)} vorgeschlagene Verbindungen, "
                       f"{len(removed_connections)} entfernte Verbindungen, "
                       f"{len(splits)} Splits, {len(merges)} Merges, "
                       f"{len(missing_elements)} fehlende Elemente, "
                       f"{len(dangling_connections)} dangling Verbindungen")
            
            # Map validated connections back to original connections
            validated_conns = []
            for val_conn in validated_connections:
                # Find matching original connection
                from_id = val_conn.get('from_id', '')
                to_id = val_conn.get('to_id', '')
                
                matching_conn = None
                for conn in connections:
                    conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
                    if (conn_dict.get('from_id', '') == from_id or 
                        conn_dict.get('from_id', '').replace(' ', '-') == from_id.replace(' ', '-')) and \
                       (conn_dict.get('to_id', '') == to_id or 
                        conn_dict.get('to_id', '').replace(' ', '-') == to_id.replace(' ', '-')):
                        matching_conn = conn
                        break
                
                if matching_conn:
                    # Update connection with reasoning
                    conn_dict = matching_conn if isinstance(matching_conn, dict) else matching_conn.model_dump() if hasattr(matching_conn, 'model_dump') else matching_conn.__dict__ if hasattr(matching_conn, '__dict__') else {}
                    conn_dict['from_id'] = from_id
                    conn_dict['to_id'] = to_id
                    conn_dict['confidence'] = val_conn.get('confidence', conn_dict.get('confidence', 0.5))
                    conn_dict['reasoning'] = val_conn.get('reasoning', '')
                    validated_conns.append(matching_conn)
                else:
                    # New connection (from validated)
                    validated_conns.append(val_conn)
            
            # Remove connections that should be removed
            removed_conns = []
            for rem_conn in removed_connections:
                from_id = rem_conn.get('from_id', '')
                to_id = rem_conn.get('to_id', '')
                
                for conn in validated_conns:
                    conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
                    if (conn_dict.get('from_id', '') == from_id or 
                        conn_dict.get('from_id', '').replace(' ', '-') == from_id.replace(' ', '-')) and \
                       (conn_dict.get('to_id', '') == to_id or 
                        conn_dict.get('to_id', '').replace(' ', '-') == to_id.replace(' ', '-')):
                        validated_conns.remove(conn)
                        removed_conns.append(conn)
                        break
            
            # Add suggested connections
            for sug_conn in suggested_connections:
                # Check if connection already exists
                from_id = sug_conn.get('from_id', '')
                to_id = sug_conn.get('to_id', '')
                
                exists = False
                for val_conn in validated_conns:
                    val_conn_dict = val_conn if isinstance(val_conn, dict) else val_conn.model_dump() if hasattr(val_conn, 'model_dump') else val_conn.__dict__ if hasattr(val_conn, '__dict__') else {}
                    if (val_conn_dict.get('from_id', '') == from_id or 
                        val_conn_dict.get('from_id', '').replace(' ', '-') == from_id.replace(' ', '-')) and \
                       (val_conn_dict.get('to_id', '') == to_id or 
                        val_conn_dict.get('to_id', '').replace(' ', '-') == to_id.replace(' ', '-')):
                        exists = True
                        break
                
                if not exists:
                    # Add suggested connection
                    sug_conn['reasoning'] = sug_conn.get('reasoning', '')
                    validated_conns.append(sug_conn)
            
            logger.info(f"Chain-of-Thought Reasoning: Finale {len(validated_conns)} Verbindungen "
                       f"({len(suggested_connections)} hinzugefügt, {len(removed_conns)} entfernt)")
            
            # Return enriched results with splits, merges, and missing elements
            result_metadata = {
                'splits': splits,
                'merges': merges,
                'missing_elements': missing_elements,
                'dangling_connections': dangling_connections
            }
            
            # Add metadata to validated connections for later processing
            for conn in validated_conns:
                if isinstance(conn, dict):
                    conn['_metadata'] = result_metadata
            
            return validated_conns, suggested_connections, result_metadata
        else:
            logger.warning(f"Chain-of-Thought Reasoning: Response war kein Dict, sondern {type(response)}")
            empty_metadata = {
                'splits': [],
                'merges': [],
                'missing_elements': [],
                'dangling_connections': []
            }
            return connections, [], empty_metadata
            
    except Exception as e:
        logger.error(f"Error in Chain-of-Thought Reasoning: {e}", exc_info=True)
        empty_metadata = {
            'splits': [],
            'merges': [],
            'missing_elements': [],
            'dangling_connections': []
        }
        return connections, [], empty_metadata

