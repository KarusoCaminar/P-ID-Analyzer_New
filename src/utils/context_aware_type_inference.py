"""
Context-Aware Type Inference using P&ID Naming Conventions.

Uses LLM to infer element types based on:
- Labels (FT = Flow Transmitter, Fv = Flow Valve, P = Pump, etc.)
- Context (adjacent elements, flow direction)
- Visual patterns (symbol shapes)
"""

import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


def infer_types_from_context(
    elements: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    llm_client: Any,
    image_path: str,
    model_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Infer element types using context-aware reasoning (P&ID naming conventions).
    
    Args:
        elements: List of elements (may have "Unknown" or incorrect types)
        connections: List of connections for context
        llm_client: LLM client for inference
        image_path: Path to image for visual context
        model_info: Model configuration
        
    Returns:
        List of elements with inferred types
    """
    logger.info("Context-Aware Type Inference: Inferring types from labels and context...")
    
    # Prepare elements data for LLM
    # NOTE: This robust conversion handles both dicts and Pydantic models.
    # If you can ensure only Pydantic models arrive here, you could simplify to: el.model_dump()
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
    
    # Prepare connections for context
    connections_for_llm = []
    for conn in connections:
        conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
        connections_for_llm.append({
            'from_id': conn_dict.get('from_id', ''),
            'to_id': conn_dict.get('to_id', '')
        })
    
    prompt = f"""Du bist ein Experte für P&ID Diagramme und Naming Conventions. Inferiere Element-Types basierend auf Labels und Kontext.

**P&ID NAMING CONVENTIONS (ENHANCED WITH UNI-BILDER PATTERNS):**
- **FT** = Flow Transmitter → Type: "Volume Flow Sensor"
- **Fv** = Flow Valve → Type: "Valve"
- **MV** = Motor Valve → Type: "Valve" (Uni-Bilder: MV3121A, MV3131B)
- **PU** = Pump → Type: "Pump" (Uni-Bilder: PU3121, PU3131, PU3111)
- **P** = Pump → Type: "Pump"
- **M** = Mixer → Type: "Mixer"
- **V** = Valve → Type: "Valve"
- **T** = Tank → Type: "Tank"
- **R** = Reactor → Type: "Reactor"
- **S** = Sample Point → Type: "Sample Point"
- **Source** = Source → Type: "Source"
- **CHP** = Combined Heat Power → Type: "Source" (Uni-Bilder: CHP1, CHP2)
- **HP** = Heat Pump → Type: "Source" (Uni-Bilder: HP_1)
- **HEX** = Heat Exchanger → Type: "Heat Exchanger" (Uni-Bilder: HEX1_HNLT)
- **Sink** = Sink → Type: "Sink"

**ERKANNTE ELEMENTE:**
{json.dumps(elements_for_llm, indent=2, ensure_ascii=False)}

**ERKANNTE VERBINDUNGEN (für Kontext):**
{json.dumps(connections_for_llm, indent=2, ensure_ascii=False)}

**AUFGABE:**
1. **Label-basierte Inferenz**: Nutze P&ID Naming Conventions
   - "FT 10" → Type: "Volume Flow Sensor" (FT = Flow Transmitter)
   - "Fv-3-3040" → Type: "Valve" (Fv = Flow Valve)
   - "P-201" → Type: "Pump" (P = Pump)
   - "M-08" → Type: "Mixer" (M = Mixer)
   - "S" → Type: "Sample Point" (S = Sample Point)

2. **Kontext-basierte Inferenz**: Nutze Verbindungen und benachbarte Elemente
   - Wenn Element zwischen Pump und Mixer verbunden ist → wahrscheinlich Valve oder Flow Sensor
   - Wenn Element nach Valve verbunden ist → wahrscheinlich Flow Sensor
   - Wenn Element nach Flow Sensor verbunden ist → wahrscheinlich Mixer

3. **Korrigiere falsche Types**:
   - "machine" → "Mixer"
   - "Sensor" → "Volume Flow Sensor"
   - "valve" → "Valve"

**RETURN FORMAT (JSON):**
{{
  "inferred_elements": [
    {{
      "id": "element_id",
      "label": "label",
      "type": "inferred_type",
      "bbox": {{"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.05}},
      "confidence": 0.9,
      "inference_method": "label_based" | "context_based" | "both",
      "reasoning": "FT 10 → FT = Flow Transmitter → Type: Volume Flow Sensor"
    }}
  ],
  "inference_summary": {{
    "label_based_inferences": 5,
    "context_based_inferences": 3,
    "corrections": 2
  }}
}}

**WICHTIG:**
- Nutze P&ID Naming Conventions: FT, Fv, P, M, V, T, R, S
- Nutze Kontext: Verbindungen und benachbarte Elemente
- Nur inferieren wenn du sicher bist (>80% confidence)
- Behalte Original-Type wenn Inferenz unsicher ist
"""
    
    try:
        response = llm_client.call_llm(
            model_info,
            system_prompt="Du bist ein Experte für P&ID Diagramme und Naming Conventions. Du inferierst Types basierend auf Labels und Kontext.",
            user_prompt=prompt,
            image_path=image_path
        )
        
        if isinstance(response, dict):
            inferred_elements_data = response.get('inferred_elements', elements_for_llm)
            inference_summary = response.get('inference_summary', {})
            
            logger.info(f"Context-Aware Type Inference: {inference_summary.get('label_based_inferences', 0)} label-basierte, "
                       f"{inference_summary.get('context_based_inferences', 0)} kontext-basierte Inferenzen, "
                       f"{inference_summary.get('corrections', 0)} Korrekturen")
            
            # Map inferred elements back to original elements
            inferred_elements = []
            element_id_map = {el.get('id', ''): el for el in elements}
            
            for inf_el_data in inferred_elements_data:
                original_id = inf_el_data.get('id', '')
                inferred_type = inf_el_data.get('type', '')
                
                original_el = element_id_map.get(original_id)
                if original_el:
                    # Update original element with inferred type
                    el_dict = original_el if isinstance(original_el, dict) else original_el.model_dump() if hasattr(original_el, 'model_dump') else original_el.__dict__ if hasattr(original_el, '__dict__') else {}
                    
                    original_type = el_dict.get('type', '')
                    if inferred_type != original_type and inferred_type:
                        el_dict['type'] = inferred_type
                        el_dict['inference_method'] = inf_el_data.get('inference_method', 'unknown')
                        el_dict['inference_reasoning'] = inf_el_data.get('reasoning', '')
                        
                        if hasattr(original_el, 'type'):
                            try:
                                original_el.type = inferred_type
                            except Exception:
                                pass
                        
                        logger.info(f"Type Inference: {original_id} '{original_type}' → '{inferred_type}' "
                                   f"({inf_el_data.get('inference_method', 'unknown')})")
                    
                    inferred_elements.append(original_el)
                else:
                    # Element not found, use inferred data
                    inferred_elements.append(inf_el_data)
            
            return inferred_elements
        else:
            logger.warning(f"Context-Aware Type Inference: Response war kein Dict, sondern {type(response)}")
            return elements
            
    except Exception as e:
        logger.error(f"Error in Context-Aware Type Inference: {e}", exc_info=True)
        return elements

