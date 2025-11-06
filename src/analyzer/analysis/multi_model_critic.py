"""
Multi-Model Critic - Comprehensive validation and critique system.

Provides:
- Multi-model perspective validation
- Legend comparison and plausibility checks
- Critical error identification
- Self-correction recommendations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MultiModelCritic:
    """
    Multi-Model Critic for comprehensive validation and critique.
    
    Uses multiple LLM models to validate analysis results from different
    perspectives and identify critical errors.
    """
    
    def __init__(
        self,
        llm_client: Any,
        knowledge_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Multi-Model Critic.
        
        Args:
            llm_client: LLM client for model access
            knowledge_manager: Knowledge manager for type validation
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.config = config
        
        # Model strategies for different perspectives
        self.critic_models = {
            'analyzer': None,  # Main analysis model
            'critic': None,    # Critique model (usually Pro for accuracy)
            'planner': None    # Planning/optimization model
        }
    
    def validate_and_critique(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        legend_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation and critique using multiple models.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            legend_data: Legend data (symbol_map, line_map)
            metadata: Analysis metadata
            model_strategy: Model strategy override
            
        Returns:
            Comprehensive validation report with:
            - critical_errors: List of critical errors
            - warnings: List of warnings
            - legend_violations: Elements not matching legend
            - plausibility_issues: Implausible patterns
            - recommendations: Improvement recommendations
        """
        logger.info("=== Multi-Model Critic: Starting comprehensive validation ===")
        
        report = {
            'critical_errors': [],
            'warnings': [],
            'legend_violations': [],
            'plausibility_issues': [],
            'recommendations': [],
            'validation_score': 100.0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. Legend comparison and validation
            if legend_data:
                legend_report = self._validate_against_legend(
                    elements, connections, legend_data
                )
                report['legend_violations'] = legend_report.get('violations', [])
                report['warnings'].extend(legend_report.get('warnings', []))
            
            # 2. Plausibility checks
            plausibility_report = self._check_plausibility(
                elements, connections, metadata
            )
            report['plausibility_issues'] = plausibility_report.get('issues', [])
            report['warnings'].extend(plausibility_report.get('warnings', []))
            
            # 3. Critical error identification
            critical_report = self._identify_critical_errors(
                elements, connections
            )
            # FIX: _identify_critical_errors returns a list directly, not a dict
            if isinstance(critical_report, list):
                report['critical_errors'] = critical_report
            else:
                report['critical_errors'] = critical_report.get('errors', [])
            
            # 4. Generate recommendations
            recommendations = self._generate_recommendations(
                report, elements, connections
            )
            report['recommendations'] = recommendations
            
            # 5. LANGFRISTIGE VERBESSERUNG: Error Explanation durch LLM
            use_error_explanation = self.config.get('logic_parameters', {}).get('use_error_explanation', True)
            if use_error_explanation:
                # Prepare errors dict for explanation
                errors_dict = {
                    'missed_elements': [e for e in elements if e.get('id') in [w.get('element_id') for w in report.get('warnings', [])]],
                    'hallucinated_elements': [e for e in elements if e.get('confidence', 1.0) < 0.3],
                    'missed_connections': len([c for c in connections if not c.get('from_id') or not c.get('to_id')]),
                    'hallucinated_connections': len([c for c in connections if c.get('confidence', 1.0) < 0.3])
                }
                
                error_explanation = self._explain_errors_with_llm(
                    elements,
                    connections,
                    errors_dict,
                    image_path=None,  # Could pass image_path if available
                    debug_map_path=None,  # Could pass debug_map_path if available
                    model_strategy=model_strategy
                )
                
                if error_explanation:
                    report['error_explanation'] = error_explanation
                    logger.info("Error Explanation: LLM-basierte Fehleranalyse abgeschlossen")
            
            # 6. Calculate validation score
            validation_score = self._calculate_validation_score(report)
            report['validation_score'] = validation_score
            
            logger.info(f"Validation complete: {len(report['critical_errors'])} critical errors, "
                       f"{len(report['warnings'])} warnings, "
                       f"{len(report['recommendations'])} recommendations")
            
        except Exception as e:
            logger.error(f"Error in multi-model critic: {e}", exc_info=True)
            report['critical_errors'].append({
                'type': 'critic_error',
                'message': f'Error in validation: {str(e)}',
                'severity': 'high'
            })
        
        return report
    
    def _validate_against_legend(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        legend_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate elements and connections against legend data.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            legend_data: Legend data with symbol_map and line_map
            
        Returns:
            Validation report with violations and warnings
        """
        logger.info("Validating against legend...")
        
        report = {
            'violations': [],
            'warnings': []
        }
        
        symbol_map = legend_data.get('symbol_map', {})
        line_map = legend_data.get('line_map', {})
        
        # Check element types against symbol map
        for element in elements:
            el_type = element.get('type', '').strip()
            el_label = element.get('label', '').strip()
            
            # Check if type matches legend symbols
            type_match = False
            if el_type:
                # Direct match
                if el_type in symbol_map.values():
                    type_match = True
                # Label match
                elif el_label in symbol_map:
                    legend_type = symbol_map.get(el_label)
                    if el_type == legend_type:
                        type_match = True
                    else:
                        # Mismatch between detected type and legend
                        report['violations'].append({
                            'element_id': element.get('id'),
                            'type': 'legend_mismatch',
                            'detected_type': el_type,
                            'legend_type': legend_type,
                            'label': el_label,
                            'severity': 'medium'
                        })
            
            # Warning if type not in legend but element detected
            if not type_match and el_type:
                report['warnings'].append({
                    'element_id': element.get('id'),
                    'type': 'type_not_in_legend',
                    'detected_type': el_type,
                    'label': el_label,
                    'severity': 'low'
                })
        
        # Check connection types against line map
        for connection in connections:
            conn_type = connection.get('kind', '').strip()
            
            # Check if connection type matches line map
            if conn_type and line_map:
                type_match = False
                for line_type, line_info in line_map.items():
                    if conn_type.lower() in str(line_type).lower():
                        type_match = True
                        break
                
                if not type_match:
                    report['warnings'].append({
                        'connection_id': connection.get('id'),
                        'type': 'connection_not_in_legend',
                        'detected_type': conn_type,
                        'severity': 'low'
                    })
        
        logger.info(f"Legend validation: {len(report['violations'])} violations, "
                   f"{len(report['warnings'])} warnings")
        
        return report
    
    def _check_plausibility(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check plausibility of detected elements and connections.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            metadata: Analysis metadata
            
        Returns:
            Plausibility report with issues and warnings
        """
        logger.info("Checking plausibility...")
        
        report = {
            'issues': [],
            'warnings': []
        }
        
        # 1. Check for isolated elements (no connections)
        element_ids = {el.get('id') for el in elements}
        connected_elements = set()
        
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            if from_id:
                connected_elements.add(from_id)
            if to_id:
                connected_elements.add(to_id)
        
        isolated_elements = element_ids - connected_elements
        if isolated_elements:
            for el_id in isolated_elements:
                el = next((e for e in elements if e.get('id') == el_id), None)
                if el:
                    # Check if it's a known standalone element type
                    el_type = el.get('type', '')
                    standalone_types = ['Diagram_Inlet', 'Diagram_Outlet', 'Valve', 'Sensor']
                    
                    if el_type not in standalone_types:
                        report['issues'].append({
                            'element_id': el_id,
                            'type': 'isolated_element',
                            'element_type': el_type,
                            'severity': 'medium',
                            'message': f'Element {el_type} has no connections'
                        })
        
        # 2. Check for duplicate elements (same position)
        element_positions = {}
        for el in elements:
            bbox = el.get('bbox', {})
            if isinstance(bbox, dict):
                center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
                center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
                pos_key = (round(center_x, 3), round(center_y, 3))
                
                if pos_key in element_positions:
                    report['issues'].append({
                        'element_id': el.get('id'),
                        'type': 'duplicate_position',
                        'severity': 'high',
                        'message': f'Element at position {pos_key} already exists',
                        'existing_element_id': element_positions[pos_key]
                    })
                else:
                    element_positions[pos_key] = el.get('id')
        
        # 3. Check for impossible connections (same element to itself)
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            if from_id == to_id:
                report['critical_errors'].append({
                    'connection_id': conn.get('id'),
                    'type': 'self_connection',
                    'severity': 'high',
                    'message': f'Connection from {from_id} to itself'
                })
        
        # 4. Check for low confidence elements
        for el in elements:
            confidence = el.get('confidence', 1.0)
            if confidence < 0.5:
                report['warnings'].append({
                    'element_id': el.get('id'),
                    'type': 'low_confidence',
                    'confidence': confidence,
                    'element_type': el.get('type'),
                    'severity': 'medium',
                    'message': f'Element {el.get("type")} has low confidence ({confidence:.2f})'
                })
        
        logger.info(f"Plausibility check: {len(report['issues'])} issues, "
                   f"{len(report['warnings'])} warnings")
        
        return report
    
    def _identify_critical_errors(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify critical errors that need immediate attention.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            
        Returns:
            List of critical errors
        """
        logger.info("Identifying critical errors...")
        
        critical_errors = []
        
        # 1. Invalid bounding boxes
        for el in elements:
            bbox = el.get('bbox', {})
            if isinstance(bbox, dict):
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
                x = bbox.get('x', 0)
                y = bbox.get('y', 0)
                
                if width <= 0 or height <= 0:
                    critical_errors.append({
                        'element_id': el.get('id'),
                        'type': 'invalid_bbox',
                        'severity': 'critical',
                        'message': f'Element {el.get("id")} has invalid bounding box: width={width}, height={height}'
                    })
                
                if x < 0 or y < 0 or x > 1 or y > 1:
                    critical_errors.append({
                        'element_id': el.get('id'),
                        'type': 'bbox_out_of_bounds',
                        'severity': 'critical',
                        'message': f'Element {el.get("id")} bbox out of bounds: x={x}, y={y}'
                    })
        
        # 2. Missing required fields
        for el in elements:
            if not el.get('id'):
                critical_errors.append({
                    'element_id': None,
                    'type': 'missing_id',
                    'severity': 'critical',
                    'message': 'Element missing ID'
                })
            
            if not el.get('type'):
                critical_errors.append({
                    'element_id': el.get('id'),
                    'type': 'missing_type',
                    'severity': 'high',
                    'message': f'Element {el.get("id")} missing type'
                })
        
        # 3. Orphaned connections (reference non-existent elements)
        element_ids = {el.get('id') for el in elements}
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if from_id and from_id not in element_ids:
                critical_errors.append({
                    'connection_id': conn.get('id'),
                    'type': 'orphaned_connection',
                    'severity': 'high',
                    'message': f'Connection references non-existent element: {from_id}'
                })
            
            if to_id and to_id not in element_ids:
                critical_errors.append({
                    'connection_id': conn.get('id'),
                    'type': 'orphaned_connection',
                    'severity': 'high',
                    'message': f'Connection references non-existent element: {to_id}'
                })
        
        logger.info(f"Identified {len(critical_errors)} critical errors")
        
        return critical_errors
    
    def validate_with_visual_feedback(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        original_image_path: str,
        debug_map_path: str,
        legend_data: Optional[Dict[str, Any]] = None,
        model_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate and critique analysis results using visual feedback.
        
        CRITICAL: This method uses visual feedback (original image + debug map) to
        enable the LLM to "see" its own work and identify errors visually.
        
        Process:
        1. LLM receives original image + debug map (visualization of current analysis)
        2. LLM compares debug map with original image
        3. LLM identifies visual errors (missing boxes, wrong boxes, wrong connections)
        4. LLM generates corrections based on visual comparison
        
        Args:
            elements: Detected elements
            connections: Detected connections
            original_image_path: Path to original P&ID image
            debug_map_path: Path to debug map (visualization of current analysis)
            legend_data: Legend data (symbol_map, line_map)
            model_strategy: Model strategy configuration
            
        Returns:
            Dictionary with visual corrections:
            - visual_errors: List of visually identified errors
            - corrections: List of corrections to apply
            - validation_score: Visual validation score
        """
        logger.info("=== Visual Feedback Validation: Starting visual critique ===")
        
        # Get model info (use Pro model for visual critique)
        if model_strategy:
            model_info = model_strategy.get('critic_model') or model_strategy.get('meta_model') or model_strategy.get('detail_model')
        else:
            model_info = self.config.get('models', {}).get('Google Gemini 2.5 Pro', {})
        
        if not model_info:
            models_cfg = self.config.get('models', {})
            model_info = list(models_cfg.values())[0] if models_cfg else {}
        
        # Normalize model info
        if isinstance(model_info, dict):
            critic_model_info = model_info
        elif hasattr(model_info, 'model_dump'):
            critic_model_info = model_info.model_dump()
        elif hasattr(model_info, 'dict'):
            critic_model_info = model_info.dict()
        else:
            critic_model_info = {}
        
        if not critic_model_info:
            logger.warning("No critic model available. Skipping visual feedback validation.")
            return {
                'visual_errors': [],
                'corrections': [],
                'validation_score': 0.0
            }
        
        # Get visual feedback prompt from config
        visual_feedback_prompt_template = self.config.get('prompts', {}).get('visual_feedback_critique_prompt_template', '')
        
        if not visual_feedback_prompt_template:
            logger.warning("No visual_feedback_critique_prompt_template found in config. Using default.")
            visual_feedback_prompt_template = self._get_default_visual_feedback_prompt()
        
        # Prepare prompt with current analysis data
        import json
        elements_json = json.dumps(elements[:50], indent=2, ensure_ascii=False)  # Limit to 50 for prompt size
        connections_json = json.dumps(connections[:50], indent=2, ensure_ascii=False)  # Limit to 50
        
        legend_context = ""
        if legend_data:
            symbol_map = legend_data.get('symbol_map', {})
            line_map = legend_data.get('line_map', {})
            legend_context = f"""
**LEGEND DATA:**
- Symbol Map: {json.dumps(symbol_map, indent=2, ensure_ascii=False)}
- Line Map: {json.dumps(line_map, indent=2, ensure_ascii=False)}
"""
        
        prompt = visual_feedback_prompt_template.format(
            elements_json=elements_json,
            connections_json=connections_json,
            legend_context=legend_context
        )
        
        system_prompt = """Du bist ein P&ID-Chef-Inspektor mit visueller Expertise. Deine Aufgabe ist es, die Arbeit eines untergeordneten KI-Analysten zu überprüfen und zu korrigieren, indem du das Originalbild mit der Visualisierung der Analyse vergleichst."""
        
        try:
            # Call LLM with debug map image
            # Note: We send the debug map as the primary image
            # The original image is referenced in the prompt for context
            # TODO: Future enhancement: Support multiple images in call_llm for side-by-side comparison
            
            # Enhance prompt to include original image path for context
            enhanced_prompt = f"""{prompt}

**WICHTIGER HINWEIS:**
- Das Bild, das du siehst, ist das [Debug-Bild] (Visualisierung der aktuellen Analyse)
- Das [Originalbild] ist unter folgendem Pfad verfügbar: {original_image_path}
- Du musst das Debug-Bild mit dem Originalbild vergleichen, um Fehler zu identifizieren
- Wenn du das Originalbild sehen musst, kannst du es in einem separaten Request anfordern
- Für jetzt: Nutze das Debug-Bild und die JSON-Daten, um visuelle Fehler zu identifizieren"""
            
            response = self.llm_client.call_llm(
                critic_model_info,
                system_prompt,
                enhanced_prompt,
                debug_map_path,  # Primary image: debug map (shows current analysis)
                expected_json_keys=["visual_errors", "corrections", "validation_score"]
            )
            
            if response and isinstance(response, dict):
                visual_errors = response.get('visual_errors', [])
                corrections = response.get('corrections', [])
                validation_score = response.get('validation_score', 0.0)
                
                logger.info(f"Visual Feedback: {len(visual_errors)} visual errors identified, "
                           f"{len(corrections)} corrections suggested, score: {validation_score:.2f}")
                
                return {
                    'visual_errors': visual_errors,
                    'corrections': corrections,
                    'validation_score': float(validation_score)
                }
            else:
                logger.warning(f"Visual Feedback: Invalid response type: {type(response)}")
                return {
                    'visual_errors': [],
                    'corrections': [],
                    'validation_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in visual feedback validation: {e}", exc_info=True)
            return {
                'visual_errors': [],
                'corrections': [],
                'validation_score': 0.0
            }
    
    def _get_default_visual_feedback_prompt(self) -> str:
        """Get default visual feedback prompt if not in config."""
        return """**TASK:** Du bist ein P&ID-Chef-Inspektor. Du wirst zwei Bilder sehen:

1. **[Originalbild]:** Das zu analysierende P&ID-Diagramm
2. **[Debug-Bild]:** Eine Visualisierung der aktuellen Analyse (was die untergeordnete KI erkannt hat)

**AKTUELLE ANALYSE (JSON):**
- Elemente: {elements_json}
- Verbindungen: {connections_json}

{legend_context}

**AUFGABE:**
1. **Vergleiche** das [Debug-Bild] mit dem [Originalbild]
2. **Identifiziere** alle visuellen Fehler:
   - Fehlende Bounding Boxes (Elemente im Original, aber nicht im Debug-Bild)
   - Falsch gezeichnete Bounding Boxes (zu groß, zu klein, falsche Position)
   - Halluzinierte Elemente (Bounding Boxes im Debug-Bild, aber keine Komponente im Original)
   - Fehlende Verbindungen (Linien im Original, aber nicht im Debug-Bild)
   - Falsche Verbindungen (Linien im Debug-Bild, aber nicht im Original)
3. **Generiere** Korrekturen als JSON

**RETURN FORMAT (STRICT JSON):**
{{
  "visual_errors": [
    {{
      "type": "missing_element",
      "element_id": "P-201",
      "reason": "Element P-201 ist im Originalbild sichtbar, aber nicht im Debug-Bild",
      "suggested_bbox": {{"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04}}
    }},
    {{
      "type": "hallucinated_element",
      "element_id": "FT-11",
      "reason": "Bounding Box für FT-11 im Debug-Bild, aber keine Komponente an dieser Position im Originalbild"
    }},
    {{
      "type": "wrong_bbox",
      "element_id": "V-01",
      "reason": "Bounding Box zu groß, schließt Text-Label ein",
      "current_bbox": {{"x": 0.3, "y": 0.4, "width": 0.1, "height": 0.1}},
      "corrected_bbox": {{"x": 0.32, "y": 0.42, "width": 0.05, "height": 0.05}}
    }}
  ],
  "corrections": [
    {{
      "action": "add_element",
      "element": {{
        "id": "P-201",
        "type": "Pump",
        "label": "Pump P-201",
        "bbox": {{"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04}},
        "confidence": 0.9
      }}
    }},
    {{
      "action": "remove_element",
      "element_id": "FT-11",
      "reason": "Halluzination: Keine Komponente an dieser Position"
    }},
    {{
      "action": "resize_bbox",
      "element_id": "V-01",
      "new_bbox": {{"x": 0.32, "y": 0.42, "width": 0.05, "height": 0.05}},
      "reason": "BBox zu groß, schließt Text ein"
    }}
  ],
  "validation_score": 75.0
}}

**CRITICAL:** Nur Korrekturen vorschlagen, wenn du dir visuell sicher bist. Wenn du unsicher bist, lasse das Element/Verbindung unverändert."""
    
    def _explain_errors_with_llm(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        errors: Dict[str, Any],
        image_path: Optional[str] = None,
        debug_map_path: Optional[str] = None,
        model_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to explain why errors occurred and suggest fixes.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            errors: Error dictionary with missed/hallucinated elements
            image_path: Optional path to image for visual context
            model_strategy: Model strategy configuration
            
        Returns:
            Dictionary with error explanations and suggestions
        """
        logger.info("Error Explanation: Analyzing errors with LLM...")
        
        import json
        
        # Get model info
        if model_strategy:
            model_info = model_strategy.get('critic_model_name') or model_strategy.get('meta_model')
        else:
            model_info = self.config.get('models', {}).get('Google Gemini 2.5 Pro', {})
        
        if not model_info:
            models_cfg = self.config.get('models', {})
            model_info = list(models_cfg.values())[0] if models_cfg else {}
        
        # Prepare error summary
        missed_elements = errors.get('missed_elements', [])
        hallucinated_elements = errors.get('hallucinated_elements', [])
        missed_connections = errors.get('missed_connections', 0)
        hallucinated_connections = errors.get('hallucinated_connections', 0)
        
        prompt = f"""Du bist ein Experte für P&ID Diagramme und Fehleranalyse. Analysiere warum diese Fehler aufgetreten sind.

**ERKANNTE ELEMENTE:**
{json.dumps(elements[:20], indent=2, ensure_ascii=False)}  # Limit to 20 for prompt size

**ERKANNTE VERBINDUNGEN:**
{json.dumps(connections[:20], indent=2, ensure_ascii=False)}  # Limit to 20

**FEHLER:**
- Verpasste Elemente: {len(missed_elements)}
- Halluzinierte Elemente: {len(hallucinated_elements)}
- Verpasste Verbindungen: {missed_connections}
- Halluzinierte Verbindungen: {hallucinated_connections}

**VERPASSTE ELEMENTE (Beispiele):**
{json.dumps(missed_elements[:10], indent=2, ensure_ascii=False) if missed_elements else "Keine"}

**HALLUZINIERTE ELEMENTE (Beispiele):**
{json.dumps(hallucinated_elements[:10], indent=2, ensure_ascii=False) if hallucinated_elements else "Keine"}

**AUFGABE:**
1. **Fehlerursachen identifizieren**:
   - Warum wurden Elemente verpasst? (BBox zu groß/zu klein, Confidence zu niedrig, Excluded Zones zu aggressiv?)
   - Warum wurden Elemente halluziniert? (Confidence zu niedrig, Type-Mismatch?)
   - Warum wurden Verbindungen verpasst? (IDs nicht normalisiert, Elemente nicht gefunden?)

2. **Verbesserungsvorschläge**:
   - Was sollte geändert werden? (Confidence Threshold, Excluded Zones, Type-Normalisierung?)
   - Welche Parameter sollten angepasst werden?

3. **Konkrete Korrekturen**:
   - Welche Elemente sollten noch einmal analysiert werden?
   - Welche Verbindungen sollten hinzugefügt werden?

**RETURN FORMAT (JSON):**
{{
  "error_analysis": {{
    "missed_elements_reasons": [
      {{
        "element_id": "P-201",
        "reason": "Confidence zu niedrig (0.65 < 0.6 Threshold)",
        "suggestion": "Confidence Threshold senken oder Re-Analyse mit höherer Confidence"
      }}
    ],
    "hallucinated_elements_reasons": [
      {{
        "element_id": "FT-10",
        "reason": "Type-Mismatch: 'machine' statt 'Mixer'",
        "suggestion": "Type-Normalisierung verbessern"
      }}
    ],
    "missing_connections_reasons": [
      {{
        "from_id": "P-504",
        "to_id": "Mixer-M-08",
        "reason": "IDs nicht normalisiert: 'P-504' vs 'P504'",
        "suggestion": "ID-Normalisierung vor Connection-Erstellung"
      }}
    ]
  }},
  "improvement_suggestions": [
    "Confidence Threshold von 0.6 auf 0.5 senken für mehr Elemente",
    "Excluded Zones weniger aggressiv (nur bei 100% Overlap ausschließen)",
    "Type-Normalisierung: 'machine' → 'Mixer' automatisch korrigieren"
  ],
  "specific_corrections": [
    {{
      "action": "re_analyze_element",
      "element_id": "P-201",
      "reason": "Confidence zu niedrig, aber wahrscheinlich korrekt"
    }}
  ],
  "explanation": "Die meisten Fehler entstehen durch: 1) Confidence Threshold zu hoch, 2) ID-Normalisierung fehlt, 3) Type-Mismatch"
}}
"""
        
        try:
            response = self.llm_client.call_llm(
                model_info,
                system_prompt="Du bist ein Experte für P&ID Diagramme und Fehleranalyse. Du erklärst warum Fehler auftreten und schlägst konkrete Verbesserungen vor.",
                user_prompt=prompt,
                image_path=image_path
            )
            
            if isinstance(response, dict):
                error_analysis = response.get('error_analysis', {})
                improvement_suggestions = response.get('improvement_suggestions', [])
                specific_corrections = response.get('specific_corrections', [])
                explanation = response.get('explanation', '')
                
                logger.info(f"Error Explanation: {len(improvement_suggestions)} Verbesserungsvorschläge, "
                           f"{len(specific_corrections)} konkrete Korrekturen")
                
                if explanation:
                    logger.info(f"Fehlerursachen: {explanation}")
                
                return {
                    'error_analysis': error_analysis,
                    'improvement_suggestions': improvement_suggestions,
                    'specific_corrections': specific_corrections,
                    'explanation': explanation
                }
            else:
                logger.warning(f"Error Explanation: Response war kein Dict, sondern {type(response)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in Error Explanation: {e}", exc_info=True)
            return {}
    
    def _generate_recommendations(
        self,
        report: Dict[str, Any],
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improvement.
        
        Args:
            report: Validation report
            elements: Detected elements
            connections: Detected connections
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Recommendation based on critical errors
        critical_count = len(report.get('critical_errors', []))
        if critical_count > 0:
            recommendations.append({
                'type': 'fix_critical_errors',
                'priority': 'high',
                'message': f'Fix {critical_count} critical errors before proceeding',
                'action': 'Re-analyze with error feedback'
            })
        
        # Recommendation based on legend violations
        legend_violations = len(report.get('legend_violations', []))
        if legend_violations > 0:
            recommendations.append({
                'type': 'validate_legend',
                'priority': 'medium',
                'message': f'{legend_violations} elements do not match legend',
                'action': 'Compare detected types with legend and correct'
            })
        
        # Recommendation based on isolated elements
        isolated = [e for e in report.get('plausibility_issues', [])
                   if e.get('type') == 'isolated_element']
        if isolated:
            recommendations.append({
                'type': 'check_isolated_elements',
                'priority': 'medium',
                'message': f'{len(isolated)} isolated elements found',
                'action': 'Verify if these elements should have connections'
            })
        
        # Recommendation based on low confidence
        low_confidence = [e for e in elements if e.get('confidence', 1.0) < 0.5]
        if low_confidence:
            recommendations.append({
                'type': 'improve_confidence',
                'priority': 'low',
                'message': f'{len(low_confidence)} elements have low confidence',
                'action': 'Re-analyze low confidence areas or increase confidence threshold'
            })
        
        return recommendations
    
    def _calculate_validation_score(
        self,
        report: Dict[str, Any]
    ) -> float:
        """
        Calculate overall validation score based on issues found.
        
        Args:
            report: Validation report
            
        Returns:
            Validation score (0-100)
        """
        score = 100.0
        
        # Penalize critical errors
        critical_errors = len(report.get('critical_errors', []))
        score -= critical_errors * 10.0  # -10 per critical error
        
        # Penalize legend violations
        legend_violations = len(report.get('legend_violations', []))
        score -= legend_violations * 3.0  # -3 per violation
        
        # Penalize plausibility issues
        plausibility_issues = len(report.get('plausibility_issues', []))
        score -= plausibility_issues * 2.0  # -2 per issue
        
        # Penalize warnings
        warnings = len(report.get('warnings', []))
        score -= warnings * 0.5  # -0.5 per warning
        
        return max(0.0, min(100.0, score))

