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
            report['critical_errors'] = critical_report.get('errors', [])
            
            # 4. Generate recommendations
            recommendations = self._generate_recommendations(
                report, elements, connections
            )
            report['recommendations'] = recommendations
            
            # 5. Calculate validation score
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

