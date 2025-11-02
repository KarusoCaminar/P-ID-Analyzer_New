"""
Pattern Matcher - Matches analysis patterns and suggests corrections.

Provides functionality for:
- Pattern matching in analysis results
- Suggesting corrections based on learned patterns
- Pattern-based validation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class PatternMatcher:
    """
    Matches patterns in analysis results and suggests corrections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pattern Matcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.patterns: List[Dict[str, Any]] = []
        self._load_patterns()
    
    def _load_patterns(self) -> None:
        """Load pattern definitions from config."""
        try:
            # Load patterns from config
            pattern_config = self.config.get('logic_parameters', {}).get('validation_patterns', [])
            self.patterns = pattern_config if isinstance(pattern_config, list) else []
            logger.info(f"Loaded {len(self.patterns)} validation patterns")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}", exc_info=True)
            self.patterns = []
    
    def match_patterns(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match patterns in analysis results.
        
        Args:
            elements: List of element dictionaries
            connections: List of connection dictionaries
            
        Returns:
            List of matched patterns with suggestions
        """
        matches = []
        
        try:
            # Check each pattern
            for pattern in self.patterns:
                pattern_type = pattern.get('type')
                pattern_rules = pattern.get('rules', {})
                
                if pattern_type == 'element_validation':
                    element_matches = self._match_element_pattern(elements, pattern_rules)
                    matches.extend(element_matches)
                elif pattern_type == 'connection_validation':
                    connection_matches = self._match_connection_pattern(connections, elements, pattern_rules)
                    matches.extend(connection_matches)
            
            return matches
        except Exception as e:
            logger.error(f"Error matching patterns: {e}", exc_info=True)
            return []
    
    def _match_element_pattern(
        self,
        elements: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Match element validation patterns."""
        matches = []
        
        try:
            required_types = rules.get('required_types', [])
            min_count = rules.get('min_count', 0)
            max_count = rules.get('max_count', float('inf'))
            
            for req_type in required_types:
                matching_elements = [
                    el for el in elements
                    if el.get('type', '').lower() == req_type.lower()
                ]
                count = len(matching_elements)
                
                if count < min_count:
                    matches.append({
                        'type': 'missing_element',
                        'element_type': req_type,
                        'expected_count': min_count,
                        'actual_count': count,
                        'suggestion': f"Expected at least {min_count} elements of type '{req_type}', found {count}"
                    })
                elif count > max_count:
                    matches.append({
                        'type': 'excessive_element',
                        'element_type': req_type,
                        'expected_count': max_count,
                        'actual_count': count,
                        'suggestion': f"Expected at most {max_count} elements of type '{req_type}', found {count}"
                    })
        except Exception as e:
            logger.error(f"Error matching element patterns: {e}", exc_info=True)
        
        return matches
    
    def _match_connection_pattern(
        self,
        connections: List[Dict[str, Any]],
        elements: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Match connection validation patterns."""
        matches = []
        
        try:
            # Check for orphaned connections
            element_ids = {el.get('id') for el in elements if el.get('id')}
            
            for conn in connections:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                
                if from_id not in element_ids:
                    matches.append({
                        'type': 'orphaned_connection',
                        'connection_id': conn.get('id'),
                        'issue': 'from_element_missing',
                        'element_id': from_id,
                        'suggestion': f"Connection references non-existent element '{from_id}'"
                    })
                
                if to_id not in element_ids:
                    matches.append({
                        'type': 'orphaned_connection',
                        'connection_id': conn.get('id'),
                        'issue': 'to_element_missing',
                        'element_id': to_id,
                        'suggestion': f"Connection references non-existent element '{to_id}'"
                    })
            
            # Check for circular connections
            circular = self._detect_circular_connections(connections, elements)
            matches.extend(circular)
            
        except Exception as e:
            logger.error(f"Error matching connection patterns: {e}", exc_info=True)
        
        return matches
    
    def _detect_circular_connections(
        self,
        connections: List[Dict[str, Any]],
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect circular connection patterns."""
        matches = []
        
        try:
            # Build graph
            graph: Dict[str, List[str]] = {}
            for el in elements:
                el_id = el.get('id')
                if el_id:
                    graph[el_id] = []
            
            for conn in connections:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                if from_id and to_id and from_id in graph:
                    graph[from_id].append(to_id)
            
            # Detect cycles using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node: str, path: List[str]) -> bool:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if has_cycle(neighbor, path.copy()):
                            return True
                    elif neighbor in rec_stack:
                        # Cycle detected
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        matches.append({
                            'type': 'circular_connection',
                            'cycle': cycle,
                            'suggestion': f"Circular connection detected: {' -> '.join(cycle)}"
                        })
                        return True
                
                rec_stack.remove(node)
                return False
            
            for node in graph:
                if node not in visited:
                    has_cycle(node, [])
        
        except Exception as e:
            logger.error(f"Error detecting circular connections: {e}", exc_info=True)
        
        return matches
    
    def suggest_corrections(
        self,
        patterns: List[Dict[str, Any]],
        current_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest corrections based on matched patterns.
        
        Args:
            patterns: List of matched patterns
            current_data: Current analysis data
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        for pattern in patterns:
            pattern_type = pattern.get('type')
            suggestion = pattern.get('suggestion')
            
            if pattern_type == 'missing_element':
                suggestions.append({
                    'type': 'add_element',
                    'element_type': pattern.get('element_type'),
                    'suggestion': suggestion,
                    'priority': 'high'
                })
            elif pattern_type == 'orphaned_connection':
                suggestions.append({
                    'type': 'fix_connection',
                    'connection_id': pattern.get('connection_id'),
                    'suggestion': suggestion,
                    'priority': 'high'
                })
            elif pattern_type == 'circular_connection':
                suggestions.append({
                    'type': 'review_connection',
                    'cycle': pattern.get('cycle'),
                    'suggestion': suggestion,
                    'priority': 'medium'
                })
        
        return suggestions


