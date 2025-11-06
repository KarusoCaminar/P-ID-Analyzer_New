"""
Legend/OCR Consistency Critic.

Checks symbol frequency distribution against legend data.
Flags inconsistencies between legend definitions and detected symbols.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LegendConsistencyCritic:
    """
    Validates consistency between legend and detected symbols.
    
    Checks:
    - Symbol frequency: Are symbols in legend actually detected?
    - Unexpected symbols: Are symbols detected that aren't in legend?
    - Frequency anomalies: Unusual occurrence counts (e.g., 60x same symbol in small area)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Legend Consistency Critic.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logic_parameters = config.get('logic_parameters', {})
    
    def validate_consistency(
        self,
        elements: List[Dict[str, Any]],
        legend_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate consistency between legend and detected symbols.
        
        Args:
            elements: Detected elements
            legend_data: Legend data with symbol_map
            
        Returns:
            Dictionary with validation results:
            - missing_symbols: Symbols in legend but not detected
            - unexpected_symbols: Symbols detected but not in legend
            - frequency_anomalies: Unusual occurrence counts
            - validation_score: Overall validation score (0-100)
        """
        logger.info("=== Starting legend consistency validation ===")
        
        try:
            missing_symbols = []
            unexpected_symbols = []
            frequency_anomalies = []
            
            if not legend_data:
                logger.warning("No legend data provided. Skipping legend consistency check.")
                return {
                    'missing_symbols': [],
                    'unexpected_symbols': [],
                    'frequency_anomalies': [],
                    'validation_score': 100.0  # Perfect score if no legend
                }
            
            symbol_map = legend_data.get('symbol_map', {})
            
            if not symbol_map:
                logger.warning("No symbol_map in legend data. Skipping legend consistency check.")
                return {
                    'missing_symbols': [],
                    'unexpected_symbols': [],
                    'frequency_anomalies': [],
                    'validation_score': 100.0
                }
            
            # Count detected symbols by type
            detected_types = {}
            for el in elements:
                el_type = el.get('type', '')
                if el_type:
                    detected_types[el_type] = detected_types.get(el_type, 0) + 1
            
            # Check for missing symbols (in legend but not detected)
            for legend_symbol, legend_type in symbol_map.items():
                if legend_type not in detected_types:
                    missing_symbols.append({
                        'legend_symbol': legend_symbol,
                        'legend_type': legend_type,
                        'issue': f"Symbol '{legend_symbol}' ({legend_type}) in legend but not detected"
                    })
            
            # Check for unexpected symbols (detected but not in legend)
            legend_types = set(symbol_map.values())
            for detected_type, count in detected_types.items():
                if detected_type not in legend_types:
                    # Check if it's a common system type (allow these)
                    common_types = ['Line_Split', 'Line_Merge', 'Diagram_Inlet', 'Diagram_Outlet']
                    if detected_type not in common_types:
                        unexpected_symbols.append({
                            'detected_type': detected_type,
                            'count': count,
                            'issue': f"Symbol '{detected_type}' detected ({count}x) but not in legend"
                        })
            
            # Check for frequency anomalies
            total_elements = len(elements)
            if total_elements > 0:
                for detected_type, count in detected_types.items():
                    frequency = count / total_elements
                    
                    # Flag if frequency > 50% (very common) or < 1% (rare)
                    if frequency > 0.5:
                        frequency_anomalies.append({
                            'type': detected_type,
                            'count': count,
                            'frequency': frequency,
                            'issue': f"Symbol '{detected_type}' appears {count}x ({frequency*100:.1f}%) - unusually high frequency"
                        })
                    elif frequency < 0.01 and count > 10:
                        # Rare but appears many times - could be anomaly
                        frequency_anomalies.append({
                            'type': detected_type,
                            'count': count,
                            'frequency': frequency,
                            'issue': f"Symbol '{detected_type}' appears {count}x ({frequency*100:.1f}%) - rare symbol with high count"
                        })
            
            # Calculate validation score
            total_issues = len(missing_symbols) + len(unexpected_symbols) + len(frequency_anomalies)
            validation_score = max(0, 100 - (total_issues * 10))  # -10 points per issue
            
            logger.info(f"Legend consistency validation complete: {total_issues} issues found, "
                       f"score: {validation_score:.2f}")
            
            return {
                'missing_symbols': missing_symbols,
                'unexpected_symbols': unexpected_symbols,
                'frequency_anomalies': frequency_anomalies,
                'validation_score': validation_score,
                'total_issues': total_issues
            }
            
        except Exception as e:
            logger.error(f"Error in legend consistency validation: {e}", exc_info=True)
            return {
                'missing_symbols': [],
                'unexpected_symbols': [],
                'frequency_anomalies': [],
                'validation_score': 0.0,
                'total_issues': 0
            }

