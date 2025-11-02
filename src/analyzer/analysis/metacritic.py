"""
Metacritic - Cross-validation between Monolith and Swarm analysis.

Compares two different analysis methods (Monolith vs Swarm) to identify:
- Hallucinated elements (present in one but not the other)
- Missed connections (found in one but not the other)
- Global inconsistencies (major structural differences)

This provides a powerful cross-validation mechanism to catch systematic errors.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Metacritic:
    """
    Metacritic for cross-validation between Monolith and Swarm analysis.
    
    Compares results from two different analysis approaches to identify
    systematic discrepancies and potential hallucinations.
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any],
        model_strategy: Dict[str, Any]
    ):
        """
        Initialize Metacritic.
        
        Args:
            llm_client: LLM client for model access
            config: Configuration dictionary
            model_strategy: Model strategy configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.model_strategy = model_strategy
        
        # Get prompts - Handle both dict and Pydantic models
        prompts = config.get('prompts', {}) if isinstance(config, dict) else getattr(config, 'prompts', {})
        if isinstance(prompts, dict):
            self.metacritic_prompt_template = prompts.get('metacritic_prompt_template')
        else:
            # Pydantic model - use attribute access
            self.metacritic_prompt_template = getattr(prompts, 'metacritic_prompt_template', None)
        
        # Get model - try multiple config paths
        self.critic_model = (
            model_strategy.get('critic_model_name') or
            model_strategy.get('critic_model') or
            config.get('logic_parameters', {}).get('metacritic_model') or
            'Google Gemini 2.5 Pro'  # Default: Pro for accuracy
        )
    
    def review(
        self,
        monolith_result: Dict[str, Any],
        swarm_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Review and compare Monolith vs Swarm analysis results.
        
        Args:
            monolith_result: Results from monolith analysis
            swarm_result: Results from swarm analysis
            
        Returns:
            List of discrepancies, each with:
            - id: Discrepancy type (Hallucinated_Element, Missed_Connection, Global_Inconsistency)
            - description: Detailed description of the discrepancy
            - suggested_correction_action: Recommended action to fix
        """
        if not self.metacritic_prompt_template:
            logger.warning("No metacritic_prompt_template found in config. Skipping metacritic review.")
            return []
        
        logger.info("=== Metacritic: Starting cross-validation review ===")
        
        try:
            # Format prompt
            monolith_str = json.dumps(monolith_result, ensure_ascii=False, indent=2)
            swarm_str = json.dumps(swarm_result, ensure_ascii=False, indent=2)
            
            user_prompt = self.metacritic_prompt_template.format(
                monolith_json=monolith_str,
                swarm_json=swarm_str
            )
            
            # Get model info from config
            models_cfg = self.config.get('models', {})
            if isinstance(self.critic_model, dict):
                model_info = self.critic_model
            else:
                model_info = models_cfg.get(str(self.critic_model), {'id': str(self.critic_model)})
            
            # Call LLM
            response = self.llm_client.call_llm(
                model_info,
                system_prompt="",
                user_prompt=user_prompt,
                image_path=None  # Metacritic doesn't need image
            )
            
            # Parse response
            discrepancies = self._parse_response(response)
            
            if discrepancies:
                logger.info(f"Metacritic found {len(discrepancies)} discrepancies:")
                for disc in discrepancies:
                    logger.info(f"  - {disc.get('id')}: {disc.get('description', '')[:80]}...")
            else:
                logger.info("Metacritic found no major discrepancies. Good work!")
            
            return discrepancies
            
        except Exception as e:
            logger.error(f"Error in metacritic review: {e}", exc_info=True)
            return []
    
    def _parse_response(self, response: Any) -> List[Dict[str, Any]]:
        """
        Parse LLM response into list of discrepancies.
        
        Args:
            response: LLM response (dict or str)
            
        Returns:
            List of discrepancy dictionaries
        """
        if not response:
            logger.warning("Metacritic response is empty/None.")
            return []
        
        # Handle string response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Metacritic returned string but not valid JSON.")
                return []
        
        # Extract discrepancies
        if isinstance(response, dict):
            discrepancies = response.get('discrepancies', [])
            if isinstance(discrepancies, list):
                # Validate discrepancy structure
                validated = []
                for disc in discrepancies:
                    if isinstance(disc, dict) and disc.get('id'):
                        validated.append({
                            'id': disc.get('id'),
                            'description': disc.get('description', ''),
                            'suggested_correction_action': disc.get('suggested_correction_action', '')
                        })
                return validated
        
        logger.warning("Metacritic response does not contain valid 'discrepancies' field.")
        return []
    
    def apply_discrepancies_to_corrections(
        self,
        discrepancies: List[Dict[str, Any]],
        error_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert metacritic discrepancies into error correction format.
        
        Args:
            discrepancies: List of discrepancies from metacritic
            error_dict: Existing error dictionary
            
        Returns:
            Updated error dictionary with metacritic issues
        """
        if not discrepancies:
            return error_dict
        
        # Add metacritic issues to error dict
        metacritic_issues = []
        for disc in discrepancies:
            metacritic_issues.append({
                'type': f"METACRITIC_{disc.get('id', 'UNKNOWN')}",
                'description': disc.get('description', ''),
                'suggested_action': disc.get('suggested_correction_action', ''),
                'source': 'metacritic'
            })
        
        # Add to error dict
        error_dict['metacritic_discrepancies'] = metacritic_issues
        
        logger.info(f"Applied {len(metacritic_issues)} metacritic discrepancies to error corrections.")
        
        return error_dict

