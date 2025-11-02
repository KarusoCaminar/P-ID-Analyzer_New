"""
Prompt manager for template management.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates for LLM calls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompts = config.get('prompts', {})
    
    def get_prompt(self, key: str, default: Optional[str] = None) -> str:
        """Get prompt template by key."""
        return self.prompts.get(key, default or "")
    
    def format_prompt(
        self,
        key: str,
        **kwargs
    ) -> str:
        """
        Get and format prompt template.
        
        Args:
            key: Prompt template key
            **kwargs: Variables to format into template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt(key)
        if not template:
            logger.warning(f"Prompt template '{key}' not found")
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template '{key}': {e}")
            return template

