"""
Embedding service for text and image embeddings.
"""

import logging
from typing import List, Optional, Union, Any
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self, llm_client: Any):  # LLMClient type
        self.llm_client = llm_client
    
    def get_image_embedding(
        self,
        image_input: Union[str, Image.Image]
    ) -> Optional[List[float]]:
        """
        Get embedding for image.
        
        Args:
            image_input: Image path (str) or PIL.Image
            
        Returns:
            Embedding vector or None
        """
        return self.llm_client.get_image_embedding(image_input)
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text (if LLM client supports it).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        # Placeholder - implement if text embeddings are needed
        logger.warning("Text embeddings not yet implemented")
        return None

