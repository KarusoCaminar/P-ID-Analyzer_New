"""
Symbol Library - Manages visual symbol library with embeddings.

Provides functionality for:
- Adding symbols to the library
- Finding similar symbols by visual similarity
- Managing symbol metadata
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SymbolLibrary:
    """
    Manages a library of visual symbols with embeddings for similarity search.
    """
    
    def __init__(self, llm_client: Any):
        """
        Initialize Symbol Library.
        
        Args:
            llm_client: LLM client for generating image embeddings
        """
        self.llm_client = llm_client
        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._symbol_ids: List[str] = []
    
    def add_symbol(
        self,
        symbol_id: str,
        image: Image.Image,
        element_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a symbol to the library.
        
        Args:
            symbol_id: Unique identifier for the symbol
            image: PIL Image of the symbol
            element_type: Type of element (e.g., "Pump", "Valve")
            metadata: Optional metadata dictionary
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.llm_client.get_image_embedding(image)
            if embedding is None:
                logger.error(f"Failed to generate embedding for symbol {symbol_id}")
                return False
            
            # Store symbol data
            self.symbols[symbol_id] = {
                'element_type': element_type,
                'metadata': metadata or {},
                'added_timestamp': logging.time.time() if hasattr(logging, 'time') else None
            }
            
            self.embeddings[symbol_id] = np.array(embedding)
            self._update_embedding_matrix()
            
            logger.info(f"Added symbol {symbol_id} to library")
            return True
        except Exception as e:
            logger.error(f"Error adding symbol {symbol_id}: {e}", exc_info=True)
            return False
    
    def find_similar_symbols(
        self,
        image: Image.Image,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find similar symbols in the library.
        
        Args:
            image: PIL Image to search for
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (symbol_id, similarity_score, metadata)
        """
        if not self.embeddings:
            return []
        
        try:
            # Generate embedding for query image
            query_embedding = self.llm_client.get_image_embedding(image)
            if query_embedding is None:
                return []
            
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            if self._embedding_matrix is None:
                self._update_embedding_matrix()
            
            if self._embedding_matrix is None:
                return []
            
            similarities = cosine_similarity(query_vector, self._embedding_matrix)[0]
            
            # Get top k results above threshold
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    symbol_id = self._symbol_ids[idx]
                    results.append((
                        symbol_id,
                        float(similarities[idx]),
                        self.symbols.get(symbol_id, {})
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar symbols: {e}", exc_info=True)
            return []
    
    def _update_embedding_matrix(self) -> None:
        """Update the embedding matrix for fast similarity search."""
        if not self.embeddings:
            self._embedding_matrix = None
            self._symbol_ids = []
            return
        
        self._symbol_ids = list(self.embeddings.keys())
        self._embedding_matrix = np.array([
            self.embeddings[symbol_id] for symbol_id in self._symbol_ids
        ])
    
    def get_symbol(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """Get symbol data by ID."""
        return self.symbols.get(symbol_id)
    
    def remove_symbol(self, symbol_id: str) -> bool:
        """Remove a symbol from the library."""
        if symbol_id in self.symbols:
            del self.symbols[symbol_id]
            del self.embeddings[symbol_id]
            self._update_embedding_matrix()
            logger.info(f"Removed symbol {symbol_id} from library")
            return True
        return False
    
    def get_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Get all symbols in the library."""
        return self.symbols.copy()
    
    def get_symbol_count(self) -> int:
        """Get the number of symbols in the library."""
        return len(self.symbols)


