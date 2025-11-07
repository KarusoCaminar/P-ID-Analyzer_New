"""
Symbol Library - Manages visual symbol library with embeddings.

Provides functionality for:
- Adding symbols to the library
- Finding similar symbols by visual similarity
- Managing symbol metadata
"""

import logging
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

logger = logging.getLogger(__name__)


class SymbolLibrary:
    """
    Manages a library of visual symbols with embeddings for similarity search.
    """
    
    def __init__(
        self, 
        llm_client: Any, 
        learning_db_path: Optional[Path] = None,
        images_dir: Optional[Path] = None
    ):
        """
        Initialize Symbol Library.
        
        Args:
            llm_client: LLM client for generating image embeddings
            learning_db_path: Optional path to learning database for persistence
            images_dir: Optional directory for storing symbol images (for viewshots)
        """
        self.llm_client = llm_client
        self.learning_db_path = learning_db_path
        self.images_dir = images_dir
        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._symbol_ids: List[str] = []
        
        # CRITICAL FIX: Add threading lock for thread-safe access
        self.lock = threading.Lock()
        
        # Create images directory if provided
        if self.images_dir:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load symbols from learning database if available
        if learning_db_path:
            self.load_from_learning_db()
    
    def add_symbol(
        self,
        symbol_id: str,
        image: Image.Image,
        element_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_immediately: bool = True
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
        # CRITICAL FIX: Thread-safe access to symbols and embeddings
        with self.lock:
            try:
                # Generate embedding
                embedding = self.llm_client.get_image_embedding(image)
                if embedding is None:
                    logger.error(f"Failed to generate embedding for symbol {symbol_id}")
                    return False
                
                # Store symbol data
                image_path = None
                if self.images_dir:
                    # Save image to disk for viewshot generation
                    type_dir_name = element_type.lower().replace(' ', '_')
                    type_dir = self.images_dir / type_dir_name
                    type_dir.mkdir(exist_ok=True)
                    # CRITICAL: symbol_id already contains OCR label + type + UUID
                    # Format: {ocr_label}_{type}_{short_uuid} or {type}_{uuid}
                    # This makes filenames meaningful for the AI
                    image_path = type_dir / f"{symbol_id}.png"
                    image.save(image_path)
                    logger.debug(f"Saved symbol image: {image_path}")
                
                self.symbols[symbol_id] = {
                    'element_type': element_type,
                    'metadata': metadata or {},
                    'added_timestamp': datetime.now().isoformat(),
                    'image_path': str(image_path) if image_path else None
                }
                
                self.embeddings[symbol_id] = np.array(embedding)
                self._update_embedding_matrix()
                
                # Save to learning database if available (only if requested)
                if self.learning_db_path and save_immediately:
                    self.save_to_learning_db()
                
                logger.debug(f"Added symbol {symbol_id} to library")
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
        # CRITICAL FIX: Thread-safe read access to embeddings
        with self.lock:
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
        # CRITICAL FIX: Thread-safe access to embeddings
        with self.lock:
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
        # CRITICAL FIX: Thread-safe read access
        with self.lock:
            return self.symbols.get(symbol_id)
    
    def remove_symbol(self, symbol_id: str) -> bool:
        """Remove a symbol from the library."""
        # CRITICAL FIX: Thread-safe write access
        with self.lock:
            if symbol_id in self.symbols:
                del self.symbols[symbol_id]
                if symbol_id in self.embeddings:
                    del self.embeddings[symbol_id]
                self._update_embedding_matrix()
                logger.info(f"Removed symbol {symbol_id} from library")
                return True
            return False
    
    def get_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Get all symbols in the library."""
        # CRITICAL FIX: Thread-safe read access
        with self.lock:
            return self.symbols.copy()
    
    def get_symbol_count(self) -> int:
        """Get the number of symbols in the library."""
        # CRITICAL FIX: Thread-safe read access
        with self.lock:
            return len(self.symbols)
    
    def load_from_learning_db(self) -> int:
        """
        Load symbols from learning database.
        
        Returns:
            Number of symbols loaded
        """
        # CRITICAL FIX: Thread-safe write access
        with self.lock:
            if not self.learning_db_path or not self.learning_db_path.exists():
                return 0
            
            try:
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    learning_db = json.load(f)
                
                symbol_library_data = learning_db.get('symbol_library', {})
                if not symbol_library_data:
                    return 0
                
                loaded_count = 0
                for symbol_id, symbol_data in symbol_library_data.items():
                    try:
                        # Restore symbol metadata
                        self.symbols[symbol_id] = {
                            'element_type': symbol_data.get('element_type', 'Unknown'),
                            'metadata': symbol_data.get('metadata', {}),
                            'added_timestamp': symbol_data.get('added_timestamp', datetime.now().isoformat()),
                            'image_path': symbol_data.get('image_path')  # Restore image path for viewshot generation
                        }
                        
                        # Restore embedding (stored as list in JSON)
                        embedding_list = symbol_data.get('embedding', [])
                        if embedding_list:
                            self.embeddings[symbol_id] = np.array(embedding_list)
                            loaded_count += 1
                        else:
                            logger.warning(f"Symbol {symbol_id} has no embedding, skipping")
                            
                    except Exception as e:
                        logger.warning(f"Error loading symbol {symbol_id}: {e}")
                        continue
                
                if loaded_count > 0:
                    self._update_embedding_matrix()
                    logger.info(f"Loaded {loaded_count} symbols from learning database")
                
                return loaded_count
                
            except Exception as e:
                logger.warning(f"Error loading symbols from learning database: {e}")
                return 0
    
    def save_to_learning_db(self) -> bool:
        """
        Save symbols to learning database.
        
        Returns:
            True if saved successfully, False otherwise
        """
        # CRITICAL FIX: Thread-safe read access (read-only for save)
        with self.lock:
            if not self.learning_db_path:
                return False
            
            try:
                # Load existing learning database
                learning_db = {}
                if self.learning_db_path.exists():
                    try:
                        with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                            learning_db = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error reading learning database: {e}")
                        learning_db = {}
                
                # Prepare symbol library data (embeddings as lists for JSON serialization)
                symbol_library_data = {}
                for symbol_id, symbol_info in self.symbols.items():
                    embedding = self.embeddings.get(symbol_id)
                    if embedding is not None:
                        symbol_library_data[symbol_id] = {
                            'element_type': symbol_info.get('element_type', 'Unknown'),
                            'metadata': symbol_info.get('metadata', {}),
                            'added_timestamp': symbol_info.get('added_timestamp', datetime.now().isoformat()),
                            'embedding': embedding.tolist(),  # Convert numpy array to list for JSON
                            'image_path': symbol_info.get('image_path')  # Save image path for viewshot generation
                        }
                
                # Update learning database
                learning_db['symbol_library'] = symbol_library_data
                
                # Write back to file (atomic write with backup)
                backup_path = self.learning_db_path.with_suffix('.json.bak')
                if self.learning_db_path.exists():
                    import shutil
                    shutil.copy2(self.learning_db_path, backup_path)
                
                with open(self.learning_db_path, 'w', encoding='utf-8') as f:
                    json.dump(learning_db, f, indent=2, ensure_ascii=False)
                
                # Remove backup if save successful
                if backup_path.exists():
                    backup_path.unlink()
                
                logger.debug(f"Saved {len(symbol_library_data)} symbols to learning database")
                return True
                
            except Exception as e:
                logger.error(f"Error saving symbols to learning database: {e}", exc_info=True)
                return False
    


