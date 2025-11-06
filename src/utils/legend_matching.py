"""
Legend Matching Utilities - Match legend symbols with diagram symbols.

Uses visual embedding similarity to match:
1. Legend symbols → Diagram symbols
2. Legend line styles → Diagram line paths
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def match_legend_symbols_with_diagram(
    legend_data: Dict[str, Any],
    diagram_elements: List[Dict[str, Any]],
    llm_client,
    image_path: str,
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Match legend symbols with diagram symbols using visual similarity.
    
    CRITICAL FIX: Implements proper N-to-M matching instead of "apple-to-fruit-basket" comparison.
    
    Old approach (WRONG):
    - Generated ONE embedding for entire legend area
    - Compared each diagram element with this "fruit basket" embedding
    - Result: Low, unreliable similarity scores
    
    New approach (CORRECT):
    - Extracts individual legend symbol bboxes from legend_extractor
    - Generates N embeddings (one per legend symbol)
    - Generates M embeddings (one per diagram element)
    - Computes N×M similarity matrix
    - Finds best matches using Hungarian algorithm or greedy matching
    
    Args:
        legend_data: Legend data with symbol_map and symbol_bboxes
        diagram_elements: List of detected diagram elements
        llm_client: LLM client for embedding generation
        image_path: Path to full diagram image
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Dictionary with matches and updated elements
    """
    matches = {
        'legend_to_diagram': {},  # legend_symbol -> diagram_element_id
        'diagram_to_legend': {},  # diagram_element_id -> legend_symbol
        'matched_elements': [],
        'updated_elements': []
    }
    
    try:
        symbol_map = legend_data.get('symbol_map', {})
        if not symbol_map:
            logger.info("No symbol_map in legend data, skipping legend matching")
            return matches
        
        # NEW: Get individual symbol bboxes (critical for N-to-M matching)
        symbol_bboxes = legend_data.get('symbol_bboxes', {})
        if not symbol_bboxes:
            logger.warning("No symbol_bboxes in legend data. Cannot perform proper N-to-M matching.")
            logger.warning("Falling back to old method (less reliable). Consider updating legend_extractor.")
            # Fallback to old method if bboxes not available
            return _match_legend_symbols_fallback(legend_data, diagram_elements, llm_client, image_path, similarity_threshold)
        
        # Load full image
        full_image = Image.open(image_path)
        img_width, img_height = full_image.size
        
        # STEP 1: Generate embeddings for each legend symbol (N embeddings)
        legend_symbol_embeddings = {}
        legend_symbol_keys = list(symbol_map.keys())
        
        logger.info(f"Generating embeddings for {len(legend_symbol_keys)} legend symbols...")
        
        for symbol_key in legend_symbol_keys:
            symbol_bbox = symbol_bboxes.get(symbol_key)
            if not symbol_bbox:
                logger.debug(f"No bbox for legend symbol '{symbol_key}', skipping")
                continue
            
            # Crop individual legend symbol
            sym_x = int(symbol_bbox.get('x', 0) * img_width)
            sym_y = int(symbol_bbox.get('y', 0) * img_height)
            sym_w = int(symbol_bbox.get('width', 0) * img_width)
            sym_h = int(symbol_bbox.get('height', 0) * img_height)
            
            # Add small padding
            padding = 5
            sym_x_pad = max(0, sym_x - padding)
            sym_y_pad = max(0, sym_y - padding)
            sym_w_pad = min(img_width - sym_x_pad, sym_w + 2 * padding)
            sym_h_pad = min(img_height - sym_y_pad, sym_h + 2 * padding)
            
            try:
                symbol_crop = full_image.crop((sym_x_pad, sym_y_pad, sym_x_pad + sym_w_pad, sym_y_pad + sym_h_pad))
                
                # Generate embedding for individual legend symbol
                symbol_embedding = llm_client.get_image_embedding(symbol_crop)
                if symbol_embedding:
                    legend_symbol_embeddings[symbol_key] = np.array(symbol_embedding)
                    logger.debug(f"Generated embedding for legend symbol '{symbol_key}'")
            except Exception as e:
                logger.debug(f"Error generating embedding for legend symbol '{symbol_key}': {e}")
                continue
        
        if not legend_symbol_embeddings:
            logger.warning("Could not generate embeddings for any legend symbols")
            return matches
        
        logger.info(f"Generated {len(legend_symbol_embeddings)} legend symbol embeddings")
        
        # STEP 2: Generate embeddings for each diagram element (M embeddings)
        diagram_element_embeddings = {}
        valid_elements = []
        
        logger.info(f"Generating embeddings for {len(diagram_elements)} diagram elements...")
        
        for element in diagram_elements:
            element_id = element.get('id')
            element_bbox = element.get('bbox')
            
            if not element_id or not element_bbox:
                continue
            
            # Crop element from diagram
            el_x = int(element_bbox.get('x', 0) * img_width)
            el_y = int(element_bbox.get('y', 0) * img_height)
            el_w = int(element_bbox.get('width', 0) * img_width)
            el_h = int(element_bbox.get('height', 0) * img_height)
            
            # Add padding
            padding = 20
            el_x_pad = max(0, el_x - padding)
            el_y_pad = max(0, el_y - padding)
            el_w_pad = min(img_width - el_x_pad, el_w + 2 * padding)
            el_h_pad = min(img_height - el_y_pad, el_h + 2 * padding)
            
            try:
                element_crop = full_image.crop((el_x_pad, el_y_pad, el_x_pad + el_w_pad, el_y_pad + el_h_pad))
                
                # Generate embedding for diagram element
                element_embedding = llm_client.get_image_embedding(element_crop)
                if element_embedding:
                    diagram_element_embeddings[element_id] = np.array(element_embedding)
                    valid_elements.append(element)
                    logger.debug(f"Generated embedding for diagram element '{element_id}'")
            except Exception as e:
                logger.debug(f"Error generating embedding for element {element_id}: {e}")
                continue
        
        if not diagram_element_embeddings:
            logger.warning("Could not generate embeddings for any diagram elements")
            return matches
        
        logger.info(f"Generated {len(diagram_element_embeddings)} diagram element embeddings")
        
        # STEP 3: Compute N×M similarity matrix
        logger.info(f"Computing {len(legend_symbol_embeddings)}×{len(diagram_element_embeddings)} similarity matrix...")
        
        similarity_matrix = {}
        for symbol_key, symbol_embedding in legend_symbol_embeddings.items():
            symbol_vector = symbol_embedding.reshape(1, -1)
            similarities = {}
            
            for element_id, element_embedding in diagram_element_embeddings.items():
                element_vector = element_embedding.reshape(1, -1)
                similarity = cosine_similarity(symbol_vector, element_vector)[0][0]
                similarities[element_id] = float(similarity)
            
            similarity_matrix[symbol_key] = similarities
        
        # STEP 4: Find best matches (greedy matching: each legend symbol matches best diagram element)
        # Use greedy matching: for each legend symbol, find best matching diagram element
        # (More sophisticated: Hungarian algorithm for optimal assignment)
        
        used_elements = set()
        
        for symbol_key, similarities in similarity_matrix.items():
            # Sort by similarity (descending)
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Find best match that hasn't been used yet
            for element_id, similarity in sorted_matches:
                if element_id in used_elements:
                    continue
                
                # Check if similarity meets threshold
                if similarity >= similarity_threshold:
                    # Additional validation: check if type/label matches
                    element = next((e for e in valid_elements if e.get('id') == element_id), None)
                    if element:
                        element_type = element.get('type', '')
                        element_label = element.get('label', '')
                        legend_type = symbol_map.get(symbol_key, '')
                        
                        # Type/label matching is optional (visual similarity is primary)
                        type_match = (element_type == legend_type or 
                                     element_label == symbol_key or
                                     element_label.startswith(symbol_key))
                        
                        # Accept match if visual similarity is high OR type/label matches
                        if similarity >= similarity_threshold * 1.1 or type_match:  # Slightly higher threshold for visual-only
                            matches['legend_to_diagram'][symbol_key] = element_id
                            matches['diagram_to_legend'][element_id] = symbol_key
                            matches['matched_elements'].append({
                                'element_id': element_id,
                                'legend_symbol': symbol_key,
                                'similarity': similarity,
                                'element_type': element_type,
                                'legend_type': legend_type,
                                'type_match': type_match
                            })
                            
                            # Update element with legend confirmation
                            element['legend_matched'] = True
                            element['legend_symbol'] = symbol_key
                            element['confidence'] = min(1.0, element.get('confidence', 0.5) + 0.1)  # Boost confidence
                            matches['updated_elements'].append(element)
                            
                            used_elements.add(element_id)
                            logger.debug(f"Matched legend symbol '{symbol_key}' with diagram element '{element_id}' (similarity: {similarity:.3f})")
                            break
        
        logger.info(f"Legend matching: {len(matches['matched_elements'])} elements matched with legend symbols (N-to-M method)")
        
    except Exception as e:
        logger.error(f"Error in legend symbol matching: {e}", exc_info=True)
    
    return matches


def _match_legend_symbols_fallback(
    legend_data: Dict[str, Any],
    diagram_elements: List[Dict[str, Any]],
    llm_client,
    image_path: str,
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Fallback method: Old "apple-to-fruit-basket" approach (less reliable).
    
    Used when symbol_bboxes are not available in legend_data.
    """
    matches = {
        'legend_to_diagram': {},
        'diagram_to_legend': {},
        'matched_elements': [],
        'updated_elements': []
    }
    
    logger.warning("Using fallback legend matching method (less reliable)")
    
    # Old implementation (kept for backward compatibility)
    symbol_map = legend_data.get('symbol_map', {})
    legend_bbox = legend_data.get('legend_bbox')
    
    if not symbol_map or not legend_bbox:
        return matches
    
    full_image = Image.open(image_path)
    img_width, img_height = full_image.size
    
    legend_x = int(legend_bbox.get('x', 0) * img_width)
    legend_y = int(legend_bbox.get('y', 0) * img_height)
    legend_w = int(legend_bbox.get('width', 0) * img_width)
    legend_h = int(legend_bbox.get('height', 0) * img_height)
    
    legend_crop = full_image.crop((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h))
    legend_embedding = llm_client.get_image_embedding(legend_crop)
    
    if not legend_embedding:
        return matches
    
    legend_vector = np.array(legend_embedding).reshape(1, -1)
    
    for element in diagram_elements:
        element_id = element.get('id')
        element_bbox = element.get('bbox')
        
        if not element_id or not element_bbox:
            continue
        
        el_x = int(element_bbox.get('x', 0) * img_width)
        el_y = int(element_bbox.get('y', 0) * img_height)
        el_w = int(element_bbox.get('width', 0) * img_width)
        el_h = int(element_bbox.get('height', 0) * img_height)
        
        padding = 20
        el_x_pad = max(0, el_x - padding)
        el_y_pad = max(0, el_y - padding)
        el_w_pad = min(img_width - el_x_pad, el_w + 2 * padding)
        el_h_pad = min(img_height - el_y_pad, el_h + 2 * padding)
        
        try:
            element_crop = full_image.crop((el_x_pad, el_y_pad, el_x_pad + el_w_pad, el_y_pad + el_h_pad))
            element_embedding = llm_client.get_image_embedding(element_crop)
            
            if not element_embedding:
                continue
            
            element_vector = np.array(element_embedding).reshape(1, -1)
            similarity = cosine_similarity(element_vector, legend_vector)[0][0]
            
            if similarity >= similarity_threshold:
                element_type = element.get('type', '')
                element_label = element.get('label', '')
                
                matched_legend_symbol = None
                for legend_symbol, legend_type in symbol_map.items():
                    if (element_type == legend_type or 
                        element_label == legend_symbol or
                        element_label.startswith(legend_symbol)):
                        matched_legend_symbol = legend_symbol
                        break
                
                if matched_legend_symbol:
                    matches['legend_to_diagram'][matched_legend_symbol] = element_id
                    matches['diagram_to_legend'][element_id] = matched_legend_symbol
                    matches['matched_elements'].append({
                        'element_id': element_id,
                        'legend_symbol': matched_legend_symbol,
                        'similarity': float(similarity),
                        'element_type': element_type
                    })
                    
                    element['legend_matched'] = True
                    element['legend_symbol'] = matched_legend_symbol
                    element['confidence'] = min(1.0, element.get('confidence', 0.5) + 0.1)
                    matches['updated_elements'].append(element)
        
        except Exception as e:
            logger.debug(f"Error matching element {element_id}: {e}")
            continue
    
    return matches


def match_legend_lines_with_diagram_paths(
    legend_data: Dict[str, Any],
    diagram_connections: List[Dict[str, Any]],
    image_path: str,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Match legend line styles with diagram line paths.
    
    Args:
        legend_data: Legend data with line_map
        diagram_connections: List of detected connections
        image_path: Path to full diagram image
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Dictionary with line matches
    """
    matches = {
        'legend_to_paths': {},  # legend_line_style -> list of connection_ids
        'path_to_legend': {},   # connection_id -> legend_line_style
        'matched_paths': []
    }
    
    try:
        line_map = legend_data.get('line_map', {})
        if not line_map:
            logger.info("No line_map in legend data, skipping line matching")
            return matches
        
        # For now, use color/style matching from line_map
        # In future, could use visual similarity for line styles
        for connection in diagram_connections:
            connection_id = connection.get('id') or f"{connection.get('from_id')}_{connection.get('to_id')}"
            
            # Check if connection has color/style info
            conn_color = connection.get('color', '')
            conn_style = connection.get('style', '')
            
            # Match with legend line_map
            for legend_line_key, legend_line_info in line_map.items():
                if isinstance(legend_line_info, dict):
                    legend_color = legend_line_info.get('color', '')
                    legend_style = legend_line_info.get('style', '')
                    
                    if (conn_color and legend_color and conn_color.lower() == legend_color.lower()) or \
                       (conn_style and legend_style and conn_style.lower() == legend_style.lower()):
                        # Match found
                        if legend_line_key not in matches['legend_to_paths']:
                            matches['legend_to_paths'][legend_line_key] = []
                        matches['legend_to_paths'][legend_line_key].append(connection_id)
                        matches['path_to_legend'][connection_id] = legend_line_key
                        matches['matched_paths'].append({
                            'connection_id': connection_id,
                            'legend_line': legend_line_key,
                            'color': conn_color or legend_color,
                            'style': conn_style or legend_style
                        })
                        break
        
        logger.info(f"Line matching: {len(matches['matched_paths'])} connections matched with legend line styles")
        
    except Exception as e:
        logger.error(f"Error in legend line matching: {e}", exc_info=True)
    
    return matches

