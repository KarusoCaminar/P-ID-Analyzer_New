#!/usr/bin/env python3
"""
Extract Viewshot Examples from Pretraining PDF/PNG Collection.

This script extracts symbol examples from the pretraining PDF collection (Pid-symbols-PDF_sammlung.png)
and organizes them by type. Uses LLM to identify symbol types and positions, then extracts and crops symbols.

CRITICAL: This creates a large database of visual symbol examples that can be used as optical feedback
for the LLM during analysis, improving type recognition accuracy.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file automatically
try:
    from src.utils.env_loader import load_env_automatically
    load_env_automatically()
except (ImportError, Exception):
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_symbol_regions_cv(image_path: Path) -> List[Dict[str, Any]]:
    """
    Detect symbol regions in the pretraining image using Computer Vision.
    
    Uses contour detection to find individual symbols.
    
    Args:
        image_path: Path to pretraining image
        
    Returns:
        List of detected regions with bboxes
    """
    regions = []
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return regions
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (symbols should be reasonably sized)
        min_symbol_size = min(width, height) // 50  # At least 2% of image size
        max_symbol_size = min(width, height) // 5   # At most 20% of image size
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size
            if (min_symbol_size <= w <= max_symbol_size and 
                min_symbol_size <= h <= max_symbol_size and
                area > min_symbol_size * min_symbol_size * 0.3):  # At least 30% filled
                
                # Normalize bbox (0-1)
                normalized_bbox = {
                    'x': x / width,
                    'y': y / height,
                    'width': w / width,
                    'height': h / height
                }
                
                regions.append({
                    'bbox': normalized_bbox,
                    'pixel_bbox': (x, y, w, h),
                    'area': area
                })
        
        logger.info(f"Detected {len(regions)} potential symbol regions using CV")
        
    except Exception as e:
        logger.error(f"Error detecting symbol regions: {e}", exc_info=True)
    
    return regions


def identify_symbol_type_with_llm(
    symbol_crop: Image.Image,
    llm_client: Any,
    model_info: Dict[str, Any],
    known_types: List[str]
) -> Optional[str]:
    """
    Identify symbol type using LLM.
    
    Args:
        symbol_crop: Cropped symbol image
        model_info: LLM model configuration
        known_types: List of known element types
        
    Returns:
        Symbol type (e.g., "Valve", "Pump") or None
    """
    try:
        # Save crop to temp file
        temp_path = project_root / "temp_symbol_crop.png"
        symbol_crop.save(temp_path)
        
        # Build prompt
        known_types_str = ", ".join(f"'{t}'" for t in known_types)
        prompt = f"""**TASK:** Identify the type of this P&ID symbol.

**KNOWN TYPES (use EXACT spelling):**
{known_types_str}

**RULES:**
1. Use EXACT type names from the list above (case-sensitive)
2. If unsure, return "Unknown"
3. Return ONLY the type name, nothing else

**SYMBOL IMAGE:**
[Image will be provided]

**RESPONSE FORMAT:**
Return ONLY the type name (e.g., "Valve", "Pump", "Volume Flow Sensor")"""
        
        system_prompt = "You are an expert in P&ID symbol recognition. Identify the exact type of the symbol shown."
        
        # Call LLM
        response = llm_client.call_llm(
            model_info,
            system_prompt,
            prompt,
            str(temp_path),
            expected_json_keys=None  # Expect plain text response
        )
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        if response:
            # Extract type from response
            if isinstance(response, dict):
                response = str(response)
            
            response_str = str(response).strip()
            
            # Check if response matches a known type
            for known_type in known_types:
                if known_type.lower() in response_str.lower():
                    return known_type
            
            # If no match, return first word (might be the type)
            words = response_str.split()
            if words:
                potential_type = words[0].strip('"\'.,;:')
                if potential_type in known_types:
                    return potential_type
        
        return None
        
    except Exception as e:
        logger.debug(f"Error identifying symbol type: {e}")
        return None


def extract_symbols_from_pretraining_image(
    image_path: Path,
    output_dir: Path,
    llm_client: Optional[Any] = None,
    model_info: Optional[Dict[str, Any]] = None,
    known_types: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Extract symbols from pretraining image and organize by type.
    
    Args:
        image_path: Path to pretraining image
        output_dir: Output directory for viewshots
        llm_client: Optional LLM client for type identification
        model_info: Optional LLM model configuration
        known_types: Optional list of known element types
        
    Returns:
        Dictionary with extraction statistics
    """
    stats = {}
    
    try:
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        logger.info(f"Processing pretraining image: {img_width}x{img_height}px")
        
        # Detect symbol regions using CV
        regions = detect_symbol_regions_cv(image_path)
        
        if not regions:
            logger.warning("No symbol regions detected. Trying alternative approach...")
            # Alternative: Divide image into grid and process each cell
            # (This is a fallback if CV detection fails)
            return stats
        
        # If LLM is available, identify types
        if llm_client and model_info and known_types:
            logger.info(f"Using LLM to identify {len(regions)} symbol types...")
            
            for idx, region in enumerate(regions):
                # Crop symbol
                x, y, w, h = region['pixel_bbox']
                padding = 10
                crop_x = max(0, x - padding)
                crop_y = max(0, y - padding)
                crop_w = min(img_width - crop_x, w + 2 * padding)
                crop_h = min(img_height - crop_y, h + 2 * padding)
                
                try:
                    symbol_crop = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    # Identify type using LLM
                    symbol_type = identify_symbol_type_with_llm(
                        symbol_crop,
                        llm_client,
                        model_info,
                        known_types
                    )
                    
                    if symbol_type:
                        # Normalize type name (lowercase, replace spaces with underscores)
                        type_dir_name = symbol_type.lower().replace(' ', '_')
                        
                        # Create output directory
                        type_dir = output_dir / type_dir_name
                        type_dir.mkdir(exist_ok=True)
                        
                        # Save symbol
                        output_path = type_dir / f"{type_dir_name}_{idx:04d}.png"
                        symbol_crop.save(output_path)
                        
                        stats[symbol_type] = stats.get(symbol_type, 0) + 1
                        logger.info(f"Extracted {symbol_type} symbol: {output_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing region {idx}: {e}")
                    continue
        
        else:
            # Without LLM: Save all regions to "unknown" folder for manual review
            logger.warning("LLM not available. Saving all regions to 'unknown' folder for manual review.")
            unknown_dir = output_dir / "unknown"
            unknown_dir.mkdir(exist_ok=True)
            
            for idx, region in enumerate(regions):
                x, y, w, h = region['pixel_bbox']
                padding = 10
                crop_x = max(0, x - padding)
                crop_y = max(0, y - padding)
                crop_w = min(img_width - crop_x, w + 2 * padding)
                crop_h = min(img_height - crop_y, h + 2 * padding)
                
                try:
                    symbol_crop = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    output_path = unknown_dir / f"symbol_{idx:04d}.png"
                    symbol_crop.save(output_path)
                    stats['unknown'] = stats.get('unknown', 0) + 1
                except Exception as e:
                    logger.warning(f"Error processing region {idx}: {e}")
                    continue
        
    except Exception as e:
        logger.error(f"Error processing pretraining image: {e}", exc_info=True)
    
    return stats


def main():
    """Main function to extract viewshots from pretraining PDF/PNG."""
    project_root = Path(__file__).parent.parent.parent
    
    # Define paths
    pretraining_image = project_root / "training_data" / "pretraining_symbols" / "Pid-symbols-PDF_sammlung.png"
    output_dir = project_root / "training_data" / "viewshot_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create type directories
    known_types = [
        'Valve', 'Pump', 'Volume Flow Sensor', 'Mixer', 'Source', 'Sink', 
        'Sample Point', 'Storage', 'Heat Exchanger', 'Tank', 'Reactor',
        'Filter', 'Separator', 'Compressor', 'Turbine', 'Heat Exchanger',
        'Control Valve', 'Check Valve', 'Gate Valve', 'Ball Valve'
    ]
    
    for type_name in known_types:
        type_dir_name = type_name.lower().replace(' ', '_')
        (output_dir / type_dir_name).mkdir(exist_ok=True)
    
    if not pretraining_image.exists():
        logger.error(f"Pretraining image not found: {pretraining_image}")
        logger.info("Please ensure the pretraining image exists at:")
        logger.info(f"  {pretraining_image}")
        return
    
    logger.info("=== Extracting Viewshots from Pretraining PDF/PNG ===")
    logger.info(f"Input: {pretraining_image}")
    logger.info(f"Output: {output_dir}")
    
    # Try to load LLM client (optional)
    llm_client = None
    model_info = None
    
    try:
        from src.services.config_service import ConfigService
        from src.analyzer.ai.llm_client import LLMClient
        
        config_service = ConfigService()
        config = config_service.get_config()
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
        
        # Get project_id and location from environment or config
        import os
        project_id = os.getenv('GCP_PROJECT_ID')
        default_location = os.getenv('GCP_LOCATION', 'us-central1')
        
        if not project_id:
            logger.warning("GCP_PROJECT_ID not set. Cannot initialize LLM client.")
            raise ValueError("GCP_PROJECT_ID not set")
        
        # Initialize LLM client correctly
        llm_client = LLMClient(
            project_id=project_id,
            default_location=default_location,
            config=config_dict
        )
        
        # Get model info (use Flash for speed)
        models_config = config_dict.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            model_info = list(models_config.values())[0] if models_config else None
        
        if model_info:
            logger.info("LLM client loaded. Will identify symbol types automatically.")
        else:
            logger.warning("LLM model info not found. Will save symbols to 'unknown' folder.")
    except Exception as e:
        logger.warning(f"Could not load LLM client: {e}", exc_info=True)
        logger.info("Will save all symbols to 'unknown' folder for manual review.")
    
    # Extract symbols
    stats = extract_symbols_from_pretraining_image(
        pretraining_image,
        output_dir,
        llm_client,
        model_info,
        known_types
    )
    
    # Summary
    logger.info("\n=== Extraction Summary ===")
    for type_name, count in sorted(stats.items()):
        logger.info(f"{type_name}: {count} symbols extracted")
    
    logger.info(f"\nTotal symbols extracted: {sum(stats.values())}")
    logger.info(f"Viewshots saved to: {output_dir}")
    
    if stats.get('unknown', 0) > 0:
        logger.info(f"\n[WARNING] {stats['unknown']} symbols saved to 'unknown' folder for manual review.")
        logger.info("   You can manually move them to the correct type folders.")


if __name__ == "__main__":
    main()

