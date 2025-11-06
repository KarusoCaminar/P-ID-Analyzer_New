#!/usr/bin/env python3
"""
Extract Viewshot Examples from Uni-Bilder.

This script extracts symbol examples from Uni-Bilder images and organizes them by type.
Uses truth data to identify symbol types and positions, then extracts and crops symbols.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_truth_data(truth_path: Path) -> Optional[Dict[str, Any]]:
    """Load truth data JSON file."""
    try:
        with open(truth_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading truth data from {truth_path}: {e}")
        return None


def extract_symbols_from_image(
    image_path: Path,
    truth_data: Dict[str, Any],
    output_dir: Path,
    img_width: int,
    img_height: int
) -> Dict[str, int]:
    """Extract symbols from image based on truth data."""
    stats = {}
    
    try:
        with Image.open(image_path) as img:
            elements = truth_data.get('elements', [])
            
            for idx, element in enumerate(elements):
                element_type = element.get('type', '').lower().replace(' ', '_')
                element_id = element.get('id', f'unknown_{idx}')
                bbox = element.get('bbox')
                
                # Skip if no bbox or invalid type
                if not bbox or not element_type:
                    continue
                
                # Normalize bbox (convert from absolute to normalized if needed)
                if isinstance(bbox, dict):
                    x = bbox.get('x', 0)
                    y = bbox.get('y', 0)
                    width = bbox.get('width', 0)
                    height = bbox.get('height', 0)
                    
                    # If bbox is normalized (0-1), convert to absolute
                    if x < 1 and y < 1 and width < 1 and height < 1:
                        x = int(x * img_width)
                        y = int(y * img_height)
                        width = int(width * img_width)
                        height = int(height * img_height)
                else:
                    continue
                
                # Validate bbox
                if width <= 0 or height <= 0:
                    continue
                
                # Crop symbol from image
                try:
                    # Add padding (10% on each side)
                    padding_x = int(width * 0.1)
                    padding_y = int(height * 0.1)
                    
                    crop_x = max(0, x - padding_x)
                    crop_y = max(0, y - padding_y)
                    crop_w = min(img_width - crop_x, width + 2 * padding_x)
                    crop_h = min(img_height - crop_y, height + 2 * padding_y)
                    
                    symbol_crop = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    # Create output directory for this type
                    type_dir = output_dir / element_type
                    type_dir.mkdir(exist_ok=True)
                    
                    # Save symbol
                    output_path = type_dir / f"{element_type}_{element_id}_{idx}.png"
                    symbol_crop.save(output_path)
                    
                    stats[element_type] = stats.get(element_type, 0) + 1
                    logger.info(f"Extracted {element_type} symbol: {element_id} -> {output_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Error cropping symbol {element_id}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
    
    return stats


def extract_viewshots_from_pretraining_symbols(
    pretraining_dir: Path,
    output_dir: Path
) -> Dict[str, int]:
    """Extract symbols from pretraining_symbols directory (already extracted symbols)."""
    stats = {}
    
    if not pretraining_dir.exists():
        logger.warning(f"Pretraining symbols directory not found: {pretraining_dir}")
        return stats
    
    # Look for symbol images in pretraining directory
    symbol_images = list(pretraining_dir.glob("*.png"))
    
    logger.info(f"Found {len(symbol_images)} symbol images in pretraining directory")
    
    # Note: These are already extracted symbols, so we can use them directly
    # But we need to identify their types - this would require LLM or manual labeling
    # For now, we'll create a placeholder structure
    
    return stats


def main():
    """Main function to extract viewshots."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    uni_bilder_dir = project_root / "training_data" / "complex_pids"
    output_dir = project_root / "training_data" / "viewshot_examples"
    output_dir.mkdir(exist_ok=True)
    
    # Create type directories
    for type_name in ['valve', 'pump', 'flow_sensor', 'mixer', 'source', 'sink', 'sample_point']:
        (output_dir / type_name).mkdir(exist_ok=True)
    
    logger.info("=== Extracting Viewshots from Uni-Bilder ===")
    
    # Process each Uni-Bild (page_1 through page_4)
    total_stats = {}
    
    for page_num in range(1, 5):
        image_path = uni_bilder_dir / f"page_{page_num}_original.png"
        truth_path = uni_bilder_dir / f"page_{page_num}_original_truth_cgm.json"
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        if not truth_path.exists():
            logger.warning(f"Truth data not found: {truth_path}")
            continue
        
        logger.info(f"\nProcessing page_{page_num}...")
        
        # Load truth data
        truth_data = load_truth_data(truth_path)
        if not truth_data:
            continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            continue
        
        # Extract symbols
        stats = extract_symbols_from_image(
            image_path,
            truth_data,
            output_dir,
            img_width,
            img_height
        )
        
        # Update total stats
        for type_name, count in stats.items():
            total_stats[type_name] = total_stats.get(type_name, 0) + count
    
    # Summary
    logger.info("\n=== Extraction Summary ===")
    for type_name, count in sorted(total_stats.items()):
        logger.info(f"{type_name}: {count} symbols extracted")
    
    logger.info(f"\nTotal symbols extracted: {sum(total_stats.values())}")
    logger.info(f"Viewshots saved to: {output_dir}")


if __name__ == "__main__":
    main()

