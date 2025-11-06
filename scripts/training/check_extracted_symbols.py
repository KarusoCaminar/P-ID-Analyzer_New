#!/usr/bin/env python3
"""
Check extracted symbols - visualize and count actual symbols
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.symbol_extraction import extract_symbols_with_cv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Check extracted symbols."""
    logger.info("=== CHECKING EXTRACTED SYMBOLS ===")
    
    # Test image
    test_image_path = Path("training_data/pretraining_symbols/Pid-symbols-PDF_sammlung.png")
    logger.info(f"Loading image: {test_image_path}")
    
    if not test_image_path.exists():
        logger.error(f"Image not found: {test_image_path}")
        return
    
    image = Image.open(test_image_path)
    logger.info(f"Image size: {image.size} (WxH)")
    
    # Extract symbols with CV
    logger.info("Extracting symbols with CV...")
    cv_symbols = extract_symbols_with_cv(
        str(test_image_path),
        min_symbol_size=30,
        max_symbol_size=800,
        text_padding=15
    )
    
    logger.info(f"CV extraction found {len(cv_symbols)} candidate regions")
    
    # Analyze symbol sizes
    sizes = []
    for idx, cv_symbol in enumerate(cv_symbols):
        bbox = cv_symbol.get('bbox', {})
        w = bbox.get('width', 0)
        h = bbox.get('height', 0)
        area = w * h
        sizes.append({
            'idx': idx,
            'width': w,
            'height': h,
            'area': area,
            'method': cv_symbol.get('method', 'unknown'),
            'confidence': cv_symbol.get('confidence', 0.0)
        })
    
    # Sort by area
    sizes.sort(key=lambda x: x['area'], reverse=True)
    
    # Show statistics
    logger.info("\n=== SYMBOL SIZE STATISTICS ===")
    logger.info(f"Total regions: {len(sizes)}")
    if sizes:
        logger.info(f"Largest: {sizes[0]['width']}x{sizes[0]['height']} (area: {sizes[0]['area']})")
        logger.info(f"Smallest: {sizes[-1]['width']}x{sizes[-1]['height']} (area: {sizes[-1]['area']})")
        
        # Count by method
        by_method = {}
        for s in sizes:
            method = s['method']
            by_method[method] = by_method.get(method, 0) + 1
        
        logger.info(f"\nBy detection method:")
        for method, count in by_method.items():
            logger.info(f"  {method}: {count}")
        
        # Show first 20 symbols
        logger.info(f"\n=== FIRST 20 SYMBOLS (by area) ===")
        for s in sizes[:20]:
            logger.info(f"Symbol {s['idx']}: {s['width']}x{s['height']} (area: {s['area']}, method: {s['method']}, conf: {s['confidence']:.2f})")
    
    # Visualize on image
    logger.info("\n=== Creating visualization... ===")
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Draw bounding boxes
    for idx, cv_symbol in enumerate(cv_symbols[:50]):  # Only first 50 to avoid clutter
        bbox = cv_symbol.get('bbox', {})
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        w = bbox.get('width', 0)
        h = bbox.get('height', 0)
        
        # Draw rectangle
        color = 'red' if cv_symbol.get('method') == 'contour' else 'blue'
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # Draw index
        try:
            draw.text((x, y - 15), str(idx), fill=color)
        except:
            pass
    
    # Save visualization
    output_path = Path("outputs/symbol_extraction_check.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis_image.save(output_path)
    logger.info(f"Visualization saved to: {output_path}")
    logger.info(f"Image shows first 50 detected regions (red=contour, blue=edge)")

if __name__ == "__main__":
    main()

