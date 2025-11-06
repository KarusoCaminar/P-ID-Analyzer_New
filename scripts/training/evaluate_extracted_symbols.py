#!/usr/bin/env python3
"""
Evaluate extracted symbols - Check quality, naming, and completeness.

This script checks:
1. Do symbols have OCR labels (real names) or generic names?
2. Are symbols complete (not cut off)?
3. Do symbols contain text (for pretraining)?
4. Are viewshots correct (symbol only, no text)?
5. Are file names meaningful (contain OCR labels)?
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_image_completeness(image_path: Path) -> Dict[str, Any]:
    """
    Check if symbol image is complete (not cut off).
    
    Returns:
        Dict with completeness metrics
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Check if edges are cut off (white/black borders on edges)
        edge_threshold = 10  # pixels from edge
        h, w = gray.shape
        
        # Check top edge
        top_edge = gray[:edge_threshold, :]
        top_white = np.sum(top_edge > 240) / top_edge.size
        
        # Check bottom edge
        bottom_edge = gray[-edge_threshold:, :]
        bottom_white = np.sum(bottom_edge > 240) / bottom_edge.size
        
        # Check left edge
        left_edge = gray[:, :edge_threshold]
        left_white = np.sum(left_edge > 240) / left_edge.size
        
        # Check right edge
        right_edge = gray[:, -edge_threshold:]
        right_white = np.sum(right_edge > 240) / right_edge.size
        
        # If edges are mostly white, symbol might be cut off
        edge_white_ratio = (top_white + bottom_white + left_white + right_white) / 4
        
        # Check if symbol touches edges (bad - means cut off)
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        touches_edge = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_contour, h_contour = cv2.boundingRect(largest_contour)
            
            # Check if contour touches any edge
            margin = 5
            touches_edge = (
                x < margin or y < margin or
                x + w_contour > w - margin or
                y + h_contour > h - margin
            )
        
        return {
            'complete': not touches_edge and edge_white_ratio < 0.8,
            'touches_edge': touches_edge,
            'edge_white_ratio': edge_white_ratio,
            'size': img.size
        }
    except Exception as e:
        logger.error(f"Error checking completeness for {image_path}: {e}")
        return {'complete': False, 'error': str(e)}


def check_has_text(image_path: Path) -> bool:
    """
    Check if image contains text (for pretraining).
    
    Uses simple heuristic: text regions are usually horizontal lines.
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Use OCR-like detection (horizontal lines)
        # Apply horizontal line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Count horizontal line pixels
        line_pixels = np.sum(detected_lines > 0)
        total_pixels = gray.size
        
        # If > 2% of image is horizontal lines, likely has text
        has_text = (line_pixels / total_pixels) > 0.02
        
        return has_text
    except Exception as e:
        logger.error(f"Error checking text for {image_path}: {e}")
        return False


def analyze_symbols():
    """Analyze extracted symbols."""
    logger.info("=== EVALUATING EXTRACTED SYMBOLS ===")
    
    # Load learning database
    learning_db_path = Path("training_data/learning_db.json")
    if not learning_db_path.exists():
        logger.error(f"Learning database not found: {learning_db_path}")
        return
    
    with open(learning_db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    symbols = data.get('symbol_library', {})
    logger.info(f"Found {len(symbols)} symbols in database")
    
    # Statistics
    stats = {
        'total': len(symbols),
        'with_ocr_label': 0,
        'with_generic_name': 0,
        'complete': 0,
        'cut_off': 0,
        'with_text': 0,
        'without_text': 0,
        'by_type': {},
        'naming_issues': []
    }
    
    # Analyze each symbol
    for symbol_id, symbol_data in symbols.items():
        element_type = symbol_data.get('element_type', 'Unknown')
        metadata = symbol_data.get('metadata', {})
        label = metadata.get('label', '')
        image_path_str = symbol_data.get('image_path')
        
        # Count by type
        if element_type not in stats['by_type']:
            stats['by_type'][element_type] = 0
        stats['by_type'][element_type] += 1
        
        # Check naming
        if label and not label.startswith('Pid-symbols-PDF_sammlung_sym_'):
            # Has OCR label (real name)
            stats['with_ocr_label'] += 1
        else:
            # Generic name
            stats['with_generic_name'] += 1
            stats['naming_issues'].append({
                'symbol_id': symbol_id,
                'label': label,
                'type': element_type
            })
        
        # Check image if path exists
        if image_path_str:
            image_path = Path(image_path_str)
            if image_path.exists():
                # Check completeness
                completeness = check_image_completeness(image_path)
                if completeness.get('complete', False):
                    stats['complete'] += 1
                else:
                    stats['cut_off'] += 1
                
                # Check for text (pretraining should have text)
                has_text = check_has_text(image_path)
                if has_text:
                    stats['with_text'] += 1
                else:
                    stats['without_text'] += 1
    
    # Print report
    logger.info("\n=== EVALUATION REPORT ===")
    logger.info(f"Total symbols: {stats['total']}")
    
    if stats['total'] == 0:
        logger.warning("⚠️  No symbols found in database. Run pretraining first!")
        return
    
    logger.info(f"\nNaming:")
    logger.info(f"  With OCR labels (real names): {stats['with_ocr_label']} ({stats['with_ocr_label']/stats['total']*100:.1f}%)")
    logger.info(f"  With generic names: {stats['with_generic_name']} ({stats['with_generic_name']/stats['total']*100:.1f}%)")
    
    logger.info(f"\nCompleteness:")
    logger.info(f"  Complete (not cut off): {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
    logger.info(f"  Cut off: {stats['cut_off']} ({stats['cut_off']/stats['total']*100:.1f}%)")
    
    logger.info(f"\nText content (for pretraining):")
    logger.info(f"  With text: {stats['with_text']} ({stats['with_text']/stats['total']*100:.1f}%)")
    logger.info(f"  Without text: {stats['without_text']} ({stats['without_text']/stats['total']*100:.1f}%)")
    
    logger.info(f"\nBy type:")
    for type_name, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {type_name}: {count}")
    
    if stats['naming_issues']:
        logger.info(f"\n⚠️  Naming issues (first 10):")
        for issue in stats['naming_issues'][:10]:
            logger.info(f"  {issue['symbol_id']}: label='{issue['label']}', type={issue['type']}")
    
    # Recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    if stats['with_generic_name'] > stats['with_ocr_label']:
        logger.warning("⚠️  Most symbols have generic names. OCR label extraction needs improvement.")
    
    if stats['cut_off'] > stats['complete']:
        logger.warning("⚠️  Many symbols are cut off. Bounding box refinement needs improvement.")
    
    if stats['without_text'] > stats['with_text']:
        logger.warning("⚠️  Most symbols don't contain text. Pretraining images should include text.")
    
    # Save report
    report_path = Path("outputs/symbol_evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    analyze_symbols()

