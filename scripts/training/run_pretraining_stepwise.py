#!/usr/bin/env python3
"""
Stepwise Pretraining Script - Tests Uni-Legenden-Bilder first, then PDF collection.

This script:
1. First tests the 4 Uni-Legenden-Bilder (page_1-4_original.png)
2. Extracts symbols from these images as viewshots
3. Learns symbols from these images
4. Then processes PDF collection page by page (if Uni-Legenden work)

Usage:
    python scripts/training/run_pretraining_stepwise.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService

# Setup logging - ALWAYS save to outputs/
output_dir = project_root / "outputs" / "pretraining_stepwise"
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"pretraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LoggingService.setup_logging(
    log_level=logging.INFO,
    log_file=log_file
)
logger = logging.getLogger(__name__)


def process_uni_legend_images(
    active_learner: ActiveLearner,
    model_info: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process the 4 Uni-Legenden-Bilder first.
    
    Args:
        active_learner: ActiveLearner instance
        model_info: Model configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary with processing results
    """
    logger.info("=" * 60)
    logger.info("[STEP 1] Processing Uni-Legenden-Bilder (page_1-4_original.png)")
    logger.info("=" * 60)
    
    results = {
        "step": "uni_legenden",
        "success": False,
        "images_processed": 0,
        "symbols_extracted": 0,
        "symbols_learned": 0,
        "viewshots_extracted": 0,
        "errors": []
    }
    
    # Find Uni-Legenden-Bilder
    uni_images_dir = project_root / "training_data" / "complex_pids"
    uni_images = [
        uni_images_dir / "page_1_original.png",
        uni_images_dir / "page_2_original.png",
        uni_images_dir / "page_3_original.png",
        uni_images_dir / "page_4_original.png"
    ]
    
    # Check which images exist
    existing_images = [img for img in uni_images if img.exists()]
    
    if not existing_images:
        error_msg = "No Uni-Legenden-Bilder found. Expected: page_1-4_original.png in training_data/complex_pids/"
        logger.error(f"[ERROR] {error_msg}")
        results["errors"].append(error_msg)
        return results
    
    logger.info(f"[INFO] Found {len(existing_images)} Uni-Legenden-Bilder")
    
    # Process each image
    for idx, image_path in enumerate(existing_images, 1):
        logger.info("=" * 60)
        logger.info(f"[PROCESSING] Image {idx}/{len(existing_images)}: {image_path.name}")
        logger.info("=" * 60)
        
        try:
            # Create temporary pretraining directory for this image
            temp_pretraining_dir = output_dir / "temp_uni_legenden" / image_path.stem
            temp_pretraining_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image to temp directory
            import shutil
            temp_image_path = temp_pretraining_dir / image_path.name
            shutil.copy2(image_path, temp_image_path)
            
            # Process this image with active learner
            report = active_learner.learn_from_pretraining_symbols(
                pretraining_path=temp_pretraining_dir,
                model_info=model_info
            )
            
            results["images_processed"] += 1
            results["symbols_extracted"] += report.get('symbols_processed', 0)
            results["symbols_learned"] += report.get('symbols_learned', 0)
            
            logger.info(f"[OK] Processed {image_path.name}:")
            logger.info(f"  Symbols extracted: {report.get('symbols_processed', 0)}")
            logger.info(f"  Symbols learned: {report.get('symbols_learned', 0)}")
            
            # Extract viewshots from this image
            viewshots_dir = output_dir / "viewshots" / image_path.stem
            viewshots_count = extract_viewshots_from_image(
                image_path=image_path,
                truth_path=uni_images_dir / f"{image_path.stem}_truth_cgm.json",
                output_dir=viewshots_dir
            )
            
            results["viewshots_extracted"] += viewshots_count
            logger.info(f"  Viewshots extracted: {viewshots_count}")
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {e}"
            logger.error(f"[ERROR] {error_msg}", exc_info=True)
            results["errors"].append(error_msg)
    
    results["success"] = len(results["errors"]) == 0
    
    logger.info("=" * 60)
    logger.info(f"[RESULTS] Uni-Legenden-Bilder Processing Complete")
    logger.info(f"  Images processed: {results['images_processed']}")
    logger.info(f"  Symbols extracted: {results['symbols_extracted']}")
    logger.info(f"  Symbols learned: {results['symbols_learned']}")
    logger.info(f"  Viewshots extracted: {results['viewshots_extracted']}")
    logger.info(f"  Errors: {len(results['errors'])}")
    logger.info("=" * 60)
    
    return results


def extract_viewshots_from_image(
    image_path: Path,
    truth_path: Path,
    output_dir: Path
) -> int:
    """
    Extract viewshots from a single image using truth data.
    
    Args:
        image_path: Path to image file
        truth_path: Path to truth data JSON
        output_dir: Output directory for viewshots
        
    Returns:
        Number of viewshots extracted
    """
    try:
        from PIL import Image
        
        # Load truth data
        if not truth_path.exists():
            logger.warning(f"[WARNING] Truth data not found: {truth_path}")
            return 0
        
        with open(truth_path, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)
        
        # Load image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            elements = truth_data.get('elements', [])
            viewshots_count = 0
            
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
                    type_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save symbol
                    output_path = type_dir / f"{element_type}_{element_id}_{idx}.png"
                    symbol_crop.save(output_path)
                    viewshots_count += 1
                    
                except Exception as e:
                    logger.warning(f"[WARNING] Error extracting viewshot for {element_id}: {e}")
                    continue
        
        return viewshots_count
        
    except Exception as e:
        logger.error(f"[ERROR] Error extracting viewshots from {image_path.name}: {e}", exc_info=True)
        return 0


def process_pdf_collection(
    active_learner: ActiveLearner,
    model_info: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Process PDF collection page by page (if Uni-Legenden worked).
    
    Args:
        active_learner: ActiveLearner instance
        model_info: Model configuration
        output_dir: Output directory for results
        
    Returns:
        Dictionary with processing results
    """
    logger.info("=" * 60)
    logger.info("[STEP 2] Processing PDF Collection (page by page)")
    logger.info("=" * 60)
    
    results = {
        "step": "pdf_collection",
        "success": False,
        "pages_processed": 0,
        "symbols_extracted": 0,
        "symbols_learned": 0,
        "errors": []
    }
    
    # Find PDF collection
    pretraining_path = project_root / "training_data" / "pretraining_symbols"
    pdf_collection_path = pretraining_path / "Pid-symbols-PDF_sammlung.png"
    
    if not pdf_collection_path.exists():
        logger.warning(f"[WARNING] PDF collection not found: {pdf_collection_path}")
        logger.info("[INFO] Skipping PDF collection processing")
        return results
    
    logger.info(f"[INFO] Found PDF collection: {pdf_collection_path}")
    
    # TODO: Split PDF collection into pages and process page by page
    # For now, process the whole collection
    try:
        # Create temporary pretraining directory for PDF collection
        temp_pretraining_dir = output_dir / "temp_pdf_collection"
        temp_pretraining_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy PDF collection to temp directory
        import shutil
        temp_pdf_path = temp_pretraining_dir / pdf_collection_path.name
        shutil.copy2(pdf_collection_path, temp_pdf_path)
        
        # Process PDF collection
        report = active_learner.learn_from_pretraining_symbols(
            pretraining_path=temp_pretraining_dir,
            model_info=model_info
        )
        
        results["pages_processed"] = 1  # TODO: Count actual pages
        results["symbols_extracted"] = report.get('symbols_processed', 0)
        results["symbols_learned"] = report.get('symbols_learned', 0)
        results["success"] = True
        
        logger.info(f"[OK] Processed PDF collection:")
        logger.info(f"  Symbols extracted: {results['symbols_extracted']}")
        logger.info(f"  Symbols learned: {results['symbols_learned']}")
        
    except Exception as e:
        error_msg = f"Error processing PDF collection: {e}"
        logger.error(f"[ERROR] {error_msg}", exc_info=True)
        results["errors"].append(error_msg)
    
    return results


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("[START] Stepwise Pretraining Script")
    logger.info("=" * 60)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "steps": [],
        "summary": {}
    }
    
    try:
        # Initialize services
        logger.info("[INIT] Initializing services...")
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get paths from config
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        
        # Initialize LLM client
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            logger.error("[ERROR] GCP_PROJECT_ID environment variable not set")
            return
        
        llm_client = LLMClient(
            project_id=project_id,
            default_location=config.get('models', {}).get('Google Gemini 2.5 Flash', {}).get('location', 'us-central1'),
            config=config
        )
        
        # Initialize knowledge manager
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Initialize symbol library
        learned_symbols_images_dir = Path(config.get('paths', {}).get('learned_symbols_images_dir', 'learned_symbols_images'))
        symbol_library = SymbolLibrary(
            llm_client=llm_client,
            learning_db_path=learning_db_path,
            images_dir=learned_symbols_images_dir
        )
        
        # Initialize active learner
        active_learner = ActiveLearner(
            knowledge_manager=knowledge_manager,
            symbol_library=symbol_library,
            llm_client=llm_client,
            config=config
        )
        
        # Get model info
        models_config = config.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            model_info = list(models_config.values())[0] if models_config else {}
        
        logger.info(f"[INFO] Using model: {model_info.get('id', 'unknown')}")
        
        # STEP 1: Process Uni-Legenden-Bilder
        uni_results = process_uni_legend_images(
            active_learner=active_learner,
            model_info=model_info,
            output_dir=output_dir
        )
        all_results["steps"].append(uni_results)
        
        # STEP 2: Process PDF collection (only if Uni-Legenden worked)
        if uni_results["success"]:
            pdf_results = process_pdf_collection(
                active_learner=active_learner,
                model_info=model_info,
                output_dir=output_dir
            )
            all_results["steps"].append(pdf_results)
        else:
            logger.warning("[WARNING] Skipping PDF collection because Uni-Legenden processing failed")
        
        # Summary
        all_results["summary"] = {
            "total_images_processed": sum(step.get("images_processed", 0) for step in all_results["steps"]),
            "total_symbols_extracted": sum(step.get("symbols_extracted", 0) for step in all_results["steps"]),
            "total_symbols_learned": sum(step.get("symbols_learned", 0) for step in all_results["steps"]),
            "total_viewshots_extracted": sum(step.get("viewshots_extracted", 0) for step in all_results["steps"]),
            "total_errors": sum(len(step.get("errors", [])) for step in all_results["steps"])
        }
        
        all_results["success"] = all_results["summary"]["total_errors"] == 0
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("[SUMMARY] Final Pretraining Summary")
        logger.info("=" * 60)
        logger.info(f"Success: {all_results['success']}")
        logger.info(f"Total images processed: {all_results['summary']['total_images_processed']}")
        logger.info(f"Total symbols extracted: {all_results['summary']['total_symbols_extracted']}")
        logger.info(f"Total symbols learned: {all_results['summary']['total_symbols_learned']}")
        logger.info(f"Total viewshots extracted: {all_results['summary']['total_viewshots_extracted']}")
        logger.info(f"Total errors: {all_results['summary']['total_errors']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"[ERROR] Error in stepwise pretraining: {e}", exc_info=True)
        all_results["success"] = False
    
    # Save results
    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[SAVE] Results saved to: {results_file}")
    
    if not all_results["success"]:
        logger.error("[ERROR] Stepwise pretraining failed!")
        sys.exit(1)
    
    logger.info("[OK] Stepwise pretraining completed successfully!")


if __name__ == "__main__":
    main()

