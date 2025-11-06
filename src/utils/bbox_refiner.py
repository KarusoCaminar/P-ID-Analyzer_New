"""
BBox Refiner - Iterative bounding box refinement with visual feedback.

Refines legend and metadata bounding boxes by:
1. Visualizing the current bbox on the image
2. Asking LLM if the bbox is optimal (not too big, not too small)
3. Iteratively refining until optimal size is reached
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import json
import tempfile

logger = logging.getLogger(__name__)


def visualize_bbox_on_image(
    image_path: str,
    bbox: Dict[str, float],
    output_path: Optional[str] = None,
    color: Tuple[int, int, int] = (255, 0, 0),  # Red
    line_width: int = 3
) -> str:
    """
    Visualize bounding box on image and save to file.
    
    Args:
        image_path: Path to original image
        bbox: Bounding box (normalized: x, y, width, height)
        output_path: Optional output path. If None, creates temp file
        color: RGB color for bbox outline
        line_width: Width of bbox outline
        
    Returns:
        Path to visualization image
    """
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Convert normalized bbox to pixel coordinates
        x = int(bbox['x'] * img_width)
        y = int(bbox['y'] * img_height)
        w = int(bbox['width'] * img_width)
        h = int(bbox['height'] * img_height)
        
        # Ensure within bounds
        x = max(0, min(img_width - 1, x))
        y = max(0, min(img_height - 1, y))
        w = max(1, min(img_width - x, w))
        h = max(1, min(img_height - y, h))
        
        # Create copy for drawing
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
        
        # Add label
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        label = f"BBox: ({x}, {y}, {w}, {h})"
        draw.text((x, y - 20), label, fill=color, font=font)
        
        # Save visualization
        if output_path is None:
            output_path = str(Path(tempfile.gettempdir()) / f"bbox_vis_{Path(image_path).stem}.png")
        
        img_copy.save(output_path)
        logger.debug(f"BBox visualization saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing bbox: {e}", exc_info=True)
        return image_path  # Fallback to original image


def refine_metadata_bbox_iteratively(
    image_path: str,
    initial_bbox: Dict[str, float],
    llm_client: Any,
    model_info: Dict[str, Any],
    system_prompt: str,
    max_iterations: int = 3,
    min_reduction: float = 0.05  # Minimum 5% reduction per iteration
) -> Dict[str, float]:
    """
    Iteratively refine metadata bounding box with LLM feedback.
    
    Similar to refine_legend_bbox_iteratively, but for metadata area.
    
    Args:
        image_path: Path to P&ID image
        initial_bbox: Initial bounding box (normalized)
        llm_client: LLM client for feedback
        model_info: Model configuration
        system_prompt: System prompt for LLM
        max_iterations: Maximum refinement iterations
        min_reduction: Minimum reduction per iteration (0.05 = 5%)
        
    Returns:
        Refined bounding box (normalized)
    """
    current_bbox = initial_bbox.copy()
    
    for iteration in range(max_iterations):
        logger.info(f"Metadata BBox refinement iteration {iteration + 1}/{max_iterations}")
        
        # Visualize current bbox
        vis_path = visualize_bbox_on_image(
            image_path,
            current_bbox,
            output_path=None  # Will create temp file
        )
        
        # Build refinement prompt
        refinement_prompt = f"""**TASK:** You will see a P&ID diagram with a RED bounding box drawn around the metadata area (typically bottom-right corner with project info, title, version, date).

**CURRENT BOUNDING BOX:**
- x: {current_bbox['x']:.3f} (normalized, 0-1)
- y: {current_bbox['y']:.3f} (normalized, 0-1)
- width: {current_bbox['width']:.3f} (normalized, 0-1)
- height: {current_bbox['height']:.3f} (normalized, 0-1)

**QUESTION:** Look at the red bounding box. Is it optimal?

**OPTIMAL BBOX CRITERIA:**
1. **NOT TOO BIG:** The bbox should be as small as possible, excluding unnecessary white space or non-metadata content
2. **NOT TOO SMALL:** The bbox must include ALL metadata content (project name, title, version, date, etc.)
3. **TIGHT FIT:** The bbox should tightly fit around the metadata content

**YOUR RESPONSE:**
Provide a JSON object with:
- `is_optimal`: boolean - true if bbox is optimal, false if it can be improved
- `can_be_smaller`: boolean - true if bbox can be made smaller without losing content
- `refined_bbox`: object with `x`, `y`, `width`, `height` (normalized 0-1) - the refined bbox if it can be improved, or the current bbox if optimal
- `reason`: string - explanation of your decision

**CRITICAL:** Only reduce the bbox if you are CERTAIN that no metadata content will be excluded. When in doubt, keep the current bbox."""
        
        try:
            # Call LLM for refinement feedback
            response = llm_client.call_llm(
                model_info,
                system_prompt,
                refinement_prompt,
                vis_path,  # Send visualization image
                expected_json_keys=["is_optimal", "can_be_smaller", "refined_bbox", "reason"]
            )
            
            if response and isinstance(response, dict):
                is_optimal = response.get("is_optimal", False)
                can_be_smaller = response.get("can_be_smaller", False)
                refined_bbox = response.get("refined_bbox")
                reason = response.get("reason", "No reason provided")
                
                logger.info(f"Refinement iteration {iteration + 1}: is_optimal={is_optimal}, can_be_smaller={can_be_smaller}")
                logger.info(f"Reason: {reason}")
                
                if is_optimal or not can_be_smaller:
                    logger.info(f"BBox is optimal after {iteration + 1} iterations. Stopping refinement.")
                    return current_bbox
                
                if refined_bbox:
                    # Validate refined bbox
                    if all(k in refined_bbox for k in ['x', 'y', 'width', 'height']):
                        # Check if reduction is significant (at least min_reduction)
                        old_area = current_bbox['width'] * current_bbox['height']
                        new_area = refined_bbox['width'] * refined_bbox['height']
                        reduction = (old_area - new_area) / old_area
                        
                        if reduction >= min_reduction:
                            logger.info(f"BBox reduced by {reduction:.1%} (area: {old_area:.3f} -> {new_area:.3f})")
                            current_bbox = refined_bbox
                        else:
                            logger.info(f"BBox reduction too small ({reduction:.1%} < {min_reduction:.1%}). Stopping refinement.")
                            return current_bbox
                    else:
                        logger.warning("Refined bbox missing required keys. Using current bbox.")
                        return current_bbox
                else:
                    logger.warning("No refined_bbox in response. Using current bbox.")
                    return current_bbox
            else:
                logger.warning(f"LLM response invalid: {response}")
                return current_bbox
                
        except Exception as e:
            logger.error(f"Error in refinement iteration {iteration + 1}: {e}", exc_info=True)
            return current_bbox
    
    logger.info(f"Reached max iterations ({max_iterations}). Returning current bbox.")
    return current_bbox


def refine_legend_bbox_iteratively(
    image_path: str,
    initial_bbox: Dict[str, float],
    llm_client: Any,
    model_info: Dict[str, Any],
    system_prompt: str,
    max_iterations: int = 3,
    min_reduction: float = 0.05  # Minimum 5% reduction per iteration
) -> Dict[str, float]:
    """
    Iteratively refine legend bounding box with LLM feedback.
    
    Process:
    1. Visualize current bbox on image
    2. Send to LLM with question: "Is this bbox optimal? Can it be smaller?"
    3. If LLM says yes, refine bbox (make it smaller)
    4. Repeat 2-3 times until optimal
    
    Args:
        image_path: Path to P&ID image
        initial_bbox: Initial bounding box (normalized)
        llm_client: LLM client for feedback
        model_info: Model configuration
        system_prompt: System prompt for LLM
        max_iterations: Maximum refinement iterations
        min_reduction: Minimum reduction per iteration (0.05 = 5%)
        
    Returns:
        Refined bounding box (normalized)
    """
    current_bbox = initial_bbox.copy()
    
    for iteration in range(max_iterations):
        logger.info(f"Legend BBox refinement iteration {iteration + 1}/{max_iterations}")
        
        # Visualize current bbox
        vis_path = visualize_bbox_on_image(
            image_path,
            current_bbox,
            output_path=None  # Will create temp file
        )
        
        # Build refinement prompt
        refinement_prompt = f"""**TASK:** You will see a P&ID diagram with a RED bounding box drawn around the legend area.

**CURRENT BOUNDING BOX:**
- x: {current_bbox['x']:.3f} (normalized, 0-1)
- y: {current_bbox['y']:.3f} (normalized, 0-1)
- width: {current_bbox['width']:.3f} (normalized, 0-1)
- height: {current_bbox['height']:.3f} (normalized, 0-1)

**QUESTION:** Look at the red bounding box. Is it optimal?

**OPTIMAL BBOX CRITERIA:**
1. **NOT TOO BIG:** The bbox should be as small as possible, excluding unnecessary white space or non-legend content
2. **NOT TOO SMALL:** The bbox must include ALL legend content (all symbols, all text, all line definitions)
3. **TIGHT FIT:** The bbox should tightly fit around the legend content

**YOUR RESPONSE:**
Provide a JSON object with:
- `is_optimal`: boolean - true if bbox is optimal, false if it can be improved
- `can_be_smaller`: boolean - true if bbox can be made smaller without losing content
- `refined_bbox`: object with `x`, `y`, `width`, `height` (normalized 0-1) - the refined bbox if it can be improved, or the current bbox if optimal
- `reason`: string - explanation of your decision

**EXAMPLE (if bbox is too big):**
```json
{{
  "is_optimal": false,
  "can_be_smaller": true,
  "refined_bbox": {{
    "x": 0.02,
    "y": 0.01,
    "width": 0.15,
    "height": 0.25
  }},
  "reason": "The bbox includes too much white space on the right and bottom. The legend content is tighter, so I reduced width by 10% and height by 5%."
}}
```

**EXAMPLE (if bbox is optimal):**
```json
{{
  "is_optimal": true,
  "can_be_smaller": false,
  "refined_bbox": {{
    "x": 0.02,
    "y": 0.01,
    "width": 0.18,
    "height": 0.28
  }},
  "reason": "The bbox tightly fits all legend content. No further reduction possible without losing symbols or text."
}}
```

**CRITICAL:** Only reduce the bbox if you are CERTAIN that no legend content will be excluded. When in doubt, keep the current bbox."""
        
        try:
            # Call LLM for refinement feedback
            response = llm_client.call_llm(
                model_info,
                system_prompt,
                refinement_prompt,
                vis_path,  # Send visualization image
                expected_json_keys=["is_optimal", "can_be_smaller", "refined_bbox", "reason"]
            )
            
            if response and isinstance(response, dict):
                is_optimal = response.get("is_optimal", False)
                can_be_smaller = response.get("can_be_smaller", False)
                refined_bbox = response.get("refined_bbox")
                reason = response.get("reason", "No reason provided")
                
                logger.info(f"Refinement iteration {iteration + 1}: is_optimal={is_optimal}, can_be_smaller={can_be_smaller}")
                logger.info(f"Reason: {reason}")
                
                if is_optimal or not can_be_smaller:
                    logger.info(f"BBox is optimal after {iteration + 1} iterations. Stopping refinement.")
                    return current_bbox
                
                if refined_bbox:
                    # Validate refined bbox
                    if all(k in refined_bbox for k in ['x', 'y', 'width', 'height']):
                        # Check if reduction is significant (at least min_reduction)
                        old_area = current_bbox['width'] * current_bbox['height']
                        new_area = refined_bbox['width'] * refined_bbox['height']
                        reduction = (old_area - new_area) / old_area
                        
                        if reduction >= min_reduction:
                            logger.info(f"BBox reduced by {reduction:.1%} (area: {old_area:.3f} -> {new_area:.3f})")
                            current_bbox = refined_bbox
                        else:
                            logger.info(f"BBox reduction too small ({reduction:.1%} < {min_reduction:.1%}). Stopping refinement.")
                            return current_bbox
                    else:
                        logger.warning("Refined bbox missing required keys. Using current bbox.")
                        return current_bbox
                else:
                    logger.warning("No refined_bbox in response. Using current bbox.")
                    return current_bbox
            else:
                logger.warning(f"LLM response invalid: {response}")
                return current_bbox
                
        except Exception as e:
            logger.error(f"Error in refinement iteration {iteration + 1}: {e}", exc_info=True)
            return current_bbox
    
    logger.info(f"Reached max iterations ({max_iterations}). Returning current bbox.")
    return current_bbox

