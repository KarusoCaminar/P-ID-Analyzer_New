"""
Visualizer - Generates visualizations for analysis results.

Provides:
- Uncertainty heatmaps
- Debug maps
- Score curves
- KPI visualizations
- Confidence maps
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generates visualizations for analysis results.
    """
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize visualizer.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
    
    def draw_uncertainty_heatmap(
        self,
        image_path: str,
        uncertain_zones: List[Dict[str, Any]],
        output_path: str
    ) -> bool:
        """
        Draw uncertainty heatmap overlay with smooth gradients.
        
        Args:
            image_path: Path to original image
            uncertain_zones: List of uncertain zones with normalized coordinates
            output_path: Path to save heatmap
            
        Returns:
            True if successful
        """
        try:
            import numpy as np
            from scipy import ndimage
            
            # Load image and copy to avoid context manager issues
            with Image.open(image_path) as img_temp:
                img = img_temp.convert('RGB').copy()
            
            # Create base overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Create uncertainty map
            uncertainty_map = np.zeros((self.image_height, self.image_width), dtype=np.float32)
            
            for zone in uncertain_zones:
                x = int(zone.get('x', 0) * self.image_width)
                y = int(zone.get('y', 0) * self.image_height)
                w = int(zone.get('width', 0) * self.image_width)
                h = int(zone.get('height', 0) * self.image_height)
                uncertainty = zone.get('uncertainty', 0.5)
                
                # Ensure within bounds
                x = max(0, min(self.image_width - 1, x))
                y = max(0, min(self.image_height - 1, y))
                w = max(1, min(self.image_width - x, w))
                h = max(1, min(self.image_height - y, h))
                
                # Fill uncertainty map
                uncertainty_map[y:y+h, x:x+w] = np.maximum(uncertainty_map[y:y+h, x:x+w], uncertainty)
            
            # Apply Gaussian blur for smooth gradients
            if uncertainty_map.max() > 0:
                uncertainty_map = ndimage.gaussian_filter(uncertainty_map, sigma=5)
            
            # Convert to heatmap colors using vectorized operations (PERFORMANCE OPTIMIZATION)
            heatmap_img = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
            
            # Vectorized color mapping (much faster than nested loops)
            uncertainty_flat = uncertainty_map.flatten()
            mask = uncertainty_flat > 0
            
            if mask.any():
                # High uncertainty (red)
                high_mask = (uncertainty_flat > 0.7) & mask
                heatmap_img[:, :, 0].flat[high_mask] = 255
                heatmap_img[:, :, 1].flat[high_mask] = (255 * (1 - uncertainty_flat[high_mask])).astype(np.uint8)
                heatmap_img[:, :, 3].flat[high_mask] = (180 * uncertainty_flat[high_mask]).astype(np.uint8)
                
                # Medium uncertainty (yellow)
                medium_mask = (uncertainty_flat > 0.4) & (uncertainty_flat <= 0.7) & mask
                heatmap_img[:, :, 0].flat[medium_mask] = 255
                heatmap_img[:, :, 1].flat[medium_mask] = 255
                heatmap_img[:, :, 2].flat[medium_mask] = (255 * (1 - uncertainty_flat[medium_mask] * 2)).astype(np.uint8)
                heatmap_img[:, :, 3].flat[medium_mask] = (180 * uncertainty_flat[medium_mask]).astype(np.uint8)
                
                # Low uncertainty (green)
                low_mask = (uncertainty_flat <= 0.4) & mask
                heatmap_img[:, :, 0].flat[low_mask] = (255 * uncertainty_flat[low_mask]).astype(np.uint8)
                heatmap_img[:, :, 1].flat[low_mask] = 255
                heatmap_img[:, :, 3].flat[low_mask] = (180 * uncertainty_flat[low_mask]).astype(np.uint8)
            
            # Create overlay from heatmap
            heatmap_overlay = Image.fromarray(heatmap_img, 'RGBA')
            
            # Draw zone outlines
            for zone in uncertain_zones:
                x = int(zone.get('x', 0) * self.image_width)
                y = int(zone.get('y', 0) * self.image_height)
                w = int(zone.get('width', 0) * self.image_width)
                h = int(zone.get('height', 0) * self.image_height)
                
                # Ensure within bounds
                x = max(0, min(self.image_width - 1, x))
                y = max(0, min(self.image_height - 1, y))
                w = max(1, min(self.image_width - x, w))
                h = max(1, min(self.image_height - y, h))
                
                draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255, 255), width=2)
                
                # Add uncertainty label
                uncertainty = zone.get('uncertainty', 0.5)
                try:
                    text = f"{uncertainty:.2f}"
                    draw.text((x + 5, y + 5), text, fill=(255, 255, 255, 255))
                except (OSError, IOError, AttributeError) as e:
                    logger.debug(f"Error drawing uncertainty text: {e}")
                    pass
            
            # Blend overlay with original image (optimize: convert only once)
            result_img = img.convert('RGBA')  # Only convert once
            result = Image.alpha_composite(result_img, heatmap_overlay)
            result = result.convert('RGB')  # Final conversion
            result.save(output_path, quality=95, dpi=(150, 150))
            
            logger.info(f"Uncertainty heatmap saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error drawing uncertainty heatmap: {e}", exc_info=True)
            # Fallback to simple version
            try:
                with Image.open(image_path) as img_temp:
                    img = img_temp.convert('RGBA').copy()
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                for zone in uncertain_zones:
                    x = int(zone.get('x', 0) * self.image_width)
                    y = int(zone.get('y', 0) * self.image_height)
                    w = int(zone.get('width', 0) * self.image_width)
                    h = int(zone.get('height', 0) * self.image_height)
                    uncertainty = zone.get('uncertainty', 0.5)
                    
                    alpha = int(180 * uncertainty)
                    color = (255, 0, 0, alpha)
                    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 0, 0, 255), width=2)
                
                result = Image.alpha_composite(img, overlay)
                result = result.convert('RGB')
                result.save(output_path)
                return True
            except (OSError, IOError, ValueError, AttributeError) as e:
                logger.warning(f"Error saving debug map: {e}")
                return False
    
    def draw_debug_map(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        output_path: str
    ) -> bool:
        """
        Draw clean debug map with elements and connections.
        
        Args:
            image_path: Path to original image
            elements: List of element dictionaries
            connections: List of connection dictionaries
            output_path: Path to save debug map
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img_temp:
                img = img_temp.convert('RGB').copy()
            
            # Create overlay for cleaner visualization
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            base_draw = ImageDraw.Draw(img)
            
            # Draw connections first (behind elements) with smooth lines
            for conn in connections:
                from_el = next((el for el in elements if el.get('id') == conn.get('from_id')), None)
                to_el = next((el for el in elements if el.get('id') == conn.get('to_id')), None)
                
                if from_el and to_el and from_el.get('bbox') and to_el.get('bbox'):
                    from_bbox = from_el['bbox']
                    to_bbox = to_el['bbox']
                    
                    from_x = int((from_bbox['x'] + from_bbox['width'] / 2) * self.image_width)
                    from_y = int((from_bbox['y'] + from_bbox['height'] / 2) * self.image_height)
                    to_x = int((to_bbox['x'] + to_bbox['width'] / 2) * self.image_width)
                    to_y = int((to_bbox['y'] + to_bbox['height'] / 2) * self.image_height)
                    
                    confidence = conn.get('confidence', 0.5)
                    
                    # Color based on confidence (green = high, yellow = medium, red = low)
                    if confidence > 0.7:
                        color = (0, 200, 0, 200)  # Green
                        width = 3
                    elif confidence > 0.4:
                        color = (255, 200, 0, 200)  # Yellow
                        width = 2
                    else:
                        color = (255, 0, 0, 150)  # Red
                        width = 2
                    
                    draw.line([(from_x, from_y), (to_x, to_y)], fill=color, width=width)
            
            # Draw elements with clean bounding boxes
            for el in elements:
                if el.get('bbox'):
                    bbox = el['bbox']
                    x = int(bbox['x'] * self.image_width)
                    y = int(bbox['y'] * self.image_height)
                    w = int(bbox['width'] * self.image_width)
                    h = int(bbox['height'] * self.image_height)
                    
                    # Ensure within bounds
                    x = max(0, min(self.image_width - 1, x))
                    y = max(0, min(self.image_height - 1, y))
                    w = max(1, min(self.image_width - x, w))
                    h = max(1, min(self.image_height - y, h))
                    
                    confidence = el.get('confidence', 0.5)
                    
                    # Color based on confidence
                    if confidence > 0.7:
                        outline_color = (0, 255, 0)  # Green
                        fill_color = (0, 255, 0, 30)
                    elif confidence > 0.4:
                        outline_color = (255, 200, 0)  # Yellow
                        fill_color = (255, 200, 0, 30)
                    else:
                        outline_color = (255, 0, 0)  # Red
                        fill_color = (255, 0, 0, 30)
                    
                    # Draw bounding box with fill
                    draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=outline_color, width=2)
                    
                    # Draw label with background
                    label = el.get('label') or el.get('type') or 'Unknown'
                    if label and isinstance(label, str):
                        label = label[:20]  # Limit label length
                        try:
                            # Calculate text size
                            bbox_text = draw.textbbox((0, 0), label)
                            text_width = bbox_text[2] - bbox_text[0]
                            text_height = bbox_text[3] - bbox_text[1]
                            
                            # Draw background
                            bg_padding = 2
                            base_draw.rectangle([
                                x - bg_padding,
                                y - text_height - bg_padding - 2,
                                x + text_width + bg_padding,
                                y - 2
                            ], fill=(0, 0, 0, 200))
                            
                            # Draw text
                            base_draw.text((x, y - text_height - 2), label, fill=(255, 255, 255))
                        except (OSError, IOError, AttributeError) as e:
                            logger.debug(f"Error drawing full label, trying truncated: {e}")
                            try:
                                base_draw.text((x, y - 15), label[:15], fill=(0, 255, 0))
                            except (OSError, IOError, AttributeError) as e2:
                                logger.debug(f"Error drawing truncated label: {e2}")
                                pass
            
            # Composite overlay (optimize: convert only once)
            result_img_rgba = img.convert('RGBA')  # Only convert once
            result = Image.alpha_composite(result_img_rgba, overlay)
            result = result.convert('RGB')  # Final conversion
            result.save(output_path, quality=95, dpi=(150, 150))
            
            logger.info(f"Debug map saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error drawing debug map: {e}", exc_info=True)
            return False
    
    def plot_score_curve(
        self,
        score_history: List[float],
        output_path: str
    ) -> bool:
        """
        Plot score improvement curve.
        
        Args:
            score_history: List of quality scores over iterations
            output_path: Path to save plot
            
        Returns:
            True if successful
        """
        fig = None
        try:
            # CRITICAL FIX: Handle empty or single-value score_history
            if not score_history or len(score_history) == 0:
                logger.warning("Empty score_history provided. Cannot plot score curve.")
                return False
            
            # If only one score, create a proper plot with iteration index
            if len(score_history) == 1:
                logger.warning(f"Only one score in history: {score_history[0]}. Plotting single point.")
                # Create iteration indices [0] for single point
                iterations = [0]
                scores = score_history
            else:
                # Normal case: multiple scores
                iterations = list(range(len(score_history)))
                scores = score_history
            
            fig = plt.figure(figsize=(10, 6))
            plt.plot(iterations, scores, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Quality Score', fontsize=12)
            plt.title('Quality Score Improvement Over Iterations', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 100])
            
            # Add annotations for each point
            for i, score in zip(iterations, scores):
                plt.annotate(f'{score:.1f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Score curve saved to: {output_path} ({len(score_history)} iterations)")
            return True
        except Exception as e:
            logger.error(f"Error plotting score curve: {e}", exc_info=True)
            return False
        finally:
            if fig:
                plt.close(fig)
    
    def plot_kpi_dashboard(
        self,
        kpis: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Plot KPI dashboard.
        
        Args:
            kpis: Dictionary of KPIs
            output_path: Path to save dashboard
            
        Returns:
            True if successful
        """
        fig = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('KPI Dashboard', fontsize=16, fontweight='bold')
            
            # Extract key KPIs
            element_metrics = {
                'Total Elements': kpis.get('total_elements', 0),
                'Connected Elements': kpis.get('connected_elements', 0),
                'Isolated Elements': kpis.get('isolated_elements', 0)
            }
            
            connection_metrics = {
                'Total Connections': kpis.get('total_connections', 0),
                'Unique Types': kpis.get('unique_element_types', 0)
            }
            
            # Plot 1: Element metrics bar chart
            axes[0, 0].bar(element_metrics.keys(), element_metrics.values(), color=['#4CAF50', '#2196F3', '#FF9800'])
            axes[0, 0].set_title('Element Metrics', fontweight='bold')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Connection metrics bar chart
            axes[0, 1].bar(connection_metrics.keys(), connection_metrics.values(), color=['#2196F3', '#9C27B0'])
            axes[0, 1].set_title('Connection Metrics', fontweight='bold')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Precision/Recall if available
            if 'element_precision' in kpis and 'element_recall' in kpis:
                metrics = ['Precision', 'Recall']
                values = [kpis.get('element_precision', 0), kpis.get('element_recall', 0)]
                axes[1, 0].bar(metrics, values, color=['#4CAF50', '#2196F3'])
                axes[1, 0].set_title('Quality Metrics', fontweight='bold')
                axes[1, 0].set_ylabel('Score (0-1)')
                axes[1, 0].set_ylim([0, 1])
            
            # Plot 4: Quality score
            quality_score = kpis.get('quality_score', 0)
            axes[1, 1].bar(['Quality Score'], [quality_score], color='#4CAF50')
            axes[1, 1].set_title('Overall Quality Score', fontweight='bold')
            axes[1, 1].set_ylabel('Score (0-100)')
            axes[1, 1].set_ylim([0, 100])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"KPI dashboard saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error plotting KPI dashboard: {e}", exc_info=True)
            return False
        finally:
            if fig:
                plt.close(fig)
    
    def draw_confidence_map(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        output_path: str
    ) -> bool:
        """
        Draw confidence map showing detection confidence.
        
        Args:
            image_path: Path to original image
            elements: List of element dictionaries with confidence scores
            output_path: Path to save confidence map
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img_temp:
                img = img_temp.convert('RGBA').copy()
            
            # Create confidence overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for el in elements:
                if el.get('bbox'):
                    bbox = el['bbox']
                    x = int(bbox['x'] * self.image_width)
                    y = int(bbox['y'] * self.image_height)
                    w = int(bbox['width'] * self.image_width)
                    h = int(bbox['height'] * self.image_height)
                    
                    confidence = el.get('confidence', 0.5)
                    
                    # Green for high confidence, yellow for medium, red for low
                    if confidence > 0.7:
                        color = (0, 255, 0, int(128 * confidence))  # Green
                    elif confidence > 0.4:
                        color = (255, 255, 0, int(128 * confidence))  # Yellow
                    else:
                        color = (255, 0, 0, int(128 * confidence))  # Red
                    
                    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 255, 255, 255), width=2)
                    
                    # Draw confidence text
                    try:
                        text = f"{confidence:.2f}"
                        draw.text((x + 5, y + 5), text, fill=(255, 255, 255, 255))
                    except (OSError, IOError, AttributeError) as e:
                        logger.debug(f"Error drawing confidence text: {e}")
                        pass
            
            # Blend overlay with original image (optimize: convert only once)
            result = Image.alpha_composite(img, overlay)  # img is already RGBA
            result = result.convert('RGB')  # Final conversion
            result.save(output_path, quality=95, dpi=(150, 150))  # Consistent quality settings
            
            logger.info(f"Confidence map saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error drawing confidence map: {e}", exc_info=True)
            return False

