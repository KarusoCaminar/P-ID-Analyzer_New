"""
Complexity Analyzer - Hybrid CV/LLM approach for P&ID complexity detection.

Phase 0: Determines the optimal strategy based on complexity analysis.
- CV-Schnelltest (Canny edge detection) as "doorman"
- LLM-Fein-Check (optional) for complex cases
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """
    Hybrid complexity analyzer for P&ID images.
    
    Uses CV for fast filtering (90% of cases) and LLM for intelligent
    semantic analysis (10% complex cases).
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize complexity analyzer.
        
        Args:
            llm_client: Optional LLM client for fine-grained analysis
        """
        self.llm_client = llm_client
    
    def analyze_complexity(
        self,
        image_path: str,
        use_llm_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze P&ID complexity using LLM-First approach (Intelligence before Speed).
        
        INVERTED LOGIC: LLM is primary, CV is fallback only.
        
        Args:
            image_path: Path to P&ID image
            use_llm_fallback: Whether to use CV as fallback if LLM fails
            
        Returns:
            Dictionary with complexity analysis results:
            {
                'complexity': 'simple' | 'moderate' | 'complex' | 'very_complex',
                'strategy': 'simple_pid_strategy' | 'optimal_swarm_monolith' | 'heavy',
                'pixel_density': float,
                'edge_density': float,
                'cv_confidence': float,
                'llm_used': bool,
                'reasoning': str
            }
        """
        logger.info("=== Phase 0: Complexity Analysis (LLM-First) ===")
        
        # Step 1: LLM-Fein-Check (Der Experte) - PRIMARY METHOD
        if use_llm_fallback and self.llm_client:
            try:
                logger.info("Requesting LLM fine-check (primary method)...")
                llm_result = self._llm_fine_check(image_path)
                
                if llm_result and llm_result.get('complexity'):
                    complexity = llm_result['complexity']
                    
                    # Determine strategy based on complexity
                    if complexity == 'simple':
                        strategy = 'simple_pid_strategy'
                    elif complexity == 'very_complex':
                        strategy = 'optimal_swarm_monolith'  # Use heavy strategy for very complex
                    else:  # moderate or complex
                        strategy = 'optimal_swarm_monolith'
                    
                    # Get CV metrics for reporting (but don't use them for decision)
                    cv_result = self._cv_quick_test(image_path)
                    
                    logger.info(f"LLM-First Result: complexity={complexity}, strategy={strategy}")
                    
                    return {
                        'complexity': complexity,
                        'strategy': strategy,
                        'pixel_density': cv_result.get('pixel_density', 0.0),
                        'edge_density': cv_result.get('edge_density', 0.0),
                        'cv_confidence': cv_result.get('confidence', 0.0),
                        'llm_used': True,
                        'llm_confidence': llm_result.get('confidence', 0.85),
                        'reasoning': llm_result.get('reasoning', f'LLM analysis: {complexity} complexity based on semantic density and component structure.')
                    }
            except Exception as e:
                logger.warning(f"LLM fine-check failed: {e}. Using default 'complex' strategy (no CV fallback).")
                # Step 2: Default to 'complex' + 'optimal_swarm_monolith' if LLM fails
                # CV-Fallback-Logik entfernt - CV ist unzuverlässig und darf keine Strategie-Entscheidungen treffen
                cv_result = self._cv_quick_test(image_path)  # Only for reporting metrics
                
                logger.info(f"LLM failed - Default Result: complexity=complex, strategy=optimal_swarm_monolith")
                
                return {
                    'complexity': 'complex',
                    'strategy': 'optimal_swarm_monolith',
                    'pixel_density': cv_result.get('pixel_density', 0.0),
                    'edge_density': cv_result.get('edge_density', 0.0),
                    'cv_confidence': cv_result.get('confidence', 0.0),
                    'llm_used': False,
                    'reasoning': f'LLM check failed: {e}. Defaulting to complex strategy (no CV fallback).'
                }
        
        # Step 2: LLM not available - Default to 'complex' + 'optimal_swarm_monolith'
        logger.info("LLM not available - Using default 'complex' strategy (no CV fallback)...")
        cv_result = self._cv_quick_test(image_path)  # Only for reporting metrics
        
        logger.info(f"Default Result: complexity=complex, strategy=optimal_swarm_monolith")
        
        return {
            'complexity': 'complex',
            'strategy': 'optimal_swarm_monolith',
            'pixel_density': cv_result.get('pixel_density', 0.0),
            'edge_density': cv_result.get('edge_density', 0.0),
            'cv_confidence': cv_result.get('confidence', 0.0),
            'llm_used': False,
            'reasoning': 'LLM not available. Defaulting to complex strategy (no CV fallback).'
        }
    
    def analyze_complexity_cv_advanced(self, image_path: str) -> Dict[str, Any]:
        """
        Erweiterte CV-basierte Komplexitätsanalyse mit mehreren Metriken.
        
        PRIORITÄT 1: Multi-Metrik CV-Ansatz (Verbesserung 1)
        Ersetzt langsame LLM-Analyse durch schnelle, präzise CV-Metriken.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with complexity analysis results:
            {
                'complexity': 'simple' | 'moderate' | 'complex' | 'very_complex',
                'score': float,  # 0.0-1.0
                'metrics': {
                    'edge_density': float,
                    'object_density': float,
                    'color_variance': float,
                    'structural_complexity': float
                }
            }
        """
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return {
                    'complexity': 'moderate',
                    'score': 0.5,
                    'metrics': {}
                }
            
            img_height, img_width = img.shape
            total_pixels = img_height * img_width
            
            # METRIK 1: Edge Density (Canny) - 30% Gewichtung
            edges = cv2.Canny(img, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels
            
            # METRIK 2: Objektdichte (Contour-basiert) - 30% Gewichtung
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter small contours (noise)
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            object_density = len(significant_contours) / (total_pixels / 1_000_000)  # Objekte pro Mio. Pixel
            
            # METRIK 3: Farbkontrast (HSV-Varianz) - 20% Gewichtung
            img_color = cv2.imread(image_path)
            if img_color is not None:
                hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                color_variance = np.var(hsv[:, :, 0])  # Hue-Varianz
            else:
                color_variance = 0.0
            
            # METRIK 4: Strukturelle Komplexität (Junction-Punkte) - 20% Gewichtung
            # Skeletonization für Junction-Detection
            skeleton = self._skeletonize(edges)
            junctions = self._detect_junctions(skeleton)
            structural_complexity = len(junctions) / (total_pixels / 1_000_000)  # Junctions pro Mio. Pixel
            
            # GEWICHTETE BEWERTUNG
            complexity_score = (
                edge_density * 0.3 +
                min(object_density / 100, 1.0) * 0.3 +
                min(color_variance / 10000, 1.0) * 0.2 +
                min(structural_complexity / 50, 1.0) * 0.2
            )
            
            # KATEGORISIERUNG
            if complexity_score < 0.2:
                complexity = 'simple'
            elif complexity_score < 0.5:
                complexity = 'moderate'
            elif complexity_score < 0.8:
                complexity = 'complex'
            else:
                complexity = 'very_complex'
            
            logger.info(f"CV Advanced Complexity: {complexity} (score={complexity_score:.3f}, "
                       f"edge={edge_density:.3f}, objects={object_density:.1f}, "
                       f"color={color_variance:.1f}, junctions={structural_complexity:.1f})")
            
            return {
                'complexity': complexity,
                'score': float(complexity_score),
                'metrics': {
                    'edge_density': float(edge_density),
                    'object_density': float(object_density),
                    'color_variance': float(color_variance),
                    'structural_complexity': float(structural_complexity)
                }
            }
        except Exception as e:
            logger.error(f"Error in advanced CV complexity analysis: {e}", exc_info=True)
            return {
                'complexity': 'moderate',
                'score': 0.5,
                'metrics': {}
            }
    
    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Skeletonize binary image using morphological operations."""
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary_work = binary.copy()
        
        while True:
            opened = cv2.morphologyEx(binary_work, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(binary_work, opened)
            eroded = cv2.erode(binary_work, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_work = eroded.copy()
            
            if cv2.countNonZero(binary_work) == 0:
                break
        
        return skeleton
    
    def _detect_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Detect junction points (points with degree > 2) in skeleton."""
        junctions = []
        y_coords, x_coords = np.where(skeleton > 0)
        
        for y, x in zip(y_coords, x_coords):
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] > 0:
                            neighbors += 1
            
            if neighbors > 2:  # Junction point
                junctions.append((x, y))
        
        return junctions
    
    def _cv_quick_test(self, image_path: str) -> Dict[str, Any]:
        """
        Fast CV-based complexity test using Canny edge detection.
        DEPRECATED: Use analyze_complexity_cv_advanced() instead.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with CV analysis results
        """
        try:
            # Load image as grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return {
                    'pixel_density': 0.0,
                    'edge_density': 0.0,
                    'confidence': 0.0
                }
            
            height, width = img.shape
            total_pixels = height * width
            
            # Calculate pixel density (non-white pixels / total pixels)
            # Invert for white background P&IDs: black pixels are content
            threshold = 250  # Consider pixels darker than this as content
            content_pixels = np.sum(img < threshold)
            pixel_density = content_pixels / total_pixels
            
            # Calculate edge density using Canny
            edges = cv2.Canny(img, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels
            
            # Confidence based on both metrics
            confidence = min(0.95, (pixel_density * 0.6 + edge_density * 0.4) * 2.0)
            
            return {
                'pixel_density': float(pixel_density),
                'edge_density': float(edge_density),
                'confidence': float(confidence)
            }
        except Exception as e:
            logger.error(f"Error in CV quick test: {e}", exc_info=True)
            return {
                'pixel_density': 0.1,  # Default to moderate
                'edge_density': 0.05,
                'confidence': 0.5
            }
    
    def _llm_fine_check(
        self,
        image_path: str,
        cv_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        LLM-based fine-grained complexity check (PRIMARY METHOD).
        
        Args:
            image_path: Path to image
            cv_result: Optional CV results (for reporting only, not used for decision)
            
        Returns:
            Dictionary with LLM analysis results
        """
        if not self.llm_client:
            raise ValueError("LLM client not available for fine check")
        
        try:
            # Use Flash-Lite for fast, cheap analysis
            # Get model info from config
            from src.services.config_service import ConfigService
            config_service = ConfigService()
            config = config_service.get_config()
            
            # Use Flash model directly (Flash-Lite is not available)
            model_name = 'Google Gemini 2.5 Flash'
            model_info = None
            
            # Try to get model from config
            if hasattr(config, 'models'):
                if model_name in config.models:
                    model_config = config.models[model_name]
                    model_info = model_config.model_dump() if hasattr(model_config, 'model_dump') else model_config
                    # Override max_output_tokens for complexity check (fast, cheap analysis)
                    if 'generation_config' in model_info:
                        model_info['generation_config']['max_output_tokens'] = 100
            
            if not model_info:
                # Ultimate fallback: use Flash model info
                model_info = {
                    'id': 'gemini-2.5-flash',
                    'access_method': 'gemini',
                    'location': 'us-central1',
                    'generation_config': {
                        'temperature': 0.0,
                        'max_output_tokens': 100
                    }
                }
            
            system_prompt = "You are an expert in analyzing P&ID diagrams. Analyze the complexity and respond with a single word."
            
            # Simplified prompt (no CV info needed - LLM is primary)
            user_prompt = """Bewerte die Komplexität dieses P&ID Diagramms.

**AUFGABE:**
Bewerte die Komplexität basierend auf:
1. Visuelle Dichte der Komponenten (Symbole pro Fläche)
2. Anzahl der Verbindungen (Linien)
3. Semantische Komplexität (verschiedene Komponententypen, verschachtelte Strukturen)

**ANTWORTE NUR MIT EINEM WORT:**
- 'simple' = Wenige Komponenten (<15), einfache Struktur, klare Flüsse
- 'moderate' = Mittlere Komplexität (15-50 Komponenten), einige verzweigte Strukturen
- 'complex' = Hohe Komplexität (50-150 Komponenten), viele verzweigte Strukturen, mehrere parallele Flüsse
- 'very_complex' = Sehr hohe Komplexität (>150 Komponenten), verschachtelte Strukturen, viele parallele/verzweigte Flüsse

**WICHTIG:** Antworte NUR mit einem der vier Wörter: 'simple', 'moderate', 'complex', oder 'very_complex'."""
            
            response = self.llm_client.call_llm(
                model_info=model_info,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=image_path,
                use_cache=False,  # Don't cache complexity checks
                timeout=30  # Fast timeout for complexity check
            )
            
            # Parse response
            if response is None:
                raise ValueError("LLM returned None response")
            
            # Handle different response types
            if isinstance(response, dict):
                # If response is already a dict, extract text from common keys
                response_text = (
                    response.get('text', '') or 
                    response.get('response', '') or 
                    str(response)
                ).strip().lower()
            elif hasattr(response, 'text'):
                response_text = response.text.strip().lower()
            else:
                response_text = str(response).strip().lower()
            
            # Extract complexity keyword
            complexity = 'moderate'  # Default
            if 'very_complex' in response_text or 'very complex' in response_text:
                complexity = 'very_complex'
            elif 'complex' in response_text and 'very' not in response_text:
                complexity = 'complex'
            elif 'simple' in response_text:
                complexity = 'simple'
            elif 'moderate' in response_text:
                complexity = 'moderate'
            
            logger.info(f"LLM fine check: complexity={complexity}, response={response_text[:100]}")
            
            return {
                'complexity': complexity,
                'confidence': 0.85,  # High confidence for LLM analysis
                'reasoning': f'LLM analysis: {complexity} complexity based on semantic density and component structure.'
            }
        except Exception as e:
            logger.error(f"Error in LLM fine check: {e}", exc_info=True)
            raise  # Re-raise to trigger CV fallback

