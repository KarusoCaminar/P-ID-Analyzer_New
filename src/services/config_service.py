"""
Configuration service using Pydantic for type-safe config management.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import logging
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration."""
    id: str
    access_method: str
    location: Optional[str] = None
    description: Optional[str] = None
    generation_config: Dict[str, Any] = Field(default_factory=dict)


class LogicParameters(BaseModel):
    """Logic parameters for pipeline control."""
    max_self_correction_iterations: int = 3
    target_quality_score: float = 98.0
    llm_executor_workers: int = 12
    llm_default_timeout: int = 240
    llm_max_retries: int = 3
    llm_disk_cache_size_gb: int = 2
    adaptive_target_tile_count: int = 50
    max_total_tiles: int = 80
    analysis_batch_size: int = 5
    iou_match_threshold: float = 0.1
    graph_completion_distance_threshold: float = 0.05
    cgm_main_components: list[str] = Field(default_factory=lambda: [
        "Boiler", "Pump", "Heat Exchanger", "Buffer Storage",
        "Thermal Consumer", "Line_Split", "Line_Merge"
    ])


class PathsConfig(BaseModel):
    """Paths configuration."""
    element_type_list: str = "element_type_list.json"
    learning_db: str = "learning_db.json"
    temp_symbol_dir: str = "temp_symbols_for_embeddings"
    llm_cache_dir: str = ".pni_analyzer_cache"
    learned_symbols_images_dir: str = "learned_symbols_images"

class PromptsConfig(BaseModel):
    """Prompts configuration."""
    general_system_prompt: str = ""
    metadata_extraction_user_prompt: str = ""
    legend_extraction_user_prompt: str = ""
    symbol_detection_user_prompt: str = ""
    raster_analysis_user_prompt_template: str = ""
    monolithic_analysis_prompt_template: str = ""
    polyline_extraction_user_prompt: str = ""

class AppConfig(BaseModel):
    """Main application configuration."""
    # Pydantic automatically converts dicts to these model types
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    strategies: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    logic_parameters: LogicParameters = Field(default_factory=LogicParameters)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)


class ConfigService:
    """Service for loading and managing configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self._config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self._config = AppConfig()
                return
            
            with self.config_path.open("r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Pydantic handles all validation and type conversion automatically
            # Convert models dict to ModelConfig objects (Pydantic will handle the rest)
            if "models" in raw_config:
                models_dict = {}
                for name, model_data in raw_config["models"].items():
                    if isinstance(model_data, dict):
                        models_dict[name] = ModelConfig(**model_data)
                    else:
                        models_dict[name] = model_data
                raw_config["models"] = models_dict
            
            # Pydantic automatically converts dicts to PathsConfig, LogicParameters, PromptsConfig
            self._config = AppConfig(**raw_config)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except ValidationError as e:
            logger.error(f"Config validation error: {e}")
            self._config = AppConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            self._config = AppConfig()
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw config dict for backward compatibility."""
        try:
            config = self.get_config()
            if hasattr(config, 'model_dump'):
                return config.model_dump()
            elif isinstance(config, dict):
                return config
            else:
                # Fallback: try to reload config
                self._load_config()
                config = self.get_config()
                if hasattr(config, 'model_dump'):
                    return config.model_dump()
                elif isinstance(config, dict):
                    return config
                else:
                    logger.warning("Could not convert config to dict, returning empty dict")
                    return {}
        except Exception as e:
            logger.error(f"Error getting raw config: {e}", exc_info=True)
            # Return empty dict as fallback
            return {}
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration (does not persist to file)."""
        current = self.get_config().model_dump()
        current.update(updates)
        self._config = AppConfig(**current)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def get_path(self, path_key: str) -> Optional[Path]:
        """Get path by key from config."""
        if self._config is None:
            return None
        # No isinstance check needed - Pydantic ensures paths is always PathsConfig
        path_str = getattr(self._config.paths, path_key, None)
        return Path(path_str) if path_str else None
    
    def get_logic_parameters(self) -> Dict[str, Any]:
        """Get logic parameters."""
        if self._config is None:
            return {}
        lp = self._config.logic_parameters
        return lp.model_dump() if hasattr(lp, 'model_dump') else lp
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        if self._config is None:
            return None
        return self._config.models.get(model_name)
    
    def get_prompt(self, prompt_key: str) -> Optional[str]:
        """Get prompt by key."""
        if self._config is None:
            return None
        # No isinstance check needed - Pydantic ensures prompts is always PromptsConfig
        return getattr(self._config.prompts, prompt_key, None)

