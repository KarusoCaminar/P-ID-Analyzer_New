"""
Test Harness Utilities - Save intermediate results and configuration snapshots.

This module provides functions to save:
- Intermediate results after each phase
- Configuration snapshots (config.yaml, prompts)
- Test metadata
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def save_intermediate_result(phase_name: str, result: Dict[str, Any], output_dir: str) -> None:
    """
    Save intermediate result after each phase for test harness.
    
    Args:
        phase_name: Name of the phase (e.g., 'phase_2a_swarm', 'phase_2b_guardrails')
        result: Result dictionary to save
        output_dir: Output directory for test artifacts
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        result_file = output_path / f"output_{phase_name}.json"
        from src.utils.json_encoder import json_dump_safe
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json_dump_safe(result, f, indent=2)
        
        logger.info(f"Saved intermediate result: {result_file}")
    except Exception as e:
        logger.warning(f"Could not save intermediate result for {phase_name}: {e}")


def save_config_snapshot(config_service: Any, output_dir: str) -> None:
    """
    Save configuration snapshot for test harness.
    
    Args:
        config_service: ConfigService instance
        output_dir: Output directory for test artifacts
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config.yaml snapshot
        config_snapshot_path = output_path / "config_snapshot.yaml"
        config_path = Path("config.yaml")
        if config_path.exists():
            shutil.copy2(config_path, config_snapshot_path)
            logger.info(f"Saved config snapshot: {config_snapshot_path}")
        
        # Save prompts snapshot
        prompts_snapshot_path = output_path / "prompts_snapshot.json"
        config = config_service.get_config()
        prompts = config.prompts
        
        # Convert prompts to dict
        if hasattr(prompts, 'model_dump'):
            prompts_dict = prompts.model_dump()
        elif isinstance(prompts, dict):
            prompts_dict = prompts
        else:
            prompts_dict = {}
        
        with open(prompts_snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved prompts snapshot: {prompts_snapshot_path}")
        
    except Exception as e:
        logger.warning(f"Could not save config snapshot: {e}")


def save_test_metadata(
    output_dir: str,
    test_name: str,
    test_description: str,
    model_strategy: Dict[str, Any],
    logic_parameters: Dict[str, Any]
) -> None:
    """
    Save test metadata for test harness.
    
    Args:
        output_dir: Output directory for test artifacts
        test_name: Name of the test
        test_description: Description of the test
        model_strategy: Model strategy dictionary
        logic_parameters: Logic parameters dictionary
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_path / "test_metadata.md"
        
        # Get strategy info
        strategy_name = 'N/A'
        swarm_model = 'N/A'
        monolith_model = 'N/A'
        if isinstance(model_strategy, dict):
            strategy_name = model_strategy.get('name', 'N/A')
            swarm_model = model_strategy.get('swarm_model', 'N/A')
            monolith_model = model_strategy.get('monolith_model', 'N/A')
        
        metadata_content = f"""# Test Metadata

**Test Name:** {test_name}
**Date:** {datetime.now().isoformat()}
**Description:** {test_description}

## Configuration

**Strategy:** {strategy_name}
**Swarm Model:** {swarm_model}
**Monolith Model:** {monolith_model}

## Feature Flags

- use_swarm_analysis: {logic_parameters.get('use_swarm_analysis', False)}
- use_monolith_analysis: {logic_parameters.get('use_monolith_analysis', False)}
- use_self_correction_loop: {logic_parameters.get('use_self_correction_loop', False)}
- use_predictive_completion: {logic_parameters.get('use_predictive_completion', False)}
- use_polyline_refinement: {logic_parameters.get('use_polyline_refinement', False)}
"""
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        
        logger.info(f"Saved test metadata: {metadata_file}")
    except Exception as e:
        logger.warning(f"Could not save test metadata: {e}")

