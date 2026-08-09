"""Configuration utilities for CACR-SP VCR."""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path (str): Path to the YAML file.
        
    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    if not os.path.exists(path):
        logger.warning(f"Config file not found: {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config if config else {}

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    Override values take precedence.
    
    Args:
        base (Dict[str, Any]): Base configuration.
        override (Dict[str, Any]): Override configuration.
        
    Returns:
        Dict[str, Any]: Merged configuration.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config_with_overrides(
    base_path: str, 
    override_path: Optional[str] = None, 
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load a base config and apply overrides from a file and CLI.
    
    Args:
        base_path (str): Path to the base YAML config.
        override_path (Optional[str]): Path to the override YAML config.
        cli_overrides (Optional[Dict[str, Any]]): Dictionary of CLI overrides.
        
    Returns:
        Dict[str, Any]: The final merged configuration.
    """
    config = load_config(base_path)
    
    if override_path:
        override_config = load_config(override_path)
        config = merge_configs(config, override_config)
        
    if cli_overrides:
        # Simple flat override handling - deeper logic can be added if needed
        # RESEARCH_DECISION: simplified CLI overrides for now
        config = merge_configs(config, cli_overrides)
        
    validate_config(config)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Basic validation for required config fields.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    required_sections = ['data', 'model', 'training', 'experiment']
    for section in required_sections:
        if section not in config:
            logger.warning(f"Config is missing required section: '{section}'")

def get_nested(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using dot-notation.
    
    Args:
        config (Dict[str, Any]): The dictionary to search.
        key (str): Dot-separated key string (e.g., 'model.vlm_name').
        default (Any): Default value if key is not found.
        
    Returns:
        Any: The value if found, else default.
    """
    keys = key.split('.')
    current = config
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current
