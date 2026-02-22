"""Configuration loader"""

from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

from .config_models import PROJECT_ROOT
from .config_classes import AppConfig

load_dotenv()

_global_config: AppConfig = None


def load_config_from_yaml(config_path: Path) -> dict:
    """Load config from YAML file"""
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs) -> dict:
    """Merge multiple config dictionaries"""
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def initialize_config(config_paths: list = None) -> AppConfig:
    """Initialize global configuration"""
    global _global_config
    
    if config_paths is None:
        config_paths = [
            PROJECT_ROOT / "configs" / "rag_config.yaml",
            PROJECT_ROOT / "configs" / "safety_config.yaml",
            PROJECT_ROOT / "configs" / "model_config.yaml",
        ]
    
    configs = [load_config_from_yaml(p) for p in config_paths]
    merged = merge_configs(*configs)
    
    _global_config = AppConfig(**merged)
    return _global_config


def get_global_config() -> AppConfig:
    """Get global config, initialize if needed"""
    global _global_config
    
    if _global_config is None:
        _global_config = initialize_config()
    
    return _global_config
