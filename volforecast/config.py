"""YAML configuration loader (carried over verbatim from the legacy project).

Converts a YAML file into a nested SimpleNamespace for dot-notation access
(e.g. ``cfg.model.lookback``).
"""
import os
import yaml
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict into a SimpleNamespace for dot access."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)


def get_config(config_path: str) -> SimpleNamespace:
    """Load a YAML config from ``config_path`` into a nested SimpleNamespace."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return _dict_to_namespace(cfg_dict)
