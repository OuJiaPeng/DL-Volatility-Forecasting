import os
import yaml
from types import SimpleNamespace

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    Recursively convert a dict into a SimpleNamespace,
    so you can write cfg.model.seq_len instead of cfg['model']['seq_len'].
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)

def get_config(config_path: str = None) -> SimpleNamespace:
    """
    Load YAML config from `config_path`. If None, defaults to ../default.yaml.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'default.yaml')
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    return _dict_to_namespace(cfg_dict)
