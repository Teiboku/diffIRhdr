# utils/config.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    @property
    def config(self) -> Dict:
        return self._config