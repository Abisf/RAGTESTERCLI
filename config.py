"""Configuration management for RAGTESTERCLI."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for RAGTESTERCLI."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        possible_paths = [
            "ragtester.yaml",
            "ragtester.yml",
            os.path.expanduser("~/.ragtester.yaml"),
            os.path.expanduser("~/.ragtester.yml"),
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path or not Path(self.config_path).exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            return {}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        llm_config = self.config_data.get('llm', {})
        
        # Default configuration
        defaults = {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': 0.0,
            'max_tokens': 1000
        }
        
        # Override with config file values
        for key, default_value in defaults.items():
            if key not in llm_config:
                llm_config[key] = default_value
        
        return llm_config
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config_data.get('output', {
            'format': 'json',
            'include_reasoning': False,
            'decimal_places': 3
        })


# Global config instance
config = Config() 