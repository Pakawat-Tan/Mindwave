"""
ConfigLoader.py
Centralized configuration loading system for all rules
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json


class ConfigLoader:
    """Centralized loader for JSON rule configurations."""
    
    def __init__(self, config_base_dir: Optional[str] = None):
        """Initialize config loader.
        
        Parameters
        ----------
        config_base_dir : str, optional
            Base directory for configuration files
        """
        if config_base_dir:
            self.config_base_dir = Path(config_base_dir)
        else:
            # Default to Brain/Rules directory
            self.config_base_dir = Path(__file__).parent
        
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_name: str, subdirectory: str = "") -> Dict[str, Any]:
        """Load a JSON configuration file with caching.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file (without .json extension)
        subdirectory : str
            Subdirectory within the config base directory
            
        Returns
        -------
        Dict[str, Any]
            Loaded configuration dictionary
        """
        cache_key = f"{subdirectory}/{config_name}"
        
        # Return from cache if available
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Construct file path
        if subdirectory:
            config_path = self.config_base_dir / subdirectory / f"{config_name}.json"
        else:
            config_path = self.config_base_dir / f"{config_name}.json"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._cache[cache_key] = config
                    return config
            else:
                print(f"Warning: Configuration file not found: {config_path}")
                return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {config_path}: {e}")
            return {}
        except Exception as e:
            print(f"Error loading configuration {config_path}: {e}")
            return {}
    
    def get_config_value(self, config_name: str, key_path: str, 
                        subdirectory: str = "", default: Any = None) -> Any:
        """Get a specific value from configuration using dot notation.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file
        key_path : str
            Dot-separated path to value (e.g., "resource_management.memory.max_usage_percent")
        subdirectory : str
            Subdirectory within the config base directory
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            The configuration value or default
        """
        config = self.load_config(config_name, subdirectory)
        
        # Navigate through nested dictionary using dot notation
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def load_all_system_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all system rule configurations.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            All loaded configurations
        """
        all_configs = {}
        
        # List of configuration files to load
        config_files = [
            ("RuntimePolicy", "System"),
            ("SystemRule", "System"),
            ("SafetyPolicy", "Safety"),
            ("RoutingRule", "Routing"),
            ("MemoryRule", "Memory"),
            ("AcquisitionRule", "Memory"),
            ("LearningRule", "Learning"),
            ("AdaptionRule", "Adaption")
        ]
        
        for config_name, subdirectory in config_files:
            config = self.load_config(config_name, subdirectory)
            if config:
                all_configs[config_name] = config
        
        return all_configs
    
    def reload_config(self, config_name: str, subdirectory: str = "") -> Dict[str, Any]:
        """Reload a configuration file, bypassing cache.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file
        subdirectory : str
            Subdirectory within the config base directory
            
        Returns
        -------
        Dict[str, Any]
            Reloaded configuration
        """
        cache_key = f"{subdirectory}/{config_name}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return self.load_config(config_name, subdirectory)
    
    def validate_config(self, config_name: str, required_keys: list,
                       subdirectory: str = "") -> bool:
        """Validate that a configuration has required keys.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file
        required_keys : list
            List of required top-level keys
        subdirectory : str
            Subdirectory within the config base directory
            
        Returns
        -------
        bool
            True if all required keys are present
        """
        config = self.load_config(config_name, subdirectory)
        
        for key in required_keys:
            if key not in config:
                print(f"Warning: Required key '{key}' not found in {config_name}")
                return False
        
        return True
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached configurations.
        
        Returns
        -------
        int
            Number of cached items
        """
        return len(self._cache)
