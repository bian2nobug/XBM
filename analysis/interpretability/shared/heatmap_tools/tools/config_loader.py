"""
Generic configuration loader utility
Used to load and manage various YAML config files

Features:
- supports relative and absolute paths
- robust error handling and messages
- type hints
- flexible config validation
- nested key-path access
"""

import yaml
import os
from typing import Any, Optional, Dict, List


class ConfigLoader:
    """
    Generic configuration loader class

    Supports:
    - loading YAML config files
    - nested key-path access (e.g. 'database.host')
    - config hot reload
    - custom config validation
    """

    def __init__(self, config_path: str):
        """
        Initialize the config loader

        Args:
            config_path: path to the config file (relative or absolute)

        Raises:
            FileNotFoundError: config file does not exist
            yaml.YAMLError: invalid YAML format
        """
        # handle relative paths: relative to the calling script's directory
        if not os.path.isabs(config_path):
            # if relative, resolve against the current working directory
            config_path = os.path.abspath(config_path)

        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the YAML config file

        Returns:
            config dictionary

        Raises:
            FileNotFoundError: config file does not exist
            yaml.YAMLError: invalid YAML format
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Make sure the config file exists, or use an absolute path"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid config file format: {self.config_path}\n"
                f"YAML parsing failed: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config file: {self.config_path}\n"
                f"Error: {e}"
            )
        
        if config is None:
            raise ValueError(f"Config file is empty: {self.config_path}")
        
        if not isinstance(config, dict):
            raise TypeError(
                f"Invalid config file format: {self.config_path}\n"
                f"Expected a dict, got {type(config)}"
            )
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a config value, supporting nested key paths

        Args:
            key_path: config key path, e.g. 'smoothing.enable' or 'scale_adjustment'
            default: default value (returned when the key does not exist)

        Returns:
            the config value, or default if it does not exist
            
        Example:
            >>> loader.get('smoothing.enable')
            True
            >>> loader.get('non.existent.key', default=False)
            False
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                if default is None:
                    # give a friendlier message
                    print(f"Warning: config item '{key_path}' does not exist, returning None")
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Dynamically set a config value (in-memory only, not written to file)

        Args:
            key_path: config key path
            value: the new value
            
        Example:
            >>> loader.set('appearance.alpha', 0.5)
        """
        keys = key_path.split('.')
        config = self.config
        
        # navigate to the second-to-last level
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        # set the value at the last level
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """
        Reload the config file

        Note: this discards all in-memory changes made via set()
        """
        self.config = self._load_config()
        print(f"Config reloaded: {self.config_path}")
    
    def validate(self, required_keys: Optional[List[str]] = None) -> bool:
        """
        Validate the completeness of the config file

        Args:
            required_keys: list of required config keys (supports nested paths, e.g. 'database.host')
                          if None, validation is skipped and True is returned

        Returns:
            True if the config is complete, False if there are missing items

        Example:
            >>> loader.validate(['database.host', 'database.port', 'api.key'])
            True
        """
        if required_keys is None:
            return True

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            print(f"Warning: config file is missing the following required items:")
            for key in missing_keys:
                print(f"  - {key}")
            return False

        return True
    
    def print_config(self, title: str = "Configuration", indent: int = 0) -> None:
        """
        Print the current config (formatted output, supports nested structures)

        Args:
            title: config title
            indent: indentation level (used for recursive printing of nested config)
        """
        if indent == 0:
            print("=" * 60)
            print(f"{title:^60}")
            print("=" * 60)
            print(f"Config file path: {self.config_path}")
            print("-" * 60)

        self._print_dict(self.config, indent)

        if indent == 0:
            print("=" * 60)

    def _print_dict(self, data: Dict[str, Any], indent: int = 0) -> None:
        """
        Recursively print a dict (internal method)

        Args:
            data: the dict to print
            indent: current indentation level
        """
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: {value}")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigLoader(config_path='{self.config_path}')"
    
    def __str__(self) -> str:
        """Friendly string representation"""
        return f"ConfigLoader: {self.config_path}"

