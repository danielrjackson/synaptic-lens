"""
Configuration Loader

Handles loading and validation of SynapticLens configuration from YAML files.

Attributes:
    ConfigLoader: Handles loading and validation of SynapticLens configuration from YAML files

© 2025 Daniel Jackson
"""

# Standard Libraries
import  os
from    pathlib import Path
from    typing import Any, Dict, List, Optional

# Third-party Libraries
import  yaml

class ConfigLoader:
    """
    Handles loading and validation of SynapticLens configuration from YAML files.
    """

    def __init__(self,
                 config_path: str,
                 config_file: str = "config.yaml",
                 required_sections: Optional[List[str]] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: The path to the configuration directory
            config_file: The name of the configuration file
            required_sections: List of required sections in the configuration file
        """
        # Default config path
        if config_path is None:
            # throw
            raise ValueError("Configuration path must be provided")

        # combine path and file name
        self.config_path = os.path.join(config_path, config_file)
        self.required_sections = required_sections

        self.config = self._load_config()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.

        Returns:
            Configuration dictionary
        """
        return self.config

    def save_config(self, config: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            output_path: Path to save the configuration to (defaults to loaded path)
        """
        if output_path is None:
            output_path = self.config_path
        else:
            output_path = Path(output_path)

        # Convert Path objects to strings for serialization
        config_to_save = self._recursively_serialize_config(config)

        # Save configuration
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        # Check if config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        # Load configuration
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate configuration
        self._validate_config(config)

        return config

    def _recursively_serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively serialize configuration to strings.

        Args:
            config: Configuration dictionary to serialize

        Returns:
            Serialized configuration dictionary
        """
        serialized_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                serialized_config[key] = self._recursively_serialize_config(value)
            elif isinstance(value, Path):
                serialized_config[key] = str(value)
            else:
                serialized_config[key] = value

        return serialized_config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and values.

        Args:
            config: Configuration dictionary to validate
        """
        if self.required_sections is None:
            return

        # Ensure required sections exist
        for section in self.required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section '{section}' not found")
