"""
Target Model Configuration

Contains the configuration for a target model, including:
    - Name
    - Architecture
    - Purpose
    - Description
    - Base path for model files
    - File type for model files
    - Test data directory
    - Test data file
    - Test label file

© 2025 Daniel Jackson
"""

# Local package imports

# The location of the synaptic_lens package configuration files
from config import SynapticLensProjectConfigFiles

# The configuration loader
from util.config_loader import ConfigLoader

# Keys
# Stored in Windows Credential Manager
# keys:
#   - name: OpenAI_DevKey
#     url: OpenAI_API
#     username: My_Dev_Key

class KeyConfig:
    """
    Configuration for accessing an API key.

    Attributes:
        name: Name of the API key
        url: URL for the API
        username: Username for the API
    """

    def __init__(self,
                 key_config: dict):
        """
        Initialize the API key configuration.

        Args:
            key_config: Configuration dictionary for the API key
        """
        self.name = key_config['name']
        self.url = key_config['url']
        self.username = key_config['username']

        self._validate()

    def _validate(self):
        """
        Validate the API key configuration.
        """
        assert self.name,       "API key name is empty"
        assert self.url,        "API key URL is empty"
        assert self.username,   "API key username is empty"

    def __str__(self):
        """
        Build a string representation of the target model configuration.
        """
        return f"{{{self.name}: url:{self.url} username:{self.username}}}"


def get_key_configs():
    """
    Get the API key configurations.

    Returns:
        List of API key configurations
    """
    required_sections = ['keys']
    config_loader = ConfigLoader(
        config_path         = SynapticLensProjectConfigFiles().project_keys,
        required_sections   = required_sections)
    config = config_loader.get_config()

    key_configs = []
    for key_config_setting_dict in config['keys']:
        key_configs.append(KeyConfig(key_config_setting_dict))

    return key_configs
