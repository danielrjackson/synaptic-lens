"""
Target Model Config Module

This module contains paths for target model configuration files.

Copyright 2025 Daniel Robert Jackson
"""

import os

class TargetModelConfigPath:
    """
    Path for target model configuration folder.

    Attributes:
        target_model_config_root: Path to the target model configuration directory
    """
    def __init__(self):
        self.target_model_config_root = os.path.dirname(__file__)

class TargetModelConfigFile:
    """
    Paths for target model configuration files.

    Attributes:
        target_model_config: Path to the target model configuration file
    """
    def __init__(self):
        self.target_model_config = os.path.join(TargetModelConfigPath.target_model_config_root, "config.yaml")
