"""
SynapticLens package configuration module.

This module contains paths for the SynapticLens package configuration files.

© 2025 Daniel Jackson
"""

import os

class SynapticLensPackageConfigPath:
    """
    Path for package configuration folder.

    Attributes:
        package_config: Path to the package configuration directory
    """
    def __init__(self):
        self.package_config_root = os.path.dirname(__file__)

class SynapticLensPackageConfigFiles:
    """
    Paths for package configuration files.

    Attributes:
        package_config: Path to the package configuration file
    """
    def __init__(self):
        self.package_config = os.path.join(SynapticLensPackageConfigPath.package_config_root, "config.yaml")
