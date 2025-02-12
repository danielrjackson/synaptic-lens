"""
SynapticLens Project Configuration

This file contains paths for the SynapticLens project configuration files.

© 2025 Daniel Jackson
"""

import os

class SynapticLensProjectConfigPath:
    """
    Folder paths for the project config.

    Attributes:
        project_config: Path to the project configuration directory
    """

    def __init__(self):
        """
        Initialize the project paths.
        """
        # Path to the project configuration directory
        self.project_config = os.path.dirname(__file__)

class SynapticLensProjectPaths(SynapticLensProjectConfigPath):
    """
    Folder paths for the project.

    Contains the Paths inherited from:
        SynapticLensProjectConfigPaths: Contains path for the project config

    Attributes:
        project_root: Path to the project root directory
        project_config_root: Path to the project configuration directory
        data: Path to the project data directory where training data is cached
        logs: Path to the project logs directory where the training logs are saved
        models: Path to the project models directory where the trained SynapticLens models are saved
        results: Path to the project results directory where the evaluation results are saved
        target_models: Path to the project target models directory where the target models are cached
    """

    def __init__(self):
        """
        Initialize the project paths.
        """
        super().__init__()
        # Path to the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path to the project data directory where training data is cached
        self.data = os.path.join(self.project_root, "data")
        # Path to the project logs directory where the training logs are saved
        self.logs = os.path.join(self.project_root, "logs")
        # Path to the project models directory where the trained SynapticLens models are saved
        self.models = os.path.join(self.project_root, "models")
        # Path to the project results directory where the evaluation results are saved
        self.results = os.path.join(self.project_root, "results")
        # Path to the project target models directory where the target models are cached
        self.target_models = os.path.join(self.data, "target_models")


class SynapticLensProjectConfigFiles:
    """
    File paths for the project config.

    Attributes:
        project_keys: Path to the project keys configuration file
    """

    def __init__(self):
        """
        Initialize the project files.
        """
        # Path to the project keys configuration file with information about how to access credentials
        self.project_keys = os.path.join(SynapticLensProjectPaths.project_config, "keys.yaml")