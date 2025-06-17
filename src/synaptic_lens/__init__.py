"""
SynapticLens package initialization file.

This file contains the SynapticLens package paths and configuration file paths.

© 2025 Daniel Jackson
"""

import os
from config import SynapticLensProjectPaths, SynapticLensProjectConfigFiles
from synaptic_lens.config import SynapticLensPackageConfigPath, SynapticLensPackageConfigFiles
from synaptic_lens.target_models.config import TargetModelConfigPath, TargetModelConfigFile

class SynapticLensPaths(SynapticLensProjectPaths, SynapticLensPackageConfigPath, TargetModelConfigPath):
    """
    Path configuration for the SynapticLens package folder.

    Contains the Paths inherited from:
        SynapticLensProjectPaths: Contains paths for the project
        SynapticLensPackageConfigPath: Contains path for the package config
        TargetModelConfigPath: Contains path for the target model config

    Attributes:
        project_root: Absolute path to the project root directory
        package_root: Absolute path to the package root directory
        project_config_root: Absolute path to the project configuration
        package_config_root: Absolute path to the package configuration directory
        target_model_config_root: Absolute path to the target model configuration directory
        data: Path to the project data directory where training data is cached
        logs: Path to the project logs directory where the training logs are saved
        models: Path to the project models directory where the trained SynapticLens models are saved
        results: Path to the project results directory where the evaluation results are saved
        target_models: Path to the project target models directory where the target models are cached
    """
    def __init__(self):
        super().__init__()
        # Path to the package root directory
        self.package_root = os.path.dirname(__file__)

class SynapticLensConfigFiles(SynapticLensProjectConfigFiles, SynapticLensPackageConfigFiles, TargetModelConfigFile):
    """
    File paths for the SynapticLens configs.

    Contains the Files inherited from:
        SynapticLensProjectFiles: Contains file paths for the project
        SynapticLensPackageConfigFiles: Contains file paths for the package config
        TargetModelConfigFile: Contains file paths for the target model config

    Attributes:
        project_keys: Path to the project keys configuration file
        package_config: Path to the package configuration file
        target_model_config: Path to the target model configuration file
    """
    def __init__(self):
        super().__init__()
