"""
Target Model Configuration

Contains the configuration for a target model, including:
    - name:             The name of the target model
    - architecture:     The architecture of the target model
    - purpose:          The kind of task the target model is designed for
    - desc:             A description of the target model
    - models_base_path: The base path for the target model files
    - model_file_type:  The file type for the target model files
    - test_data_file:   The path to the test data file
    - test_label_file:  The path to the test label file

© 2025 Daniel Jackson
"""

# Standard library imports

# Build file paths
import os
from typing import Optional, Dict

# Local package imports

# The location of the target model configuration file
from synaptic_lens.target_models.config import TargetModelConfigFile

# The configuration loader
from util.config_loader import ConfigLoader

class TargetModelConfig:
    """
    Configuration for a target model.

    Attributes:
        name: Name of the target model
        architecture: Architecture of the target model
        purpose: Purpose of the target model
        desc: Description of the target model
        models_base_path: Base path for the target model files
        model_file_type: File type for the target model files
        test_data_file: Path to the test data file
        test_label_file: Path to the test label file
    """

    def __init__(self,
                 model_config: Optional[Dict]):
        """
        Initialize the target model configuration.

        Args:
            model_config: Configuration dictionary for the target model
        """

        self.name               = model_config['name']
        self.architecture       = model_config['architecture']
        self.purpose            = model_config['purpose']
        self.desc               = model_config['desc']
        self.models_base_path   = model_config['models_base_path']
        self.model_file_type    = model_config['model_file_type']
        test                    = model_config['test']
        assert 'base_path'  in test,            "Test base path field is empty"
        assert 'files'      in test,            "Test files field is empty"
        test_base_path          = test['base_path']
        assert os.path.exists(test_base_path), f"Path does not exist: {test_base_path}"
        test_files              = test['files']
        assert 'data'       in test_files,      "Test data file field is empty"
        assert 'label'      in test_files,      "Test label file field is empty"
        test_data_file_name     = test_files['data']
        test_label_file_name    = test_files['label']
        self.test_data_file     = os.path.join(test_base_path, test_data_file_name)
        self.test_label_file    = os.path.join(test_base_path, test_label_file_name)

        self._validate()

    def _validate(self):
        """
        Validate the target model configuration.
        """
        # ensure fields aren't empty
        assert self.name,               "Name field is empty"
        assert self.architecture,       "Architecture field is empty"
        assert self.purpose,            "Purpose field is empty"
        assert self.desc,               "Description field is empty"
        assert self.models_base_path,   "Models base path field is empty"
        assert self.model_file_type,    "Model file type field is empty"
        assert self.test_data_file,     "Test data file field is empty"
        assert self.test_label_file,    "Test label file field is empty"

        # validate paths
        assert os.path.exists(self.models_base_path),   f"Path does not exist: {self.models_base_path}"
        assert os.path.exists(self.test_data_file),     f"Path does not exist: {self.test_data_file}"
        assert os.path.exists(self.test_label_file),    f"Path does not exist: {self.test_label_file}"

    def __str__(self):
        """
        Build a string representation of the target model configuration.
        """
        return f'{{{self.name}: Architecture:{self.architecture} Purpose:{self.purpose} Description:"{self.desc}" ModelsBasePath:{self.models_base_path} ModelFileType:{self.model_file_type} TestDataFile:{self.test_data_file} TestLabelFile:{self.test_label_file}}}'

class TargetModelConfigs:
    """
    Target Model Configurations

    Contains a list of configurations for target models, each including:
        - name:             The name of the target model
        - architecture:     The architecture of the target model
        - purpose:          The kind of task the target model is designed for
        - desc:             A description of the target model
        - models_base_path: The base path for the target model files
        - model_file_type:  The file type for the target model files
        - test_data_file:   The path to the test data file
        - test_label_file:  The path to the test label file
    """

    def __init__(self):
        """
        Load the target model configurations.
        """
        required_sections = ['target_models']
        config_loader     = ConfigLoader(
            config_path         = TargetModelConfigFile.target_model_config,
            required_sections   = required_sections)
        config = config_loader.get_config()

        target_model_config_settings = config['target_models']

        self.configs = []

        for target_model_config_setting_dict in target_model_config_settings:
            self.configs.append(TargetModelConfig(target_model_config_setting_dict))

        assert self.configs, "No target model configurations found"

    def __str__(self):
        """
        Build a string representation of the target model configurations.
        """
        return str(self.configs)
