"""
Target Model Loader Configuration

Contains the configuration for a target model, including:
  - max_models:           Maximum number of models to process (null for all)
  - include_batch_models: Whether to include intermediate batch models
  - cache_models:         Whether to cache the model data and test files
  - cache_test_results:   Whether to cache the model test results
  - cache_performance:    Whether to cache model performance results

Copyright 2025 Daniel Robert Jackson
"""

# Local package imports

# The location of the target model configuration file
from synaptic_lens.target_models.config import TargetModelConfigFile

# The configuration loader
from util.config_loader import ConfigLoader

class TargetModelLoaderConfig:
    """
    Configuration for a target model.

    Attributes:
        max_models:             Maximum number of models to process (null for all)
        include_batch_models:   Whether to include intermediate batch models
        cache_models:           Whether to cache the model data and test files
        cache_test_results:     Whether to cache the model test results
        cache_performance:      Whether to cache model performance
    """

    def __init__(self):
        """
        Initialize the target model configuration.
        """
        required_sections   = ['model_loading']
        config_loader       = ConfigLoader(
            config_path         = TargetModelConfigFile.target_model_config,
            required_sections   = required_sections)
        config          = config_loader.get_config()
        model_loading   = config['model_loading']

        self.max_models             = model_loading['max_models']
        self.include_batch_models   = model_loading['include_batch_models']
        self.cache_models           = model_loading['cache_models']
        self.cache_test_results     = model_loading['cache_test_results']
        self.cache_performance      = model_loading['cache_performance']

    def __str__(self):
        """
        Build a string representation of the target model configuration.
        """
        return f"{{MaxModels:{self.max_models} IncludeBatchModels:{self.include_batch_models} CacheModels:{self.cache_models} CacheTestResults:{self.cache_test_results} CachePerformance:{self.cache_performance}}}"
