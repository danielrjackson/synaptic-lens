"""
Parameter Encoder

This file contains the ParameterEncoder class for encoding neural network parameters
into a standardized format for analysis. Supports both 2D grid representations and
structured representations.

© 2025 Daniel Jackson
"""

import torch
import numpy as np
from collections import defaultdict

class ParameterEncoder:
    """
    Encodes neural network parameters into a standardized format for analysis.
    Supports both 2D grid representations and structured representations.
    """

    def __init__(self, fixed_structure=True, representation_type='2d'):
        """
        Initialize the parameter encoder.

        Args:
            fixed_structure: Whether all models have the same structure
            representation_type: Type of representation ('2d' or 'structured')
        """
        self.fixed_structure = fixed_structure
        self.representation_type = representation_type
        self.param_shapes = {}
        self.param_metadata = defaultdict(list)
        self.global_stats = {}
        self.key_order = None

    def analyze_parameter_space(self, models_data):
        """
        Analyze the parameter space across multiple models to calculate
        statistics for normalization.

        Args:
            models_data: List of model data entries

        Returns:
            Parameter shapes and metadata
        """
        print("Analyzing parameter space...")

        # Reset metadata
        self.param_metadata = defaultdict(list)
        self.global_stats = {}

        # Collect parameter shapes from the first model
        if not self.param_shapes:
            first_model_data = models_data[0]
            first_params = first_model_data["params"]

            for name, param in first_params.items():
                self.param_shapes[name] = param.shape

            # Create a consistent key order for 2D representation
            self.key_order = sorted(self.param_shapes.keys())

        # Collect statistics for each parameter across all models
        for model_data in models_data:
            params = model_data["params"]

            for name, param in params.items():
                if name in self.param_shapes:
                    # Convert to float tensor if not already
                    if not isinstance(param, torch.Tensor):
                        param = torch.tensor(param, dtype=torch.float32)

                    # Store statistics for this parameter
                    stats = {
                        'mean': float(param.mean().item()),
                        'std': float(param.std().item()),
                        'min': float(param.min().item()),
                        'max': float(param.max().item())
                    }
                    self.param_metadata[name].append(stats)

        # Calculate global statistics for each parameter
        for name, stats_list in self.param_metadata.items():
            means = [stats['mean'] for stats in stats_list]
            stds = [stats['std'] for stats in stats_list]
            mins = [stats['min'] for stats in stats_list]
            maxs = [stats['max'] for stats in stats_list]

            self.global_stats[name] = {
                'global_mean': sum(means) / len(means),
                'global_std': sum(stds) / len(stds),
                'global_min': min(mins),
                'global_max': max(maxs)
            }

        return self.param_shapes, self.global_stats

    def encode_parameters(self, params, normalize=True):
        """
        Encode model parameters into a format suitable for analysis.

        Args:
            params: Dictionary of parameter tensors
            normalize: Whether to normalize the parameters

        Returns:
            Dictionary of encoded parameter tensors
        """
        encoded = {}

        for name, param in params.items():
            if name not in self.param_shapes:
                continue

            # Convert to tensor if not already
            if not isinstance(param, torch.Tensor):
                param = torch.tensor(param, dtype=torch.float32)

            # Make a copy to avoid modifying the original
            encoded_param = param.clone()

            if normalize and name in self.global_stats:
                stats = self.global_stats[name]
                global_mean = stats['global_mean']
                global_std = stats['global_std']

                # Standardize the parameter (z-score normalization)
                if global_std > 0:
                    encoded_param = (encoded_param - global_mean) / global_std
                else:
                    encoded_param = encoded_param - global_mean

            encoded[name] = encoded_param

        return encoded

    def create_2d_representation(self, encoded_params):
        """
        Create a 2D grid representation of parameters for input to a CNN.
        Arranges all parameters into a single square-ish grid.

        Args:
            encoded_params: Dictionary of encoded parameter tensors

        Returns:
            2D tensor representing the parameters
        """
        # Use consistent key order if available
        key_order = self.key_order if self.key_order else sorted(encoded_params.keys())

        # Flatten each parameter tensor
        flat_params = []
        for key in key_order:
            if key in encoded_params:
                # Get the parameter tensor
                param = encoded_params[key]

                # Flatten to 1D
                flat_params.append(param.flatten())

        # Concatenate all flattened parameters
        if not flat_params:
            raise ValueError("No parameters to encode")

        all_params = torch.cat(flat_params)

        # Determine a reasonable grid size (square-ish)
        total_elements = all_params.numel()
        grid_side = int(np.ceil(np.sqrt(total_elements)))

        # Create a square-ish grid, padding with zeros if necessary
        grid = torch.zeros(grid_side, grid_side)
        flat_grid = grid.view(-1)
        flat_grid[:total_elements] = all_params

        return grid

    def create_structured_representation(self, encoded_params):
        """
        Create a structured representation preserving parameter relationships.
        Groups parameters by their role in the network.

        Args:
            encoded_params: Dictionary of encoded parameter tensors

        Returns:
            Dictionary of structured parameter representations
        """
        structured = {}

        # Group parameters by type
        weight_params = {}
        bias_params = {}
        norm_params = {}
        other_params = {}

        for name, param in encoded_params.items():
            if 'weight' in name:
                weight_params[name] = param
            elif 'bias' in name:
                bias_params[name] = param
            elif 'norm' in name:
                norm_params[name] = param
            else:
                other_params[name] = param

        # Process each group
        for group_name, group_params in [
            ('weights', weight_params),
            ('biases', bias_params),
            ('norms', norm_params),
            ('others', other_params)
        ]:
            if not group_params:
                continue

            # Concatenate parameters in each group
            param_list = []
            for name in sorted(group_params.keys()):
                param = group_params[name]

                # Reshape based on parameter type
                if len(param.shape) >= 2:
                    # Keep 2D matrices as is
                    param_list.append(param)
                else:
                    # Reshape vectors to row vectors
                    param_list.append(param.view(1, -1))

            # Combine parameters in this group
            if param_list:
                structured[group_name] = torch.cat(param_list, dim=0)

        return structured

    def encode_model(self, model_data):
        """
        Encode a model's parameters based on the configured representation type.

        Args:
            model_data: Dictionary containing model information

        Returns:
            Encoded parameter representation
        """
        # Extract parameters
        params = model_data["params"]

        # Encode parameters
        encoded_params = self.encode_parameters(params)

        # Create representation based on type
        if self.representation_type == '2d':
            representation = self.create_2d_representation(encoded_params)
        else:  # structured
            representation = self.create_structured_representation(encoded_params)

        return representation
