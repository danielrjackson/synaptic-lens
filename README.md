# SynapticLens

SynapticLens is an AI model for classifying structures within artificial neural networks. It treats neural network parameters as data that can be analyzed and classified to understand what makes networks effective for specific tasks.

## Project Vision

SynapticLens aims to develop a meta-learning approach that can:

1. Understand the internal structure of neural networks by analyzing their parameters
2. Predict network performance based solely on parameter patterns
3. Identify which parameters are critical for specific network functionalities
4. Eventually classify networks by their function, not just their architecture

This approach is inspired by image recognition techniques, applying similar pattern recognition concepts to the parameter space of neural networks.

## Features

- **Model Analysis**: Load and analyze pre-trained neural network models
- **Parameter Encoding**: Convert model parameters into standardized representations
- **Performance Prediction**: Predict network accuracy and loss based on parameter patterns
- **Importance Analysis**: Identify which parameters most influence network performance
- **Visualization**: Generate insightful visualizations of parameter patterns and importance

## Documentation

- [Core Concepts] - Fundamental concepts and approach behind SynapticLens
- [Extending SynapticLens] - Guide for supporting new architectures and customizing analysis

## Installation

SynapticLens uses Poetry for dependency management. To install:

```bash
# Clone the repository
git clone https://github.com/danielrjackson/synaptic_lens.git
cd synaptic_lens

# Install dependencies using Poetry
uv init .
```

## Usage

### Basic Usage

> [!IMPORTANT]
> Usage TBD, the following is just an example.

```bash
# Run the analysis with default configuration
python -m synaptic_lens.main

# Run with a custom configuration file
python -m synaptic_lens.main --config path/to/config.yaml
```

### Configuration

SynapticLens is configured using YAML files. The default configuration is located at `synaptic_lens/config/config.yaml`. This file contains settings for:

- Model loading options
- Parameter encoding methods
- Neural network analyzer architecture
- Training parameters
- Visualization preferences
- Analysis options

> [!IMPORTANT]
> The project is very new and is in transition. The utility files are being factored out.

## Project Structure

```text
synaptic_lens/
├── .venv/                                  # Python virtual environment folder
│   ├── Include/                            # Include files
│   ├── Lib/                                # Python libraries
│   │   └── site-packages/                  # Python 3rd-party packages
│   │       └── ...                         # Python packages used by SynapticLens
│   ├── Scripts/                            # Python environment executables
│   │   └── ...                             # Executable files
│   ├── .gitignore                          # Git config file excluding the contents of this directory
│   └── pyvenv.cfg                          # Config file for the Python virtual environment
├── .vscode/                                # VSCode config folder
│   ├── launch.json                         # VSCode launch configuration
│   ├── settings.json                       # VSCode project configuration
│   └── synaptic_lens.code-workspace        # VSCode workspace file
├── config/                                 # Configuration folder for the project
│   ├── __init__.py                         # Package file for SynapticLens project config
│   └── keys.yaml                           # Configuration file with key/credential access information
├── data/                                   # Data storage
├── docs/                                   # Documentation
│   ├── images/                             # Documentation images
│   │   ├── architecture_diagram.mermaid    # SynapticLens architecture diagram
│   │   └── license.png                     # Image for the license for this project
│   ├── test_coverage/                      # Test coverage results
|   └── core_concepts_doc.md                # Core concepts for this project
|   └── extending_doc.md                    # Future ideas for extending this project
├── logs/                                   # Logs from running SynapticLens
├── models/                                 # Contains models created by SynapticLens
├── results/                                # Analysis results
├── synaptic_lens/                          # Main package
│   ├── config/                             # Configuration folder for the package
│   │   ├── __init__.py                     # Package file for SynapticLens package config
│   │   └── config.yaml                     # Configuration file for the package
│   ├── loaders/                            # AI model loader folder
│   │   ├── __init__.py                     # Package file for SynapticLens loaders module
│   │   └── model_loader.py                 # Model loading and evaluation
│   ├── __init__.py                         # Package file for SynapticLens package
│   ├── network_analyzer.py                 # Neural network for analyzing parameters
│   ├── parameter_encoder.py                # Parameter encoding and normalization
│   └── synaptic_lens.py                    # Main entry point
├── util/                                   # Documentation
│   ├── __init__.py                         # Package file for SynapticLens util module
│   ├── config_loader.py                    # Configuration loading utilities
│   └── visualization.py                    # Visualization utilities
├── .gitignore                              # Git config file excluding certain files and folders
├── LICENSE.md                              # The license file for SynapticLens
├── pyproject.toml                          # Poetry configuration
└── README.md                               # This file
```

## Current Status

The project currently supports analysis of transformer networks trained on the MNIST dataset. These networks serve as a proof of concept for the approach.

Future work will focus on:

- Supporting diverse network architectures
- Handling varying network structures
- Identifying functional components across different networks
- Classifying networks by their capabilities rather than just architecture

---

*Copyright © 2025 Daniel Jackson. [All Rights Reserved.]*

[//]: # (Links)
[Core Concepts]:            docs/core_concepts_doc
[Extending SynapticLens]:   docs/extending_doc
[All Rights Reserved.]:          LICENSE
