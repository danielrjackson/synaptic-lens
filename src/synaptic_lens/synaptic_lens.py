"""
SynapticLens: Neural Network Parameter Analysis

This file contains the main SynapticLens class for analyzing neural network parameters.
The SynapticLens class loads neural network models, encodes their parameters, trains
a network analyzer, and evaluates the analyzer on the models. It also analyzes the
importance of parameters for predicting performance.

© 2025 Daniel Jackson
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import SynapticLens modules
from synaptic_lens.config_loader import ConfigLoader
from synaptic_lens.target_models.model_loader import MNISTModelLoader
from synaptic_lens.parameter_encoder import ParameterEncoder
from synaptic_lens.network_analyzer import NetworkAnalyzer, NetworkDataset
from util.visualization import (
    visualize_predictions,
    plot_parameter_importance,
    visualize_parameter_distributions,
    visualize_training_progress
)

class SynapticLens:
    """
    Main class for the SynapticLens project.
    Analyzes neural network parameters to predict performance.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize SynapticLens.

        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()

        # Extract configuration sections
        self.paths = self.config['paths']
        self.model_loading_config = self.config['model_loading']
        self.encoding_config = self.config['encoding']
        self.analyzer_config = self.config['analyzer']
        self.training_config = self.config['training']
        self.visualization_config = self.config.get('visualization', {})
        self.analysis_config = self.config.get('analysis', {})

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize components
        self.model_loader = None
        self.param_encoder = None
        self.analyzer = None
        self.dataset = None

        # Set random seed for reproducibility
        seed = self.training_config.get('random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        os.makedirs(self.paths['output_dir'], exist_ok=True)

        # Create run directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.paths['output_dir'] / f"run_{self.timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        # Save configuration for this run
        self.config_loader.save_config(self.config, self.run_dir / "config.yaml")

    def load_models(self) -> List[Dict[str, Any]]:
        """
        Load and process neural network models.

        Returns:
            List of model data entries
        """
        print("Loading neural network models...")

        # Initialize the model loader
        self.model_loader = MNISTModelLoader(
            base_dir=self.paths['models_dir'],
            test_data_dir=self.paths['models_dir'] / "data" / "MNIST"
        )

        # Load performance cache if available and enabled
        if self.model_loading_config.get('cache_performance', True):
            cache_path = self.paths['data_dir'] / "performance_cache.json"
            if cache_path.exists():
                print(f"Loading performance cache from {cache_path}")
                self.model_loader.load_performance_cache(cache_path)

        # Prepare the dataset
        models_data = self.model_loader.prepare_models_dataset(
            output_dir=self.paths['data_dir'],
            max_models=self.model_loading_config.get('max_models'),
            include_batch_models=self.model_loading_config.get('include_batch_models', False)
        )

        print(f"Loaded {len(models_data)} models")

        # Visualize training progress if enabled
        if self.analysis_config.get('training_progress', True):
            visualize_training_progress(
                models_data,
                output_dir=self.run_dir,
                show_plots=self.visualization_config.get('show_plots', True),
                save_plots=self.visualization_config.get('save_plots', True),
                dpi=self.visualization_config.get('plot_dpi', 300)
            )

        return models_data

    def create_dataset(self, models_data: List[Dict[str, Any]]) -> NetworkDataset:
        """
        Create a dataset for analyzing neural network parameters.

        Args:
            models_data: List of model data entries

        Returns:
            NetworkDataset instance
        """
        print("Creating parameter dataset...")

        # Initialize the parameter encoder
        self.param_encoder = ParameterEncoder(
            fixed_structure=self.encoding_config.get('fixed_structure', True),
            representation_type=self.encoding_config.get('representation_type', '2d'),
            normalize_params=self.encoding_config.get('normalize_params', True)
        )

        # Create the dataset
        self.dataset = NetworkDataset(
            models_data=models_data,
            param_encoder=self.param_encoder,
            representation_type=self.encoding_config.get('representation_type', '2d')
        )

        print(f"Created dataset with {len(self.dataset)} samples")

        # Analyze parameter distributions if enabled
        if self.analysis_config.get('distribution_analysis', True):
            visualize_parameter_distributions(
                models_data,
                output_dir=self.run_dir,
                show_plots=self.visualization_config.get('show_plots', True),
                save_plots=self.visualization_config.get('save_plots', True),
                dpi=self.visualization_config.get('plot_dpi', 300)
            )

        return self.dataset

    def train_analyzer(self, dataset: NetworkDataset) -> Tuple[NetworkAnalyzer, Dict[str, List[float]]]:
        """
        Train the network analyzer.

        Args:
            dataset: NetworkDataset instance

        Returns:
            Trained NetworkAnalyzer instance and training history
        """
        print("Training network analyzer...")

        # Split dataset into train and validation sets
        train_ratio = self.training_config.get('train_ratio', 0.8)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"Split dataset: {train_size} training samples, {val_size} validation samples")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.get('batch_size', 16),
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.get('batch_size', 16)
        )

        # Initialize the analyzer
        input_shape = dataset.get_representation_shape()
        hidden_dim = self.analyzer_config.get('hidden_dim', 256)
        dropout_rate = self.analyzer_config.get('dropout_rate', 0.5)

        self.analyzer = NetworkAnalyzer(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        print(f"Initialized analyzer with input shape: {input_shape}")

        # Train the analyzer
        history = self.analyzer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.training_config.get('num_epochs', 50),
            lr=self.training_config.get('learning_rate', 0.001),
            device=self.device
        )

        # Save the trained analyzer
        model_path = self.run_dir / f"analyzer_model.pt"
        torch.save(self.analyzer.state_dict(), model_path)
        print(f"Saved trained analyzer to {model_path}")

        # Save training history
        history_path = self.run_dir / "training_history.pt"
        torch.save(history, history_path)

        # Plot training history
        self._plot_training_history(history)

        return self.analyzer, history

    def _plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot and save the training history.

        Args:
            history: Training history dictionary
        """
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Training History')

        # Save the plot
        if self.visualization_config.get('save_plots', True):
            plot_path = self.run_dir / "training_history.png"
            plt.savefig(
                plot_path,
                dpi=self.visualization_config.get('plot_dpi', 300),
                bbox_inches='tight'
            )

        if self.visualization_config.get('show_plots', True):
            plt.show()
        else:
            plt.close()

    def evaluate_analyzer(self, test_loader: Optional[DataLoader] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the trained analyzer.

        Args:
            test_loader: DataLoader for test data (if None, uses validation data)

        Returns:
            Evaluation metrics, predictions, and targets
        """
        if self.analyzer is None:
            raise ValueError("Analyzer not trained yet")

        print("Evaluating analyzer...")

        # If no test loader provided, create one from the validation set
        if test_loader is None:
            # Split dataset into train and validation sets
            train_ratio = self.training_config.get('train_ratio', 0.8)
            train_size = int(train_ratio * len(self.dataset))
            val_size = len(self.dataset) - train_size

            _, val_dataset = random_split(self.dataset, [train_size, val_size])

            test_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.get('batch_size', 16)
            )

        # Evaluate the analyzer
        metrics = self.analyzer.evaluate(test_loader, device=self.device)

        print(f"Evaluation metrics: {metrics}")

        # Visualize predictions
        predictions, targets = visualize_predictions(
            self.analyzer,
            test_loader,
            device=self.device,
            output_dir=self.run_dir if self.visualization_config.get('save_plots', True) else None,
            show_plots=self.visualization_config.get('show_plots', True),
            dpi=self.visualization_config.get('plot_dpi', 300)
        )

        # Save evaluation metrics
        metrics_path = self.run_dir / "evaluation_metrics.pt"
        torch.save(metrics, metrics_path)

        return metrics, predictions, targets

    def analyze_parameter_importance(self) -> Any:
        """
        Analyze which parameters are most important for predicting performance.

        Returns:
            Parameter importance analysis results
        """
        if not self.analysis_config.get('importance_analysis', True):
            print("Parameter importance analysis disabled in configuration")
            return None

        if self.analyzer is None:
            raise ValueError("Analyzer not trained yet")

        print("Analyzing parameter importance...")

        # Create a data loader for the entire dataset
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.training_config.get('batch_size', 16)
        )

        # Analyze parameter importance
        importance = self.analyzer.analyze_parameter_importance(
            self.dataset, data_loader, device=self.device
        )

        # Plot parameter importance
        plot_parameter_importance(
            importance,
            self.dataset.representation_type,
            output_dir=self.run_dir if self.visualization_config.get('save_plots', True) else None,
            show_plots=self.visualization_config.get('show_plots', True),
            dpi=self.visualization_config.get('plot_dpi', 300)
        )

        # Save importance analysis
        if isinstance(importance, dict):
            # For structured representation, save as JSON
            import json
            with open(self.run_dir / "parameter_importance.json", 'w') as f:
                json.dump({str(k): float(v) for k, v in importance.items()}, f, indent=2)
        else:
            # For 2D representation, save as tensor
            torch.save(importance, self.run_dir / "parameter_importance.pt")

        return importance

    def run(self) -> Dict[str, Any]:
        """
        Run the complete SynapticLens analysis pipeline.

        Returns:
            Dictionary of results
        """
        print(f"Starting SynapticLens run {self.timestamp}")
        print(f"Results will be saved to {self.run_dir}")

        # Load models
        models_data = self.load_models()

        # Create dataset
        dataset = self.create_dataset(models_data)

        # Train analyzer
        analyzer, history = self.train_analyzer(dataset)

        # Evaluate analyzer
        metrics, predictions, targets = self.evaluate_analyzer()

        # Analyze parameter importance
        importance = self.analyze_parameter_importance()

        # Collect results
        results = {
            "metrics": metrics,
            "history": history,
            "importance": importance,
            "timestamp": self.timestamp,
            "config": self.config
        }

        print(f"Analysis complete. All results saved to {self.run_dir}")

        return results


def main():
    """Main function to run SynapticLens."""
    import argparse

    parser = argparse.ArgumentParser(description='SynapticLens: Neural Network Parameter Analysis')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')

    args = parser.parse_args()

    # Initialize and run SynapticLens
    synaptic_lens = SynapticLens(args.config)
    results = synaptic_lens.run()

    # Print final metrics
    print("\nFinal Evaluation Metrics:")
    for metric, value in results["metrics"].items():
        print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()
