"""
Target Model Loader

This class is used to load target models for training the SynapticLens project.

Copyright 2025 Daniel Robert Jackson
"""

import  os
import  torch
import  re
import  json
from    pathlib                 import  Path
from    datetime                import  datetime
import  torchvision.datasets    as      datasets
import  torchvision.transforms  as      transforms
from    torch.utils.data        import  DataLoader
import  numpy                   as      np

from synaptic_lens.target_models.config.target_model_config import TargetModelConfig

class MNISTModelLoader:
    """
    Loader for MNIST models trained in the Training-Models project.
    Loads models and evaluates their performance on the MNIST test set.
    """

    def __init__(self,
                 base_dir: str,
                 test_data_dir=None):
        """
        Initialize the MNIST model loader.

        Args:
            base_dir: Base directory for models (C:/Users/Daniel/code/training_models/training_models/mnist)
            test_data_dir: Directory containing MNIST test data (defaults to base_dir/data/MNIST)
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.test_data_dir = Path(test_data_dir) if test_data_dir else self.base_dir / "data" / "MNIST"

        # Compiled regex for extracting epoch and batch from filenames
        self.model_pattern = re.compile(r'mnist_model_e(\d+)(?:_b(\d+))?\.pth')

        # Load test data
        self.test_loader = self._load_test_data()

        # Device for evaluation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache for model performance
        self.performance_cache = {}

    def _load_test_data(self):
        """Load MNIST test data."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load test dataset
        test_dataset = datasets.MNIST(
            root=str(self.test_data_dir),
            train=False,
            download=True,
            transform=transform
        )

        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False
        )

        return test_loader

    def find_model_files(self, include_batch_models=False):
        """
        Find all model files in the models directory.

        Args:
            include_batch_models: Whether to include intermediate batch models

        Returns:
            List of model file paths
        """
        model_files = []

        # Iterate through timestamp directories
        for timestamp_dir in self.models_dir.glob("*"):
            if not timestamp_dir.is_dir():
                continue

            # Find model files in this directory
            for model_file in timestamp_dir.glob("mnist_model_*.pth"):
                # Check if this is a batch model
                match = self.model_pattern.search(model_file.name)
                if match:
                    epoch, batch = match.groups()

                    # Skip batch models if not requested
                    if batch is not None and not include_batch_models:
                        continue

                    model_files.append(model_file)

        return sorted(model_files)

    def extract_model_info(self, model_path):
        """
        Extract model information from the file path.

        Args:
            model_path: Path to the model file

        Returns:
            Dictionary with model information
        """
        # Extract timestamp from parent directory
        timestamp_str = model_path.parent.name

        # Try to convert timestamp to datetime
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M")
        except ValueError:
            timestamp = None

        # Extract epoch and batch from filename
        match = self.model_pattern.search(model_path.name)
        if match:
            epoch, batch = match.groups()
            epoch = int(epoch)
            batch = int(batch) if batch else None
        else:
            epoch, batch = None, None

        return {
            "path": model_path,
            "timestamp": timestamp,
            "epoch": epoch,
            "batch": batch,
            "is_final": batch is None
        }

    def load_model(self, model_path):
        """
        Load a PyTorch model from the given path.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded PyTorch model
        """
        return torch.load(model_path, map_location=self.device)

    def evaluate_model(self, model):
        """
        Evaluate a model on the MNIST test set.

        Args:
            model: PyTorch model to evaluate

        Returns:
            Dictionary with loss and accuracy
        """
        model.to(self.device)
        model.eval()

        # Use CrossEntropyLoss for evaluation
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss
                total_loss += loss.item() * images.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy
        avg_loss = total_loss / total
        accuracy = correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }

    def get_model_performance(self, model_path, force_evaluate=False):
        """
        Get the performance of a model, evaluating if necessary.

        Args:
            model_path: Path to the model file
            force_evaluate: Whether to force evaluation even if cached

        Returns:
            Dictionary with loss and accuracy
        """
        # Convert path to string for use as dict key
        path_str = str(model_path)

        # Return cached result if available
        if not force_evaluate and path_str in self.performance_cache:
            return self.performance_cache[path_str]

        # Load and evaluate the model
        model = self.load_model(model_path)
        performance = self.evaluate_model(model)

        # Cache the result
        self.performance_cache[path_str] = performance

        return performance

    def save_performance_cache(self, output_file):
        """
        Save the performance cache to a JSON file.

        Args:
            output_file: Path to the output file
        """
        with open(output_file, 'w') as f:
            json.dump(self.performance_cache, f, indent=2)

    def load_performance_cache(self, input_file):
        """
        Load the performance cache from a JSON file.

        Args:
            input_file: Path to the input file
        """
        try:
            with open(input_file, 'r') as f:
                self.performance_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Could not load performance cache from {input_file}")

    def prepare_models_dataset(self, output_dir, max_models=None, include_batch_models=False):
        """
        Prepare a dataset of models and their performance for analysis.

        Args:
            output_dir: Directory to save the dataset
            max_models: Maximum number of models to include
            include_batch_models: Whether to include intermediate batch models

        Returns:
            List of model data entries
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Find model files
        model_files = self.find_model_files(include_batch_models)

        # Limit the number of models if specified
        if max_models is not None:
            model_files = model_files[:max_models]

        print(f"Processing {len(model_files)} models...")

        # Process each model
        models_data = []
        for i, model_path in enumerate(model_files):
            print(f"Processing model {i+1}/{len(model_files)}: {model_path.name}")

            # Extract model info
            model_info = self.extract_model_info(model_path)

            # Load model and get performance
            model = self.load_model(model_path)
            performance = self.get_model_performance(model_path)

            # Extract parameters
            params = {name: param.data.clone().cpu() for name, param in model.named_parameters()}

            # Save model data
            model_data = {
                "info": model_info,
                "performance": performance,
                "params": params
            }

            models_data.append(model_data)

            # Save performance cache periodically
            if (i + 1) % 10 == 0:
                self.save_performance_cache(output_dir / "performance_cache.json")

        # Save final performance cache
        self.save_performance_cache(output_dir / "performance_cache.json")

        # Save a summary of the dataset
        summary = {
            "num_models": len(models_data),
            "models": [
                {
                    "path": str(data["info"]["path"].relative_to(self.base_dir)),
                    "epoch": data["info"]["epoch"],
                    "batch": data["info"]["batch"],
                    "is_final": data["info"]["is_final"],
                    "performance": data["performance"]
                }
                for data in models_data
            ]
        }

        with open(output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return models_data


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MNIST Model Loader for SynapticLens')
    parser.add_argument('--base_dir', type=str,
                        default='C:/Users/Daniel/code/training_models/training_models/mnist',
                        help='Base directory for models')
    parser.add_argument('--output_dir', type=str, default='./mnist_models_dataset',
                        help='Output directory for the processed dataset')
    parser.add_argument('--max_models', type=int, default=None,
                        help='Maximum number of models to process')
    parser.add_argument('--include_batch', action='store_true',
                        help='Include intermediate batch models')

    args = parser.parse_args()

    # Initialize the loader
    loader = MNISTModelLoader(args.base_dir)

    # Prepare the dataset
    models_data = loader.prepare_models_dataset(
        args.output_dir,
        max_models=args.max_models,
        include_batch_models=args.include_batch
    )

    print(f"Dataset prepared with {len(models_data)} models")
