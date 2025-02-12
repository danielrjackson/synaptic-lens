"""
Network Analyzer

This file contains classes for analyzing neural network parameters and predicting performance metrics.

© 2025 Daniel Jackson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class NetworkDataset(Dataset):
    """
    Dataset for neural network parameters and their performance metrics.
    """

    def __init__(self, models_data, param_encoder, representation_type='2d'):
        """
        Initialize the dataset.

        Args:
            models_data: List of model data entries (each containing params and performance)
            param_encoder: ParameterEncoder instance
            representation_type: Type of parameter representation ('2d' or 'structured')
        """
        self.models_data = models_data
        self.param_encoder = param_encoder
        self.representation_type = representation_type

        # Analyze parameter space for normalization
        self.param_encoder.analyze_parameter_space(models_data)

        # Create encoded representations
        self.encoded_data = []

        print(f"Encoding {len(models_data)} models...")
        for model_data in models_data:
            # Skip models without performance data
            performance = model_data["performance"]
            if None in performance.values():
                continue

            # Encode parameters
            representation = self.param_encoder.encode_model(model_data)

            # Extract performance metrics
            performance_tensor = torch.tensor([
                performance.get('loss', 0.0),
                performance.get('accuracy', 0.0)
            ], dtype=torch.float32)

            self.encoded_data.append((representation, performance_tensor))

        print(f"Created dataset with {len(self.encoded_data)} encoded models")

    def __len__(self):
        """Return the number of models in the dataset."""
        return len(self.encoded_data)

    def __getitem__(self, idx):
        """Get a model representation and its performance metrics."""
        return self.encoded_data[idx]

    def get_representation_shape(self):
        """Get the shape of the parameter representation."""
        if not self.encoded_data:
            return None

        # Get the first representation
        first_representation = self.encoded_data[0][0]

        # Return shape based on representation type
        if self.representation_type == '2d':
            return first_representation.shape
        else:  # structured
            return {key: value.shape for key, value in first_representation.items()}


class NetworkAnalyzer(nn.Module):
    """
    Neural network for analyzing other neural networks' parameters.
    Predicts performance metrics (loss and accuracy) from parameter representations.
    """

    def __init__(self, input_shape, hidden_dim=256, dropout_rate=0.5):
        """
        Initialize the network analyzer.

        Args:
            input_shape: Shape of the input parameter representation
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super(NetworkAnalyzer, self).__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Architecture depends on the input representation
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            # 2D grid representation -> CNN architecture
            self._init_cnn_architecture()
            self.forward_fn = self._forward_cnn
        else:
            # Structured representation -> Custom architecture
            self._init_structured_architecture()
            self.forward_fn = self._forward_structured

    def _init_cnn_architecture(self):
        """Initialize CNN architecture for 2D grid representation."""
        # First convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions and pooling
        h, w = self.input_shape
        # After 3 pooling layers of stride 2, dimensions are divided by 8
        h_out = h // 8
        w_out = w // 8
        # Handle cases where dimensions become too small
        h_out = max(1, h_out)
        w_out = max(1, w_out)

        # Flatten dimension
        self.flatten_size = 128 * h_out * w_out

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc_out = nn.Linear(self.hidden_dim // 2, 2)  # loss and accuracy

    def _init_structured_architecture(self):
        """Initialize architecture for structured representation."""
        # For structured inputs, we need a network for each parameter group
        self.group_embeddings = nn.ModuleDict()
        self.group_sizes = {}

        # Get the total size of each group
        if isinstance(self.input_shape, dict):
            for group_name, shape in self.input_shape.items():
                # Get the total number of elements in this group
                if len(shape) == 2:
                    size = shape[0] * shape[1]
                else:
                    size = np.prod(shape)

                self.group_sizes[group_name] = size

                # Create an embedding layer for this group
                self.group_embeddings[group_name] = nn.Sequential(
                    nn.Linear(size, self.hidden_dim // len(self.input_shape)),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate / 2)
                )
        else:
            # If input_shape is not a dict, handle as a single group
            total_size = np.prod(self.input_shape)
            self.group_sizes = {"all": total_size}
            self.group_embeddings["all"] = nn.Sequential(
                nn.Linear(total_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate / 2)
            )

        # Create shared layers after embedding
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc_out = nn.Linear(self.hidden_dim // 2, 2)  # loss and accuracy

    def _forward_cnn(self, x):
        """Forward pass for CNN architecture (2D grid)."""
        # Add batch and channel dimensions if needed
        if len(x.shape) == 2:
            # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Add channel dimension: [B, H, W] -> [B, 1, H, W]
            x = x.unsqueeze(1)

        # Apply convolution layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)

        return x

    def _forward_structured(self, x):
        """Forward pass for structured architecture."""
        # Process each parameter group
        if isinstance(x, dict):
            # List to store embeddings for each group
            embeddings = []

            for group_name, layer in self.group_embeddings.items():
                if group_name in x:
                    # Flatten the input
                    group_input = x[group_name].view(x[group_name].size(0), -1)
                    # Apply embedding
                    group_embedding = layer(group_input)
                    embeddings.append(group_embedding)

            # Concatenate group embeddings if multiple groups
            if len(embeddings) > 1:
                x = torch.cat(embeddings, dim=1)
            elif len(embeddings) == 1:
                x = embeddings[0]
            else:
                raise ValueError("No valid parameter groups found in input")
        else:
            # Handle as a single group
            # Flatten the input
            x = x.view(x.size(0), -1)
            # Apply embedding
            x = self.group_embeddings["all"](x)

        # Apply shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)

        return x

    def forward(self, x):
        """Forward pass through the network."""
        return self.forward_fn(x)

    def train(self, train_loader, val_loader=None, num_epochs=50, lr=0.001, device=None):
        """
        Train the network analyzer.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on

        Returns:
            Training history
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # For tracking metrics
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            epoch_loss = 0.0

            for inputs, targets in train_loader:
                # Move data to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                targets = targets.to(device)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0) if isinstance(inputs, torch.Tensor) else loss.item()

            # Calculate average training loss
            avg_train_loss = epoch_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion, device)
                history['val_loss'].append(val_loss)

                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}')

        return history

    def evaluate(self, data_loader, criterion=None, device=None):
        """
        Evaluate the network analyzer.

        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function (default: MSELoss)
            device: Device to evaluate on

        Returns:
            Evaluation metrics
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if criterion is None:
            criterion = nn.MSELoss()

        self.to(device)
        self.eval()

        total_loss = 0.0
        predictions = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                targets = targets.to(device)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                # Accumulate loss
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size

                # Store predictions and targets
                predictions.append(outputs.cpu())
                targets_list.append(targets.cpu())

        # Concatenate predictions and targets
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)

        # Extract predicted and actual loss/accuracy
        pred_loss = all_predictions[:, 0]
        pred_accuracy = all_predictions[:, 1]
        actual_loss = all_targets[:, 0]
        actual_accuracy = all_targets[:, 1]

        # Calculate error metrics
        loss_mae = F.l1_loss(pred_loss, actual_loss).item()
        accuracy_mae = F.l1_loss(pred_accuracy, actual_accuracy).item()

        # Calculate R-squared (coefficient of determination)
        loss_r2 = 1 - torch.sum((actual_loss - pred_loss) ** 2) / torch.sum((actual_loss - actual_loss.mean()) ** 2)
        accuracy_r2 = 1 - torch.sum((actual_accuracy - pred_accuracy) ** 2) / torch.sum((actual_accuracy - actual_accuracy.mean()) ** 2)

        # Create metrics dictionary
        metrics = {
            'avg_loss': avg_loss,
            'loss_mae': loss_mae,
            'accuracy_mae': accuracy_mae,
            'loss_r2': loss_r2.item(),
            'accuracy_r2': accuracy_r2.item()
        }

        print(f'Evaluation: Loss MAE: {loss_mae:.4f}, Accuracy MAE: {accuracy_mae:.4f}')
        print(f'Evaluation: Loss R²: {loss_r2.item():.4f}, Accuracy R²: {accuracy_r2.item():.4f}')

        return metrics

    def analyze_parameter_importance(self, dataset, data_loader, device=None):
        """
        Analyze which parameters are most important for predicting performance.

        Args:
            dataset: NetworkDataset instance
            data_loader: DataLoader for data
            device: Device to run analysis on

        Returns:
            Parameter importance analysis results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
        self.eval()

        # Get baseline predictions
        baseline_preds = []
        baseline_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self(inputs)
                baseline_preds.append(outputs.cpu())
                baseline_targets.append(targets)

        baseline_preds = torch.cat(baseline_preds, dim=0)
        baseline_targets = torch.cat(baseline_targets, dim=0)
        baseline_loss = F.mse_loss(baseline_preds, baseline_targets).item()

        print(f"Baseline loss: {baseline_loss:.6f}")

        # Analyze importance differently based on representation type
        if dataset.representation_type == '2d':
            # For 2D grid, use occlusion sensitivity
            return self._analyze_2d_importance(dataset, data_loader, baseline_preds, device)
        else:
            # For structured representation, perturb each parameter group
            return self._analyze_structured_importance(dataset, data_loader, baseline_preds, device)

    def _analyze_2d_importance(self, dataset, data_loader, baseline_preds, device):
        """
        Analyze importance for 2D grid representation using occlusion.

        Args:
            dataset: NetworkDataset instance
            data_loader: DataLoader for data
            baseline_preds: Baseline predictions
            device: Device to run analysis on

        Returns:
            Importance map
        """
        # Get the shape of the 2D representation
        input_shape = dataset.get_representation_shape()
        importance_map = torch.zeros(input_shape)

        # Define patch size for occlusion (adjust based on input size)
        patch_size = max(1, min(input_shape) // 10)
        patch_stride = max(1, patch_size // 2)

        print(f"Analyzing importance with patch size {patch_size} and stride {patch_stride}...")

        # Track progress
        total_patches = ((input_shape[0] - patch_size) // patch_stride + 1) * ((input_shape[1] - patch_size) // patch_stride + 1)
        patches_done = 0

        # Iterate over patches in the input
        for i in range(0, input_shape[0] - patch_size + 1, patch_stride):
            for j in range(0, input_shape[1] - patch_size + 1, patch_stride):
                # Create occluded versions of all inputs
                occluded_inputs = []

                for orig_input, _ in data_loader.dataset:
                    # Clone the input
                    occluded = orig_input.clone()

                    # Set patch to zero (occlude it)
                    occluded[i:i+patch_size, j:j+patch_size] = 0

                    occluded_inputs.append(occluded)

                # Convert to batch tensor
                occluded_tensor = torch.stack(occluded_inputs).to(device)

                # Get predictions with occluded inputs
                with torch.no_grad():
                    occluded_preds = self(occluded_tensor)

                # Calculate how much the predictions changed
                pred_diff = F.mse_loss(occluded_preds.cpu(), baseline_preds).item()

                # Assign importance to this patch (higher diff = more important)
                importance_map[i:i+patch_size, j:j+patch_size] += pred_diff

                # Update progress
                patches_done += 1
                if patches_done % 10 == 0:
                    print(f"Processed {patches_done}/{total_patches} patches...")

        # Normalize importance map
        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()

        return importance_map

    def _analyze_structured_importance(self, dataset, data_loader, baseline_preds, device):
        """
        Analyze importance for structured representation by perturbing groups.

        Args:
            dataset: NetworkDataset instance
            data_loader: DataLoader for data
            baseline_preds: Baseline predictions
            device: Device to run analysis on

        Returns:
            Dictionary of importance scores by parameter group
        """
        importance_scores = {}

        # Get the first input to determine structure
        first_input, _ = next(iter(data_loader.dataset))

        if isinstance(first_input, dict):
            # For structured representation, analyze each group
            for group_name in first_input.keys():
                print(f"Analyzing importance of group: {group_name}")

                # Create perturbed versions of all inputs
                perturbed_preds = []

                for inputs, _ in data_loader:
                    if isinstance(inputs, dict):
                        # Clone the inputs
                        perturbed = {k: v.clone() for k, v in inputs.items()}

                        # Set this group to zero
                        perturbed[group_name] = torch.zeros_like(perturbed[group_name])

                        # Move to device
                        perturbed = {k: v.to(device) for k, v in perturbed.items()}

                        # Get predictions
                        with torch.no_grad():
                            outputs = self(perturbed)
                            perturbed_preds.append(outputs.cpu())

                # Concatenate predictions
                all_perturbed_preds = torch.cat(perturbed_preds, dim=0)

                # Calculate how much the predictions changed
                pred_diff = F.mse_loss(all_perturbed_preds, baseline_preds).item()

                # Higher difference means this group is more important
                importance_scores[group_name] = pred_diff
        else:
            # For 2D representation that is being treated as structured,
            # we need a different approach
            # This is a placeholder - we should not reach here
            pass

        # Normalize importance scores
        max_score = max(importance_scores.values())
        if max_score > 0:
            importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return importance_scores
