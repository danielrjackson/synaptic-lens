"""
Visualization Utility Methods

This module contains utility functions for visualizing neural network predictions,
parameter importance, and training progress.

Copyright 2025 Daniel Robert Jackson
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns

def visualize_predictions(analyzer, data_loader, device=None, output_dir=None):
    """
    Visualize the predictions of the neural network analyzer.

    Args:
        analyzer: Trained NetworkAnalyzer instance
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        output_dir: Directory to save visualization

    Returns:
        Tuple of predictions and targets
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    analyzer.to(device)
    analyzer.eval()

    # Lists to store predictions and targets
    all_predictions = []
    all_targets = []

    # Get predictions
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move data to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = analyzer(inputs)

            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss predictions
    axes[0].scatter(targets[:, 0], predictions[:, 0], alpha=0.7)

    # Add perfect prediction line
    min_loss = min(targets[:, 0].min(), predictions[:, 0].min())
    max_loss = max(targets[:, 0].max(), predictions[:, 0].max())
    axes[0].plot([min_loss, max_loss], [min_loss, max_loss], 'r--')

    axes[0].set_xlabel('Actual Loss')
    axes[0].set_ylabel('Predicted Loss')
    axes[0].set_title('Neural Network Loss Prediction')

    # Calculate R² for loss
    loss_corr = np.corrcoef(targets[:, 0], predictions[:, 0])[0, 1]
    loss_r2 = loss_corr ** 2
    axes[0].text(0.05, 0.95, f'R² = {loss_r2:.4f}', transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Plot accuracy predictions
    axes[1].scatter(targets[:, 1], predictions[:, 1], alpha=0.7)

    # Add perfect prediction line
    min_acc = min(targets[:, 1].min(), predictions[:, 1].min())
    max_acc = max(targets[:, 1].max(), predictions[:, 1].max())
    axes[1].plot([min_acc, max_acc], [min_acc, max_acc], 'r--')

    axes[1].set_xlabel('Actual Accuracy')
    axes[1].set_ylabel('Predicted Accuracy')
    axes[1].set_title('Neural Network Accuracy Prediction')

    # Calculate R² for accuracy
    acc_corr = np.corrcoef(targets[:, 1], predictions[:, 1])[0, 1]
    acc_r2 = acc_corr ** 2
    axes[1].text(0.05, 0.95, f'R² = {acc_r2:.4f}', transform=axes[1].transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_dir / f"predictions_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction visualization to {fig_path}")

    plt.show()

    return predictions, targets


def plot_parameter_importance(importance, representation_type, output_dir=None):
    """
    Visualize the importance of neural network parameters.

    Args:
        importance: Importance analysis results (map or dictionary)
        representation_type: Type of representation ('2d' or 'structured')
        output_dir: Directory to save visualization
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    if representation_type == '2d' and isinstance(importance, torch.Tensor):
        # For 2D grid representation
        plt.figure(figsize=(10, 8))

        # Convert to numpy if needed
        if isinstance(importance, torch.Tensor):
            importance = importance.numpy()

        # Create heatmap
        plt.imshow(importance, cmap='viridis')
        plt.colorbar(label='Importance Score')
        plt.title('Parameter Importance Map')

        # Save figure if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = output_dir / f"importance_map_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved importance map to {fig_path}")

        plt.show()

    elif representation_type == 'structured' and isinstance(importance, dict):
        # For structured representation
        plt.figure(figsize=(12, 6))

        # Sort groups by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        group_names = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        # Create bar chart
        bars = plt.barh(range(len(group_names)), scores, align='center')
        plt.yticks(range(len(group_names)), group_names)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(max(score + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', va='center')

        plt.xlabel('Importance Score')
        plt.title('Parameter Group Importance Analysis')
        plt.tight_layout()

        # Save figure if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = output_dir / f"parameter_importance_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved parameter importance chart to {fig_path}")

        plt.show()
    else:
        print(f"Unsupported importance format or representation type: {representation_type}")


def visualize_parameter_distributions(model_data, output_dir=None):
    """
    Visualize the distributions of parameters across neural networks.

    Args:
        model_data: List of model data entries
        output_dir: Directory to save visualizations
    """
    if not model_data:
        print("No model data provided")
        return

    # Extract parameter names from the first model
    first_model = model_data[0]
    param_names = list(first_model["params"].keys())

    # Organize by parameter type
    weight_params = [name for name in param_names if 'weight' in name]
    bias_params = [name for name in param_names if 'bias' in name]
    other_params = [name for name in param_names if 'weight' not in name and 'bias' not in name]

    # Create figure directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        fig_dir = output_dir / "parameter_distributions"
        fig_dir.mkdir(exist_ok=True, parents=True)

    # Function to plot parameter distributions
    def plot_param_distributions(param_name):
        # Collect parameter values across models
        values = []
        performances = []

        for model in model_data:
            param = model["params"][param_name]
            values.append(param.flatten().numpy())
            performances.append(model["performance"]["accuracy"])

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot parameter distributions
        for i, (vals, acc) in enumerate(zip(values, performances)):
            if i % max(1, len(values) // 10) == 0:  # Plot a subset to avoid cluttering
                sns.kdeplot(vals, ax=axes[0], alpha=0.5, label=f'Model {i} (Acc: {acc:.4f})')

        # Plot parameter statistics across all models
        means = [torch.mean(model["params"][param_name]).item() for model in model_data]
        stds = [torch.std(model["params"][param_name]).item() for model in model_data]
        accuracies = [model["performance"]["accuracy"] for model in model_data]

        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]
        sorted_accs = [accuracies[i] for i in sorted_indices]

        # Plot means vs accuracy
        axes[1].scatter(sorted_accs, sorted_means, label='Mean', alpha=0.7)
        axes[1].scatter(sorted_accs, sorted_stds, label='Std Dev', alpha=0.7, marker='x')

        # Set labels and title
        axes[0].set_title(f'Distribution of {param_name}')
        axes[0].set_xlabel('Parameter Value')
        axes[0].set_ylabel('Density')
        axes[0].legend(loc='upper right')

        axes[1].set_title(f'Statistics of {param_name} vs Accuracy')
        axes[1].set_xlabel('Model Accuracy')
        axes[1].set_ylabel('Parameter Statistic')
        axes[1].legend()

        plt.tight_layout()

        # Save figure if output directory is provided
        if output_dir is not None:
            safe_name = param_name.replace('.', '_')
            fig_path = fig_dir / f"{safe_name}_distribution.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close()

    # Plot distributions for selected parameters
    print("Generating parameter distribution visualizations...")

    # Plot for all weight parameters
    for param_name in weight_params:
        plot_param_distributions(param_name)

    # Plot for all bias parameters
    for param_name in bias_params:
        plot_param_distributions(param_name)

    print(f"Generated parameter distribution visualizations for {len(weight_params)} weights and {len(bias_params)} biases")


def visualize_training_progress(models_data, output_dir=None):
    """
    Visualize how model performance changes during training.

    Args:
        models_data: List of model data entries
        output_dir: Directory to save visualization
    """
    # Extract epoch, batch, and performance information
    epochs = []
    batches = []
    losses = []
    accuracies = []

    for model_data in models_data:
        info = model_data["info"]
        performance = model_data["performance"]

        epochs.append(info["epoch"])
        batches.append(info["batch"] if info["batch"] is not None else -1)  # -1 for final models
        losses.append(performance["loss"])
        accuracies.append(performance["accuracy"])

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot loss over epochs
    axes[0].scatter(epochs, losses, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Progression During Training')

    # Plot accuracy over epochs
    axes[1].scatter(epochs, accuracies, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Progression During Training')

    plt.tight_layout()

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        fig_path = output_dir / "training_progress.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved training progress visualization to {fig_path}")

    plt.show()
