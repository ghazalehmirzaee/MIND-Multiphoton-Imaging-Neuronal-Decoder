"""
Feature importance visualization utilities for neural decoding models.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_temporal_importance(importance_matrix: np.ndarray,
                             model_name: str,
                             signal_name: str,
                             output_dir: Optional[Union[str, Path]] = None,
                             fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot temporal importance pattern.

    Parameters
    ----------
    importance_matrix : np.ndarray
        Feature importance matrix, shape (window_size, n_neurons)
    model_name : str
        Name of the model
    signal_name : str
        Name of the signal type
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (10, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting temporal importance for {model_name} on {signal_name}")

    # Calculate temporal importance by averaging across neurons
    temporal_importance = np.mean(importance_matrix, axis=1)

    # Normalize temporal importance
    if temporal_importance.sum() > 0:
        temporal_importance = temporal_importance / temporal_importance.sum()

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot temporal importance
    ax.bar(range(len(temporal_importance)), temporal_importance)

    # Set labels
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Importance")
    ax.set_title(f"Temporal Importance - {model_name} - {signal_name}")

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"temporal_importance_{model_name}_{signal_name}.png", dpi=300, bbox_inches='tight')
        logger.info(
            f"Saved temporal importance to {output_dir / f'temporal_importance_{model_name}_{signal_name}.png'}")

    return fig


def plot_neuron_importance(importance_matrix: np.ndarray,
                           top_neuron_indices: np.ndarray,
                           model_name: str,
                           signal_name: str,
                           output_dir: Optional[Union[str, Path]] = None,
                           n_top: int = 20,
                           fig_size: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot neuron importance for top neurons.

    Parameters
    ----------
    importance_matrix : np.ndarray
        Feature importance matrix, shape (window_size, n_neurons)
    top_neuron_indices : np.ndarray
        Indices of top neurons, shape (n_top,)
    model_name : str
        Name of the model
    signal_name : str
        Name of the signal type
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    n_top : int, optional
        Number of top neurons to plot, by default 20
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting neuron importance for {model_name} on {signal_name}")

    # Calculate neuron importance by averaging across time steps
    neuron_importance = np.mean(importance_matrix, axis=0)

    # Normalize neuron importance
    if neuron_importance.sum() > 0:
        neuron_importance = neuron_importance / neuron_importance.sum()

    # Get top neuron indices
    n_top = min(n_top, len(top_neuron_indices))
    top_indices = top_neuron_indices[:n_top]

    # Get importance values for top neurons
    top_importances = neuron_importance[top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create bar plot with colors based on importance
    bars = ax.bar(range(n_top), top_importances)

    # Set colors based on importance
    colors = plt.cm.viridis(top_importances / top_importances.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Set labels
    ax.set_xlabel("Neuron Rank")
    ax.set_ylabel("Importance")
    ax.set_title(f"Top {n_top} Neuron Importance - {model_name} - {signal_name}")
    ax.set_xticks(range(n_top))
    ax.set_xticklabels([f"N{idx}" for idx in top_indices], rotation=90)

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"neuron_importance_{model_name}_{signal_name}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved neuron importance to {output_dir / f'neuron_importance_{model_name}_{signal_name}.png'}")

    return fig


def plot_importance_heatmap(importance_matrix: np.ndarray,
                            model_name: str,
                            signal_name: str,
                            output_dir: Optional[Union[str, Path]] = None,
                            fig_size: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot feature importance heatmap.

    Parameters
    ----------
    importance_matrix : np.ndarray
        Feature importance matrix, shape (window_size, n_neurons)
    model_name : str
        Name of the model
    signal_name : str
        Name of the signal type
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 8)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting importance heatmap for {model_name} on {signal_name}")

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot heatmap
    sns.heatmap(importance_matrix, cmap='viridis', cbar_kws={'label': 'Importance'}, ax=ax)

    # Set labels
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Time Step")
    ax.set_title(f"Feature Importance Heatmap - {model_name} - {signal_name}")

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"importance_heatmap_{model_name}_{signal_name}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved importance heatmap to {output_dir / f'importance_heatmap_{model_name}_{signal_name}.png'}")

    return fig


def plot_importance_comparison(importance_matrices: Dict[str, np.ndarray],
                               model_name: str,
                               output_dir: Optional[Union[str, Path]] = None,
                               fig_size: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot comparison of importance patterns across signal types.

    Parameters
    ----------
    importance_matrices : Dict[str, np.ndarray]
        Dictionary of importance matrices for different signal types
    model_name : str
        Name of the model
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (15, 5)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting importance comparison for {model_name}")

    # Get signal types
    signal_types = list(importance_matrices.keys())

    # Calculate temporal importance for each signal type
    temporal_importances = {}
    for signal_type, matrix in importance_matrices.items():
        temporal_importance = np.mean(matrix, axis=1)
        if temporal_importance.sum() > 0:
            temporal_importance = temporal_importance / temporal_importance.sum()
        temporal_importances[signal_type] = temporal_importance

    # Create figure
    fig, axes = plt.subplots(1, len(signal_types), figsize=fig_size)

    # Handle case with only one signal type
    if len(signal_types) == 1:
        axes = [axes]

    # Plot temporal importance for each signal type
    for i, signal_type in enumerate(signal_types):
        axes[i].bar(range(len(temporal_importances[signal_type])), temporal_importances[signal_type])
        axes[i].set_title(f"{signal_type}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Importance")

    # Set overall title
    fig.suptitle(f"Temporal Importance Comparison - {model_name}", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"temporal_importance_comparison_{model_name}.png", dpi=300, bbox_inches='tight')
        logger.info(
            f"Saved temporal importance comparison to {output_dir / f'temporal_importance_comparison_{model_name}.png'}")

    return fig


def plot_cross_model_importance(temporal_importances: Dict[str, Dict[str, np.ndarray]],
                                signal_type: str,
                                output_dir: Optional[Union[str, Path]] = None,
                                fig_size: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot temporal importance patterns across different models for a specific signal type.

    Parameters
    ----------
    temporal_importances : Dict[str, Dict[str, np.ndarray]]
        Dictionary of temporal importances for different models and signal types
    signal_type : str
        Signal type to plot
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 8)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting cross-model importance for {signal_type}")

    # Get models
    models = list(temporal_importances.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot temporal importance for each model
    for model in models:
        if signal_type in temporal_importances[model]:
            ax.plot(temporal_importances[model][signal_type], label=model)

    # Set labels
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Importance")
    ax.set_title(f"Temporal Importance Patterns - {signal_type}")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"cross_model_importance_{signal_type}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved cross-model importance to {output_dir / f'cross_model_importance_{signal_type}.png'}")

    return fig


def find_important_time_windows(importance_matrix: np.ndarray,
                                percentile: float = 90) -> List[Tuple[int, int]]:
    """
    Find time windows with high feature importance.

    Parameters
    ----------
    importance_matrix : np.ndarray
        Feature importance matrix, shape (window_size, n_neurons)
    percentile : float, optional
        Percentile threshold for importance, by default 90

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) time windows
    """
    # Calculate temporal importance
    temporal_importance = np.mean(importance_matrix, axis=1)

    # Calculate threshold
    threshold = np.percentile(temporal_importance, percentile)

    # Find time points above threshold
    above_threshold = temporal_importance > threshold

    # Find contiguous segments
    segments = []
    start = None

    for i, above in enumerate(above_threshold):
        if above and start is None:
            start = i
        elif not above and start is not None:
            segments.append((start, i - 1))
            start = None

    # Handle case where the last segment extends to the end
    if start is not None:
        segments.append((start, len(above_threshold) - 1))

    logger.info(f"Found {len(segments)} important time windows with percentile {percentile}")

    return segments


