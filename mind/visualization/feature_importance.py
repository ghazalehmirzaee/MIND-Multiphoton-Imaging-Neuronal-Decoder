# mind/visualization/feature_importance.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os


def plot_feature_importance_heatmap(feature_importances: np.ndarray,
                                    window_size: int,
                                    num_neurons: int,
                                    title: str = 'Feature Importance Heatmap',
                                    output_dir: Optional[str] = None,
                                    save_filename: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot feature importance heatmap.

    Parameters
    ----------
    feature_importances : np.ndarray
        Feature importance values
    window_size : int
        Size of the sliding window
    num_neurons : int
        Number of neurons
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Reshape feature importances to (window_size, num_neurons)
    feature_importances_reshaped = feature_importances.reshape(window_size, num_neurons)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(feature_importances_reshaped, aspect='auto', cmap='viridis')

    # Set labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Neuron Index', fontsize=14)
    ax.set_ylabel('Time Step', fontsize=14)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_temporal_feature_importance(feature_importances: np.ndarray,
                                     window_size: int,
                                     num_neurons: int,
                                     title: str = 'Temporal Feature Importance',
                                     output_dir: Optional[str] = None,
                                     save_filename: Optional[str] = None,
                                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot temporal feature importance.

    Parameters
    ----------
    feature_importances : np.ndarray
        Feature importance values
    window_size : int
        Size of the sliding window
    num_neurons : int
        Number of neurons
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Reshape feature importances to (window_size, num_neurons)
    feature_importances_reshaped = feature_importances.reshape(window_size, num_neurons)

    # Compute mean importance across neurons for each time step
    temporal_importance = np.mean(feature_importances_reshaped, axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bar chart
    ax.bar(range(window_size), temporal_importance, color='skyblue')

    # Set labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Mean Feature Importance', fontsize=14)

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_neuron_feature_importance(feature_importances: np.ndarray,
                                   window_size: int,
                                   num_neurons: int,
                                   top_n: int = 20,
                                   title: str = 'Top Neuron Feature Importance',
                                   output_dir: Optional[str] = None,
                                   save_filename: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot feature importance for top neurons.

    Parameters
    ----------
    feature_importances : np.ndarray
        Feature importance values
    window_size : int
        Size of the sliding window
    num_neurons : int
        Number of neurons
    top_n : int, optional
        Number of top neurons to plot
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Reshape feature importances to (window_size, num_neurons)
    feature_importances_reshaped = feature_importances.reshape(window_size, num_neurons)

    # Compute mean importance across time steps for each neuron
    neuron_importance = np.mean(feature_importances_reshaped, axis=0)

    # Get indices of top neurons
    top_neuron_indices = np.argsort(neuron_importance)[-top_n:][::-1]
    top_neuron_importances = neuron_importance[top_neuron_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal bar chart
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_neuron_importances, color='skyblue')

    # Set labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Mean Feature Importance', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Neuron {idx}' for idx in top_neuron_indices])

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_all_feature_importances(all_feature_importances: Dict[str, Dict[str, np.ndarray]],
                                 window_size: int,
                                 num_neurons: int,
                                 output_dir: Optional[str] = None,
                                 save_filename_prefix: Optional[str] = 'feature_importance',
                                 figsize: Tuple[int, int] = (20, 15)) -> None:
    """
    Plot all feature importance visualizations for each model and signal type.

    Parameters
    ----------
    all_feature_importances : Dict[str, Dict[str, np.ndarray]]
        Nested dictionary with structure {signal_type: {model_name: feature_importances}}
    window_size : int
        Size of the sliding window
    num_neurons : int
        Number of neurons
    output_dir : str, optional
        Directory to save the plots
    save_filename_prefix : str, optional
        Prefix for the saved filenames
    figsize : Tuple[int, int], optional
        Figure size
    """
    # Extract signal types and model names
    signal_types = list(all_feature_importances.keys())
    model_names = list(all_feature_importances[signal_types[0]].keys())

    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Plot heatmaps
    for signal_type in signal_types:
        for model_name in model_names:
            feature_importances = all_feature_importances[signal_type][model_name]

            # Plot heatmap
            title = f'Feature Importance Heatmap - {model_name} - {signal_type}'
            save_filename = f'{save_filename_prefix}_heatmap_{model_name}_{signal_type}.png'
            plot_feature_importance_heatmap(
                feature_importances, window_size, num_neurons,
                title=title, output_dir=output_dir, save_filename=save_filename
            )

            # Plot temporal importance
            title = f'Temporal Feature Importance - {model_name} - {signal_type}'
            save_filename = f'{save_filename_prefix}_temporal_{model_name}_{signal_type}.png'
            plot_temporal_feature_importance(
                feature_importances, window_size, num_neurons,
                title=title, output_dir=output_dir, save_filename=save_filename
            )

            # Plot neuron importance
            title = f'Top Neuron Feature Importance - {model_name} - {signal_type}'
            save_filename = f'{save_filename_prefix}_neuron_{model_name}_{signal_type}.png'
            plot_neuron_feature_importance(
                feature_importances, window_size, num_neurons,
                title=title, output_dir=output_dir, save_filename=save_filename
            )

    # Also create comparison plots

    # Plot temporal importance comparison across signal types for each model
    for model_name in model_names:
        fig, axes = plt.subplots(1, len(signal_types), figsize=figsize, sharey=True)

        # Make sure axes is a list
        if len(signal_types) == 1:
            axes = [axes]

        for i, (signal_type, ax) in enumerate(zip(signal_types, axes)):
            feature_importances = all_feature_importances[signal_type][model_name]
            feature_importances_reshaped = feature_importances.reshape(window_size, num_neurons)
            temporal_importance = np.mean(feature_importances_reshaped, axis=1)

            # Plot bar chart
            ax.bar(range(window_size), temporal_importance, color='skyblue')

            # Set labels and title
            ax.set_title(f'{signal_type}', fontsize=14)
            ax.set_xlabel('Time Step', fontsize=12)
            if i == 0:
                ax.set_ylabel('Mean Feature Importance', fontsize=12)

        # Set overall title
        fig.suptitle(f'Temporal Feature Importance Comparison - {model_name}', fontsize=16)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save figure
        if output_dir is not None and save_filename_prefix is not None:
            save_path = os.path.join(output_dir,
                                     f'{save_filename_prefix}_temporal_comparison_{model_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

