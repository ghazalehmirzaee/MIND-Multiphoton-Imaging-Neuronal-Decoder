"""Signal visualization functions."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


def plot_raw_signals(
        data: Dict[str, np.ndarray],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        num_neurons: int = 20,
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot raw signals for each signal type.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    signal_types : List[str], optional
        List of signal types to plot, by default ['calcium', 'deltaf', 'deconv']
    num_neurons : int, optional
        Number of neurons to plot, by default 20
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing raw signal figures
    """
    logger.info("Plotting raw signals")

    # Initialize figures dictionary
    figures = {}

    # Plot each signal type
    for signal_type in signal_types:
        # Get raw data key
        raw_key = f'raw_{signal_type}'

        if raw_key not in data:
            logger.warning(f"Raw data not found for {signal_type}")
            continue

        # Get raw data
        raw_data = data[raw_key]

        # Select neurons
        if 'top_neurons' in data and signal_type in data['top_neurons']:
            top_neurons = data['top_neurons'][signal_type]
            selected_neurons = top_neurons[:num_neurons]
        else:
            # Randomly select neurons
            np.random.seed(42)
            selected_neurons = np.random.choice(raw_data.shape[1], num_neurons, replace=False)

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot each selected neuron
        for i, neuron_idx in enumerate(selected_neurons):
            # Get normalized signal data
            signal = raw_data[:, neuron_idx]
            signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

            # Plot with vertical offset
            ax.plot(signal_norm + i, linewidth=1, alpha=0.8, label=f'Neuron {neuron_idx}')

        # Set title and labels
        ax.set_title(f'{signal_type.capitalize()} Signal - Top {num_neurons} Neurons')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Neuron (offset)')

        # Add y-tick labels for neurons
        ax.set_yticks(np.arange(len(selected_neurons)))
        ax.set_yticklabels([f'Neuron {int(n)}' for n in selected_neurons])

        # Save figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{signal_type}_raw_signals.png')
            plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{signal_type}_raw_signals'] = fig

    return figures

def plot_signal_heatmaps(data, signal_types=['calcium', 'deltaf', 'deconv'], num_neurons=250, output_dir=None):
    """Plot signal heatmaps for each signal type without any grids."""
    logger.info("Plotting signal heatmaps without grids")

    # Initialize figures dictionary
    figures = {}

    # Use a clean style without grid
    plt.style.use('default')

    # Plot each signal type
    for signal_type in signal_types:
        # Get raw data key
        raw_key = f'raw_{signal_type}'

        if raw_key not in data:
            logger.warning(f"Raw data not found for {signal_type}")
            continue

        # Get raw data
        raw_data = data[raw_key]

        # Select neurons
        if 'feature_importance' in data:
            # Try to get top neurons from feature importance
            fi_key = f"{signal_type}_rf"
            if fi_key in data['feature_importance']:
                neuron_importance = data['feature_importance'][fi_key].get('neuron_importance')
                if neuron_importance is not None:
                    top_neurons = np.argsort(neuron_importance)[-num_neurons:]
                    selected_neurons = top_neurons
                else:
                    # Randomly select neurons
                    np.random.seed(42)
                    selected_neurons = np.random.choice(raw_data.shape[1], min(num_neurons, raw_data.shape[1]),
                                                      replace=False)
            else:
                # Randomly select neurons
                np.random.seed(42)
                selected_neurons = np.random.choice(raw_data.shape[1], min(num_neurons, raw_data.shape[1]),
                                                  replace=False)
        else:
            # Randomly select neurons
            np.random.seed(42)
            selected_neurons = np.random.choice(raw_data.shape[1], min(num_neurons, raw_data.shape[1]), replace=False)

        # Get data for selected neurons
        selected_data = raw_data[:, selected_neurons]

        # Create figure without grid
        fig, ax = plt.subplots(figsize=(15, 10))

        # Turn off the grid explicitly
        ax.grid(False)

        # Remove all box lines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Plot heatmap without grid using imshow
        im = ax.imshow(selected_data.T, aspect='auto', cmap='viridis', interpolation='none')

        # Set title and labels
        ax.set_title(f'{signal_type.capitalize()} Signal Heatmap - Top {len(selected_neurons)} Neurons', fontsize=16)
        ax.set_xlabel('Time Frame', fontsize=14)
        ax.set_ylabel('Neuron Index', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Signal Intensity')

        # Save figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{signal_type}_heatmap.png')
            plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{signal_type}_heatmap'] = fig

    # Restore original style
    plt.style.use('default')

    return figures


def plot_signal_scatter(
        data: Dict[str, np.ndarray],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        num_neurons: int = 20,
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot signal scatter plots for each signal type.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    signal_types : List[str], optional
        List of signal types to plot, by default ['calcium', 'deltaf', 'deconv']
    num_neurons : int, optional
        Number of neurons to plot, by default 20
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing signal scatter figures
    """
    logger.info("Plotting signal scatter plots")

    # Initialize figures dictionary
    figures = {}

    # Plot each signal type
    for signal_type in signal_types:
        # Get raw data key
        raw_key = f'raw_{signal_type}'

        if raw_key not in data:
            logger.warning(f"Raw data not found for {signal_type}")
            continue

        # Get raw data
        raw_data = data[raw_key]

        # Select neurons
        if 'top_neurons' in data and signal_type in data['top_neurons']:
            top_neurons = data['top_neurons'][signal_type]
            selected_neurons = top_neurons[:num_neurons]
        else:
            # Randomly select neurons
            np.random.seed(42)
            selected_neurons = np.random.choice(raw_data.shape[1], num_neurons, replace=False)

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot each selected neuron
        for i, neuron_idx in enumerate(selected_neurons):
            # Get signal data
            signal = raw_data[:, neuron_idx]

            # Plot scatter with vertical offset
            ax.scatter(range(len(signal)), signal + i * np.max(signal), s=10, alpha=0.5, label=f'Neuron {neuron_idx}')

        # Set title and labels
        ax.set_title(f'{signal_type.capitalize()} Signal Scatter - Top {num_neurons} Neurons')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Signal Value (offset)')

        # Add legend
        ax.legend(loc='upper right')

        # Save figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{signal_type}_scatter.png')
            plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{signal_type}_scatter'] = fig

    return figures

def plot_signal_comparison(
        data: Dict[str, np.ndarray],
        output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of different signal types for the same neurons.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    plt.Figure
        Signal comparison figure
    """
    logger.info("Plotting signal comparison")

    # Check if all signal types are available
    signal_types = ['calcium', 'deltaf', 'deconv']
    for signal_type in signal_types:
        raw_key = f'raw_{signal_type}'
        if raw_key not in data:
            logger.warning(f"Raw data not found for {signal_type}")
            return None

    # Get raw data
    raw_calcium = data['raw_calcium']
    raw_deltaf = data['raw_deltaf']
    raw_deconv = data['raw_deconv']

    # Get valid neurons from deltaf and deconv
    if 'valid_neurons' in data:
        valid_neurons = data['valid_neurons']
    else:
        # Use intersection of neuron indices
        valid_neurons = np.arange(min(raw_calcium.shape[1], raw_deltaf.shape[1], raw_deconv.shape[1]))

    # Select a few example neurons
    np.random.seed(42)
    num_neurons = 5
    example_neurons = np.random.choice(len(valid_neurons), num_neurons, replace=False)

    # Create figure
    fig, axes = plt.subplots(num_neurons, 3, figsize=(18, 15), sharex=True)

    # Set titles for columns
    for i, signal_type in enumerate(signal_types):
        axes[0, i].set_title(f'{signal_type.capitalize()} Signal')

    # Plot each neuron
    for i, neuron_idx in enumerate(example_neurons):
        # Get the actual neuron index in each signal type
        calcium_idx = neuron_idx
        deltaf_idx = neuron_idx
        deconv_idx = neuron_idx

        # Plot raw calcium signal
        if calcium_idx < raw_calcium.shape[1]:
            axes[i, 0].plot(raw_calcium[:, calcium_idx], color='blue')
            axes[i, 0].set_ylabel(f'Neuron {calcium_idx}')

        # Plot ∆F/F signal
        if deltaf_idx < raw_deltaf.shape[1]:
            axes[i, 1].plot(raw_deltaf[:, deltaf_idx], color='orange')

        # Plot deconvolved signal
        if deconv_idx < raw_deconv.shape[1]:
            axes[i, 2].plot(raw_deconv[:, deconv_idx], color='green')

    # Add x-axis label to bottom row
    for i in range(3):
        axes[-1, i].set_xlabel('Time Frame')

    # Adjust layout
    plt.tight_layout()

    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'signal_type_comparison.png')
        plt.savefig(fig_path, dpi=300)

    return fig


def plot_signal_vertical_comparison(
        data: Dict[str, np.ndarray],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        num_neurons: int = 20,
        output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Plot vertical comparison of different signal types.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    signal_types : List[str], optional
        List of signal types to plot, by default ['calcium', 'deltaf', 'deconv']
    num_neurons : int, optional
        Number of neurons to include, by default 20
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    plt.Figure
        Vertical signal comparison figure
    """
    logger.info("Plotting vertical signal comparison")

    # Check if all signal types are available
    for signal_type in signal_types:
        raw_key = f'raw_{signal_type}'
        if raw_key not in data:
            logger.warning(f"Raw data not found for {signal_type}")
            return None

    # Get top neurons
    if 'feature_importance' in data:
        top_neurons = {}
        for signal_type in signal_types:
            fi_key = f"{signal_type}_rf"
            if fi_key in data['feature_importance']:
                neuron_importance = data['feature_importance'][fi_key].get('neuron_importance')
                if neuron_importance is not None:
                    top_neurons[signal_type] = np.argsort(neuron_importance)[-num_neurons:][::-1]

            if signal_type not in top_neurons:
                # Get raw data
                raw_key = f'raw_{signal_type}'
                raw_data = data[raw_key]

                # Randomly select neurons
                np.random.seed(42)
                top_neurons[signal_type] = np.random.choice(raw_data.shape[1], min(num_neurons, raw_data.shape[1]),
                                                            replace=False)
    else:
        top_neurons = {}
        for signal_type in signal_types:
            # Get raw data
            raw_key = f'raw_{signal_type}'
            raw_data = data[raw_key]

            # Randomly select neurons
            np.random.seed(42)
            top_neurons[signal_type] = np.random.choice(raw_data.shape[1], min(num_neurons, raw_data.shape[1]),
                                                        replace=False)

    # Create figure
    fig, axes = plt.subplots(len(signal_types), 1, figsize=(15, 5 * len(signal_types)), sharex=True)

    # Make axes indexable if only one signal type
    if len(signal_types) == 1:
        axes = [axes]

    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        # Get raw data
        raw_key = f'raw_{signal_type}'
        raw_data = data[raw_key]

        # Get selected neurons
        selected_neurons = top_neurons[signal_type]

        # Create a vertically offset plot for each neuron
        for j, neuron_idx in enumerate(selected_neurons):
            # Get normalized signal data
            signal = raw_data[:, neuron_idx]
            signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

            # Plot with vertical offset
            axes[i].plot(signal_norm + j, linewidth=1, alpha=0.8)

        # Set title and labels
        axes[i].set_title(f'{signal_type.capitalize()} Signal')
        axes[i].set_ylabel('Neuron (offset)')

        # Add y-tick labels for neurons
        axes[i].set_yticks(np.arange(len(selected_neurons)))
        axes[i].set_yticklabels([f'N{int(n)}' for n in selected_neurons])

    # Add x-axis label to bottom plot
    axes[-1].set_xlabel('Time Frame')

    # Adjust layout
    plt.tight_layout()

    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'vertical_signal_comparison.png')
        plt.savefig(fig_path, dpi=300)

    return fig

def create_signal_visualizations(
        data: Dict[str, Any],
        output_dir: str = 'results/figures'
) -> Dict[str, plt.Figure]:
    """
    Create all signal visualizations.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing data
    output_dir : str, optional
        Output directory, by default 'results/figures'

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing all signal visualization figures
    """
    logger.info("Creating all signal visualizations")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize figures dictionary
    figures = {}

    # Plot raw signals
    raw_figures = plot_raw_signals(data, output_dir=output_dir)
    figures.update(raw_figures)

    # Plot signal heatmaps
    heatmap_figures = plot_signal_heatmaps(data, output_dir=output_dir)
    figures.update(heatmap_figures)

    # Plot signal scatter plots
    scatter_figures = plot_signal_scatter(data, output_dir=output_dir)
    figures.update(scatter_figures)

    # Plot signal comparison
    signal_comparison = plot_signal_comparison(data, output_dir=output_dir)
    if signal_comparison is not None:
        figures['signal_comparison'] = signal_comparison

    # Plot vertical signal comparison
    vertical_comparison = plot_signal_vertical_comparison(data, output_dir=output_dir)
    if vertical_comparison is not None:
        figures['vertical_signal_comparison'] = vertical_comparison

    return figures

