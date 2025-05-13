"""
Signal visualization utilities for calcium imaging data.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_signal_comparison(calcium_signal: np.ndarray,
                           deltaf_signal: np.ndarray,
                           deconv_signal: np.ndarray,
                           neuron_indices: List[int],
                           output_dir: Optional[Union[str, Path]] = None,
                           n_frames: int = 500,
                           fig_size: Tuple[int, int] = (15, 15)) -> plt.Figure:
    """
    Plot a comparison of the three signal types for selected neurons.

    Parameters
    ----------
    calcium_signal : np.ndarray
        Raw calcium signal, shape (n_frames, n_neurons)
    deltaf_signal : np.ndarray
        ΔF/F signal, shape (n_frames, n_neurons)
    deconv_signal : np.ndarray
        Deconvolved signal, shape (n_frames, n_neurons)
    neuron_indices : List[int]
        Indices of neurons to plot
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    n_frames : int, optional
        Number of frames to plot, by default 500
    fig_size : Tuple[int, int], optional
        Figure size, by default (15, 15)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting signal comparison for {len(neuron_indices)} neurons")

    # Create figure
    fig, axes = plt.subplots(len(neuron_indices), 3, figsize=fig_size, sharex=True)

    # Set titles for columns
    axes[0, 0].set_title("Raw Calcium Signal")
    axes[0, 1].set_title("ΔF/F Signal")
    axes[0, 2].set_title("Deconvolved Signal")

    # Plot each neuron
    for i, neuron_idx in enumerate(neuron_indices):
        # Make sure neuron index is valid
        if neuron_idx >= calcium_signal.shape[1] or neuron_idx >= deltaf_signal.shape[1] or neuron_idx >= \
                deconv_signal.shape[1]:
            logger.warning(f"Neuron index {neuron_idx} is out of bounds")
            continue

        # Get data for this neuron
        calcium_trace = calcium_signal[:n_frames, neuron_idx]
        deltaf_trace = deltaf_signal[:n_frames, neuron_idx]
        deconv_trace = deconv_signal[:n_frames, neuron_idx]

        # Plot raw calcium signal
        axes[i, 0].plot(calcium_trace, 'b-')
        axes[i, 0].set_ylabel(f"Neuron {neuron_idx}")

        # Plot ΔF/F signal
        axes[i, 1].plot(deltaf_trace, 'g-')

        # Plot deconvolved signal
        axes[i, 2].plot(deconv_trace, 'r-')

    # Set x-label for bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Frame")

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / "signal_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved signal comparison to {output_dir / 'signal_comparison.png'}")

    return fig


def plot_activity_heatmap(signal: np.ndarray,
                          neuron_indices: List[int],
                          signal_name: str,
                          output_dir: Optional[Union[str, Path]] = None,
                          n_frames: int = 2000,
                          fig_size: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot a heatmap of neural activity.

    Parameters
    ----------
    signal : np.ndarray
        Neural activity data, shape (n_frames, n_neurons)
    neuron_indices : List[int]
        Indices of neurons to plot
    signal_name : str
        Name of the signal type
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    n_frames : int, optional
        Number of frames to plot, by default 2000
    fig_size : Tuple[int, int], optional
        Figure size, by default (15, 8)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting activity heatmap for {signal_name}")

    # Extract data for selected neurons and frames
    data = signal[:n_frames, neuron_indices].T

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot heatmap
    sns.heatmap(data, cmap='viridis', xticklabels=100, yticklabels=neuron_indices, ax=ax)

    # Set labels
    ax.set_xlabel("Frame")
    ax.set_ylabel("Neuron")
    ax.set_title(f"{signal_name} - Activity Heatmap")

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"heatmap_{signal_name.lower().replace('/', '')}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved activity heatmap to {output_dir / f'heatmap_{signal_name.lower().replace('/', '')}.png'}")

    return fig


def plot_aligned_activity(signal: np.ndarray,
                          labels: np.ndarray,
                          neuron_idx: int,
                          signal_name: str,
                          output_dir: Optional[Union[str, Path]] = None,
                          window_size: int = 50,
                          fig_size: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot neural activity aligned with behavior.

    Parameters
    ----------
    signal : np.ndarray
        Neural activity data, shape (n_frames, n_neurons)
    labels : np.ndarray
        Behavior labels, shape (n_frames,)
    neuron_idx : int
        Index of the neuron to plot
    signal_name : str
        Name of the signal type
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    window_size : int, optional
        Size of the window around each footstep event, by default 50
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 6)

    Returns
    -------
    Tuple[plt.Figure, plt.Figure]
        The figure objects (full trace, event-triggered average)
    """
    logger.info(f"Plotting aligned activity for {signal_name} - Neuron {neuron_idx}")

    # Extract activity for the selected neuron
    activity = signal[:, neuron_idx]

    # Find footstep events (transitions from 0 to 1)
    event_frames = np.where(np.diff(labels) == 1)[0] + 1

    # Create figure for full trace
    fig1, ax1 = plt.subplots(figsize=fig_size)

    # Plot full activity trace
    ax1.plot(activity, 'gray', alpha=0.5, label='Activity')

    # Highlight footstep events
    for frame in event_frames:
        ax1.axvspan(frame, frame + labels[frame:].tolist().index(0) if 0 in labels[frame:] else len(labels),
                    alpha=0.2, color='red')

    # Set labels
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Activity")
    ax1.set_title(f"{signal_name} - Neuron {neuron_idx} Activity Aligned with Footsteps")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.2, label='Footstep')]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig1.savefig(output_dir / f"aligned_activity_{signal_name.lower().replace('/', '')}_neuron{neuron_idx}.png",
                     dpi=300, bbox_inches='tight')
        logger.info(
            f"Saved aligned activity to {output_dir / f'aligned_activity_{signal_name.lower().replace('/', '')}_neuron{neuron_idx}.png'}")

    # Create figure for event-triggered average
    fig2, ax2 = plt.subplots(figsize=fig_size)

    # Calculate event-triggered average
    event_windows = []
    for frame in event_frames:
        if frame - window_size // 2 >= 0 and frame + window_size // 2 < len(activity):
            window = activity[frame - window_size // 2:frame + window_size // 2]
            event_windows.append(window)

    if event_windows:
        event_windows = np.array(event_windows)
        avg_window = np.mean(event_windows, axis=0)
        std_window = np.std(event_windows, axis=0)

        # Plot average and standard deviation
        times = np.arange(-window_size // 2, window_size // 2)
        ax2.plot(times, avg_window, 'b', label='Average')
        ax2.fill_between(times, avg_window - std_window, avg_window + std_window, alpha=0.3, color='b')

        # Add vertical line at event onset
        ax2.axvline(x=0, color='r', linestyle='--', label='Event onset')

        # Set labels
        ax2.set_xlabel("Frames relative to footstep")
        ax2.set_ylabel("Activity")
        ax2.set_title(f"{signal_name} - Neuron {neuron_idx} Event-Triggered Average")

        # Add legend
        ax2.legend()

        # Save figure if output directory is provided
        if output_dir is not None:
            fig2.savefig(
                output_dir / f"event_triggered_avg_{signal_name.lower().replace('/', '')}_neuron{neuron_idx}.png",
                dpi=300, bbox_inches='tight')
            logger.info(
                f"Saved event-triggered average to {output_dir / f'event_triggered_avg_{signal_name.lower().replace('/', '')}_neuron{neuron_idx}.png'}")
    else:
        logger.warning(f"No valid event windows found for {signal_name} - Neuron {neuron_idx}")

    return fig1, fig2


def find_most_active_neurons(signal: np.ndarray, n: int = 20) -> np.ndarray:
    """
    Find the n most active neurons based on mean activity.

    Parameters
    ----------
    signal : np.ndarray
        Neural activity data, shape (n_frames, n_neurons)
    n : int, optional
        Number of neurons to return, by default 20

    Returns
    -------
    np.ndarray
        Indices of the n most active neurons
    """
    logger.info(f"Finding {n} most active neurons")

    # Calculate mean activity for each neuron
    mean_activity = np.mean(signal, axis=0)

    # Find indices of top n neurons
    top_indices = np.argsort(mean_activity)[::-1][:n]

    return top_indices


def find_most_informative_neurons(signal: np.ndarray, labels: np.ndarray, n: int = 20) -> np.ndarray:
    """
    Find the n most informative neurons based on correlation with behavior.

    Parameters
    ----------
    signal : np.ndarray
        Neural activity data, shape (n_frames, n_neurons)
    labels : np.ndarray
        Behavior labels, shape (n_frames,)
    n : int, optional
        Number of neurons to return, by default 20

    Returns
    -------
    np.ndarray
        Indices of the n most informative neurons
    """
    logger.info(f"Finding {n} most informative neurons")

    # Calculate correlation with behavior for each neuron
    correlations = np.zeros(signal.shape[1])

    for i in range(signal.shape[1]):
        corr = np.corrcoef(signal[:, i], labels)[0, 1]
        correlations[i] = np.abs(corr)  # Use absolute correlation

    # Find indices of top n neurons
    top_indices = np.argsort(correlations)[::-1][:n]

    return top_indices

