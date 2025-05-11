import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os


def visualize_signals(data: Dict[str, np.ndarray],
                      num_neurons: int = 20,
                      time_range: Optional[Tuple[int, int]] = None,
                      neuron_indices: Optional[List[int]] = None,
                      output_dir: Optional[str] = None,
                      save_filename: Optional[str] = None,
                      figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Visualize different types of calcium signals.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing calcium signal data
    num_neurons : int, optional
        Number of neurons to visualize
    time_range : Tuple[int, int], optional
        Range of time frames to visualize
    neuron_indices : List[int], optional
        Indices of specific neurons to visualize (overrides num_neurons)
    output_dir : str, optional
        Directory to save the visualization
    save_filename : str, optional
        Filename to save the visualization
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if required signal types are present
    signal_types = ['calcium_signal', 'deltaf', 'deconv']
    missing_signals = [s for s in signal_types if s not in data or data[s] is None]
    if missing_signals:
        print(f"Warning: Missing signal types: {missing_signals}")

    # Extract available signals
    available_signals = {s: data[s] for s in signal_types if s in data and data[s] is not None}
    if not available_signals:
        raise ValueError("No signal data available for visualization")

    # Determine the number of frames and neurons
    num_frames, num_neurons_total = next(iter(available_signals.values())).shape

    # Set time range if not provided
    if time_range is None:
        time_range = (0, num_frames)

    # Select neurons to visualize
    if neuron_indices is None:
        # If no specific neurons are provided, select top neurons based on activity
        if 'deltaf' in available_signals:
            # Use ∆F/F to determine most active neurons
            neuron_activity = np.std(available_signals['deltaf'], axis=0)
            top_neuron_indices = np.argsort(neuron_activity)[-num_neurons:][::-1]
        else:
            # Otherwise, just use the first num_neurons
            top_neuron_indices = np.arange(min(num_neurons, num_neurons_total))
    else:
        top_neuron_indices = neuron_indices

    # Create a figure with one subplot per signal type
    fig, axes = plt.subplots(len(available_signals), 1, figsize=figsize, sharex=True)

    # If there's only one signal type, ensure axes is a list
    if len(available_signals) == 1:
        axes = [axes]

    # Plot each signal type
    for i, (signal_type, signal_data) in enumerate(available_signals.items()):
        ax = axes[i]

        # Extract the data for selected neurons and time range
        selected_data = signal_data[time_range[0]:time_range[1], top_neuron_indices].T

        # Set a different color for each neuron
        cmap = plt.cm.get_cmap('tab10', len(top_neuron_indices))
        colors = [cmap(j) for j in range(len(top_neuron_indices))]

        # Plot each neuron's trace
        for j, neuron_idx in enumerate(top_neuron_indices):
            # Scale and offset each trace for better visualization
            trace = selected_data[j]

            # Plot the trace
            ax.plot(range(time_range[0], time_range[1]), trace, color=colors[j],
                    label=f'Neuron {neuron_idx}', linewidth=1)

        # Set subplot title and labels
        title_map = {
            'calcium_signal': 'Raw Calcium Signal',
            'deltaf': 'ΔF/F Signal',
            'deconv': 'Deconvolved Signal'
        }
        ax.set_title(title_map.get(signal_type, signal_type), fontsize=16)
        ax.set_ylabel('Amplitude', fontsize=14)

        # Add a legend for the first few neurons to avoid overcrowding
        if i == 0:
            # Only show legend for the first few neurons
            handles, labels = ax.get_legend_handles_labels()
            max_legend_items = min(10, len(top_neuron_indices))
            ax.legend(handles[:max_legend_items], labels[:max_legend_items],
                      fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')

    # Set common x-axis label
    axes[-1].set_xlabel('Time Frame', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def visualize_signal_comparison(data: Dict[str, np.ndarray],
                                num_neurons: int = 5,
                                neuron_indices: Optional[List[int]] = None,
                                output_dir: Optional[str] = None,
                                save_filename: Optional[str] = "signal_comparison.png",
                                figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Visualize comparison of different signal types for the same neurons.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing calcium signal data
    num_neurons : int, optional
        Number of neurons to visualize
    neuron_indices : List[int], optional
        Indices of specific neurons to visualize (overrides num_neurons)
    output_dir : str, optional
        Directory to save the visualization
    save_filename : str, optional
        Filename to save the visualization
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if required signal types are present
    signal_types = ['calcium_signal', 'deltaf', 'deconv']
    missing_signals = [s for s in signal_types if s not in data or data[s] is None]
    if missing_signals:
        print(f"Warning: Missing signal types: {missing_signals}")

    # Extract available signals
    available_signals = {s: data[s] for s in signal_types if s in data and data[s] is not None}
    if not available_signals:
        raise ValueError("No signal data available for visualization")

    # Determine the number of frames and neurons
    num_frames, num_neurons_total = next(iter(available_signals.values())).shape

    # Select neurons to visualize
    if neuron_indices is None:
        # If no specific neurons are provided, select top neurons based on activity
        if 'deltaf' in available_signals:
            # Use ∆F/F to determine most active neurons
            neuron_activity = np.std(available_signals['deltaf'], axis=0)
            top_neuron_indices = np.argsort(neuron_activity)[-num_neurons:][::-1]
        else:
            # Otherwise, just use the first num_neurons
            top_neuron_indices = np.arange(min(num_neurons, num_neurons_total))
    else:
        top_neuron_indices = neuron_indices

    # Create a figure with one subplot per neuron
    fig, axes = plt.subplots(num_neurons, 3, figsize=figsize, sharex='col', sharey='row')

    # Define subplot titles
    title_map = {
        'calcium_signal': 'Raw Calcium Signal',
        'deltaf': 'ΔF/F Signal',
        'deconv': 'Deconvolved Signal'
    }

    # Plot each neuron and signal type
    for i, neuron_idx in enumerate(top_neuron_indices):
        for j, (signal_type, signal_data) in enumerate(available_signals.items()):
            ax = axes[i, j]

            # Extract the data for the selected neuron
            trace = signal_data[:, neuron_idx]

            # Plot the trace
            ax.plot(trace, color='blue', linewidth=1)

            # Set titles and labels
            if i == 0:
                ax.set_title(title_map.get(signal_type, signal_type), fontsize=14)
            if j == 0:
                ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=12)

    # Set common x-axis label for bottom row
    for j in range(3):
        axes[-1, j].set_xlabel('Time Frame', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def visualize_signals_heatmap(data: Dict[str, np.ndarray],
                              num_neurons: int = 250,
                              neuron_indices: Optional[List[int]] = None,
                              output_dir: Optional[str] = None,
                              save_filename: Optional[str] = "signals_heatmap.png",
                              figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Visualize heatmap of different signal types.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing calcium signal data
    num_neurons : int, optional
        Number of neurons to visualize
    neuron_indices : List[int], optional
        Indices of specific neurons to visualize (overrides num_neurons)
    output_dir : str, optional
        Directory to save the visualization
    save_filename : str, optional
        Filename to save the visualization
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if required signal types are present
    signal_types = ['calcium_signal', 'deltaf', 'deconv']
    missing_signals = [s for s in signal_types if s not in data or data[s] is None]
    if missing_signals:
        print(f"Warning: Missing signal types: {missing_signals}")

    # Extract available signals
    available_signals = {s: data[s] for s in signal_types if s in data and data[s] is not None}
    if not available_signals:
        raise ValueError("No signal data available for visualization")

    # Determine the number of frames and neurons
    num_frames, num_neurons_total = next(iter(available_signals.values())).shape

    # Select neurons to visualize
    if neuron_indices is None:
        # If no specific neurons are provided, select top neurons based on activity
        if 'deltaf' in available_signals:
            # Use ∆F/F to determine most active neurons
            neuron_activity = np.std(available_signals['deltaf'], axis=0)
            top_neuron_indices = np.argsort(neuron_activity)[-num_neurons:][::-1]
        else:
            # Otherwise, just use the first num_neurons
            top_neuron_indices = np.arange(min(num_neurons, num_neurons_total))
    else:
        top_neuron_indices = neuron_indices

    # Create a figure with one subplot per signal type
    fig, axes = plt.subplots(1, len(available_signals), figsize=figsize)

    # If there's only one signal type, ensure axes is a list
    if len(available_signals) == 1:
        axes = [axes]

    # Define subplot titles
    title_map = {
        'calcium_signal': 'Raw Calcium Signal',
        'deltaf': 'ΔF/F Signal',
        'deconv': 'Deconvolved Signal'
    }

    # Plot each signal type as a heatmap
    for i, (signal_type, signal_data) in enumerate(available_signals.items()):
        ax = axes[i]

        # Extract the data for selected neurons
        selected_data = signal_data[:, top_neuron_indices]

        # Create heatmap
        im = ax.imshow(selected_data.T, aspect='auto', cmap='viridis')

        # Set titles and labels
        ax.set_title(title_map.get(signal_type, signal_type), fontsize=16)
        ax.set_xlabel('Time Frame', fontsize=14)
        ax.set_ylabel('Neuron Index', fontsize=14)

        # Add colorbar
        plt.colorbar(im, ax=ax)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig

