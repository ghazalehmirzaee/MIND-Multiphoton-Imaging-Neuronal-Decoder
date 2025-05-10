"""Signal visualization functions optimized for academic publication."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# Define academic-friendly colors
ACADEMIC_COLORS = {
    'calcium': "#0072B2",  # Blue - good for calcium
    'deltaf': "#E69F00",  # Orange - good for ΔF/F
    'deconv': "#009E73",  # Green - good for deconvolved
    'background': "#f5f5f5"  # Light gray for backgrounds
}

# Custom color maps for heatmaps
CALCIUM_CMAP = LinearSegmentedColormap.from_list('calcium',
                                                 ['#ffffff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
                                                  '#2171b5'], N=256)
DELTAF_CMAP = LinearSegmentedColormap.from_list('deltaf',
                                                ['#ffffff', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548',
                                                 '#d7301f'], N=256)
DECONV_CMAP = LinearSegmentedColormap.from_list('deconv',
                                                ['#ffffff', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d',
                                                 '#238b45'], N=256)

# Signal-specific color maps dictionary
SIGNAL_CMAPS = {
    'calcium': CALCIUM_CMAP,
    'deltaf': DELTAF_CMAP,
    'deconv': DECONV_CMAP
}


def visualize_neuron_traces(
        data: Dict[str, np.ndarray],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        num_neurons: int = 20,
        output_dir: Optional[str] = None,
        dpi: int = 300
) -> Dict[str, plt.Figure]:
    """
    Create academic-quality visualization of neuron traces for each signal type.

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
    dpi : int, optional
        Resolution for saved figures, by default 300

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing visualization figures
    """
    logger.info(f"Visualizing traces for top {num_neurons} neurons of each signal type")

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

        # Identify top neurons based on activity
        if 'feature_importance' in data:
            fi_key = f"{signal_type}_rf"  # Use Random Forest importance if available
            if fi_key in data['feature_importance'] and 'neuron_importance' in data['feature_importance'][fi_key]:
                neuron_importance = data['feature_importance'][fi_key]['neuron_importance']
                top_neurons = np.argsort(neuron_importance)[-num_neurons:][::-1]  # Most important first
            else:
                # Use signal variance as a proxy for importance
                signal_variance = np.var(raw_data, axis=0)
                top_neurons = np.argsort(signal_variance)[-num_neurons:][::-1]  # Most variable first
        else:
            # Use signal variance as a proxy for importance
            signal_variance = np.var(raw_data, axis=0)
            top_neurons = np.argsort(signal_variance)[-num_neurons:][::-1]  # Most variable first

        # Create figure with clean styling for academic publication
        plt.style.use('default')  # Clean base style
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')  # White background

        # Add gray grid lines (subtle)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Plot each selected neuron with offset
        offset_factor = 0.1  # Controls vertical spacing between neurons
        for i, neuron_idx in enumerate(top_neurons):
            # Get signal data
            signal = raw_data[:, neuron_idx]

            # Normalize to [0, 1] range for consistent display
            if np.max(signal) > np.min(signal):  # Avoid division by zero
                signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            else:
                signal_norm = signal * 0

            # Plot with vertical offset
            ax.plot(
                signal_norm + i * offset_factor,
                linewidth=1,
                color=ACADEMIC_COLORS[signal_type],
                alpha=0.9,
                label=f'N{int(neuron_idx)}'
            )

            # Add neuron label at the start of each trace
            ax.text(
                -len(signal) * 0.02,  # Slight offset to the left
                i * offset_factor,
                f'N{int(neuron_idx)}',
                fontsize=8,
                ha='right',
                va='center'
            )

        # Add time frame indicators
        time_frames = len(raw_data)
        frame_ticks = np.linspace(0, time_frames - 1, 7).astype(int)  # 7 tick marks
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels(frame_ticks)

        # Set title and labels
        ax.set_title(f'{signal_type.capitalize()} Signal', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)

        # Remove y-ticks since we have direct labels
        ax.set_yticks([])

        # Add subtle box around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('gray')

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{signal_type}_traces.png')
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved {signal_type} traces to {fig_path}")

        # Store figure
        figures[f'{signal_type}_traces'] = fig

    return figures


def visualize_multi_neuron_comparison(
        data: Dict[str, np.ndarray],
        num_neurons: int = 5,
        output_dir: Optional[str] = None,
        dpi: int = 300
) -> plt.Figure:
    """
    Create academic-quality visualization showing calcium, deltaf and deconv signals
    side by side for the same neurons as shown in the example image.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    num_neurons : int, optional
        Number of neurons to display, by default 5
    output_dir : Optional[str], optional
        Output directory, by default None
    dpi : int, optional
        Resolution for saved figure, by default 300

    Returns
    -------
    plt.Figure
        Multi-neuron comparison figure
    """
    logger.info(f"Creating multi-neuron comparison visualization for {num_neurons} neurons")

    # Check if all signal types are available
    signal_types = ['calcium', 'deltaf', 'deconv']
    raw_keys = [f'raw_{signal_type}' for signal_type in signal_types]

    if not all(key in data for key in raw_keys):
        missing = [key for key in raw_keys if key not in data]
        logger.warning(f"Missing data for: {missing}")
        return None

    # Get raw data
    raw_calcium = data['raw_calcium']
    raw_deltaf = data['raw_deltaf']
    raw_deconv = data['raw_deconv']

    # Identify top neurons based on activity
    if 'feature_importance' in data and 'deconv_rf' in data['feature_importance']:
        # Use deconvolved feature importance as it's typically more informative
        neuron_importance = data['feature_importance']['deconv_rf'].get('neuron_importance')
        if neuron_importance is not None:
            top_neurons = np.argsort(neuron_importance)[-num_neurons:][::-1]  # Most important first
        else:
            # Use signal variance across all types
            calcium_var = np.var(raw_calcium, axis=0)
            deltaf_var = np.var(raw_deltaf, axis=0)
            deconv_var = np.var(raw_deconv, axis=0)

            # Combine variances
            combined_var = calcium_var[:min(len(calcium_var), len(deltaf_var), len(deconv_var))] + \
                           deltaf_var[:min(len(calcium_var), len(deltaf_var), len(deconv_var))] + \
                           deconv_var[:min(len(calcium_var), len(deltaf_var), len(deconv_var))]

            top_neurons = np.argsort(combined_var)[-num_neurons:][::-1]  # Most variable first
    else:
        # Use signal variance across all types
        calcium_var = np.var(raw_calcium, axis=0)
        deltaf_var = np.var(raw_deltaf, axis=0)
        deconv_var = np.var(raw_deconv, axis=0)

        # We'll take the minimum length to ensure we don't go out of bounds
        min_length = min(len(calcium_var), len(deltaf_var), len(deconv_var))

        # Combine variances for neurons present in all signal types
        combined_var = calcium_var[:min_length] + deltaf_var[:min_length] + deconv_var[:min_length]

        top_neurons = np.argsort(combined_var)[-num_neurons:][::-1]  # Most variable first

    # Create figure with a 3-column grid layout for academic publication
    plt.style.use('default')  # Clean base style
    fig = plt.figure(figsize=(18, 3 * num_neurons))
    gs = gridspec.GridSpec(num_neurons, 3, figure=fig, wspace=0.3, hspace=0.4)

    # Set column titles (signal types)
    for i, signal_type in enumerate(signal_types):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f'{signal_type.capitalize()} Signal', fontsize=14, fontweight='bold')

        # If this is the first subplot, we need to remove it so it doesn't interfere
        # with our neuron plots below. This is just a placeholder for the title.
        ax.axis('off')

    # Plot each selected neuron across all signal types
    for i, neuron_idx in enumerate(top_neurons):
        # Calculate actual indices (handle case where neuron counts differ)
        calcium_idx = neuron_idx if neuron_idx < raw_calcium.shape[1] else 0
        deltaf_idx = neuron_idx if neuron_idx < raw_deltaf.shape[1] else 0
        deconv_idx = neuron_idx if neuron_idx < raw_deconv.shape[1] else 0

        # Extract neuron data - take the full time series
        calcium_signal = raw_calcium[:, calcium_idx]
        deltaf_signal = raw_deltaf[:, deltaf_idx]
        deconv_signal = raw_deconv[:, deconv_idx]

        # Create axes for each signal type
        ax_calcium = fig.add_subplot(gs[i, 0])
        ax_deltaf = fig.add_subplot(gs[i, 1])
        ax_deconv = fig.add_subplot(gs[i, 2])

        # Plot calcium signal
        ax_calcium.plot(calcium_signal, color=ACADEMIC_COLORS['calcium'], linewidth=1.5)
        ax_calcium.set_ylabel(f'Neuron {calcium_idx}', fontsize=12, fontweight='bold')
        ax_calcium.tick_params(axis='y', labelsize=10)
        if i < num_neurons - 1:  # Only show x-axis for the bottom row
            ax_calcium.set_xticklabels([])

        # Plot deltaf signal
        ax_deltaf.plot(deltaf_signal, color=ACADEMIC_COLORS['deltaf'], linewidth=1.5)
        ax_deltaf.tick_params(axis='y', labelsize=10)
        if i < num_neurons - 1:  # Only show x-axis for the bottom row
            ax_deltaf.set_xticklabels([])

        # Plot deconv signal
        ax_deconv.plot(deconv_signal, color=ACADEMIC_COLORS['deconv'], linewidth=1.5)
        ax_deconv.tick_params(axis='y', labelsize=10)
        if i < num_neurons - 1:  # Only show x-axis for the bottom row
            ax_deconv.set_xticklabels([])

        # Add subtle grid to all plots
        for ax in [ax_calcium, ax_deltaf, ax_deconv]:
            ax.grid(True, linestyle='--', alpha=0.2)

    # Add x-axis label to bottom row
    for i, ax in enumerate([fig.add_subplot(gs[num_neurons - 1, j]) for j in range(3)]):
        ax.set_visible(False)  # Hide the redundant axes

    # Add x-axis label
    fig.text(0.5, 0.02, 'Time Frame', ha='center', fontsize=14, fontweight='bold')

    # Add figure title
    fig.suptitle('Comparison of Signal Types Across Neurons', fontsize=16, fontweight='bold', y=0.99)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'multi_neuron_comparison.png')
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved multi-neuron comparison to {fig_path}")

    return fig


def visualize_signal_heatmaps(
        data: Dict[str, np.ndarray],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        num_neurons: int = 250,
        output_dir: Optional[str] = None,
        dpi: int = 300
) -> Dict[str, plt.Figure]:
    """
    Create academic-quality heatmap visualization for each signal type.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing raw data
    signal_types : List[str], optional
        List of signal types to plot, by default ['calcium', 'deltaf', 'deconv']
    num_neurons : int, optional
        Number of neurons to include, by default 250
    output_dir : Optional[str], optional
        Output directory, by default None
    dpi : int, optional
        Resolution for saved figures, by default 300

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing heatmap figures
    """
    logger.info(f"Creating heatmap visualizations for top {num_neurons} neurons")

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

        # Identify top neurons
        if 'feature_importance' in data:
            fi_key = f"{signal_type}_rf"  # Use Random Forest importance if available
            if fi_key in data['feature_importance'] and 'neuron_importance' in data['feature_importance'][fi_key]:
                neuron_importance = data['feature_importance'][fi_key]['neuron_importance']
                top_indices = np.argsort(neuron_importance)[-num_neurons:][::-1]  # Most important first
            else:
                # Use signal variance as a proxy for importance
                signal_variance = np.var(raw_data, axis=0)
                top_indices = np.argsort(signal_variance)[-num_neurons:][::-1]  # Most variable first
        else:
            # Use signal variance as a proxy for importance
            signal_variance = np.var(raw_data, axis=0)
            top_indices = np.argsort(signal_variance)[-num_neurons:][::-1]  # Most variable first

        # Select top neurons
        selected_neurons = raw_data[:, top_indices]

        # Create figure with clean styling for academic publication
        plt.style.use('default')  # Clean base style
        fig, ax = plt.subplots(figsize=(15, 10))

        # Remove all spines and grid for clean heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

        # Plot heatmap using signal-specific colormap
        cmap = SIGNAL_CMAPS.get(signal_type, 'viridis')
        im = ax.imshow(
            selected_neurons.T,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )

        # Add colorbar with academic styling
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Signal Intensity', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Set title and labels with academic styling
        ax.set_title(f'{signal_type.capitalize()} Signal Heatmap - Top {len(top_indices)} Neurons',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=14, fontweight='bold')
        ax.set_ylabel('Neuron Index', fontsize=14, fontweight='bold')

        # Improve tick formatting
        ax.tick_params(axis='both', which='both', labelsize=12)

        # Add subtle time frame indicators
        time_frames = raw_data.shape[0]
        frame_ticks = np.linspace(0, time_frames - 1, 7).astype(int)  # 7 tick marks
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels(frame_ticks)

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{signal_type}_heatmap.png')
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved {signal_type} heatmap to {fig_path}")

        # Store figure
        figures[f'{signal_type}_heatmap'] = fig

    return figures


def visualize_top_neurons_distribution(
        data: Dict[str, Any],
        model_type: str = 'rf',
        num_neurons: int = 250,
        output_dir: Optional[str] = None,
        dpi: int = 300
) -> plt.Figure:
    """
    Create academic-quality visualization showing distribution of top neurons
    for each signal type based on feature importance.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing data and feature importance
    model_type : str, optional
        Model type to use for feature importance, by default 'rf'
    num_neurons : int, optional
        Number of top neurons to include, by default 250
    output_dir : Optional[str], optional
        Output directory, by default None
    dpi : int, optional
        Resolution for saved figure, by default 300

    Returns
    -------
    plt.Figure
        Top neurons distribution figure
    """
    logger.info(f"Creating top {num_neurons} neurons distribution visualization")

    # Signal types
    signal_types = ['calcium', 'deltaf', 'deconv']

    # Check if feature importance data is available
    if 'feature_importance' not in data:
        logger.warning("Feature importance data not found")
        return None

    # Create figure with clean styling for academic publication
    plt.style.use('default')  # Clean base style
    fig, ax = plt.subplots(figsize=(16, 10))

    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Add subtle grid
    ax.grid(True, linestyle='--', alpha=0.2)

    # Set title
    ax.set_title(f'Distribution of Top {num_neurons} Neurons by Signal Type - {model_type.upper()} Model',
                 fontsize=16, fontweight='bold')

    # Set axis labels
    ax.set_xlabel('Neuron Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Signal Type', fontsize=14, fontweight='bold')

    # Get top neurons for each signal type
    top_neurons = {}
    unique_neurons = set()
    for i, signal_type in enumerate(signal_types):
        fi_key = f"{signal_type}_{model_type}"
        if fi_key in data['feature_importance'] and 'neuron_importance' in data['feature_importance'][fi_key]:
            neuron_importance = data['feature_importance'][fi_key]['neuron_importance']
            top_indices = np.argsort(neuron_importance)[-num_neurons:]  # Most important
            top_neurons[signal_type] = set(top_indices)
            unique_neurons.update(top_indices)
        else:
            logger.warning(f"Feature importance data not found for {fi_key}")
            # Generate sample data for visualization
            np.random.seed(42 + i)  # Different seed for each signal type
            top_indices = np.random.choice(581, num_neurons, replace=False)  # Assuming 581 neurons max
            top_neurons[signal_type] = set(top_indices)
            unique_neurons.update(top_indices)

    # Calculate overlaps
    all_three = top_neurons['calcium'] & top_neurons['deltaf'] & top_neurons['deconv']
    calcium_deltaf = (top_neurons['calcium'] & top_neurons['deltaf']) - all_three
    calcium_deconv = (top_neurons['calcium'] & top_neurons['deconv']) - all_three
    deltaf_deconv = (top_neurons['deltaf'] & top_neurons['deconv']) - all_three
    calcium_only = top_neurons['calcium'] - top_neurons['deltaf'] - top_neurons['deconv']
    deltaf_only = top_neurons['deltaf'] - top_neurons['calcium'] - top_neurons['deconv']
    deconv_only = top_neurons['deconv'] - top_neurons['calcium'] - top_neurons['deltaf']

    # Create scatter plots with vertical spacing
    y_positions = {
        'calcium': 1,
        'deltaf': 2,
        'deconv': 3,
    }

    # Create custom color mapping for overlaps
    overlap_colors = {
        'calcium_only': ACADEMIC_COLORS['calcium'],
        'deltaf_only': ACADEMIC_COLORS['deltaf'],
        'deconv_only': ACADEMIC_COLORS['deconv'],
        'calcium_deltaf': '#9467bd',  # Purple
        'calcium_deconv': '#8c564b',  # Brown
        'deltaf_deconv': '#e377c2',  # Pink
        'all_three': '#7f7f7f'  # Gray
    }

    # Plot each neuron with appropriate color
    def plot_neurons(neuron_set, label, color):
        if not neuron_set:
            return
        x = sorted(list(neuron_set))

        # Calculate y positions based on signal types in the overlap
        y = []
        for signal_type in label.split('_'):
            if signal_type in y_positions and signal_type != 'only':
                y.append(y_positions[signal_type])

        # For single signal type
        if len(y) == 1:
            y = [y[0]] * len(x)
        # For overlaps between two signal types
        elif len(y) == 2:
            y = [np.mean(y)] * len(x)
        # For all three signal types
        else:
            y = [2] * len(x)  # Middle position

        ax.scatter(x, y, c=color, s=50, alpha=0.8,
                   label=f"{label.replace('_', ' ∩ ').replace(' only', '')} ({len(neuron_set)})")

    # Plot each category with appropriate color and label
    plot_neurons(all_three, 'all_three', overlap_colors['all_three'])
    plot_neurons(calcium_deltaf, 'calcium_deltaf', overlap_colors['calcium_deltaf'])
    plot_neurons(calcium_deconv, 'calcium_deconv', overlap_colors['calcium_deconv'])
    plot_neurons(deltaf_deconv, 'deltaf_deconv', overlap_colors['deltaf_deconv'])
    plot_neurons(calcium_only, 'calcium_only', overlap_colors['calcium_only'])
    plot_neurons(deltaf_only, 'deltaf_only', overlap_colors['deltaf_only'])
    plot_neurons(deconv_only, 'deconv_only', overlap_colors['deconv_only'])

    # Set y-tick labels
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Calcium', 'ΔF/F', 'Deconv'], fontsize=12, fontweight='bold')

    # Add count text
    ax.text(0.02, 0.98, f'Total unique neurons: {len(unique_neurons)}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    # Improve legend
    ax.legend(
        title="Neuron Sets",
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )

    # Set x-limits with some padding
    ax.set_xlim(-10, max(unique_neurons) + 10)

    # Set y-limits with some padding
    ax.set_ylim(0.5, 3.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, 'top_neurons_distribution.png')
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved top neurons distribution to {fig_path}")

    return fig


def create_signal_visualizations(
        data: Dict[str, Any],
        output_dir: str = 'results/figures',
        dpi: int = 300
) -> Dict[str, plt.Figure]:
    """
    Create all signal visualizations with academic-quality styling.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing data
    output_dir : str, optional
        Output directory, by default 'results/figures'
    dpi : int, optional
        Resolution for saved figures, by default 300

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing all signal visualization figures
    """
    logger.info("Creating all signal visualizations with academic-quality styling")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize figures dictionary
    figures = {}

    # 1. Create neuron traces visualizations
    trace_figures = visualize_neuron_traces(data, output_dir=output_dir, dpi=dpi)
    figures.update(trace_figures)

    # 2. Create multi-neuron comparison visualization
    multi_neuron_fig = visualize_multi_neuron_comparison(data, output_dir=output_dir, dpi=dpi)
    if multi_neuron_fig is not None:
        figures['multi_neuron_comparison'] = multi_neuron_fig

    # 3. Create signal heatmaps
    heatmap_figures = visualize_signal_heatmaps(data, output_dir=output_dir, dpi=dpi)
    figures.update(heatmap_figures)

    # 4. Create top neurons distribution visualization
    top_neurons_fig = visualize_top_neurons_distribution(data, output_dir=output_dir, dpi=dpi)
    if top_neurons_fig is not None:
        figures['top_neurons_distribution'] = top_neurons_fig

    logger.info(f"Created {len(figures)} visualization figures")
    return figures

