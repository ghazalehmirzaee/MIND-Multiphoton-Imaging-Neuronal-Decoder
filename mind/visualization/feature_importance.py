"""Feature importance visualization functions."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


def plot_top_neurons_overlap(
        feature_importance: Dict[str, Dict[str, np.ndarray]],
        signal_types: List[str] = ['calcium', 'deltaf', 'deconv'],
        model_type: str = 'rf',
        num_neurons: int = 250,
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create improved visualization of overlapping top neurons across signal types.

    Parameters
    ----------
    feature_importance : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing feature importance data
    signal_types : List[str], optional
        List of signal types to compare
    model_type : str, optional
        Model type to use for comparison
    num_neurons : int, optional
        Number of top neurons to include
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing overlap visualization figures
    """
    logger.info(f"Creating top {num_neurons} neurons overlap visualization for {model_type} model")

    # Initialize figures dictionary
    figures = {}

    # Extract top neurons for each signal type
    top_neurons = {}
    for signal_type in signal_types:
        key = f"{signal_type}_{model_type}"
        if key not in feature_importance:
            logger.warning(f"Feature importance not found for {key}")
            continue

        if 'neuron_importance' not in feature_importance[key]:
            logger.warning(f"Neuron importance not found in {key}")
            continue

        neuron_importance = feature_importance[key]['neuron_importance']
        top_n = min(num_neurons, len(neuron_importance))
        top_indices = np.argsort(neuron_importance)[-top_n:][::-1]  # Descending order
        top_neurons[signal_type] = set(top_indices)

    # Calculate overlaps between signal types
    overlaps = {
        'calcium_only': top_neurons.get('calcium', set()) -
                        top_neurons.get('deltaf', set()) -
                        top_neurons.get('deconv', set()),
        'deltaf_only': top_neurons.get('deltaf', set()) -
                       top_neurons.get('calcium', set()) -
                       top_neurons.get('deconv', set()),
        'deconv_only': top_neurons.get('deconv', set()) -
                       top_neurons.get('calcium', set()) -
                       top_neurons.get('deltaf', set()),
        'calcium_deltaf': top_neurons.get('calcium', set()) &
                          top_neurons.get('deltaf', set()) -
                          top_neurons.get('deconv', set()),
        'calcium_deconv': top_neurons.get('calcium', set()) &
                          top_neurons.get('deconv', set()) -
                          top_neurons.get('deltaf', set()),
        'deltaf_deconv': top_neurons.get('deltaf', set()) &
                         top_neurons.get('deconv', set()) -
                         top_neurons.get('calcium', set()),
        'all_three': top_neurons.get('calcium', set()) &
                     top_neurons.get('deltaf', set()) &
                     top_neurons.get('deconv', set())
    }

    # Create improved Venn diagram visualization
    # Set figure size and create the figure
    fig1, ax1 = plt.subplots(figsize=(16, 14))

    # Define colors and alpha for each set
    set_colors = {
        'calcium_only': '#3274A1',  # Dark blue
        'deltaf_only': '#E1812C',  # Orange
        'deconv_only': '#3A923A',  # Green
        'calcium_deltaf': '#9C9EDE',  # Light purple
        'calcium_deconv': '#8C564B',  # Brown
        'deltaf_deconv': '#E377C2',  # Pink
        'all_three': '#7F7F7F'  # Gray
    }

    # Define circles for Venn diagram
    # Use patches for better control
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    # Calculate optimal circle size and positions
    radius = 1.2
    centers = {
        'calcium': (-1.2, 0),  # Left
        'deltaf': (1.2, 0),  # Right
        'deconv': (0, 1.5)  # Top
    }

    # Draw circles for each signal type
    for signal_type, center in centers.items():
        if signal_type not in top_neurons:
            continue

        circle = Circle(center, radius, fill=True,
                        edgecolor='black', linewidth=2,
                        facecolor=set_colors.get(f'{signal_type}_only'),
                        alpha=0.4)
        ax1.add_patch(circle)

        # Add signal type label
        ax1.text(center[0], center[1], signal_type.capitalize(),
                 ha='center', va='center', fontsize=16, fontweight='bold')

    # Add count annotations in visually appealing positions
    positions = {
        'calcium_only': (-1.6, -0.6),
        'deltaf_only': (1.6, -0.6),
        'deconv_only': (0, 2.1),
        'calcium_deltaf': (0, -0.8),
        'calcium_deconv': (-0.8, 0.8),
        'deltaf_deconv': (0.8, 0.8),
        'all_three': (0, 0.5)
    }

    # Add count labels with nice boxes
    for overlap_type, pos in positions.items():
        count = len(overlaps[overlap_type])
        if count > 0:
            ax1.text(pos[0], pos[1], f"{count}",
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8,
                               edgecolor=set_colors.get(overlap_type, 'black')))

    # Add a legend explaining the diagram
    legend_elements = []
    for overlap_type, neurons in overlaps.items():
        if len(neurons) > 0:
            label = overlap_type.replace('_', ' ∩ ').replace(' only', '')
            label = label.capitalize()
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=set_colors.get(overlap_type),
                                              markersize=15, label=f"{label} ({len(neurons)})"))

    ax1.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fontsize=12, ncol=2)

    # Set limits and remove ticks
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2, 2.5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Add title
    ax1.set_title(f'Overlap of Top {num_neurons} Neurons across Signal Types - {model_type.upper()} Model',
                  fontsize=18, y=1.05)

    # Add total counts
    for signal_type in signal_types:
        if signal_type in top_neurons:
            ax1.text(centers[signal_type][0], centers[signal_type][1] - 0.3,
                     f"Total: {len(top_neurons[signal_type])}",
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f'{model_type}_top_neurons_overlap.png')
        plt.savefig(fig_path, dpi=300)

    figures[f'{model_type}_top_neurons_overlap'] = fig1

    # Create a more detailed visualization for neuron indices by group
    fig2, ax2 = plt.subplots(figsize=(20, 12))

    # Define a consistent color palette for the visualization
    color_palette = {
        'calcium_only': '#3274A1',  # Dark blue
        'deltaf_only': '#E1812C',  # Orange
        'deconv_only': '#3A923A',  # Green
        'calcium_deltaf': '#9C9EDE',  # Light purple
        'calcium_deconv': '#8C564B',  # Brown
        'deltaf_deconv': '#E377C2',  # Pink
        'all_three': '#7F7F7F'  # Gray
    }

    # Create a horizontal bar chart showing neuron distributions
    y_positions = []
    colors = []
    labels = []
    neuron_indices = []

    # Position counter for y-axis
    y_pos = 0

    # Process each overlap group
    for overlap_type in ['all_three', 'calcium_deconv', 'deltaf_deconv',
                         'calcium_deltaf', 'calcium_only', 'deltaf_only', 'deconv_only']:
        indices = sorted(list(overlaps[overlap_type]))
        if not indices:
            continue

        # Add label for group
        nice_label = overlap_type.replace('_', ' ∩ ').replace(' only', '')
        nice_label = nice_label.capitalize()

        ax2.text(-20, y_pos + len(indices) / 2, f"{nice_label} ({len(indices)})",
                 ha='right', va='center', fontsize=14, fontweight='bold',
                 color=color_palette[overlap_type])

        # Add neurons from this group
        for idx in indices:
            y_positions.append(y_pos)
            colors.append(color_palette[overlap_type])
            labels.append(f"Neuron {idx}")
            neuron_indices.append(idx)
            y_pos += 1

        # Add space between groups
        y_pos += 2

    # Create scatter plot of neuron indices
    ax2.scatter(neuron_indices, y_positions, c=colors, s=100, alpha=0.8)

    # Add connecting lines between points for better visualization
    for group in ['all_three', 'calcium_deconv', 'deltaf_deconv',
                  'calcium_deltaf', 'calcium_only', 'deltaf_only', 'deconv_only']:
        indices = sorted(list(overlaps[group]))
        if len(indices) > 1:
            group_y_positions = [y for i, y in zip(neuron_indices, y_positions)
                                 if i in indices]
            group_indices = [i for i in neuron_indices if i in indices]
            ax2.plot(group_indices, group_y_positions, '-',
                     color=color_palette[group], alpha=0.3, linewidth=1)

    # Set labels and title
    ax2.set_xlabel('Neuron Index', fontsize=14)
    ax2.set_yticks([])
    ax2.set_title(f'Distribution of Top {num_neurons} Neurons by Signal Type Overlap - {model_type.upper()} Model',
                  fontsize=16)

    # Add grid for better readability
    ax2.grid(axis='x', linestyle='--', alpha=0.3)

    # Add legend explaining color coding
    legend_elements = []
    for group, color in color_palette.items():
        if len(overlaps[group]) > 0:
            nice_label = group.replace('_', ' ∩ ').replace(' only', '')
            nice_label = nice_label.capitalize()
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color, markersize=10,
                                              label=f"{nice_label} ({len(overlaps[group])})"))

    ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Add total unique neurons count
    total_neurons = sum(len(group) for group in overlaps.values())
    ax2.text(0.02, 0.98, f'Total unique neurons: {total_neurons}',
             transform=ax2.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure if output_dir is provided
    if output_dir:
        fig_path = os.path.join(output_dir, f'{model_type}_top_neurons_indices.png')
        plt.savefig(fig_path, dpi=300)

    figures[f'{model_type}_top_neurons_indices'] = fig2

    return figures


def plot_feature_importance_heatmap(
        importance_2d: np.ndarray,
        title: str,
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance heatmap.

    Parameters
    ----------
    importance_2d : np.ndarray
        2D feature importance array (window_size, n_neurons)
    title : str
        Plot title
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Feature importance heatmap figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(importance_2d, ax=ax, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Time Step')

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def plot_temporal_importance(
        temporal_importance: np.ndarray,
        title: str,
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal importance.

    Parameters
    ----------
    temporal_importance : np.ndarray
        Temporal importance array (window_size,)
    title : str
        Plot title
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Temporal importance figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot temporal importance
    ax.bar(range(len(temporal_importance)), temporal_importance)
    ax.set_title(title)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Feature Importance')

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def plot_neuron_importance(
        neuron_importance: np.ndarray,
        title: str,
        num_neurons: int = 20,
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot neuron importance.

    Parameters
    ----------
    neuron_importance : np.ndarray
        Neuron importance array (n_neurons,)
    title : str
        Plot title
    num_neurons : int, optional
        Number of top neurons to plot, by default 20
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Neuron importance figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get top neurons
    num_neurons = min(num_neurons, len(neuron_importance))
    top_neurons = np.argsort(neuron_importance)[-num_neurons:][::-1]

    # Plot neuron importance
    ax.bar(range(num_neurons), neuron_importance[top_neurons])
    ax.set_title(title)
    ax.set_xlabel('Neuron Rank')
    ax.set_ylabel('Mean Feature Importance')

    # Add neuron indices as x-tick labels
    ax.set_xticks(range(num_neurons))
    ax.set_xticklabels([f'N{int(n)}' for n in top_neurons], rotation=45)

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def analyze_feature_importance(
        importance_data: Dict[str, Dict[str, np.ndarray]],
        window_size: int,
        n_neurons: Dict[str, int],
        output_dir: str = 'results/figures'
) -> Dict[str, Any]:
    """
    Analyze feature importance data.

    Parameters
    ----------
    importance_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing feature importance data
    window_size : int
        Window size used for data processing
    n_neurons : Dict[str, int]
        Dictionary mapping signal types to number of neurons
    output_dir : str, optional
        Output directory, by default 'results/figures'

    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
    """
    logger.info("Analyzing feature importance")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analysis results
    analysis = {
        'top_neurons': {},
        'temporal_patterns': {},
        'figures': {}
    }

    # Analyze each signal type and model
    for key, importance in importance_data.items():
        parts = key.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid key format: {key}")
            continue

        signal_type, model_type = parts

        # Get importance data
        if 'importance_2d' not in importance:
            logger.warning(f"importance_2d not found in {key}")
            continue

        importance_2d = importance['importance_2d']

        # Identify top neurons
        neuron_importance = np.mean(importance_2d, axis=0)
        top_neurons = np.argsort(neuron_importance)[-20:][::-1]  # Top 20 in descending order

        # Identify important time windows
        temporal_importance = np.mean(importance_2d, axis=1)
        peak_time = np.argmax(temporal_importance)

        # Store analysis results
        analysis['top_neurons'][key] = top_neurons.tolist()
        analysis['temporal_patterns'][key] = {
            'peak_time': int(peak_time),
            'temporal_importance': temporal_importance.tolist()
        }

        # Create visualizations

        # Feature importance heatmap
        heatmap_fig = plot_feature_importance_heatmap(
            importance_2d,
            f'Feature Importance Heatmap - {signal_type} - {model_type}',
            os.path.join(output_dir, f'{key}_importance_heatmap.png')
        )
        analysis['figures'][f'{key}_importance_heatmap'] = heatmap_fig

        # Temporal importance
        temporal_fig = plot_temporal_importance(
            temporal_importance,
            f'Temporal Importance - {signal_type} - {model_type}',
            os.path.join(output_dir, f'{key}_temporal_importance.png')
        )
        analysis['figures'][f'{key}_temporal_importance'] = temporal_fig

        # Neuron importance
        neuron_fig = plot_neuron_importance(
            neuron_importance,
            f'Top 20 Neuron Importance - {signal_type} - {model_type}',
            20,
            os.path.join(output_dir, f'{key}_top_neurons.png')
        )
        analysis['figures'][f'{key}_top_neurons'] = neuron_fig

    return analysis


def plot_comparative_feature_importance(
        importance_data: Dict[str, Dict[str, np.ndarray]],
        output_dir: str = 'results/figures'
) -> Dict[str, plt.Figure]:
    """
    Create comparative visualizations of feature importance across signal types.

    Parameters
    ----------
    importance_data : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing feature importance data
    output_dir : str, optional
        Output directory, by default 'results/figures'

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing comparative figures
    """
    logger.info("Creating comparative feature importance visualizations")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize figures dictionary
    figures = {}

    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['rf', 'mlp']  # Random Forest and MLP

    # Extract top neurons for each signal type and model type
    top_neurons = {}
    for key, importance in importance_data.items():
        parts = key.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid key format: {key}")
            continue

        signal_type, model_type = parts

        # Get importance data
        if 'importance_2d' not in importance:
            logger.warning(f"importance_2d not found in {key}")
            continue

        importance_2d = importance['importance_2d']

        # Identify top neurons
        neuron_importance = np.mean(importance_2d, axis=0)
        top_20 = np.argsort(neuron_importance)[-20:][::-1]  # Top 20 in descending order

        # Store top neurons
        top_neurons[key] = top_20

    # Create comparative temporal importance plot (all signal types, same model)
    for model_type in model_types:
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, signal_type in enumerate(signal_types):
            key = f"{signal_type}_{model_type}"

            if key not in importance_data:
                logger.warning(f"Importance data not found for {key}")
                continue

            if 'importance_2d' not in importance_data[key]:
                logger.warning(f"importance_2d not found in {key}")
                continue

            # Get temporal importance
            importance_2d = importance_data[key]['importance_2d']
            temporal_importance = np.mean(importance_2d, axis=1)

            # Plot temporal importance
            axes[i].bar(range(len(temporal_importance)), temporal_importance)
            axes[i].set_title(f'Temporal Importance - {signal_type}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Mean Feature Importance')

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, f'{model_type}_temporal_comparison.png')
        plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{model_type}_temporal_comparison'] = fig

    # Create comparative neuron importance plot (all signal types, same model)
    for model_type in model_types:
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, signal_type in enumerate(signal_types):
            key = f"{signal_type}_{model_type}"

            if key not in importance_data:
                logger.warning(f"Importance data not found for {key}")
                continue

            if 'importance_2d' not in importance_data[key]:
                logger.warning(f"importance_2d not found in {key}")
                continue

            if key not in top_neurons:
                logger.warning(f"Top neurons not found for {key}")
                continue

            # Get top neurons
            top_20 = top_neurons[key]

            # Get neuron importance
            importance_2d = importance_data[key]['importance_2d']
            neuron_importance = np.mean(importance_2d, axis=0)

            # Plot neuron importance
            axes[i].bar(range(len(top_20)), neuron_importance[top_20])
            axes[i].set_title(f'Top 20 Neuron Importance - {signal_type}')
            axes[i].set_xlabel('Neuron Rank')
            axes[i].set_ylabel('Mean Feature Importance')

            # Add neuron indices as x-tick labels
            axes[i].set_xticks(range(len(top_20)))
            axes[i].set_xticklabels([f'N{int(n)}' for n in top_20], rotation=45)

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, f'{model_type}_neuron_comparison.png')
        plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{model_type}_neuron_comparison'] = fig

    # Create overlap analysis of top neurons across signal types
    for model_type in model_types:
        # Get top neurons for each signal type
        top_sets = {}
        for signal_type in signal_types:
            key = f"{signal_type}_{model_type}"

            if key not in top_neurons:
                logger.warning(f"Top neurons not found for {key}")
                continue

            top_sets[signal_type] = set(top_neurons[key])

        # Calculate overlaps
        overlaps = {}
        for i, signal_i in enumerate(signal_types):
            if signal_i not in top_sets:
                continue

            for j, signal_j in enumerate(signal_types[i + 1:], i + 1):
                if signal_j not in top_sets:
                    continue

                # Calculate overlap
                overlap = top_sets[signal_i].intersection(top_sets[signal_j])
                overlaps[f"{signal_i}_vs_{signal_j}"] = {
                    'neurons': list(overlap),
                    'count': len(overlap)
                }

        # Create Venn diagram-like visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw circles for each signal type
        radius = 1
        centers = [
            (0, 0),  # calcium
            (1.5, 0),  # deltaf
            (0.75, 1.3)  # deconv
        ]

        for i, signal_type in enumerate(signal_types):
            if signal_type not in top_sets:
                continue

            # Draw circle
            circle = plt.Circle(centers[i], radius, fill=False, edgecolor=f'C{i}', linewidth=2, label=signal_type)
            ax.add_artist(circle)

            # Add label
            ax.text(centers[i][0], centers[i][1], signal_type, ha='center', va='center', fontweight='bold')

        # Add overlap counts
        for pair, data in overlaps.items():
            signal_i, signal_j = pair.split('_vs_')
            i = signal_types.index(signal_i)
            j = signal_types.index(signal_j)

            # Calculate midpoint between centers
            mid_x = (centers[i][0] + centers[j][0]) / 2
            mid_y = (centers[i][1] + centers[j][1]) / 2

            # Add count
            ax.text(mid_x, mid_y, str(data['count']), ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # Set axis limits and turn off ticks
        ax.set_xlim(-1.5, 3)
        ax.set_ylim(-1.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        ax.set_title(f'Overlap of Top 20 Neurons - {model_type.upper()}')

        # Add legend
        ax.legend()

        # Save figure
        fig_path = os.path.join(output_dir, f'{model_type}_neuron_overlap.png')
        plt.savefig(fig_path, dpi=300)

        # Store figure
        figures[f'{model_type}_neuron_overlap'] = fig

    return figures

