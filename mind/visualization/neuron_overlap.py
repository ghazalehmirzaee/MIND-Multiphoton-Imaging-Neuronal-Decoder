"""
Venn diagram visualization with bubble representation for overlapping neurons.

This module creates a more visually appealing Venn diagram showing the overlap
of top contributing neurons across different signal types.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import os

# Import from existing visualization components
from mind.visualization.config import (
    SIGNAL_COLORS,
    SIGNAL_DISPLAY_NAMES,
    set_publication_style
)

# Import utilities from the neuron bubble chart module
from mind.visualization.neuron_importance import (
    load_data,
    extract_neuron_importance
)

logger = logging.getLogger(__name__)


def create_neuron_bubble(ax, x, y, neuron_id, color='#4CAF50', size=0.4):
    """
    Create a bubble representation for a single neuron.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x, y : float
        Position of the bubble center
    neuron_id : int
        The neuron ID to display
    color : str, optional
        Color of the bubble, by default '#4CAF50'
    size : float, optional
        Size of the bubble, by default 0.4
    """
    # Create the bubble circle
    circle = Circle((x, y), size, facecolor=color, edgecolor='white',
                    linewidth=2, alpha=0.9, zorder=10)
    ax.add_patch(circle)

    # Add the neuron ID text
    ax.text(x, y, f'{neuron_id}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=11)


def arrange_bubbles_in_grid(n_bubbles, center_x, center_y, spacing=0.8):
    """
    Arrange bubbles in a grid pattern around a center point.

    Parameters
    ----------
    n_bubbles : int
        Number of bubbles to arrange
    center_x, center_y : float
        Center position for the grid
    spacing : float, optional
        Spacing between bubbles, by default 0.8

    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) positions for each bubble
    """
    positions = []

    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n_bubbles)))
    rows = int(np.ceil(n_bubbles / cols))

    # Calculate starting position
    start_x = center_x - (cols - 1) * spacing / 2
    start_y = center_y - (rows - 1) * spacing / 2

    # Create positions
    for i in range(n_bubbles):
        row = i // cols
        col = i % cols
        x = start_x + col * spacing
        y = start_y + row * spacing
        positions.append((x, y))

    return positions


def plot_neuron_venn_diagram(
        calcium_signals: Dict[str, np.ndarray],
        excluded_cells: np.ndarray,
        importance_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        top_n: int = 100,
        output_path: Optional[str] = None,
        show_plot: bool = True
) -> plt.Figure:
    """
    Create an Venn diagram with bubble representation for overlapping neurons.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    excluded_cells : np.ndarray
        Array of excluded cell indices
    importance_dict : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    top_n : int, optional
        Number of top neurons to consider, by default 100
    output_path : Optional[str], optional
        Path to save the figure, by default None
    show_plot : bool, optional
        Whether to display the plot, by default True

    Returns
    -------
    plt.Figure
        Figure object
    """
    set_publication_style()

    # Import matplotlib_venn
    try:
        from matplotlib_venn import venn3, venn3_circles
    except ImportError:
        logger.error("matplotlib_venn is required for Venn diagrams")
        raise ImportError("matplotlib_venn package required for Venn diagrams. "
                          "Install with: pip install matplotlib-venn")

    # Create figure with more space
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 8)

    # Signal types
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Extract top neuron indices for each signal type
    top_neurons_by_signal = {}
    calcium_n_neurons = calcium_signals['calcium_signal'].shape[1]

    # Map between processed signals and calcium signal
    valid_indices = None
    if np.any(excluded_cells):
        valid_indices = np.setdiff1d(np.arange(calcium_n_neurons), excluded_cells)

    # Get top neuron indices for each signal type
    for signal_type in signal_types:
        if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
            top_neurons_by_signal[signal_type] = np.array([])
            continue

        if signal_type in importance_dict:
            importance, top_indices = importance_dict[signal_type]

            # Map indices for ΔF/F and deconvolved signals back to calcium signal indices
            if signal_type != 'calcium_signal' and valid_indices is not None:
                top_indices = top_indices[top_indices < len(valid_indices)]
                top_indices = valid_indices[top_indices]

            # Take top N
            top_indices = top_indices[:top_n]
            top_neurons_by_signal[signal_type] = top_indices
        else:
            top_neurons_by_signal[signal_type] = np.array([])

    # Create sets of top neurons
    calcium_set = set(top_neurons_by_signal['calcium_signal'])
    deltaf_set = set(top_neurons_by_signal['deltaf_signal'])
    deconv_set = set(top_neurons_by_signal['deconv_signal'])

    # Calculate intersections
    only_calcium = calcium_set - deltaf_set - deconv_set
    only_deltaf = deltaf_set - calcium_set - deconv_set
    only_deconv = deconv_set - calcium_set - deltaf_set
    calcium_deltaf = (calcium_set & deltaf_set) - deconv_set
    calcium_deconv = (calcium_set & deconv_set) - deltaf_set
    deltaf_deconv = (deltaf_set & deconv_set) - calcium_set
    all_three = calcium_set & deltaf_set & deconv_set

    # Create the basic Venn diagram
    subsets = [
        len(only_calcium),
        len(only_deltaf),
        len(calcium_deltaf),
        len(only_deconv),
        len(calcium_deconv),
        len(deltaf_deconv),
        len(all_three)
    ]

    # Define labels with counts
    labels = [
        f"Raw Calcium\n({len(calcium_set)} neurons)",
        f"ΔF/F\n({len(deltaf_set)} neurons)",
        f"Deconvolved\n({len(deconv_set)} neurons)"
    ]

    # Define colors for each circle
    colors = [
        SIGNAL_COLORS['calcium_signal'],
        SIGNAL_COLORS['deltaf_signal'],
        SIGNAL_COLORS['deconv_signal']
    ]

    # Create Venn diagram with more spacing
    venn = venn3(subsets=subsets, set_labels=labels, ax=ax, alpha=0.4,
                 set_colors=colors)

    # Add outlines to circles
    venn_circles = venn3_circles(subsets=subsets, ax=ax, linewidth=2,
                                 color='black')

    # Define positions for different intersection regions
    positions = {
        'A': (-3, 0),  # Only calcium
        'B': (3, 2),  # Only ΔF/F
        'C': (3, -2),  # Only deconvolved
        'AB': (0, 2.5),  # Calcium & ΔF/F
        'AC': (0, -2.5),  # Calcium & Deconvolved
        'BC': (4, 0),  # ΔF/F & Deconvolved
        'ABC': (0, 0)  # All three
    }

    # Color scheme for bubbles in different regions
    bubble_colors = {
        'A': '#5E9FD8',  # Light blue
        'B': '#6EBF8B',  # Light green
        'C': '#C97B7B',  # Light red
        'AB': '#4FA6A6',  # Teal
        'AC': '#9370DB',  # Purple
        'BC': '#FFA07A',  # Light salmon
        'ABC': '#FFD700'  # Gold
    }

    # Plot bubbles for neurons in each region
    bubble_size = 0.35

    # Function to plot neurons for a specific region
    def plot_region_neurons(neurons, region_key, center_pos):
        if len(neurons) == 0:
            return

        neurons_list = sorted(list(neurons))

        # Adjust spacing based on number of neurons
        if len(neurons_list) <= 4:
            spacing = 0.8
        elif len(neurons_list) <= 9:
            spacing = 0.7
        else:
            spacing = 0.6
            bubble_size = 0.3

        # Get positions for bubbles
        bubble_positions = arrange_bubbles_in_grid(
            len(neurons_list), center_pos[0], center_pos[1], spacing)

        # Plot each neuron as a bubble
        for idx, neuron_id in enumerate(neurons_list):
            if idx < len(bubble_positions):
                x, y = bubble_positions[idx]
                create_neuron_bubble(ax, x, y, neuron_id,
                                     color=bubble_colors[region_key],
                                     size=bubble_size)

    # Plot neurons for each region
    plot_region_neurons(only_calcium, 'A', positions['A'])
    plot_region_neurons(only_deltaf, 'B', positions['B'])
    plot_region_neurons(only_deconv, 'C', positions['C'])
    plot_region_neurons(calcium_deltaf, 'AB', positions['AB'])
    plot_region_neurons(calcium_deconv, 'AC', positions['AC'])
    plot_region_neurons(deltaf_deconv, 'BC', positions['BC'])
    plot_region_neurons(all_three, 'ABC', positions['ABC'])

    # Add a legend for the bubble colors
    legend_elements = [
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['A'],
                        label='Calcium only'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['B'],
                        label='ΔF/F only'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['C'],
                        label='Deconvolved only'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['AB'],
                        label='Calcium & ΔF/F'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['AC'],
                        label='Calcium & Deconvolved'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['BC'],
                        label='ΔF/F & Deconvolved'),
        mpatches.Circle((0, 0), 0.1, facecolor=bubble_colors['ABC'],
                        label='All three'),
    ]

    legend = ax.legend(handles=legend_elements, loc='upper left',
                       bbox_to_anchor=(0.02, 0.98), frameon=True,
                       fancybox=True, shadow=True)

    # Style the plot
    ax.set_title('Overlap of Top Contributing Neurons Across Signal Types\n',
                 fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')

    # Add summary text
    summary_text = (
        f"Total neurons identified as important: {len(calcium_set | deltaf_set | deconv_set)}\n"
        f"Neurons important in all three signal types: {len(all_three)}"
    )
    fig.text(0.5, 0.05, summary_text, ha='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    # Add note about excluded neurons
    if np.any(excluded_cells):
        note_text = (
            f"Note: {len(excluded_cells)} neurons were excluded from ΔF/F and deconvolved signals.\n"
            f"All indices are aligned to original calcium signal numbering."
        )
        fig.text(0.5, 0.01, note_text, ha='center', fontsize=11,
                 style='italic', color='gray')

    # Save the figure
    if output_path:
        # Make sure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved Venn diagram to {output_path}")

    # Show the plot
    if show_plot:
        plt.show()

    return fig


def create_neuron_venn_diagram(
        mat_file_path: str,
        model_or_results: Any,
        top_n: int = 100,
        output_path: Optional[str] = None,
        show_plot: bool = True
) -> plt.Figure:
    """
    Create a Venn diagram showing the overlap of top contributing neurons.

    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB file containing calcium signals
    model_or_results : Any
        Either a trained model or results dictionary
    top_n : int, optional
        Number of top neurons to consider, by default 100
    output_path : Optional[str], optional
        Path to save the figure, by default None
    show_plot : bool, optional
        Whether to display the plot, by default True

    Returns
    -------
    plt.Figure
        The created figure
    """
    try:
        # Load data
        calcium_signals, _, excluded_cells = load_data(mat_file_path)

        # Extract neuron importance
        importance_dict = extract_neuron_importance(model_or_results, calcium_signals, top_n)

        # Create Venn diagram
        fig = plot_neuron_venn_diagram(
            calcium_signals=calcium_signals,
            excluded_cells=excluded_cells,
            importance_dict=importance_dict,
            top_n=top_n,
            output_path=output_path,
            show_plot=show_plot
        )

        return fig

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")

        # Create error message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Missing dependency: {str(e)}\n"
                          "Install with: pip install matplotlib-venn",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='red')
        ax.axis('off')

        if output_path:
            # Make sure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save figure
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error creating Venn diagram: {e}")

        # Create error message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error creating Venn diagram:\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='red')
        ax.axis('off')

        if output_path:
            # Make sure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save figure
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()

        return fig

