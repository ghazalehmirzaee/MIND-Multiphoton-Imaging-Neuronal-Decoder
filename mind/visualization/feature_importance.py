# # mind/visualization/feature_importance.py
# """
# Feature importance visualization module with consistent styling.
# Creates heatmaps, temporal importance plots, and neuron importance visualizations.
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from typing import Dict, Optional, Any
# import logging
#
# from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES,
#                      set_publication_style, get_signal_colormap, SIGNAL_GRADIENTS,
#                      FIGURE_SIZES)
#
# logger = logging.getLogger(__name__)
#
#
# def plot_feature_importance_heatmaps(
#         results: Dict[str, Dict[str, Any]],
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Create feature importance heatmaps for each signal type.
#
#     Shows relative importance of neurons (x-axis) across time steps (y-axis)
#     using consistent color coding for each signal type.
#
#     Parameters
#     ----------
#     results : Dict[str, Dict[str, Any]]
#         Results dictionary organized by model and signal type
#     output_dir : Optional[Path]
#         Directory to save the figure
#
#     Returns
#     -------
#     plt.Figure
#         The created figure
#     """
#     set_publication_style()
#
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#
#     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
#
#     for j, signal in enumerate(signals):
#         ax = axes[j]
#
#         # Use Random Forest importance as it's most reliable
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             importance_matrix = np.array(importance_summary['importance_matrix'])
#
#             # Create custom colormap for this signal type
#             cmap = get_signal_colormap(signal)
#
#             # Plot heatmap with neurons on x-axis and time steps on y-axis
#             im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap)
#
#             ax.set_xlabel('Neuron', fontsize=12)
#             ax.set_ylabel('Time Step', fontsize=12)
#
#             # Set title with signal color
#             signal_color = SIGNAL_COLORS[signal]
#             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Feature Importance',
#                          fontsize=14, fontweight='bold', color=signal_color)
#
#             # Add colorbar
#             cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#             cbar.set_label('Importance', fontsize=10)
#
#             # Add colored border
#             for spine in ax.spines.values():
#                 spine.set_edgecolor(signal_color)
#                 spine.set_linewidth(2)
#
#             # Adjust neuron labels to avoid crowding
#             n_neurons = importance_matrix.shape[1]
#             if n_neurons > 50:
#                 # Show every 10th neuron label
#                 neuron_ticks = np.arange(0, n_neurons, 10)
#                 ax.set_xticks(neuron_ticks)
#                 ax.set_xticklabels(neuron_ticks)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No importance data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Feature Importance Heatmaps', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'feature_importance_heatmaps.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved feature importance heatmaps to {output_path}")
#
#     return fig
#
#
# def plot_temporal_importance_patterns(
#         results: Dict[str, Dict[str, Any]],
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Create temporal importance bar plots for each signal type.
#
#     Shows the average importance across all neurons for each time step,
#     using consistent color coding for each signal type.
#
#     Parameters
#     ----------
#     results : Dict[str, Dict[str, Any]]
#         Results dictionary organized by model and signal type
#     output_dir : Optional[Path]
#         Directory to save the figure
#
#     Returns
#     -------
#     plt.Figure
#         The created figure
#     """
#     set_publication_style()
#
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#
#     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
#
#     for j, signal in enumerate(signals):
#         ax = axes[j]
#
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             temporal_importance = np.array(importance_summary['temporal_importance'])
#
#             # Get signal color
#             signal_color = SIGNAL_COLORS[signal]
#             gradient = SIGNAL_GRADIENTS[signal]
#
#             # Create bars with gradient effect
#             bars = ax.bar(range(len(temporal_importance)), temporal_importance)
#
#             # Apply gradient coloring to bars
#             for i, bar in enumerate(bars):
#                 # Gradient intensity based on importance value
#                 intensity = temporal_importance[i] / temporal_importance.max()
#                 color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
#                 bar.set_color(gradient[color_idx])
#                 bar.set_edgecolor(signal_color)
#                 bar.set_linewidth(0.5)
#
#             ax.set_xlabel('Time Step', fontsize=12)
#             ax.set_ylabel('Mean Feature Importance', fontsize=12)
#             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
#                          fontsize=14, fontweight='bold', color=signal_color)
#             ax.grid(True, alpha=0.3, axis='y')
#
#             # Style improvements
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['left'].set_color(signal_color)
#             ax.spines['left'].set_linewidth(2)
#
#             # Set y-axis limits for consistency
#             ax.set_ylim(0, temporal_importance.max() * 1.1)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No temporal data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Temporal Importance Patterns', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'temporal_importance_patterns.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved temporal importance patterns to {output_path}")
#
#     return fig
#
#
# def plot_top_neuron_importance(
#         results: Dict[str, Dict[str, Any]],
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Create bar plots showing top 20 neuron importance for each signal type.
#
#     Shows the mean importance of the top 20 neurons using consistent color
#     coding for each signal type.
#
#     Parameters
#     ----------
#     results : Dict[str, Dict[str, Any]]
#         Results dictionary organized by model and signal type
#     output_dir : Optional[Path]
#         Directory to save the figure
#
#     Returns
#     -------
#     plt.Figure
#         The created figure
#     """
#     set_publication_style()
#
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#
#     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
#
#     for j, signal in enumerate(signals):
#         ax = axes[j]
#
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             neuron_importance = np.array(importance_summary['neuron_importance'])
#             top_indices = np.array(importance_summary['top_neuron_indices'])[:20]
#
#             # Get importance values for top neurons
#             top_importance = neuron_importance[top_indices]
#
#             # Get signal color and gradient
#             signal_color = SIGNAL_COLORS[signal]
#             gradient = SIGNAL_GRADIENTS[signal]
#
#             # Create bars with gradient effect
#             bars = ax.bar(range(len(top_importance)), top_importance)
#
#             # Apply gradient coloring to bars
#             for i, bar in enumerate(bars):
#                 # Gradient intensity based on importance value
#                 intensity = top_importance[i] / top_importance.max()
#                 color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
#                 bar.set_color(gradient[color_idx])
#                 bar.set_edgecolor(signal_color)
#                 bar.set_linewidth(0.5)
#
#             ax.set_xlabel('Neuron Rank', fontsize=12)
#             ax.set_ylabel('Mean Feature Importance', fontsize=12)
#             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
#                          fontsize=14, fontweight='bold', color=signal_color)
#
#             # Set x-ticks to show neuron indices
#             ax.set_xticks(range(0, 20, 5))
#             ax.set_xticklabels([f'N{top_indices[i]}' for i in range(0, 20, 5)])
#
#             ax.grid(True, alpha=0.3, axis='y')
#
#             # Style improvements
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['left'].set_color(signal_color)
#             ax.spines['left'].set_linewidth(2)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No neuron data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Top 20 Neuron Importance', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'top_neuron_importance.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved top neuron importance to {output_path}")
#
#     return fig
#


# mind/visualization/feature_importance.py
"""
Feature importance visualization module with consistent styling.
Creates heatmaps, temporal importance plots, and neuron importance visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES,
                     set_publication_style, get_signal_colormap, SIGNAL_GRADIENTS,
                     FIGURE_SIZES)

logger = logging.getLogger(__name__)


def plot_feature_importance_heatmaps(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create feature importance heatmaps for each signal type.

    Shows relative importance of neurons (x-axis) across time steps (y-axis)
    using consistent color coding for each signal type.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for j, signal in enumerate(signals):
        ax = axes[j]

        # Use Random Forest importance as it's most reliable
        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            importance_matrix = np.array(importance_summary['importance_matrix'])

            # Create custom colormap for this signal type
            cmap = get_signal_colormap(signal)

            # Create the heatmap with enhanced styling
            from matplotlib import ticker

            # Plot heatmap with neurons on x-axis and time steps on y-axis
            im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap,
                           interpolation='nearest')

            # Add thin grid lines for better academic look
            ax.set_xticks(np.arange(-0.5, importance_matrix.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, importance_matrix.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", size=0)

            # Set labels
            ax.set_xlabel('Neuron', fontsize=12)
            ax.set_ylabel('Time Step', fontsize=12)

            # Set title with signal color
            signal_color = SIGNAL_COLORS[signal]
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Feature Importance',
                         fontsize=14, fontweight='bold', color=signal_color)

            # Add colorbar with proper formatting
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Importance', fontsize=10)
            cbar.ax.tick_params(labelsize=9)

            # Format colorbar to show scientific notation if needed
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            # Add colored border around the entire plot
            for spine in ax.spines.values():
                spine.set_edgecolor(signal_color)
                spine.set_linewidth(2)

            # Fix neuron labels to avoid crowding
            n_neurons = importance_matrix.shape[1]
            n_time_steps = importance_matrix.shape[0]

            # Set major ticks for x-axis (neurons)
            if n_neurons > 50:
                # Show every 50th neuron
                neuron_ticks = np.arange(0, n_neurons, 50)
                ax.set_xticks(neuron_ticks)
                ax.set_xticklabels(neuron_ticks, fontsize=9)
            else:
                # Show every 10th neuron
                neuron_ticks = np.arange(0, n_neurons, 10)
                ax.set_xticks(neuron_ticks)
                ax.set_xticklabels(neuron_ticks, fontsize=9)

            # Set major ticks for y-axis (time steps)
            time_ticks = np.arange(0, n_time_steps, 2)
            ax.set_yticks(time_ticks)
            ax.set_yticklabels(time_ticks, fontsize=9)

            # Remove tick marks
            ax.tick_params(axis='both', which='major', length=0)

        except (KeyError, TypeError, ValueError):
            ax.text(0.5, 0.5, 'No importance data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Feature Importance Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'feature_importance_heatmaps.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance heatmaps to {output_path}")

    return fig


def plot_temporal_importance_patterns(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create temporal importance bar plots for each signal type.

    Shows the average importance across all neurons for each time step,
    using consistent color coding for each signal type.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])

    for j, signal in enumerate(signals):
        ax = axes[j]

        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            temporal_importance = np.array(importance_summary['temporal_importance'])

            # Get signal color
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create bars with gradient effect
            bars = ax.bar(range(len(temporal_importance)), temporal_importance)

            # Apply gradient coloring to bars
            for i, bar in enumerate(bars):
                # Gradient intensity based on importance value
                intensity = temporal_importance[i] / temporal_importance.max()
                color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
                bar.set_color(gradient[color_idx])
                bar.set_edgecolor(signal_color)
                bar.set_linewidth(0.5)

            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Mean Feature Importance', fontsize=12)
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
                         fontsize=14, fontweight='bold', color=signal_color)
            ax.grid(True, alpha=0.3, axis='y')

            # Style improvements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(signal_color)
            ax.spines['left'].set_linewidth(2)

            # Set y-axis limits for consistency
            ax.set_ylim(0, temporal_importance.max() * 1.1)

        except (KeyError, TypeError, ValueError):
            ax.text(0.5, 0.5, 'No temporal data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Temporal Importance Patterns', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'temporal_importance_patterns.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal importance patterns to {output_path}")

    return fig


def plot_top_neuron_importance(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar plots showing top 20 neuron importance for each signal type.

    Shows the mean importance of the top 20 neurons using consistent color
    coding for each signal type.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])

    for j, signal in enumerate(signals):
        ax = axes[j]

        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            neuron_importance = np.array(importance_summary['neuron_importance'])
            top_indices = np.array(importance_summary['top_neuron_indices'])[:20]

            # Get importance values for top neurons
            top_importance = neuron_importance[top_indices]

            # Get signal color and gradient
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create bars with gradient effect
            bars = ax.bar(range(len(top_importance)), top_importance)

            # Apply gradient coloring to bars
            for i, bar in enumerate(bars):
                # Gradient intensity based on importance value
                intensity = top_importance[i] / top_importance.max()
                color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
                bar.set_color(gradient[color_idx])
                bar.set_edgecolor(signal_color)
                bar.set_linewidth(0.5)

            ax.set_xlabel('Neuron Rank', fontsize=12)
            ax.set_ylabel('Mean Feature Importance', fontsize=12)
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
                         fontsize=14, fontweight='bold', color=signal_color)

            # Set x-ticks to show neuron indices
            ax.set_xticks(range(0, 20, 5))
            ax.set_xticklabels([f'N{top_indices[i]}' for i in range(0, 20, 5)])

            ax.grid(True, alpha=0.3, axis='y')

            # Style improvements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(signal_color)
            ax.spines['left'].set_linewidth(2)

        except (KeyError, TypeError, ValueError):
            ax.text(0.5, 0.5, 'No neuron data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Top 20 Neuron Importance', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'top_neuron_importance.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved top neuron importance to {output_path}")

    return fig

