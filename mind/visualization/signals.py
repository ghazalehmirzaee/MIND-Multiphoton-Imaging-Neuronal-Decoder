# # mind/visualization/signals.py
# """
# Signal visualization module for calcium imaging data.
# Uses consistent color scheme for all signal types.
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import Dict, Optional, Tuple
# import logging
#
# from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES,
#                      set_publication_style, FIGURE_SIZES)
#
# logger = logging.getLogger(__name__)
#
#
# def plot_signal_comparison_top20(
#         calcium_signals: Dict[str, np.ndarray],
#         top_20_indices: np.ndarray,
#         time_range: Optional[Tuple[int, int]] = None,
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Plot signal comparison for top 20 neurons with consistent colors.
#
#     Creates a vertically stacked plot showing raw, ΔF/F, and deconvolved signals
#     for the top 20 most active neurons.
#
#     Parameters
#     ----------
#     calcium_signals : Dict[str, np.ndarray]
#         Dictionary containing all three signal types
#     top_20_indices : np.ndarray
#         Indices of top 20 most active neurons
#     time_range : Optional[Tuple[int, int]]
#         Time range to plot (start, end)
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
#     signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     n_neurons = 20  # Fixed at 20 as requested
#
#     # Create figure with subplots
#     fig, axes = plt.subplots(n_neurons, 3, figsize=(15, 2.5 * n_neurons),
#                              gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
#
#     # Get frame count and determine time range
#     max_frames = calcium_signals[signal_types[0]].shape[0]
#     if time_range is None:
#         time_range = (0, min(1000, max_frames))
#     else:
#         time_range = (max(0, time_range[0]), min(max_frames, time_range[1]))
#
#     time_indices = slice(time_range[0], time_range[1])
#     time_points = np.arange(time_range[0], time_range[1])
#
#     # Plot each neuron and signal type
#     for i, neuron_idx in enumerate(top_20_indices[:n_neurons]):
#         for j, signal_type in enumerate(signal_types):
#             ax = axes[i, j]
#
#             signal = calcium_signals[signal_type]
#             signal_data = signal[time_indices, neuron_idx]
#
#             # Plot with consistent color
#             color = SIGNAL_COLORS[signal_type]
#             ax.plot(time_points, signal_data, color=color, linewidth=0.8)
#
#             # Set labels and styling
#             if i == 0:
#                 ax.set_title(SIGNAL_DISPLAY_NAMES[signal_type],
#                              fontsize=14, fontweight='bold', color=color)
#             if i == n_neurons - 1:
#                 ax.set_xlabel('Time (frames)', fontsize=12)
#             if j == 0:
#                 ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)
#
#             # Clean styling
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.grid(True, alpha=0.3, linestyle='--')
#
#             # Add colored left spine to match signal type
#             ax.spines['left'].set_color(color)
#             ax.spines['left'].set_linewidth(2)
#
#     fig.suptitle('Signal Comparison for Top 20 Active Neurons',
#                  fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'signal_comparison_top20.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved signal comparison to {output_path}")
#
#     return fig
#

# mind/visualization/signals.py
"""
Signal visualization module for calcium imaging data.
Uses consistent color scheme for all signal types.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES,
                     set_publication_style, FIGURE_SIZES)

logger = logging.getLogger(__name__)


def plot_signal_comparison_top20(
        calcium_signals: Dict[str, np.ndarray],
        top_20_indices: np.ndarray,
        time_range: Optional[Tuple[int, int]] = None,
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Plot signal comparison for top 10 neurons with consistent colors.

    Creates a vertically stacked plot showing raw, ΔF/F, and deconvolved signals
    for the top 20 most active neurons.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary containing all three signal types
    top_20_indices : np.ndarray
        Indices of top 20 most active neurons
    time_range : Optional[Tuple[int, int]]
        Time range to plot (start, end)
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    set_publication_style()

    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    n_neurons = 10  # Fixed at 20 as requested

    # Create figure with subplots
    fig, axes = plt.subplots(n_neurons, 3, figsize=(15, 2.5 * n_neurons),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    # Get frame count and determine time range
    max_frames = calcium_signals[signal_types[0]].shape[0]
    if time_range is None:
        time_range = (0, min(1000, max_frames))
    else:
        time_range = (max(0, time_range[0]), min(max_frames, time_range[1]))

    time_indices = slice(time_range[0], time_range[1])
    time_points = np.arange(time_range[0], time_range[1])

    # Plot each neuron and signal type
    for i, neuron_idx in enumerate(top_20_indices[:n_neurons]):
        for j, signal_type in enumerate(signal_types):
            ax = axes[i, j]

            signal = calcium_signals[signal_type]
            signal_data = signal[time_indices, neuron_idx]

            # Plot with consistent color
            color = SIGNAL_COLORS[signal_type]
            ax.plot(time_points, signal_data, color=color, linewidth=0.8)

            # Set labels and styling
            if i == 0:
                ax.set_title(SIGNAL_DISPLAY_NAMES[signal_type],
                             fontsize=14, fontweight='bold', color=color)
            if i == n_neurons - 1:
                ax.set_xlabel('Time (frames)', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'Neuron {neuron_idx}', fontsize=10)

            # Clean styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add colored left spine to match signal type
            ax.spines['left'].set_color(color)
            ax.spines['left'].set_linewidth(2)

    fig.suptitle('Signal Comparison for Top 20 Active Neurons',
                 fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'signal_comparison_top20.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved signal comparison to {output_path}")

    return fig

