# # # mind/visualization/feature_importance.py
# # """
# # Feature importance visualization module with consistent styling.
# # Creates heatmaps, temporal importance plots, and neuron importance visualizations.
# # """
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from pathlib import Path
# # from typing import Dict, Optional, Any
# # import logging
# #
# # from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES,
# #                      set_publication_style, get_signal_colormap, SIGNAL_GRADIENTS,
# #                      FIGURE_SIZES)
# #
# # logger = logging.getLogger(__name__)
# #
# #
# # def plot_feature_importance_heatmaps(
# #         results: Dict[str, Dict[str, Any]],
# #         output_dir: Optional[Path] = None
# # ) -> plt.Figure:
# #     """
# #     Create feature importance heatmaps for each signal type.
# #
# #     Shows relative importance of neurons (x-axis) across time steps (y-axis)
# #     using consistent color coding for each signal type.
# #
# #     Parameters
# #     ----------
# #     results : Dict[str, Dict[str, Any]]
# #         Results dictionary organized by model and signal type
# #     output_dir : Optional[Path]
# #         Directory to save the figure
# #
# #     Returns
# #     -------
# #     plt.Figure
# #         The created figure
# #     """
# #     set_publication_style()
# #
# #     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
# #
# #     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
# #
# #     for j, signal in enumerate(signals):
# #         ax = axes[j]
# #
# #         # Use Random Forest importance as it's most reliable
# #         try:
# #             importance_summary = results['random_forest'][signal]['importance_summary']
# #             importance_matrix = np.array(importance_summary['importance_matrix'])
# #
# #             # Create custom colormap for this signal type
# #             cmap = get_signal_colormap(signal)
# #
# #             # Plot heatmap with neurons on x-axis and time steps on y-axis
# #             im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap)
# #
# #             ax.set_xlabel('Neuron', fontsize=12)
# #             ax.set_ylabel('Time Step', fontsize=12)
# #
# #             # Set title with signal color
# #             signal_color = SIGNAL_COLORS[signal]
# #             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Feature Importance',
# #                          fontsize=14, fontweight='bold', color=signal_color)
# #
# #             # Add colorbar
# #             cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# #             cbar.set_label('Importance', fontsize=10)
# #
# #             # Add colored border
# #             for spine in ax.spines.values():
# #                 spine.set_edgecolor(signal_color)
# #                 spine.set_linewidth(2)
# #
# #             # Adjust neuron labels to avoid crowding
# #             n_neurons = importance_matrix.shape[1]
# #             if n_neurons > 50:
# #                 # Show every 10th neuron label
# #                 neuron_ticks = np.arange(0, n_neurons, 10)
# #                 ax.set_xticks(neuron_ticks)
# #                 ax.set_xticklabels(neuron_ticks)
# #
# #         except (KeyError, TypeError, ValueError):
# #             ax.text(0.5, 0.5, 'No importance data',
# #                     ha='center', va='center', transform=ax.transAxes)
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #
# #     fig.suptitle('Feature Importance Heatmaps', fontsize=16, fontweight='bold')
# #
# #     if output_dir:
# #         output_path = Path(output_dir) / 'feature_importance_heatmaps.png'
# #         fig.savefig(output_path, dpi=300, bbox_inches='tight')
# #         logger.info(f"Saved feature importance heatmaps to {output_path}")
# #
# #     return fig
# #
# #
# # def plot_temporal_importance_patterns(
# #         results: Dict[str, Dict[str, Any]],
# #         output_dir: Optional[Path] = None
# # ) -> plt.Figure:
# #     """
# #     Create temporal importance bar plots for each signal type.
# #
# #     Shows the average importance across all neurons for each time step,
# #     using consistent color coding for each signal type.
# #
# #     Parameters
# #     ----------
# #     results : Dict[str, Dict[str, Any]]
# #         Results dictionary organized by model and signal type
# #     output_dir : Optional[Path]
# #         Directory to save the figure
# #
# #     Returns
# #     -------
# #     plt.Figure
# #         The created figure
# #     """
# #     set_publication_style()
# #
# #     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
# #
# #     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
# #
# #     for j, signal in enumerate(signals):
# #         ax = axes[j]
# #
# #         try:
# #             importance_summary = results['random_forest'][signal]['importance_summary']
# #             temporal_importance = np.array(importance_summary['temporal_importance'])
# #
# #             # Get signal color
# #             signal_color = SIGNAL_COLORS[signal]
# #             gradient = SIGNAL_GRADIENTS[signal]
# #
# #             # Create bars with gradient effect
# #             bars = ax.bar(range(len(temporal_importance)), temporal_importance)
# #
# #             # Apply gradient coloring to bars
# #             for i, bar in enumerate(bars):
# #                 # Gradient intensity based on importance value
# #                 intensity = temporal_importance[i] / temporal_importance.max()
# #                 color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
# #                 bar.set_color(gradient[color_idx])
# #                 bar.set_edgecolor(signal_color)
# #                 bar.set_linewidth(0.5)
# #
# #             ax.set_xlabel('Time Step', fontsize=12)
# #             ax.set_ylabel('Mean Feature Importance', fontsize=12)
# #             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
# #                          fontsize=14, fontweight='bold', color=signal_color)
# #             ax.grid(True, alpha=0.3, axis='y')
# #
# #             # Style improvements
# #             ax.spines['top'].set_visible(False)
# #             ax.spines['right'].set_visible(False)
# #             ax.spines['left'].set_color(signal_color)
# #             ax.spines['left'].set_linewidth(2)
# #
# #             # Set y-axis limits for consistency
# #             ax.set_ylim(0, temporal_importance.max() * 1.1)
# #
# #         except (KeyError, TypeError, ValueError):
# #             ax.text(0.5, 0.5, 'No temporal data',
# #                     ha='center', va='center', transform=ax.transAxes)
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #
# #     fig.suptitle('Temporal Importance Patterns', fontsize=16, fontweight='bold')
# #
# #     if output_dir:
# #         output_path = Path(output_dir) / 'temporal_importance_patterns.png'
# #         fig.savefig(output_path, dpi=300, bbox_inches='tight')
# #         logger.info(f"Saved temporal importance patterns to {output_path}")
# #
# #     return fig
# #
# #
# # def plot_top_neuron_importance(
# #         results: Dict[str, Dict[str, Any]],
# #         output_dir: Optional[Path] = None
# # ) -> plt.Figure:
# #     """
# #     Create bar plots showing top 20 neuron importance for each signal type.
# #
# #     Shows the mean importance of the top 20 neurons using consistent color
# #     coding for each signal type.
# #
# #     Parameters
# #     ----------
# #     results : Dict[str, Dict[str, Any]]
# #         Results dictionary organized by model and signal type
# #     output_dir : Optional[Path]
# #         Directory to save the figure
# #
# #     Returns
# #     -------
# #     plt.Figure
# #         The created figure
# #     """
# #     set_publication_style()
# #
# #     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
# #
# #     fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])
# #
# #     for j, signal in enumerate(signals):
# #         ax = axes[j]
# #
# #         try:
# #             importance_summary = results['random_forest'][signal]['importance_summary']
# #             neuron_importance = np.array(importance_summary['neuron_importance'])
# #             top_indices = np.array(importance_summary['top_neuron_indices'])[:20]
# #
# #             # Get importance values for top neurons
# #             top_importance = neuron_importance[top_indices]
# #
# #             # Get signal color and gradient
# #             signal_color = SIGNAL_COLORS[signal]
# #             gradient = SIGNAL_GRADIENTS[signal]
# #
# #             # Create bars with gradient effect
# #             bars = ax.bar(range(len(top_importance)), top_importance)
# #
# #             # Apply gradient coloring to bars
# #             for i, bar in enumerate(bars):
# #                 # Gradient intensity based on importance value
# #                 intensity = top_importance[i] / top_importance.max()
# #                 color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
# #                 bar.set_color(gradient[color_idx])
# #                 bar.set_edgecolor(signal_color)
# #                 bar.set_linewidth(0.5)
# #
# #             ax.set_xlabel('Neuron Rank', fontsize=12)
# #             ax.set_ylabel('Mean Feature Importance', fontsize=12)
# #             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
# #                          fontsize=14, fontweight='bold', color=signal_color)
# #
# #             # Set x-ticks to show neuron indices
# #             ax.set_xticks(range(0, 20, 5))
# #             ax.set_xticklabels([f'N{top_indices[i]}' for i in range(0, 20, 5)])
# #
# #             ax.grid(True, alpha=0.3, axis='y')
# #
# #             # Style improvements
# #             ax.spines['top'].set_visible(False)
# #             ax.spines['right'].set_visible(False)
# #             ax.spines['left'].set_color(signal_color)
# #             ax.spines['left'].set_linewidth(2)
# #
# #         except (KeyError, TypeError, ValueError):
# #             ax.text(0.5, 0.5, 'No neuron data',
# #                     ha='center', va='center', transform=ax.transAxes)
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #
# #     fig.suptitle('Top 20 Neuron Importance', fontsize=16, fontweight='bold')
# #
# #     if output_dir:
# #         output_path = Path(output_dir) / 'top_neuron_importance.png'
# #         fig.savefig(output_path, dpi=300, bbox_inches='tight')
# #         logger.info(f"Saved top neuron importance to {output_path}")
# #
# #     return fig
# #
#
#
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
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
#             # Create the heatmap with enhanced styling
#             from matplotlib import ticker
#
#             # Plot heatmap with neurons on x-axis and time steps on y-axis
#             im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap,
#                            interpolation='nearest')
#
#             # Add thin grid lines for better academic look
#             ax.set_xticks(np.arange(-0.5, importance_matrix.shape[1], 1), minor=True)
#             ax.set_yticks(np.arange(-0.5, importance_matrix.shape[0], 1), minor=True)
#             ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
#             ax.tick_params(which="minor", size=0)
#
#             # Set labels
#             ax.set_xlabel('Neuron', fontsize=12)
#             ax.set_ylabel('Time Step', fontsize=12)
#
#             # Set title with signal color
#             signal_color = SIGNAL_COLORS[signal]
#             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Feature Importance',
#                          fontsize=14, fontweight='bold', color=signal_color)
#
#             # Add colorbar with proper formatting
#             cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#             cbar.set_label('Importance', fontsize=10)
#             cbar.ax.tick_params(labelsize=9)
#
#             # Format colorbar to show scientific notation if needed
#             cbar.formatter.set_powerlimits((0, 0))
#             cbar.update_ticks()
#
#             # Add colored border around the entire plot
#             for spine in ax.spines.values():
#                 spine.set_edgecolor(signal_color)
#                 spine.set_linewidth(2)
#
#             # Fix neuron labels to avoid crowding
#             n_neurons = importance_matrix.shape[1]
#             n_time_steps = importance_matrix.shape[0]
#
#             # Set major ticks for x-axis (neurons)
#             if n_neurons > 50:
#                 # Show every 50th neuron
#                 neuron_ticks = np.arange(0, n_neurons, 50)
#                 ax.set_xticks(neuron_ticks)
#                 ax.set_xticklabels(neuron_ticks, fontsize=9)
#             else:
#                 # Show every 10th neuron
#                 neuron_ticks = np.arange(0, n_neurons, 10)
#                 ax.set_xticks(neuron_ticks)
#                 ax.set_xticklabels(neuron_ticks, fontsize=9)
#
#             # Set major ticks for y-axis (time steps)
#             time_ticks = np.arange(0, n_time_steps, 2)
#             ax.set_yticks(time_ticks)
#             ax.set_yticklabels(time_ticks, fontsize=9)
#
#             # Remove tick marks
#             ax.tick_params(axis='both', which='major', length=0)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No importance data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Feature Importance Heatmaps', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#
#     if output_dir:
#         output_path = Path(output_dir) / 'feature_importance_heatmaps.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved feature importance heatmaps to {output_path}")
#
#     return fig
#
#
# # mind/visualization/feature_importance.py
# # Modified plot_temporal_importance_patterns function
#
# def plot_temporal_importance_patterns(
#         results: Dict[str, Dict[str, Any]],
#         frame_labels: Optional[np.ndarray] = None,
#         window_size: int = 15,
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Create temporal importance bar plots with contralateral footstep markers.
#
#     This enhanced version marks time points where contralateral footsteps occurred,
#     helping visualize the relationship between neural importance and movement events.
#
#     Parameters
#     ----------
#     results : Dict[str, Dict[str, Any]]
#         Results dictionary organized by model and signal type
#     frame_labels : Optional[np.ndarray]
#         Array of frame labels (0=no footstep, 1=contralateral footstep)
#     window_size : int
#         Size of the sliding window used in analysis
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
#             # Add contralateral footstep markers if frame labels are provided
#             if frame_labels is not None:
#                 # Find contralateral footstep events within the window
#                 # The importance at time step i corresponds to a window ending at that position
#                 max_importance = temporal_importance.max()
#
#                 for i in range(len(temporal_importance)):
#                     # Check if there's a contralateral footstep in the window ending at position i
#                     window_start = max(0, i)
#                     window_end = min(i + window_size, len(frame_labels))
#
#                     # Check if the last frame of the window has a contralateral footstep
#                     if window_end - 1 < len(frame_labels) and frame_labels[window_end - 1] == 1:
#                         # Add a red triangle marker above the bar
#                         ax.plot(i, temporal_importance[i] + max_importance * 0.05,
#                                 'v', color='red', markersize=8,
#                                 markeredgecolor='darkred', markeredgewidth=1)
#
#                         # Add "C" label above the marker
#                         ax.text(i, temporal_importance[i] + max_importance * 0.08, 'C',
#                                 ha='center', va='bottom', fontsize=8,
#                                 color='darkred', fontweight='bold')
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
#             # Set y-axis limits with space for markers
#             ax.set_ylim(0, temporal_importance.max() * 1.15)
#
#             # Add legend if footstep markers are shown
#             if frame_labels is not None:
#                 # Create custom legend
#                 from matplotlib.lines import Line2D
#                 legend_elements = [
#                     Line2D([0], [0], marker='v', color='red', linestyle='None',
#                            markersize=8, markeredgecolor='darkred',
#                            label='Contralateral Footstep')
#                 ]
#                 ax.legend(handles=legend_elements, loc='upper right',
#                           framealpha=0.9, fontsize=9)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No temporal data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Temporal Importance Patterns with Movement Events',
#                  fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'temporal_importance_patterns.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved temporal importance patterns to {output_path}")
#
#     return fig
#
# def plot_top_neuron_importance(
#         results: Dict[str, Dict[str, Any]],
#         output_dir: Optional[Path] = None
# ) -> plt.Figure:
#     """
#     Create horizontal bar plots with scatter overlay showing individual importance values.
#
#     This visualization shows both the mean importance (as bars) and the distribution
#     of importance values across time steps (as scatter points) for each top neuron.
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
#     fig, axes = plt.subplots(1, 3, figsize=(18, 8))
#
#     for j, signal in enumerate(signals):
#         ax = axes[j]
#
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             importance_matrix = np.array(importance_summary['importance_matrix'])
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
#             # Create horizontal bars with light color
#             y_positions = np.arange(len(top_importance))
#
#             # Plot horizontal bars with very light color (background)
#             bars = ax.barh(y_positions, top_importance,
#                            color=gradient[0], alpha=0.3, edgecolor='none')
#
#             # Get individual importance values for each neuron across time steps
#             for i, neuron_idx in enumerate(top_indices):
#                 # Extract importance values for this neuron across all time steps
#                 neuron_temporal_importance = importance_matrix[:, neuron_idx]
#
#                 # Create scatter points with jitter for better visibility
#                 jitter = np.random.normal(0, 0.1, len(neuron_temporal_importance))
#                 y_scatter = np.full(len(neuron_temporal_importance), i) + jitter
#
#                 # Plot scatter points with gradient color based on value
#                 scatter = ax.scatter(neuron_temporal_importance, y_scatter,
#                                      alpha=0.6, s=20, c=neuron_temporal_importance,
#                                      cmap=get_signal_colormap(signal),
#                                      edgecolors='white', linewidth=0.5)
#
#                 # Add mean line
#                 ax.plot([top_importance[i]], [i], 'o',
#                         color=signal_color, markersize=8,
#                         markeredgecolor='white', markeredgewidth=2,
#                         zorder=10)
#
#             # Customize y-axis
#             ax.set_yticks(y_positions)
#             ax.set_yticklabels([f'Neuron {idx}' for idx in top_indices], fontsize=10)
#
#             # Labels and title
#             ax.set_xlabel('Feature Importance', fontsize=12)
#             ax.set_ylabel('Neuron ID', fontsize=12)
#             ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
#                          fontsize=14, fontweight='bold', color=signal_color)
#
#             # Grid and styling
#             ax.grid(True, alpha=0.3, axis='x')
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['bottom'].set_color(signal_color)
#             ax.spines['bottom'].set_linewidth(2)
#
#             # Invert y-axis so top neuron is at top
#             ax.invert_yaxis()
#
#             # Add colorbar for scatter points
#             if j == 2:  # Only add colorbar to last subplot
#                 cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
#                 cbar.set_label('Time-specific\nImportance', fontsize=10)
#                 cbar.ax.tick_params(labelsize=8)
#
#             # Add legend
#             from matplotlib.lines import Line2D
#             legend_elements = [
#                 Line2D([0], [0], marker='o', color='w',
#                        markerfacecolor=gradient[0], markersize=10,
#                        alpha=0.3, label='Mean Importance'),
#                 Line2D([0], [0], marker='o', color='w',
#                        markerfacecolor=signal_color, markersize=8,
#                        markeredgecolor='white', markeredgewidth=2,
#                        label='Mean Value'),
#                 Line2D([0], [0], marker='o', color='w',
#                        markerfacecolor=gradient[2], markersize=6,
#                        alpha=0.6, label='Individual Values')
#             ]
#             ax.legend(handles=legend_elements, loc='lower right',
#                       framealpha=0.9, fontsize=9)
#
#         except (KeyError, TypeError, ValueError) as e:
#             ax.text(0.5, 0.5, f'No neuron data\n{str(e)}',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Top 20 Neuron Importance Distribution', fontsize=16, fontweight='bold')
#     plt.tight_layout()
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
Enhanced feature importance visualization module with improved styling and functionality.
Creates heatmaps, temporal importance plots with movement markers, and neuron importance visualizations.
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
    Create feature importance heatmaps for each signal type with enhanced academic styling.

    This function creates publication-quality heatmaps that show the relative importance
    of neurons (x-axis) across time steps (y-axis) using consistent color coding for
    each signal type. The visualization includes grid lines and proper formatting to
    meet academic publication standards.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure with enhanced styling
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for j, signal in enumerate(signals):
        ax = axes[j]

        # Use Random Forest importance as it provides the most interpretable results
        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            importance_matrix = np.array(importance_summary['importance_matrix'])

            # Create custom colormap for this signal type to maintain consistency
            cmap = get_signal_colormap(signal)

            # Create the heatmap with enhanced academic styling
            from matplotlib import ticker

            # Plot heatmap with neurons on x-axis and time steps on y-axis
            im = ax.imshow(importance_matrix, aspect='auto', cmap=cmap,
                           interpolation='nearest')

            # Add subtle grid lines for better readability in academic contexts
            ax.set_xticks(np.arange(-0.5, importance_matrix.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, importance_matrix.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", size=0)

            # Set axis labels with appropriate font sizing
            ax.set_xlabel('Neuron', fontsize=12)
            ax.set_ylabel('Time Step', fontsize=12)

            # Set title with signal-specific color for visual consistency
            signal_color = SIGNAL_COLORS[signal]
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Feature Importance',
                         fontsize=14, fontweight='bold', color=signal_color)

            # Add properly formatted colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Importance', fontsize=10)
            cbar.ax.tick_params(labelsize=9)

            # Format colorbar to show scientific notation when appropriate
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            # Add colored border around the plot for visual organization
            for spine in ax.spines.values():
                spine.set_edgecolor(signal_color)
                spine.set_linewidth(2)

            # Optimize tick labels to prevent overcrowding
            n_neurons = importance_matrix.shape[1]
            n_time_steps = importance_matrix.shape[0]

            # Set major ticks for x-axis (neurons) based on data size
            if n_neurons > 50:
                neuron_ticks = np.arange(0, n_neurons, 50)
                ax.set_xticks(neuron_ticks)
                ax.set_xticklabels(neuron_ticks, fontsize=9)
            else:
                neuron_ticks = np.arange(0, n_neurons, 10)
                ax.set_xticks(neuron_ticks)
                ax.set_xticklabels(neuron_ticks, fontsize=9)

            # Set major ticks for y-axis (time steps)
            time_ticks = np.arange(0, n_time_steps, 2)
            ax.set_yticks(time_ticks)
            ax.set_yticklabels(time_ticks, fontsize=9)

            # Remove tick marks for cleaner appearance
            ax.tick_params(axis='both', which='major', length=0)

        except (KeyError, TypeError, ValueError):
            # Handle cases where importance data is not available
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
        frame_labels: Optional[np.ndarray] = None,
        window_size: int = 15,
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create temporal importance bar plots with contralateral footstep markers.

    This enhanced visualization marks time points where contralateral footsteps occurred,
    helping researchers visualize the relationship between neural importance and movement
    events. This is particularly valuable for understanding the temporal dynamics of
    motor cortex activity during skilled locomotion.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    frame_labels : Optional[np.ndarray]
        Array of frame labels (0=no footstep, 1=contralateral footstep)
    window_size : int
        Size of the sliding window used in analysis (default: 15)
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure with movement event markers
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'])

    for j, signal in enumerate(signals):
        ax = axes[j]

        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            temporal_importance = np.array(importance_summary['temporal_importance'])

            # Get signal-specific color scheme for consistent visualization
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create bars with gradient effect based on importance values
            bars = ax.bar(range(len(temporal_importance)), temporal_importance)

            # Apply gradient coloring to bars for visual appeal and information encoding
            for i, bar in enumerate(bars):
                # Calculate gradient intensity based on relative importance
                intensity = temporal_importance[i] / temporal_importance.max()
                color_idx = min(int(intensity * (len(gradient) - 1)), len(gradient) - 1)
                bar.set_color(gradient[color_idx])
                bar.set_edgecolor(signal_color)
                bar.set_linewidth(0.5)

            # Add contralateral footstep markers if behavioral data is provided
            if frame_labels is not None:
                max_importance = temporal_importance.max()

                # Track marked positions to avoid overcrowding labels
                marked_positions = []

                for i in range(len(temporal_importance)):
                    # Determine the corresponding frame range for this time step
                    # The sliding window approach means each time step represents a window
                    window_start = max(0, i)
                    window_end = min(i + window_size, len(frame_labels))

                    # Check if there's a contralateral footstep at the end of this window
                    if window_end - 1 < len(frame_labels) and frame_labels[window_end - 1] == 1:
                        # Add distinctive red triangle marker above the bar
                        marker_y = temporal_importance[i] + max_importance * 0.08
                        ax.plot(i, marker_y, 'v', color='red', markersize=10,
                                markeredgecolor='darkred', markeredgewidth=1.5,
                                zorder=10)

                        # Add "CONTRA" label above the marker for clarity
                        label_y = marker_y + max_importance * 0.05
                        ax.text(i, label_y, 'CONTRA',
                                ha='center', va='bottom', fontsize=8,
                                color='darkred', fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor='white', edgecolor='darkred',
                                          alpha=0.8))

                        marked_positions.append(i)

                # Add informative legend if footstep markers are present
                if marked_positions:
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='v', color='red', linestyle='None',
                               markersize=10, markeredgecolor='darkred',
                               markeredgewidth=1.5,
                               label='Contralateral Footstep Event')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right',
                              framealpha=0.95, fontsize=9,
                              fancybox=True, shadow=True)

            # Set axis labels and styling
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Mean Feature Importance', fontsize=12)
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
                         fontsize=14, fontweight='bold', color=signal_color)
            ax.grid(True, alpha=0.3, axis='y')

            # Apply consistent styling across all subplots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(signal_color)
            ax.spines['left'].set_linewidth(2)

            # Set y-axis limits with adequate space for markers and labels
            y_max_adjustment = 1.25 if frame_labels is not None else 1.1
            ax.set_ylim(0, temporal_importance.max() * y_max_adjustment)

        except (KeyError, TypeError, ValueError):
            # Handle cases where temporal data is not available
            ax.text(0.5, 0.5, 'No temporal data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Create informative title that reflects the enhanced functionality
    title = 'Temporal Importance Patterns'
    if frame_labels is not None:
        title += ' with Movement Events'
    fig.suptitle(title, fontsize=16, fontweight='bold')

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
    Create enhanced horizontal bar plots with scatter overlay showing individual importance values.

    This visualization combines mean importance (as horizontal bars) with the distribution
    of importance values across time steps (as scatter points) for each top neuron. The
    background bars are now more visible to provide better context for the data distribution.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    output_dir : Optional[Path]
        Directory to save the figure

    Returns
    -------
    plt.Figure
        The created figure with enhanced bar visibility
    """
    set_publication_style()

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for j, signal in enumerate(signals):
        ax = axes[j]

        try:
            importance_summary = results['random_forest'][signal]['importance_summary']
            importance_matrix = np.array(importance_summary['importance_matrix'])
            neuron_importance = np.array(importance_summary['neuron_importance'])
            top_indices = np.array(importance_summary['top_neuron_indices'])[:20]

            # Get importance values for top neurons
            top_importance = neuron_importance[top_indices]

            # Get signal color and gradient for consistent styling
            signal_color = SIGNAL_COLORS[signal]
            gradient = SIGNAL_GRADIENTS[signal]

            # Create horizontal bars with enhanced visibility (darker background)
            y_positions = np.arange(len(top_importance))

            # Use a more visible background color (darker than before)
            # Choose a color that's 60% of the way through the gradient for better visibility
            background_color_idx = min(int(0.6 * (len(gradient) - 1)), len(gradient) - 1)
            background_color = gradient[background_color_idx]

            bars = ax.barh(y_positions, top_importance,
                           color=background_color, alpha=0.7,
                           edgecolor=signal_color, linewidth=1.5)

            # Get individual importance values for each neuron across time steps
            for i, neuron_idx in enumerate(top_indices):
                # Extract importance values for this neuron across all time steps
                neuron_temporal_importance = importance_matrix[:, neuron_idx]

                # Create scatter points with slight jitter for better visibility
                jitter = np.random.normal(0, 0.08, len(neuron_temporal_importance))
                y_scatter = np.full(len(neuron_temporal_importance), i) + jitter

                # Plot scatter points with gradient color mapping
                scatter = ax.scatter(neuron_temporal_importance, y_scatter,
                                     alpha=0.7, s=25, c=neuron_temporal_importance,
                                     cmap=get_signal_colormap(signal),
                                     edgecolors='white', linewidth=0.8,
                                     zorder=5)

                # Add prominent mean value marker
                ax.plot([top_importance[i]], [i], 'o',
                        color=signal_color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2.5,
                        zorder=10)

            # Customize y-axis with neuron labels
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f'Neuron {idx}' for idx in top_indices], fontsize=10)

            # Set axis labels and title
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_ylabel('Neuron ID', fontsize=12)
            ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]} - Top 20 Neurons',
                         fontsize=14, fontweight='bold', color=signal_color)

            # Add grid and styling for better readability
            ax.grid(True, alpha=0.3, axis='x')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(signal_color)
            ax.spines['bottom'].set_linewidth(2)

            # Invert y-axis so highest importance neuron appears at top
            ax.invert_yaxis()

            # Add colorbar for scatter points (only on rightmost subplot)
            if j == 2:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
                cbar.set_label('Time-specific\nImportance', fontsize=10)
                cbar.ax.tick_params(labelsize=8)

            # Create comprehensive legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=background_color, markersize=12,
                       alpha=0.7, markeredgecolor=signal_color,
                       markeredgewidth=1.5, label='Mean Importance Range'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=signal_color, markersize=10,
                       markeredgecolor='white', markeredgewidth=2.5,
                       label='Mean Value'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=gradient[2], markersize=6,
                       alpha=0.7, markeredgecolor='white',
                       markeredgewidth=0.8, label='Individual Time Points')
            ]
            ax.legend(handles=legend_elements, loc='lower right',
                      framealpha=0.95, fontsize=9,
                      fancybox=True, shadow=True)

        except (KeyError, TypeError, ValueError) as e:
            # Handle cases where neuron data is not available
            ax.text(0.5, 0.5, f'No neuron data\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Top 20 Neuron Importance Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'top_neuron_importance.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved top neuron importance to {output_path}")

    return fig

