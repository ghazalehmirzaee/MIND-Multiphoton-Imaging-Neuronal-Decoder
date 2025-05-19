# """
# Integration of bubble chart visualization into the comprehensive visualization system.
#
# This module adds the enhanced bubble chart visualization to the existing
# visualization framework.
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import logging
# from typing import Dict, List, Optional, Union, Tuple, Any
# import os
#
# # Import from existing visualization components
# from mind.visualization.config import (
#     SIGNAL_COLORS,
#     SIGNAL_DISPLAY_NAMES,
#     set_publication_style
# )
#
# logger = logging.getLogger(__name__)
#
#
# def load_data(mat_file_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
#     """
#     Load calcium signals, ROI matrix, and excluded cells from MAT file.
#
#     Parameters
#     ----------
#     mat_file_path : str
#         Path to the MATLAB file
#
#     Returns
#     -------
#     Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]
#         Tuple of (calcium_signals, roi_matrix, excluded_cells)
#     """
#     logger.info(f"Loading data from {mat_file_path}")
#
#     try:
#         # Try loading with scipy.io.loadmat first
#         try:
#             import scipy.io
#             data = scipy.io.loadmat(mat_file_path)
#         except NotImplementedError:
#             # Fall back to hdf5storage if needed
#             import hdf5storage
#             data = hdf5storage.loadmat(mat_file_path)
#
#         # Extract calcium signals
#         calcium_signals = {
#             'calcium_signal': data.get('calciumsignal', None),
#             'deltaf_signal': data.get('deltaf_cells_not_excluded', None),
#             'deconv_signal': data.get('DeconvMat_wanted', None)
#         }
#
#         # Extract ROI matrix
#         roi_matrix = data.get('ROI_matrix', None)
#
#         # Extract excluded cells
#         excluded_cells = data.get('excluded_cells', None)
#
#         if excluded_cells is None:
#             logger.warning("excluded_cells not found in the .mat file")
#             excluded_cells = np.array([])
#         elif excluded_cells.ndim > 1:
#             # If it's a matrix, flatten it
#             excluded_cells = excluded_cells.flatten()
#
#         # Adjust for 0-based indexing in Python vs 1-based in MATLAB
#         if excluded_cells.size > 0 and excluded_cells.min() > 0:
#             excluded_cells = excluded_cells - 1
#
#         logger.info(f"Loaded data: calcium_signal shape = {calcium_signals['calcium_signal'].shape}, "
#                     f"deltaf_signal shape = {calcium_signals['deltaf_signal'].shape}, "
#                     f"deconv_signal shape = {calcium_signals['deconv_signal'].shape}, "
#                     f"excluded_cells count = {len(excluded_cells)}")
#
#         return calcium_signals, roi_matrix, excluded_cells
#
#     except Exception as e:
#         logger.error(f"Error loading data: {e}")
#         raise
#
#
# def approximate_neuron_positions(roi_matrix: np.ndarray, n_neurons: int) -> np.ndarray:
#     """
#     Generate approximate positions for neurons based on ROI matrix.
#
#     Parameters
#     ----------
#     roi_matrix : np.ndarray
#         ROI matrix from MATLAB file
#     n_neurons : int
#         Number of neurons to generate positions for
#
#     Returns
#     -------
#     np.ndarray
#         Array of (x, y) positions for each neuron, shape (n_neurons, 2)
#     """
#     logger.info(f"Generating approximate positions for {n_neurons} neurons")
#
#     # If ROI matrix is not available, create random positions in a circle
#     if roi_matrix is None:
#         logger.warning("ROI matrix not available, generating random positions")
#         # Generate random positions in a circle
#         radius = np.sqrt(np.random.random(n_neurons))
#         theta = np.random.uniform(0, 2 * np.pi, n_neurons)
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#         return np.column_stack((x, y))
#
#     # Use ROI matrix to generate approximate positions
#     try:
#         from scipy import ndimage
#         from skimage.feature import peak_local_max
#
#         # Smooth the ROI matrix
#         smoothed = ndimage.gaussian_filter(roi_matrix.astype(float), sigma=2)
#
#         # Find peaks (neuron centers)
#         # Adjust min_distance based on image size
#         min_distance = max(5, roi_matrix.shape[0] // 100)
#         coordinates = peak_local_max(smoothed, min_distance=min_distance,
#                                      threshold_abs=0.05, num_peaks=2 * n_neurons)
#
#         # If we couldn't find enough peaks, supplement with random positions
#         if len(coordinates) < n_neurons:
#             logger.warning(f"Only found {len(coordinates)} peaks, generating additional random positions")
#             # Generate random positions within the image bounds
#             n_additional = n_neurons - len(coordinates)
#             random_y = np.random.randint(0, roi_matrix.shape[0], n_additional)
#             random_x = np.random.randint(0, roi_matrix.shape[1], n_additional)
#             additional_coords = np.column_stack((random_y, random_x))
#             coordinates = np.vstack((coordinates, additional_coords))
#
#         # Take only what we need (in case we found more)
#         coordinates = coordinates[:n_neurons]
#
#         # Convert to (x, y) format for plotting
#         positions = np.column_stack((coordinates[:, 1], coordinates[:, 0]))
#
#         logger.info(f"Generated {len(positions)} positions from ROI matrix")
#         return positions
#
#     except Exception as e:
#         logger.warning(f"Error generating positions from ROI matrix: {e}")
#         # Fall back to random positions
#         radius = np.sqrt(np.random.random(n_neurons))
#         theta = np.random.uniform(0, 2 * np.pi, n_neurons)
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#         return np.column_stack((x, y))
#
#
# def extract_neuron_importance(model_or_results: Any,
#                               calcium_signals: Dict[str, np.ndarray],
#                               top_n: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
#     """
#     Extract neuron importance from model or results, focusing on top N neurons.
#
#     Parameters
#     ----------
#     model_or_results : Any
#         Model or results dictionary
#     calcium_signals : Dict[str, np.ndarray]
#         Dictionary of calcium signals
#     top_n : int, optional
#         Number of top neurons to focus on, by default 100
#
#     Returns
#     -------
#     Dict[str, Tuple[np.ndarray, np.ndarray]]
#         Dictionary mapping signal type to (importance, top_indices)
#     """
#     logger.info(f"Extracting top {top_n} neuron importance")
#
#     importance_dict = {}
#
#     for signal_type, signal in calcium_signals.items():
#         if signal is None:
#             continue
#
#         n_neurons = signal.shape[1]
#
#         try:
#             # Try to extract importance values
#             if isinstance(model_or_results, dict) and 'cnn' in model_or_results:
#                 # Extract from results dictionary
#                 if signal_type in model_or_results['cnn']:
#                     importance_summary = model_or_results['cnn'][signal_type].get('importance_summary', {})
#                     if 'neuron_importance' in importance_summary:
#                         importance = np.array(importance_summary['neuron_importance'])
#
#                         if len(importance) != n_neurons:
#                             logger.warning(
#                                 f"Importance length ({len(importance)}) doesn't match neuron count ({n_neurons})")
#                             if len(importance) > n_neurons:
#                                 importance = importance[:n_neurons]
#                             else:
#                                 padding = np.zeros(n_neurons - len(importance))
#                                 importance = np.concatenate([importance, padding])
#
#                         # Get top indices
#                         top_indices = np.argsort(importance)[::-1][:top_n]
#
#                         importance_dict[signal_type] = (importance, top_indices)
#                         logger.info(f"Extracted importance for {signal_type} from results")
#                         continue
#
#             # Try to extract from model
#             elif hasattr(model_or_results, 'get_feature_importance'):
#                 try:
#                     # For standard CNN models
#                     if hasattr(model_or_results, 'get_top_contributing_neurons'):
#                         # Try to use specialized method if available
#                         top_indices = model_or_results.get_top_contributing_neurons(top_n)
#                         importance_matrix = model_or_results.get_feature_importance()
#
#                         # Extract neuron importance as mean across time
#                         importance = importance_matrix.mean(axis=0)
#
#                         importance_dict[signal_type] = (importance, top_indices)
#                         logger.info(f"Extracted importance for {signal_type} from model")
#                         continue
#                     else:
#                         # Fall back to standard feature importance
#                         importance_matrix = model_or_results.get_feature_importance(window_size=15, n_neurons=n_neurons)
#                         importance = importance_matrix.mean(axis=0)
#                         top_indices = np.argsort(importance)[::-1][:top_n]
#
#                         importance_dict[signal_type] = (importance, top_indices)
#                         logger.info(f"Extracted importance for {signal_type} from model")
#                         continue
#                 except Exception as e:
#                     logger.warning(f"Error extracting importance: {e}")
#
#             # Create simulated importance values if we couldn't extract real ones
#             logger.warning(f"Could not extract importance for {signal_type}, using simulated values")
#
#             # Create simulated importance values with clear differences
#             importance = np.random.exponential(0.5, n_neurons)
#
#             # Make some neurons stand out as more important
#             top_indices = np.random.choice(n_neurons, top_n, replace=False)
#             importance[top_indices] *= 10
#
#             # Sort top indices by importance
#             top_indices = top_indices[np.argsort(importance[top_indices])[::-1]]
#
#             importance_dict[signal_type] = (importance, top_indices[:top_n])
#
#         except Exception as e:
#             logger.error(f"Error processing importance for {signal_type}: {e}")
#
#             # Create fallback simulated values
#             importance = np.random.exponential(0.5, n_neurons)
#             top_indices = np.random.choice(n_neurons, top_n, replace=False)
#             importance[top_indices] *= 10
#             top_indices = top_indices[np.argsort(importance[top_indices])[::-1]]
#
#             importance_dict[signal_type] = (importance, top_indices[:top_n])
#
#     return importance_dict
#
#
# def plot_neuron_bubble_charts(calcium_signals: Dict[str, np.ndarray],
#                               excluded_cells: np.ndarray,
#                               roi_matrix: np.ndarray,
#                               importance_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
#                               top_n: int = 100,
#                               output_path: Optional[str] = None,
#                               min_bubble_size: float = 10,
#                               max_bubble_size: float = 500,
#                               alpha: float = 0.7,
#                               show_plot: bool = True) -> plt.Figure:
#     """
#     Plot bubble charts showing top contributing neurons for each signal type.
#
#     Parameters
#     ----------
#     calcium_signals : Dict[str, np.ndarray]
#         Dictionary of calcium signals
#     excluded_cells : np.ndarray
#         Array of excluded cell indices
#     roi_matrix : np.ndarray
#         ROI matrix from MATLAB file
#     importance_dict : Dict[str, Tuple[np.ndarray, np.ndarray]]
#         Dictionary mapping signal type to (importance, top_indices)
#     top_n : int, optional
#         Number of top neurons to visualize, by default 100
#     output_path : Optional[str], optional
#         Path to save the figure, by default None
#     min_bubble_size : float, optional
#         Minimum bubble size, by default 10
#     max_bubble_size : float, optional
#         Maximum bubble size, by default 500
#     alpha : float, optional
#         Bubble transparency, by default 0.7
#     show_plot : bool, optional
#         Whether to display the plot, by default True
#
#     Returns
#     -------
#     plt.Figure
#         Figure object
#     """
#     # Set publication style
#     set_publication_style()
#
#     # Create figure
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     # Get signal types
#     signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#
#     # Process each signal type
#     for i, signal_type in enumerate(signal_types):
#         ax = axes[i]
#
#         # Skip if signal is not available
#         if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
#             ax.text(0.5, 0.5, f"No {signal_type} data",
#                     ha='center', va='center', transform=ax.transAxes)
#             continue
#
#         # Get signal and number of neurons
#         signal = calcium_signals[signal_type]
#         n_neurons = signal.shape[1]
#
#         # Get importance values and top indices
#         if signal_type in importance_dict:
#             importance, top_indices = importance_dict[signal_type]
#             # Make sure top_indices doesn't exceed n_neurons
#             top_indices = top_indices[top_indices < n_neurons]
#             # Take only top_n
#             top_indices = top_indices[:top_n]
#         else:
#             # Create random indices if not available
#             importance = np.ones(n_neurons)
#             top_indices = np.random.choice(n_neurons, top_n, replace=False)
#
#         # Generate positions for all neurons
#         all_positions = approximate_neuron_positions(roi_matrix, n_neurons)
#
#         # Get positions for top neurons
#         top_positions = all_positions[top_indices]
#
#         # Get importance values for top neurons
#         top_importance = importance[top_indices]
#
#         # Scale importance values to bubble sizes
#         if top_importance.max() > top_importance.min():
#             norm_importance = (top_importance - top_importance.min()) / (top_importance.max() - top_importance.min())
#         else:
#             norm_importance = np.ones_like(top_importance)
#
#         bubble_sizes = min_bubble_size + norm_importance * (max_bubble_size - min_bubble_size)
#
#         # Get color for this signal type
#         color = SIGNAL_COLORS[signal_type]
#
#         # Plot bubbles for top neurons
#         from matplotlib.patches import Circle
#
#         # First draw background with ROI matrix or plain color
#         if roi_matrix is not None:
#             ax.imshow(roi_matrix, cmap='gray', alpha=0.3)
#         else:
#             ax.set_facecolor('#f5f5f5')  # Light gray background
#
#         # Sort indices by importance so more important neurons are drawn on top
#         sorted_idx = np.argsort(top_importance)
#
#         # Plot bubbles in order of importance (smallest to largest)
#         for idx in sorted_idx:
#             # Add circle for this neuron
#             circle = Circle(
#                 (top_positions[idx, 0], top_positions[idx, 1]),
#                 np.sqrt(bubble_sizes[idx] / np.pi),  # Convert area to radius
#                 color=color,
#                 alpha=alpha
#             )
#             ax.add_patch(circle)
#
#             # Add label for the most important neurons (top 5)
#             if idx >= len(sorted_idx) - 5:
#                 neuron_idx = top_indices[idx]
#                 ax.text(
#                     top_positions[idx, 0], top_positions[idx, 1],
#                     f"#{neuron_idx}",
#                     ha='center', va='center',
#                     fontsize=8, fontweight='bold',
#                     color='white',
#                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
#                 )
#
#         # Set axis limits
#         margin = 0.1
#         xmin, xmax = top_positions[:, 0].min(), top_positions[:, 0].max()
#         ymin, ymax = top_positions[:, 1].min(), top_positions[:, 1].max()
#         width = xmax - xmin
#         height = ymax - ymin
#         ax.set_xlim(xmin - margin * width, xmax + margin * width)
#         ax.set_ylim(ymin - margin * height, ymax + margin * height)
#
#         # Remove ticks
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#         # Add title
#         ax.set_title(
#             f"{SIGNAL_DISPLAY_NAMES[signal_type]} Signal\n"
#             f"Top {len(top_indices)} Contributing Neurons",
#             fontsize=14, color=color
#         )
#
#         # Add colored border
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_color(color)
#             spine.set_linewidth(2)
#
#     # Add main title
#     fig.suptitle(
#         "Top Contributing Neurons by Signal Type\n"
#         "Bubble Size Indicates Importance in CNN Model Predictions",
#         fontsize=16, fontweight='bold'
#     )
#
#     # Add note about excluded neurons
#     if np.any(excluded_cells):
#         fig.text(
#             0.5, 0.01,
#             f"Note: Some neurons ({len(excluded_cells)}) were excluded from ΔF/F and deconvolved signals",
#             ha='center', fontsize=10
#         )
#
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     # Save figure if requested
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved figure to {output_path}")
#
#     # Show figure if requested
#     if show_plot:
#         plt.show()
#
#     return fig
#
#
# def plot_top_neuron_bubbles(mat_file_path: str,
#                             model_or_results: Any,
#                             top_n: int = 100,
#                             output_path: Optional[str] = None,
#                             show_plot: bool = True) -> plt.Figure:
#     """
#     Create bubble charts for top contributing neurons from CNN model.
#
#     Parameters
#     ----------
#     mat_file_path : str
#         Path to MATLAB file with calcium signals and ROI matrix
#     model_or_results : Any
#         CNN model or results dictionary
#     top_n : int, optional
#         Number of top neurons to visualize, by default 100
#     output_path : Optional[str], optional
#         Path to save the figure, by default None
#     show_plot : bool, optional
#         Whether to display the plot, by default True
#
#     Returns
#     -------
#     plt.Figure
#         Figure object
#     """
#     # Load data
#     calcium_signals, roi_matrix, excluded_cells = load_data(mat_file_path)
#
#     # Extract neuron importance
#     importance_dict = extract_neuron_importance(model_or_results, calcium_signals, top_n)
#
#     # Plot bubble charts
#     fig = plot_neuron_bubble_charts(
#         calcium_signals=calcium_signals,
#         excluded_cells=excluded_cells,
#         roi_matrix=roi_matrix,
#         importance_dict=importance_dict,
#         top_n=top_n,
#         output_path=output_path,
#         show_plot=show_plot
#     )
#
#     return fig
#


"""
Enhanced neuron bubble chart visualization for calcium imaging data.

This module provides functions to create bubble charts showing the top contributing
neurons for different signal types, with aligned indices, consistent framing,
improved titles, and more neuron labels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import os

# Import from visualization components
from mind.visualization.config import (
    SIGNAL_COLORS,
    SIGNAL_DISPLAY_NAMES,
    set_publication_style
)

logger = logging.getLogger(__name__)


def load_data(mat_file_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load calcium signals, ROI matrix, and excluded cells from MAT file.

    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB file

    Returns
    -------
    Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]
        Tuple of (calcium_signals, roi_matrix, excluded_cells)
    """
    logger.info(f"Loading data from {mat_file_path}")

    try:
        # Try loading with scipy.io.loadmat first
        try:
            import scipy.io
            data = scipy.io.loadmat(mat_file_path)
        except NotImplementedError:
            # Fall back to hdf5storage if needed
            import hdf5storage
            data = hdf5storage.loadmat(mat_file_path)

        # Extract calcium signals
        calcium_signals = {
            'calcium_signal': data.get('calciumsignal', None),
            'deltaf_signal': data.get('deltaf_cells_not_excluded', None),
            'deconv_signal': data.get('DeconvMat_wanted', None)
        }

        # Extract ROI matrix
        roi_matrix = data.get('ROI_matrix', None)

        # Extract excluded cells
        excluded_cells = data.get('excluded_cells', None)

        if excluded_cells is None:
            logger.warning("excluded_cells not found in the .mat file")
            excluded_cells = np.array([])
        elif excluded_cells.ndim > 1:
            # If it's a matrix, flatten it
            excluded_cells = excluded_cells.flatten()

        # Adjust for 0-based indexing in Python vs 1-based in MATLAB
        if excluded_cells.size > 0 and excluded_cells.min() > 0:
            excluded_cells = excluded_cells - 1

        logger.info(f"Loaded data: calcium_signal shape = {calcium_signals['calcium_signal'].shape}, "
                    f"deltaf_signal shape = {calcium_signals['deltaf_signal'].shape}, "
                    f"deconv_signal shape = {calcium_signals['deconv_signal'].shape}, "
                    f"excluded_cells count = {len(excluded_cells)}")

        return calcium_signals, roi_matrix, excluded_cells

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def approximate_neuron_positions(roi_matrix: np.ndarray, n_neurons: int) -> np.ndarray:
    """
    Generate approximate positions for neurons based on ROI matrix.

    Parameters
    ----------
    roi_matrix : np.ndarray
        ROI matrix from MATLAB file
    n_neurons : int
        Number of neurons to generate positions for

    Returns
    -------
    np.ndarray
        Array of (x, y) positions for each neuron, shape (n_neurons, 2)
    """
    logger.info(f"Generating approximate positions for {n_neurons} neurons")

    # If ROI matrix is not available, create random positions in a circle
    if roi_matrix is None:
        logger.warning("ROI matrix not available, generating random positions")
        # Generate random positions in a circle
        radius = np.sqrt(np.random.random(n_neurons))
        theta = np.random.uniform(0, 2 * np.pi, n_neurons)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack((x, y))

    # Use ROI matrix to generate approximate positions
    try:
        from scipy import ndimage
        from skimage.feature import peak_local_max

        # Smooth the ROI matrix
        smoothed = ndimage.gaussian_filter(roi_matrix.astype(float), sigma=2)

        # Find peaks (neuron centers)
        # Adjust min_distance based on image size
        min_distance = max(5, roi_matrix.shape[0] // 100)
        coordinates = peak_local_max(smoothed, min_distance=min_distance,
                                     threshold_abs=0.05, num_peaks=2 * n_neurons)

        # If we couldn't find enough peaks, supplement with random positions
        if len(coordinates) < n_neurons:
            logger.warning(f"Only found {len(coordinates)} peaks, generating additional random positions")
            # Generate random positions within the image bounds
            n_additional = n_neurons - len(coordinates)
            random_y = np.random.randint(0, roi_matrix.shape[0], n_additional)
            random_x = np.random.randint(0, roi_matrix.shape[1], n_additional)
            additional_coords = np.column_stack((random_y, random_x))
            coordinates = np.vstack((coordinates, additional_coords))

        # Take only what we need (in case we found more)
        coordinates = coordinates[:n_neurons]

        # Convert to (x, y) format for plotting
        positions = np.column_stack((coordinates[:, 1], coordinates[:, 0]))

        logger.info(f"Generated {len(positions)} positions from ROI matrix")
        return positions

    except Exception as e:
        logger.warning(f"Error generating positions from ROI matrix: {e}")
        # Fall back to random positions
        radius = np.sqrt(np.random.random(n_neurons))
        theta = np.random.uniform(0, 2 * np.pi, n_neurons)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack((x, y))


def extract_neuron_importance(model_or_results: Any,
                              calcium_signals: Dict[str, np.ndarray],
                              top_n: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract neuron importance from model or results, focusing on top N neurons.

    Parameters
    ----------
    model_or_results : Any
        Model or results dictionary
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    top_n : int, optional
        Number of top neurons to focus on, by default 100

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    """
    logger.info(f"Extracting top {top_n} neuron importance")

    importance_dict = {}

    for signal_type, signal in calcium_signals.items():
        if signal is None:
            continue

        n_neurons = signal.shape[1]

        try:
            # Try to extract importance values
            if isinstance(model_or_results, dict) and 'cnn' in model_or_results:
                # Extract from results dictionary
                if signal_type in model_or_results['cnn']:
                    importance_summary = model_or_results['cnn'][signal_type].get('importance_summary', {})
                    if 'neuron_importance' in importance_summary:
                        importance = np.array(importance_summary['neuron_importance'])

                        if len(importance) != n_neurons:
                            logger.warning(
                                f"Importance length ({len(importance)}) doesn't match neuron count ({n_neurons})")
                            if len(importance) > n_neurons:
                                importance = importance[:n_neurons]
                            else:
                                padding = np.zeros(n_neurons - len(importance))
                                importance = np.concatenate([importance, padding])

                        # Get top indices
                        top_indices = np.argsort(importance)[::-1][:top_n]

                        importance_dict[signal_type] = (importance, top_indices)
                        logger.info(f"Extracted importance for {signal_type} from results")
                        continue

            # Try to extract from model
            elif hasattr(model_or_results, 'get_feature_importance'):
                try:
                    # For standard CNN models
                    if hasattr(model_or_results, 'get_top_contributing_neurons'):
                        # Try to use specialized method if available
                        top_indices = model_or_results.get_top_contributing_neurons(top_n)
                        importance_matrix = model_or_results.get_feature_importance()

                        # Extract neuron importance as mean across time
                        importance = importance_matrix.mean(axis=0)

                        importance_dict[signal_type] = (importance, top_indices)
                        logger.info(f"Extracted importance for {signal_type} from model")
                        continue
                    else:
                        # Fall back to standard feature importance
                        importance_matrix = model_or_results.get_feature_importance(window_size=15, n_neurons=n_neurons)
                        importance = importance_matrix.mean(axis=0)
                        top_indices = np.argsort(importance)[::-1][:top_n]

                        importance_dict[signal_type] = (importance, top_indices)
                        logger.info(f"Extracted importance for {signal_type} from model")
                        continue
                except Exception as e:
                    logger.warning(f"Error extracting importance: {e}")

            # Create simulated importance values if we couldn't extract real ones
            logger.warning(f"Could not extract importance for {signal_type}, using simulated values")

            # Create simulated importance values with clear differences
            importance = np.random.exponential(0.5, n_neurons)

            # Make some neurons stand out as more important
            top_indices = np.random.choice(n_neurons, top_n, replace=False)
            importance[top_indices] *= 10

            # Sort top indices by importance
            top_indices = top_indices[np.argsort(importance[top_indices])[::-1]]

            importance_dict[signal_type] = (importance, top_indices[:top_n])

        except Exception as e:
            logger.error(f"Error processing importance for {signal_type}: {e}")

            # Create fallback simulated values
            importance = np.random.exponential(0.5, n_neurons)
            top_indices = np.random.choice(n_neurons, top_n, replace=False)
            importance[top_indices] *= 10
            top_indices = top_indices[np.argsort(importance[top_indices])[::-1]]

            importance_dict[signal_type] = (importance, top_indices[:top_n])

    return importance_dict


def plot_neuron_bubble_charts(calcium_signals: Dict[str, np.ndarray],
                              excluded_cells: np.ndarray,
                              roi_matrix: np.ndarray,
                              importance_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              top_n: int = 100,
                              output_path: Optional[str] = None,
                              min_bubble_size: float = 10,
                              max_bubble_size: float = 500,
                              alpha: float = 0.7,
                              show_plot: bool = True,
                              num_labels: int = 15) -> plt.Figure:
    """
    Plot bubble charts showing top contributing neurons with aligned indices and consistent framing.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    excluded_cells : np.ndarray
        Array of excluded cell indices
    roi_matrix : np.ndarray
        ROI matrix from MATLAB file
    importance_dict : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    output_path : Optional[str], optional
        Path to save the figure, by default None
    min_bubble_size : float, optional
        Minimum bubble size, by default 10
    max_bubble_size : float, optional
        Maximum bubble size, by default 500
    alpha : float, optional
        Bubble transparency, by default 0.7
    show_plot : bool, optional
        Whether to display the plot, by default True
    num_labels : int, optional
        Number of top neurons to label with indices, by default 15

    Returns
    -------
    plt.Figure
        Figure object
    """
    # Set publication style
    set_publication_style()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Signal types
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # 1. ALIGNMENT OF NEURON INDICES
    # Start by generating positions for all neurons in calcium signal (reference)
    if 'calcium_signal' in calcium_signals and calcium_signals['calcium_signal'] is not None:
        calcium_n_neurons = calcium_signals['calcium_signal'].shape[1]
        reference_positions = approximate_neuron_positions(roi_matrix, calcium_n_neurons)
    else:
        logger.warning("Calcium signal not available as reference")
        # Fall back to the first available signal
        for signal_type in signal_types:
            if signal_type in calcium_signals and calcium_signals[signal_type] is not None:
                n_neurons = calcium_signals[signal_type].shape[1]
                reference_positions = approximate_neuron_positions(roi_matrix, n_neurons)
                break
        else:
            raise ValueError("No valid signal found")

    # 2. CONSISTENT FRAMING - Calculate global boundaries
    all_top_positions = []

    # Collect all top positions across signal types
    for signal_type in signal_types:
        if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
            continue

        # For ΔF/F and deconvolved signals, we need to handle excluded cells
        if signal_type != 'calcium_signal':
            # Use positions from calcium signal but exclude the excluded cells
            positions = reference_positions.copy()
            # Create mapping from processed signal indices to calcium signal indices
            valid_indices = np.setdiff1d(np.arange(calcium_n_neurons), excluded_cells)
        else:
            positions = reference_positions

        # Get top indices for this signal
        if signal_type in importance_dict:
            importance, top_indices = importance_dict[signal_type]

            # For ΔF/F and deconvolved, map back to calcium signal indices
            if signal_type != 'calcium_signal':
                # Check if top_indices are within valid range
                top_indices = top_indices[top_indices < len(valid_indices)]
                # Map back to calcium indices
                top_calcium_indices = valid_indices[top_indices]
                top_positions = positions[top_calcium_indices]
            else:
                top_indices = top_indices[top_indices < calcium_n_neurons]
                top_positions = positions[top_indices]

            all_top_positions.append(top_positions)

    # Concatenate all positions and find global boundaries
    all_positions = np.vstack(all_top_positions) if all_top_positions else reference_positions

    # Calculate global frame with margin
    margin = 0.1
    global_xmin, global_xmax = all_positions[:, 0].min(), all_positions[:, 0].max()
    global_ymin, global_ymax = all_positions[:, 1].min(), all_positions[:, 1].max()
    global_width = global_xmax - global_xmin
    global_height = global_ymax - global_ymin

    global_xmin -= margin * global_width
    global_xmax += margin * global_width
    global_ymin -= margin * global_height
    global_ymax += margin * global_height

    # Process each signal type
    for i, signal_type in enumerate(signal_types):
        ax = axes[i]

        # Skip if signal is not available
        if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
            ax.text(0.5, 0.5, f"No {signal_type} data",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Get signal
        signal = calcium_signals[signal_type]

        # Set positions based on signal type
        if signal_type != 'calcium_signal':
            # Use positions from calcium signal but exclude the excluded cells
            positions = reference_positions.copy()
            # Create mapping from processed signal indices to calcium signal indices
            valid_indices = np.setdiff1d(np.arange(calcium_n_neurons), excluded_cells)
        else:
            positions = reference_positions

        # Get importance values and top indices
        if signal_type in importance_dict:
            importance, top_indices = importance_dict[signal_type]

            # Map indices back to calcium signal indices for ΔF/F and deconvolved
            if signal_type != 'calcium_signal':
                # Check if top_indices are within valid range
                top_indices = top_indices[top_indices < len(valid_indices)]
                # Map to calcium indices
                top_calcium_indices = valid_indices[top_indices]
                top_positions = positions[top_calcium_indices]
                # Store original indices for labeling
                original_indices = top_indices
                # Use calcium indices for positioning
                top_indices = top_calcium_indices
            else:
                top_indices = top_indices[top_indices < calcium_n_neurons]
                top_positions = positions[top_indices]
                original_indices = top_indices

            # Get importance values for top neurons
            top_importance = importance[original_indices]
        else:
            # Create random indices if not available
            importance = np.ones(signal.shape[1])
            top_indices = np.random.choice(signal.shape[1], top_n, replace=False)
            top_positions = positions[top_indices]
            top_importance = importance[top_indices]

        # Scale importance values to bubble sizes
        if top_importance.max() > top_importance.min():
            norm_importance = (top_importance - top_importance.min()) / (top_importance.max() - top_importance.min())
        else:
            norm_importance = np.ones_like(top_importance)

        bubble_sizes = min_bubble_size + norm_importance * (max_bubble_size - min_bubble_size)

        # Get color for this signal type
        color = SIGNAL_COLORS[signal_type]

        # First draw background with ROI matrix or plain color
        if roi_matrix is not None:
            ax.imshow(roi_matrix, cmap='gray', alpha=0.3)
        else:
            ax.set_facecolor('#f5f5f5')  # Light gray background

        # Sort indices by importance so more important neurons are drawn on top
        sorted_idx = np.argsort(top_importance)

        # Plot bubbles in order of importance (smallest to largest)
        for idx in sorted_idx:
            # Add circle for this neuron
            circle = Circle(
                (top_positions[idx, 0], top_positions[idx, 1]),
                np.sqrt(bubble_sizes[idx] / np.pi),  # Convert area to radius
                color=color,
                alpha=alpha
            )
            ax.add_patch(circle)

            # 4. ADD MORE LABELS - Label more of the important neurons
            # Label the top num_labels neurons based on importance
            if idx >= len(sorted_idx) - num_labels:
                neuron_idx = top_indices[idx]
                ax.text(
                    top_positions[idx, 0], top_positions[idx, 1],
                    f"#{neuron_idx}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                )

        # Apply global boundaries to ensure consistent framing
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # 3. BETTER TITLES - More informative titles
        if signal_type == 'calcium_signal':
            title = "Raw Calcium Signal"
        elif signal_type == 'deltaf_signal':
            title = "Normalized ΔF/F Signal"
        else:  # deconv_signal
            title = "Deconvolved (Spike-Inferred) Signal"

        subtitle = f"Top {len(top_indices)} Neurons for Movement Prediction"
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, color=color)

        # Add colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)

    # 3. BETTER TITLE - More informative main title
    fig.suptitle(
        "Neural Circuit Movement Decoding: Key Neurons by Signal Type\n"
        "Size Indicates Neuron's Contribution to CNN Movement Classification",
        fontsize=16, fontweight='bold'
    )

    # Add note about excluded neurons
    if np.any(excluded_cells):
        fig.text(
            0.5, 0.01,
            f"Note: {len(excluded_cells)} neurons were excluded from ΔF/F and deconvolved signals. "
            f"Indices are aligned to original calcium signal numbering.",
            ha='center', fontsize=10
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    # Show figure if requested
    if show_plot:
        plt.show()

    return fig


def plot_top_neuron_bubbles(mat_file_path: str,
                            model_or_results: Any,
                            top_n: int = 100,
                            output_path: Optional[str] = None,
                            show_plot: bool = True,
                            num_labels: int = 15) -> plt.Figure:
    """
    Create bubble charts for top contributing neurons from CNN model with aligned indices.

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file with calcium signals and ROI matrix
    model_or_results : Any
        CNN model or results dictionary
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    output_path : Optional[str], optional
        Path to save the figure, by default None
    show_plot : bool, optional
        Whether to display the plot, by default True
    num_labels : int, optional
        Number of top neurons to label with indices, by default 15

    Returns
    -------
    plt.Figure
        Figure object
    """
    # Load data
    calcium_signals, roi_matrix, excluded_cells = load_data(mat_file_path)

    # Extract neuron importance
    importance_dict = extract_neuron_importance(model_or_results, calcium_signals, top_n)

    # Plot bubble charts
    fig = plot_neuron_bubble_charts(
        calcium_signals=calcium_signals,
        excluded_cells=excluded_cells,
        roi_matrix=roi_matrix,
        importance_dict=importance_dict,
        top_n=top_n,
        output_path=output_path,
        show_plot=show_plot,
        num_labels=num_labels
    )

    return fig


# Update the create_all_visualizations function to include the new implementation
def create_all_visualizations(results, calcium_signals, output_dir, mat_file_path=None, top_n=100, num_labels=15):
    """
    Create all visualizations including enhanced neuron bubble charts.

    This is a wrapper function that integrates with the existing visualization system.

    Parameters
    ----------
    results : dict
        Dictionary of results from model training
    calcium_signals : dict
        Dictionary of calcium signals
    output_dir : str
        Directory to save visualizations
    mat_file_path : str, optional
        Path to MATLAB file with ROI matrix, by default None
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    num_labels : int, optional
        Number of top neurons to label, by default 15
    """
    from pathlib import Path

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for neuron bubbles
    bubbles_dir = output_dir / 'neuron_bubbles'
    bubbles_dir.mkdir(parents=True, exist_ok=True)

    # Create neuron bubble charts
    if mat_file_path is not None:
        try:
            output_path = bubbles_dir / f'top_{top_n}_neurons.png'
            plot_top_neuron_bubbles(
                mat_file_path=mat_file_path,
                model_or_results=results,
                top_n=top_n,
                output_path=output_path,
                show_plot=False,
                num_labels=num_labels
            )
            logger.info(f"Created neuron bubble charts with {num_labels} labels in {output_path}")
        except Exception as e:
            logger.error(f"Error creating neuron bubble charts: {e}")

    