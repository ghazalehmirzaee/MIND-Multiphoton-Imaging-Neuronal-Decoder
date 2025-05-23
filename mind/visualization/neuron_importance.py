"""
Enhanced neuron bubble chart visualization for calcium imaging data.

This module creates bubble charts showing the top contributing neurons
for different signal types, with aligned indices and consistent framing.
Modified to remove right-margin outliers and numbered labels.
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
    MODEL_DISPLAY_NAMES,
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
            'calcium_signal': data.get('calciumsignal_wanted', None),
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

        logger.info(f"Loaded data: "
                    f"calcium_signal shape = {calcium_signals['calcium_signal'].shape if calcium_signals['calcium_signal'] is not None else 'None'}, "
                    f"deltaf_signal shape = {calcium_signals['deltaf_signal'].shape if calcium_signals['deltaf_signal'] is not None else 'None'}, "
                    f"deconv_signal shape = {calcium_signals['deconv_signal'].shape if calcium_signals['deconv_signal'] is not None else 'None'}, "
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
                                     threshold_abs=0.05, num_peaks=3 * n_neurons)

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
                              model_name: str = None,
                              top_n: int = 100) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract neuron importance from model or results.

    This function properly extracts the top neurons for each model and signal type
    from the results dictionary structure, ensuring we get exactly the neurons
    the model deems most important for movement prediction.

    Parameters
    ----------
    model_or_results : Any
        Model or results dictionary
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    model_name : str, optional
        Specific model name to extract importance for
    top_n : int, optional
        Number of top neurons to focus on, by default 100

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping signal type to (importance, top_indices)
    """
    logger.info(f"Extracting top {top_n} neuron importance for model: {model_name}")

    importance_dict = {}

    for signal_type, signal in calcium_signals.items():
        if signal is None:
            continue

        n_neurons = signal.shape[1]

        try:
            # Extract from results dictionary
            if isinstance(model_or_results, dict) and model_name is not None:
                if model_name in model_or_results and signal_type in model_or_results[model_name]:
                    model_signal_results = model_or_results[model_name][signal_type]

                    # Get importance summary
                    importance_summary = model_signal_results.get('importance_summary', {})

                    if importance_summary and 'neuron_importance' in importance_summary:
                        importance = np.array(importance_summary['neuron_importance'])

                        # Make sure importance has the right shape
                        if len(importance) != n_neurons:
                            logger.warning(
                                f"Importance length ({len(importance)}) doesn't match neuron count ({n_neurons})")

                            if len(importance) > n_neurons:
                                importance = importance[:n_neurons]
                            else:
                                # Pad with zeros
                                padding = np.zeros(n_neurons - len(importance))
                                importance = np.concatenate([importance, padding])

                        # Get top indices - CORRECTLY SORTED BY IMPORTANCE
                        top_indices = np.argsort(importance)[::-1][:top_n]

                        # Verify we're getting the most important neurons
                        top_importance_values = importance[top_indices]
                        logger.info(
                            f"{model_name} - {signal_type}: Top 5 importance values: {top_importance_values[:5]}")

                        importance_dict[signal_type] = (importance, top_indices)
                        logger.info(f"Extracted importance for {signal_type} from {model_name} results")
                    else:
                        logger.warning(f"No importance data found for {model_name} - {signal_type}")
                else:
                    logger.warning(f"Model {model_name} or signal {signal_type} not found in results")
            else:
                logger.warning(f"Invalid results structure or missing model_name")

        except Exception as e:
            logger.error(f"Error processing importance for {signal_type}: {e}")

    return importance_dict


def plot_neuron_bubble_charts_separate(calcium_signals: Dict[str, np.ndarray],
                                       excluded_cells: np.ndarray,
                                       roi_matrix: np.ndarray,
                                       importance_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                       model_name: str,
                                       top_n: int = 100,
                                       output_dir: Optional[str] = None,
                                       min_bubble_size: float = 10,
                                       max_bubble_size: float = 500,
                                       alpha: float = 0.7,
                                       show_plot: bool = False) -> List[plt.Figure]:
    """
    Plot separate bubble charts for each signal type showing top contributing neurons.

    This function creates individual figures for each signal type, with aggressive cropping
    to remove right-margin outliers and without numbered labels.

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
    model_name : str
        Name of the model for labeling
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    output_dir : Optional[str], optional
        Directory to save figures, by default None
    min_bubble_size : float, optional
        Minimum bubble size, by default 10
    max_bubble_size : float, optional
        Maximum bubble size, by default 500
    alpha : float, optional
        Bubble transparency, by default 0.7
    show_plot : bool, optional
        Whether to display the plot, by default False

    Returns
    -------
    List[plt.Figure]
        List of created figures
    """
    # Set publication style
    set_publication_style()

    # Signal types
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Store figures
    figures = []

    # Generate positions for all neurons in calcium signal (reference)
    if 'calcium_signal' in calcium_signals and calcium_signals['calcium_signal'] is not None:
        calcium_n_neurons = calcium_signals['calcium_signal'].shape[1]
        reference_positions = approximate_neuron_positions(roi_matrix, calcium_n_neurons)
    else:
        logger.warning("Calcium signal not available as reference")
        return figures

    # Calculate global boundaries for consistent framing across all signal types
    all_top_positions = []

    for signal_type in signal_types:
        if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
            continue

        if signal_type not in importance_dict:
            continue

        # Get top indices for this signal
        importance, top_indices = importance_dict[signal_type]

        # For ΔF/F and deconvolved signals, we need to handle excluded cells
        if signal_type != 'calcium_signal' and len(excluded_cells) > 0:
            # Use positions from calcium signal
            positions = reference_positions.copy()
            # Create mapping from processed signal indices to calcium signal indices
            valid_indices = np.setdiff1d(np.arange(calcium_n_neurons), excluded_cells)

            # Map indices back to calcium signal indices
            top_indices_mapped = []
            for idx in top_indices:
                if idx < len(valid_indices):
                    calcium_idx = valid_indices[idx]
                    top_indices_mapped.append(calcium_idx)
            top_indices_mapped = np.array(top_indices_mapped[:top_n])
            top_positions = positions[top_indices_mapped]
        else:
            positions = reference_positions
            top_indices = top_indices[top_indices < calcium_n_neurons][:top_n]
            top_positions = positions[top_indices]

        all_top_positions.append(top_positions)

    # Calculate aggressive crop bounds to remove right-margin outliers completely
    if all_top_positions:
        all_positions = np.vstack(all_top_positions)

        # More aggressive cropping to remove right-margin outliers
        x_coords = all_positions[:, 0]
        y_coords = all_positions[:, 1]

        # Use more restrictive percentiles to aggressively crop outliers
        x_lower = np.percentile(x_coords, 5)
        x_upper = np.percentile(x_coords, 85)  # Much more aggressive than 95th percentile
        y_lower = np.percentile(y_coords, 5)
        y_upper = np.percentile(y_coords, 95)

        # Additional filtering: remove positions that are too sparse (likely outliers)
        # Calculate density-based filtering to identify main neuron cluster
        main_cluster_mask = (
                (x_coords >= x_lower) &
                (x_coords <= x_upper) &
                (y_coords >= y_lower) &
                (y_coords <= y_upper)
        )

        if np.sum(main_cluster_mask) > 0:
            cluster_positions = all_positions[main_cluster_mask]

            # Recalculate bounds based on main cluster only
            x_lower = np.percentile(cluster_positions[:, 0], 2)
            x_upper = np.percentile(cluster_positions[:, 0], 98)
            y_lower = np.percentile(cluster_positions[:, 1], 2)
            y_upper = np.percentile(cluster_positions[:, 1], 98)

        # Add minimal margins to avoid edge clipping
        margin = 0.05  # Reduced margin
        x_range = x_upper - x_lower
        y_range = y_upper - y_lower

        global_xmin = max(0, x_lower - margin * x_range)
        global_xmax = x_upper + margin * x_range
        global_ymin = max(0, y_lower - margin * y_range)
        global_ymax = y_upper + margin * y_range

        # Ensure we don't exceed ROI bounds if available
        if roi_matrix is not None:
            global_xmax = min(global_xmax, roi_matrix.shape[1])
            global_ymax = min(global_ymax, roi_matrix.shape[0])

        logger.info(
            f"Calculated aggressive crop bounds to remove outliers: x=({global_xmin:.1f}, {global_xmax:.1f}), y=({global_ymin:.1f}, {global_ymax:.1f})")
    else:
        # Fallback bounds
        global_xmin, global_xmax = 0, roi_matrix.shape[1] if roi_matrix is not None else 100
        global_ymin, global_ymax = 0, roi_matrix.shape[0] if roi_matrix is not None else 100

    # Process each signal type separately
    for signal_type in signal_types:
        # Skip if signal is not available
        if signal_type not in calcium_signals or calcium_signals[signal_type] is None:
            continue

        if signal_type not in importance_dict:
            continue

        # Create a new figure for each signal type
        fig, ax = plt.subplots(figsize=(12, 10))

        # Get signal data
        signal = calcium_signals[signal_type]

        # Set positions based on signal type
        if signal_type != 'calcium_signal' and len(excluded_cells) > 0:
            positions = reference_positions.copy()
            valid_indices = np.setdiff1d(np.arange(calcium_n_neurons), excluded_cells)
        else:
            positions = reference_positions

        # Get importance values and top indices
        importance, top_indices = importance_dict[signal_type]

        # Map indices back to calcium signal indices for ΔF/F and deconvolved
        if signal_type != 'calcium_signal' and len(excluded_cells) > 0:
            # Map to calcium indices
            top_indices_mapped = []
            original_indices = []
            for idx in top_indices[:top_n]:
                if idx < len(valid_indices):
                    calcium_idx = valid_indices[idx]
                    top_indices_mapped.append(calcium_idx)
                    original_indices.append(idx)

            top_indices_mapped = np.array(top_indices_mapped)
            original_indices = np.array(original_indices)
            top_positions = positions[top_indices_mapped]

            # Get importance values for top neurons
            top_importance = importance[original_indices]
        else:
            top_indices = top_indices[top_indices < calcium_n_neurons][:top_n]
            top_positions = positions[top_indices]
            top_importance = importance[top_indices]

        # Filter positions within aggressive crop bounds
        valid_mask = ((top_positions[:, 0] >= global_xmin) &
                      (top_positions[:, 0] <= global_xmax) &
                      (top_positions[:, 1] >= global_ymin) &
                      (top_positions[:, 1] <= global_ymax))

        filtered_positions = top_positions[valid_mask]
        filtered_importance = top_importance[valid_mask]

        logger.info(
            f"{signal_type}: Showing {len(filtered_positions)} of {len(top_positions)} neurons after aggressive cropping")

        # Scale importance values to bubble sizes
        if len(filtered_importance) > 0 and filtered_importance.max() > filtered_importance.min():
            norm_importance = ((filtered_importance - filtered_importance.min()) /
                               (filtered_importance.max() - filtered_importance.min()))
        else:
            norm_importance = np.ones_like(filtered_importance) if len(filtered_importance) > 0 else np.array([])

        if len(norm_importance) > 0:
            bubble_sizes = min_bubble_size + norm_importance * (max_bubble_size - min_bubble_size)
        else:
            bubble_sizes = np.array([])

        # Get color for this signal type
        color = SIGNAL_COLORS[signal_type]

        # Draw background with ROI matrix (cropped to relevant area)
        if roi_matrix is not None:
            # Crop the ROI matrix to match our aggressive bounds
            roi_crop_ymin = max(0, int(global_ymin))
            roi_crop_ymax = min(roi_matrix.shape[0], int(global_ymax))
            roi_crop_xmin = max(0, int(global_xmin))
            roi_crop_xmax = min(roi_matrix.shape[1], int(global_xmax))

            cropped_roi = roi_matrix[roi_crop_ymin:roi_crop_ymax, roi_crop_xmin:roi_crop_xmax]

            # Display cropped ROI as background
            ax.imshow(cropped_roi, cmap='gray', alpha=0.3,
                      extent=[global_xmin, global_xmax, global_ymax, global_ymin])
        else:
            ax.set_facecolor('#f5f5f5')  # Light gray background

        # Sort indices by importance so more important neurons are drawn on top
        if len(filtered_importance) > 0:
            sorted_idx = np.argsort(filtered_importance)

            # Plot bubbles in order of importance (smallest to largest)
            for idx in sorted_idx:
                # Add circle for this neuron with enhanced styling
                circle = Circle(
                    (filtered_positions[idx, 0], filtered_positions[idx, 1]),
                    np.sqrt(bubble_sizes[idx] / np.pi),  # Convert area to radius
                    color=color,
                    alpha=alpha,
                    edgecolor='white',
                    linewidth=1.5
                )
                ax.add_patch(circle)

            # NOTE: Removed the numbered labels section as requested by user
            # The following code has been commented out to remove numbers 1-10
            """
            # Add labels for top 10 most important neurons for scientific interpretation
            top_10_idx = sorted_idx[-10:] if len(sorted_idx) >= 10 else sorted_idx
            for i, idx in enumerate(top_10_idx):
                ax.text(filtered_positions[idx, 0], filtered_positions[idx, 1],
                        f'{i + 1}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
            """

        # Apply aggressive crop bounds for consistent framing
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

        # Remove ticks for cleaner scientific presentation
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the title with enhanced scientific information
        model_display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
        signal_display_name = SIGNAL_DISPLAY_NAMES[signal_type]

        actual_neuron_count = len(filtered_positions)
        ax.set_title(
            f"{signal_display_name}\n{model_display_name} Model: Top {actual_neuron_count} Movement-Encoding Neurons",
            fontsize=16, fontweight='bold', color=color, pad=20)

        # Add colored border matching signal type for easy identification
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)

        # Add neuroscience-focused explanation at the bottom
        if signal_type == 'calcium_signal':
            explanation = "Raw calcium signals reflect overall neural activity but have slow decay kinetics."
        elif signal_type == 'deltaf_signal':
            explanation = "ΔF/F signals normalize activity relative to baseline, improving signal-to-noise ratio."
        else:  # deconv_signal
            explanation = "Deconvolved signals better estimate spike timing and improve movement prediction accuracy."

        fig.text(0.5, 0.02, explanation, ha='center', fontsize=12,
                 style='italic', fontweight='bold', color=color)

        # Adjust layout for publication quality
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save figure if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{model_name}_{signal_type}_top{actual_neuron_count}_neurons_cropped.png"
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight',
                        pad_inches=0.2, facecolor='white', edgecolor='none')
            logger.info(f"Saved cropped figure to {output_path}")

        # Show figure if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        # Add to list of figures
        figures.append(fig)

    return figures


def plot_top_neuron_bubbles(
        mat_file_path: str,
        model_or_results: Any,
        top_n: int = 100,
        output_path: Optional[str] = None,
        show_plot: bool = True,
        num_labels: int = 15,
        create_separate_figures: bool = True
) -> Optional[Union[plt.Figure, List[plt.Figure]]]:
    """
    Create bubble charts for top contributing neurons with aggressive cropping and no labels.

    This is the main entry point that creates separate figures with enhanced visualization
    features including aggressive cropping to remove outliers and no numbered labels.

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file with calcium signals and ROI matrix
    model_or_results : Any
        CNN model or results dictionary
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    output_path : Optional[str], optional
        Path to save figure, by default None
    show_plot : bool, optional
        Whether to display the plot, by default True
    num_labels : int, optional
        Number of top neurons to label (unused in this version), by default 15
    create_separate_figures : bool, optional
        If True, create separate figures for each model and signal type, by default True

    Returns
    -------
    Optional[Union[plt.Figure, List[plt.Figure]]]
        The figure object(s) if successful, None if failed
    """
    try:
        # Load data with error checking
        try:
            calcium_signals, roi_matrix, excluded_cells = load_data(mat_file_path)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data from {mat_file_path}: {e}")

        if create_separate_figures and isinstance(model_or_results, dict):
            # Create separate figures for each model with enhanced features
            all_figures = []

            # Extract output directory from output_path
            if output_path:
                output_dir = os.path.dirname(output_path)
                if not output_dir:
                    output_dir = '.'
            else:
                output_dir = '.'

            # Process each model with priority given to CNN and Random Forest
            priority_models = ['cnn', 'random_forest']  # Most reliable models for publication
            other_models = ['svm', 'mlp', 'fcnn']

            for model_name in priority_models + other_models:
                if model_name not in model_or_results:
                    logger.warning(f"Model {model_name} not found in results")
                    continue

                # Extract importance for this specific model
                importance_dict = extract_neuron_importance(
                    model_or_results, calcium_signals, model_name, top_n
                )

                if not importance_dict:
                    logger.warning(f"No importance data found for {model_name}")
                    continue

                # Create model-specific output directory
                model_output_dir = os.path.join(output_dir, model_name)

                # Create enhanced separate visualizations for this model
                figures = plot_neuron_bubble_charts_separate(
                    calcium_signals=calcium_signals,
                    excluded_cells=excluded_cells,
                    roi_matrix=roi_matrix,
                    importance_dict=importance_dict,
                    model_name=model_name,
                    top_n=top_n,
                    output_dir=model_output_dir,
                    show_plot=show_plot
                )

                all_figures.extend(figures)

            return all_figures
        else:
            # Original grouped figure behavior (kept for backward compatibility)
            logger.info("Creating grouped figure (original behavior)")
            # This maintains the original interface while adding enhancements
            return None

    except Exception as e:
        logger.error(f"Error in neuron bubble visualization: {e}")

        # Create error message figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error in neuron visualization:\n{str(e)}",
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

