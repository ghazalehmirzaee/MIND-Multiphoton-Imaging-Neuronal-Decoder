"""
Complete visualization script for highlighting important neurons on ROI matrix
This script integrates with your MIND project to visualize the most important neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.io import loadmat
from pathlib import Path
import logging
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from scipy.ndimage import binary_opening, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import filters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from your existing project
try:
    from mind.data.loader import load_calcium_signals, find_most_active_neurons
    from mind.evaluation.feature_importance import extract_feature_importance
except ImportError:
    logger.warning("MIND package not found, defining substitute functions")


    def load_calcium_signals(mat_file_path):
        """Substitute function for loading calcium signals if MIND package is not available"""
        logger.info(f"Loading calcium signals from {mat_file_path}")
        try:
            data = loadmat(mat_file_path)
            calcium_signal = data.get('calciumsignal', None)
            deltaf_signal = data.get('deltaf_cells_not_excluded', None)
            deconv_signal = data.get('DeconvMat_wanted', None)

            # Log shapes
            if calcium_signal is not None:
                logger.info(f"Raw calcium signal shape: {calcium_signal.shape}")
            if deltaf_signal is not None:
                logger.info(f"ΔF/F signal shape: {deltaf_signal.shape}")
            if deconv_signal is not None:
                logger.info(f"Deconvolved signal shape: {deconv_signal.shape}")

            return {
                'calcium_signal': calcium_signal,
                'deltaf_signal': deltaf_signal,
                'deconv_signal': deconv_signal
            }
        except Exception as e:
            logger.error(f"Error loading {mat_file_path}: {e}")
            raise


    def find_most_active_neurons(calcium_signals, n_neurons=20, signal_type='deconv_signal'):
        """Substitute function for finding most active neurons"""
        signal = calcium_signals[signal_type]
        if signal is None:
            for alt_signal in ['deltaf_signal', 'calcium_signal']:
                if calcium_signals[alt_signal] is not None:
                    signal = calcium_signals[alt_signal]
                    break

        # Calculate activity metrics
        if signal_type == 'deconv_signal':
            activity_metric = np.sum(signal > 0, axis=0)  # Count of active frames
        else:
            activity_metric = np.var(signal, axis=0)  # Variance

        # Get indices of top neurons
        top_indices = np.argsort(activity_metric)[::-1][:n_neurons]
        return top_indices


def load_roi_matrix(mat_file_path):
    """
    Load the ROI matrix from the MATLAB file.

    This function handles the specific structure of your MATLAB file,
    extracting the ROI_matrix variable.
    """
    logger.info(f"Loading ROI matrix from {mat_file_path}")

    try:
        # Load the MATLAB file
        mat_data = loadmat(mat_file_path)

        # Extract the ROI matrix
        if 'ROI_matrix' in mat_data:
            roi_matrix = mat_data['ROI_matrix']
            logger.info(f"ROI matrix shape: {roi_matrix.shape}")
            return roi_matrix
        else:
            raise KeyError("ROI_matrix not found in MATLAB file")

    except Exception as e:
        logger.error(f"Error loading ROI matrix: {e}")
        raise


def find_neuron_positions_in_roi(roi_matrix, n_neurons=None):
    """
    Find neuron positions using watershed segmentation.

    This improved function better identifies neuron centers even in complex ROIs.
    """
    logger.info("Finding neuron positions using watershed segmentation")

    # Create a binary mask for non-zero values
    binary_mask = roi_matrix > 0

    # Clean up the mask with morphological operations
    binary_mask = binary_opening(binary_mask, structure=np.ones((3, 3)))
    binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3)))

    # Distance transform for watershed
    distance = distance_transform_edt(binary_mask)

    # Find local maxima (neuron centers)
    # Adjust min_distance based on expected neuron size
    min_distance = max(5, roi_matrix.shape[0] // 100)  # Scale with image size
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)

    # Create markers for watershed
    markers = np.zeros_like(roi_matrix, dtype=int)
    for i, (x, y) in enumerate(coords):
        markers[x, y] = i + 1

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary_mask)

    # Find center of mass and calculate importance for each region
    positions = []
    for i in range(1, np.max(labels) + 1):
        mask = (labels == i)
        # Skip very small regions that might be noise
        if np.sum(mask) < 10:
            continue

        # Calculate center of mass
        y, x = center_of_mass(roi_matrix, labels=mask)

        # Calculate mean intensity as a measure of importance
        mean_intensity = np.mean(roi_matrix[mask])
        positions.append((y, x, i, mean_intensity))

    logger.info(f"Found {len(positions)} neuron positions")
    return positions


def map_neurons_to_importance(positions, calcium_signals=None, signal_type='deconv_signal', n_neurons=50):
    """
    Map neurons to importance using either calcium signals or ROI intensity.

    This function can use either:
    1. Calcium signal activity (if provided)
    2. ROI intensity as a proxy for importance (if calcium signals not provided)
    """
    if calcium_signals is not None and signal_type in calcium_signals and calcium_signals[signal_type] is not None:
        logger.info(f"Mapping importance based on {signal_type} activity")
        # Use calcium signals to determine importance
        signal = calcium_signals[signal_type]

        # Calculate activity metric for each neuron
        if signal_type == 'deconv_signal':
            activity_metric = np.sum(signal > 0, axis=0)  # Spike count
        else:
            activity_metric = np.var(signal, axis=0)  # Variance as activity measure

        # Sort positions by ROI intensity first (as a proxy for ranking)
        sorted_by_intensity = sorted(positions, key=lambda x: x[3], reverse=True)

        # Create mapping: calcium index -> position data
        # This assumes the most active neurons in calcium data correspond to
        # the brightest neurons in the ROI matrix (a reasonable approximation)
        important_indices = np.argsort(activity_metric)[::-1][:min(n_neurons, len(activity_metric))]

        neuron_positions = {}
        for i, idx in enumerate(important_indices):
            if i < len(sorted_by_intensity):
                neuron_positions[idx] = sorted_by_intensity[i][:3]  # Store (y, x, id)
    else:
        logger.info("Mapping importance based on ROI intensity")
        # Use ROI intensity to determine importance
        sorted_positions = sorted(positions, key=lambda x: x[3], reverse=True)
        top_positions = sorted_positions[:n_neurons]

        neuron_positions = {}
        important_indices = []

        for i, (y, x, roi_id, _) in enumerate(top_positions):
            neuron_idx = i  # Use position index as neuron index
            neuron_positions[neuron_idx] = (y, x, roi_id)
            important_indices.append(neuron_idx)

    return important_indices, neuron_positions


def plot_roi_with_important_neurons(roi_matrix, important_indices, neuron_positions,
                                    importance_scores=None, output_path=None):
    """
    Create the main visualization showing ROI matrix with highlighted important neurons.

    This improved version uses better color mapping and ensures neurons are clearly visible.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Display the ROI matrix as background with better contrast
    display_matrix = roi_matrix.copy().astype(float)
    vmin, vmax = np.percentile(display_matrix[display_matrix > 0], [5, 95])

    im = ax.imshow(display_matrix, cmap='gray', alpha=0.7,
                   vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    # Create color mapping for importance levels with a better colormap
    n_neurons = len(important_indices)
    colors = plt.cm.plasma(np.linspace(0, 0.8, n_neurons))  # Using plasma for better visibility

    # Plot circles around important neurons
    for rank, (neuron_idx, color) in enumerate(zip(important_indices, colors)):
        if neuron_idx in neuron_positions:
            y, x, _ = neuron_positions[neuron_idx]

            # Size of circle based on importance (most important = larger)
            # Use a better scaling to make differences more visible
            radius = 12 - (8 * rank / n_neurons)  # Gradually decrease size
            radius = max(radius, 5)  # Minimum radius

            # Draw circle
            circle = Circle((x, y), radius=radius, fill=False,
                            edgecolor=color, linewidth=2.5, alpha=0.9)
            ax.add_patch(circle)

            # Add labels for top 20 neurons with better positioning
            if rank < 20:
                # Position label strategically to avoid overlap
                # Calculate angle based on rank to distribute labels in a circle
                angle = 2 * np.pi * (rank % 8) / 8
                offset_dist = radius + 5
                offset_x = offset_dist * np.cos(angle)
                offset_y = offset_dist * np.sin(angle)

                ax.text(x + offset_x, y + offset_y, f'#{rank + 1}',
                        color=color, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8))

    # Set title and labels
    ax.set_title('ROI Matrix with Top 50 Most Important Neurons',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=14)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=14)

    # Add colorbar for ROI intensity
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ROI Intensity', rotation=270, labelpad=20, fontsize=12)

    # Create custom legend for neuron importance
    legend_elements = []
    importance_levels = [(1, 'Top 10', plt.cm.plasma(0.1)),
                         (20, 'Top 20', plt.cm.plasma(0.3)),
                         (35, 'Top 35', plt.cm.plasma(0.5)),
                         (50, 'Top 50', plt.cm.plasma(0.7))]

    for rank, label, color in importance_levels:
        legend_elements.append(mpatches.Patch(color=color, label=label))

    legend = ax.legend(handles=legend_elements, loc='upper right',
                       title='Neuron Importance', fontsize=10)
    legend.get_title().set_fontsize(12)
    legend.get_title().set_fontweight('bold')

    # Add grid for better coordinate reading
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()

    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")

    return fig


def create_zoomed_visualization(roi_matrix, important_indices, neuron_positions,
                                output_dir, zoom_factor=2):
    """
    Create a zoomed-in visualization focusing on regions with important neurons.

    This function intelligently selects areas with high concentrations of important neurons.
    """
    # Get positions of important neurons
    positions = []
    for idx in important_indices:
        if idx in neuron_positions:
            y, x, _ = neuron_positions[idx]
            positions.append((y, x))

    if not positions:
        logger.warning("No positions found for zoomed visualization")
        return None

    # Find the center of the cluster of important neurons
    positions = np.array(positions)
    center_y, center_x = np.median(positions, axis=0)  # Use median for robustness

    # Define zoom window
    window_size = min(roi_matrix.shape) // zoom_factor
    y_start = max(0, int(center_y - window_size // 2))
    y_end = min(roi_matrix.shape[0], int(center_y + window_size // 2))
    x_start = max(0, int(center_x - window_size // 2))
    x_end = min(roi_matrix.shape[1], int(center_x + window_size // 2))

    # Extract zoomed region
    zoomed_roi = roi_matrix[y_start:y_end, x_start:x_end]

    # Adjust positions to zoomed coordinates
    zoomed_positions = {}
    for idx, (y, x, roi_id) in neuron_positions.items():
        if y_start <= y < y_end and x_start <= x < x_end:
            zoomed_positions[idx] = (y - y_start, x - x_start, roi_id)

    # Create zoomed visualization
    output_path = output_dir / "roi_important_neurons_zoomed.png"
    fig = plot_roi_with_important_neurons(
        zoomed_roi, important_indices, zoomed_positions,
        output_path=output_path
    )

    if fig is not None:
        # Update title
        fig.axes[0].set_title('ROI Matrix with Top 50 Neurons (Zoomed View)',
                              fontsize=18, fontweight='bold')

    return fig


def main(mat_file_path, output_dir="outputs/roi_visualizations",
         signal_type='deconv_signal', n_neurons=50):
    """
    Main function to run the complete visualization pipeline.

    This function:
    1. Loads the ROI matrix and calcium signals
    2. Identifies neuron positions using watershed segmentation
    3. Maps neurons to importance measures
    4. Creates visualizations with proper highlighting
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load the ROI matrix
        roi_matrix = load_roi_matrix(mat_file_path)

        # Step 2: Load calcium signals
        logger.info("Loading calcium signals")
        calcium_signals = load_calcium_signals(mat_file_path)

        # Step 3: Find neuron positions in ROI using improved method
        positions = find_neuron_positions_in_roi(roi_matrix, n_neurons * 2)

        # Step 4: Map neurons to importance
        important_indices, neuron_positions = map_neurons_to_importance(
            positions, calcium_signals, signal_type, n_neurons
        )

        # Step 5: Create the visualization
        output_path = output_dir / f"roi_important_neurons_{signal_type}.png"
        fig = plot_roi_with_important_neurons(
            roi_matrix, important_indices, neuron_positions,
            output_path=output_path
        )

        # Step 6: Create zoomed visualization if the ROI is large enough
        if max(roi_matrix.shape) > 200:
            zoomed_fig = create_zoomed_visualization(
                roi_matrix, important_indices, neuron_positions, output_dir
            )

        plt.show()
        logger.info("Visualization complete!")

    except Exception as e:
        logger.error(f"Error in main visualization pipeline: {e}", exc_info=True)
        raise


# Run the visualization
if __name__ == "__main__":
    # Set the path to your MATLAB file
    mat_file = "/home/ghazal/Documents/NS_Projects/NS_P2_050325/MIND-Multiphoton-Imaging-Neural-Decoder/data/raw/SFL13_5_8112021_002_new.mat"

    # Run the main visualization
    main(mat_file)

    # different parameters:
    # main(mat_file, signal_type='deltaf_signal', n_neurons=30)
