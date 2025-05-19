# # mind/visualization/comprehensive_viz.py
#
# """
# Comprehensive visualization module with neuron bubble charts.
#
# This module integrates the neuron bubble chart visualization into the
# comprehensive visualization system.
# """
# from .create_all_figures import create_all_visualizations
#
# # Re-export the main function for backward compatibility
# __all__ = ['create_all_visualizations']
#
# # Re-export configuration for consistency
# from .config import (
#     SIGNAL_COLORS,
#     SIGNAL_DISPLAY_NAMES,
#     SIGNAL_GRADIENTS,
#     set_publication_style
# )
#
# # Import all visualization functions for backward compatibility
# from .signals import plot_signal_comparison_top20
# from .performance import (
#     plot_confusion_matrix_grid,
#     plot_roc_curve_grid,
#     plot_precision_recall_grid,
#     plot_performance_radar,
#     plot_model_performance_heatmap
# )
# from .feature_importance import (
#     plot_feature_importance_heatmaps,
#     plot_temporal_importance_patterns,
#     plot_top_neuron_importance
# )
#
# # Import the new neuron bubble chart visualization
# from .neuron_importance import plot_top_neuron_bubbles
#
# # Update create_all_visualizations to include neuron bubble charts
# from .create_all_figures import create_all_visualizations as original_create_all_visualizations
#
#
# def create_all_visualizations_with_bubbles(
#         results: dict,
#         calcium_signals: dict,
#         output_dir: str,
#         mat_file_path: str = None,
#         top_n: int = 100
# ) -> None:
#     """
#     Create all visualizations including neuron bubble charts.
#
#     This function extends the original create_all_visualizations function
#     to include neuron bubble charts for top contributing neurons.
#
#     Parameters
#     ----------
#     results : dict
#         Dictionary of results from model training
#     calcium_signals : dict
#         Dictionary of calcium signals
#     output_dir : str
#         Directory to save visualizations
#     mat_file_path : str, optional
#         Path to MATLAB file with ROI matrix, by default None
#     top_n : int, optional
#         Number of top neurons to visualize, by default 100
#     """
#     # Call the original function
#     original_create_all_visualizations(results, calcium_signals, output_dir)
#
#     # Create neuron bubble charts if mat_file_path is provided
#     if mat_file_path is not None:
#         from pathlib import Path
#         import logging
#         logger = logging.getLogger(__name__)
#
#         # Create subdirectory for neuron bubbles
#         bubbles_dir = Path(output_dir) / 'neuron_bubbles'
#         bubbles_dir.mkdir(parents=True, exist_ok=True)
#
#         # Create neuron bubble charts
#         output_path = bubbles_dir / f'top_{top_n}_neurons.png'
#
#         try:
#             from .neuron_importance import plot_top_neuron_bubbles
#             plot_top_neuron_bubbles(
#                 mat_file_path=mat_file_path,
#                 model_or_results=results,
#                 top_n=top_n,
#                 output_path=output_path,
#                 show_plot=False
#             )
#             logger.info(f"Created neuron bubble charts in {output_path}")
#         except Exception as e:
#             logger.error(f"Error creating neuron bubble charts: {e}")
#
#
# # Override the original function
# create_all_visualizations = create_all_visualizations_with_bubbles
#
#
# mind/visualization/comprehensive_viz.py
"""
Comprehensive visualization module with neuron analysis.

This module orchestrates all visualization functions including:
- Signal comparisons
- Performance metrics (confusion matrices, ROC curves, etc.)
- Feature importance analyses
- Venn diagrams for neuron overlap
"""
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Import the original visualization function
from .create_all_figures import create_all_visualizations as _original_create_all_visualizations

# Import configuration for consistency
from .config import (
    SIGNAL_COLORS,
    SIGNAL_DISPLAY_NAMES,
    SIGNAL_GRADIENTS,
    set_publication_style
)

# Import all individual visualization functions for backward compatibility
from .signals import plot_signal_comparison_top20
from .performance import (
    plot_confusion_matrix_grid,
    plot_roc_curve_grid,
    plot_precision_recall_grid,
    plot_performance_radar,
    plot_model_performance_heatmap
)

from .feature_importance import (
    plot_feature_importance_heatmaps,
    plot_temporal_importance_patterns,
    plot_top_neuron_importance
)

# Import the Venn diagram function
from .neuron_overlap import create_neuron_venn_diagram

logger = logging.getLogger(__name__)


def create_all_visualizations(
        results: Dict[str, Dict[str, Any]],
        calcium_signals: Dict[str, Any],
        output_dir: str,
        mat_file_path: Optional[str] = None,
        top_n: int = 100,
        include_venn: bool = True
) -> None:
    """
    Create all visualizations including the Venn diagram.

    This is the main entry point for creating all visualizations from your
    neural decoding results. It creates:
    1. All standard visualizations (performance metrics, signal comparisons, etc.)
    2. A Venn diagram showing neuron overlap across signal types

    Parameters
    ----------
    results : [Dict[str, Any]]
        Results dictionary organized by model and signal type.
        Should contain keys like 'random_forest', 'cnn', etc., each with
        sub-keys for different signal types.
    calcium_signals : Dict[str, Any]
        Dictionary containing the three signal types:
        - 'calcium_signal': raw calcium data
        - 'deltaf_signal': ΔF/F normalized data
        - 'deconv_signal': deconvolved spike data
    output_dir : str
        Directory where all visualizations will be saved
    mat_file_path : Optional[str], optional
        Path to MATLAB file containing calcium signals.
        Required if you want to create the Venn diagram.
    top_n : int, optional
        Number of top neurons to analyze in the Venn diagram, by default 100
    include_venn : bool, optional
        Whether to create the Venn diagram, by default True
    --------
    >>> # After running the experiments and getting results
    >>> create_all_visualizations(
    ...     results=all_results,
    ...     calcium_signals=calcium_signals,
    ...     output_dir="outputs/visualizations",
    ...     mat_file_path="data/calcium_data.mat",
    ...     top_n=100
    ... )
    """
    logger.info("Starting comprehensive visualization creation...")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create all standard visualizations
    logger.info("Creating standard visualizations...")
    try:
        _original_create_all_visualizations(results, calcium_signals, str(output_dir))
        logger.info("Standard visualizations completed successfully")
    except Exception as e:
        logger.error(f"Error creating standard visualizations: {e}")
        raise

    # Step 2: Create Venn diagram if requested
    if include_venn and mat_file_path is not None:
        logger.info("Creating Venn diagram...")
        try:
            # Create subdirectory for neuron analysis
            venn_dir = output_dir / 'neuron_analysis'
            venn_dir.mkdir(parents=True, exist_ok=True)

            # Generate the Venn diagram
            venn_output_path = venn_dir / f'venn_top_{top_n}_neurons.png'

            # Use the best performing model's results (typically CNN)
            # You can modify this to use a different model or all models
            if 'cnn' in results:
                model_results = results
            else:
                # Fallback to the first available model
                model_results = results

            create_neuron_venn_diagram(
                mat_file_path=mat_file_path,
                model_or_results=model_results,
                top_n=top_n,
                output_path=str(venn_output_path),
                show_plot=False  # Don't show plot when batch processing
            )

            logger.info(f"Venn diagram saved to {venn_output_path}")

            # Create a summary text file explaining the Venn diagram
            summary_path = venn_dir / 'venn_diagram_summary.txt'
            _create_venn_summary(results, top_n, summary_path)

        except Exception as e:
            logger.error(f"Error creating Venn diagram: {e}")
            # Don't raise here - we still want other visualizations
    elif include_venn and mat_file_path is None:
        logger.warning("Cannot create Venn diagram: mat_file_path not provided")

    logger.info(f"All visualizations completed! Results saved to: {output_dir}")

    # Create a master summary file
    _create_master_summary(output_dir)


def _create_venn_summary(results: Dict[str, Dict[str, Any]],
                         top_n: int,
                         output_path: Path) -> None:
    """Create a text summary explaining the Venn diagram results."""
    try:
        with open(output_path, 'w') as f:
            f.write("Venn Diagram Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Analysis of top {top_n} contributing neurons across signal types\n\n")
            f.write("This diagram shows:\n")
            f.write("- Bubbles representing individual neurons (numbers inside)\n")
            f.write("- Color coding for different intersection regions\n")
            f.write("- Overlap patterns between signal types\n\n")
            f.write("Interpretation:\n")
            f.write("- Neurons in the center (gold) are important across all signal types\n")
            f.write("- Neurons in overlapping regions show partial consistency\n")
            f.write("- Neurons in individual circles are unique to that signal type\n\n")
            f.write("This analysis helps identify which neurons consistently contribute\n")
            f.write("to behavior prediction regardless of signal processing method.\n")
    except Exception as e:
        logger.error(f"Error creating Venn summary: {e}")


def _create_master_summary(output_dir: Path) -> None:
    """Create a master summary file listing all generated visualizations."""
    try:
        summary_path = output_dir / 'visualization_master_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Neural Decoding Visualization Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write("Generated Visualizations:\n\n")

            # List all subdirectories and their contents
            for subdir in sorted(output_dir.iterdir()):
                if subdir.is_dir():
                    f.write(f"\n{subdir.name}/\n")
                    f.write("-" * len(subdir.name) + "\n")

                    for file in sorted(subdir.glob('*.png')):
                        f.write(f"  - {file.name}\n")

            f.write("\n\nVisualization Guide:\n")
            f.write("- signals/: Signal comparison plots\n")
            f.write("- performance/: Model performance metrics\n")
            f.write("- feature_importance/: Neuron and temporal importance\n")
            f.write("- neuron_analysis/: Venn diagram of neuron overlap\n")

    except Exception as e:
        logger.error(f"Error creating master summary: {e}")


# Convenience function for creating only the Venn diagram
def create_neuron_venn_only(
        mat_file_path: str,
        results: Dict[str, Dict[str, Any]],
        output_path: str,
        top_n: int = 100,
        show_plot: bool = True
) -> None:
    """
    Create only the Venn diagram.

    This is a convenience function if you only want to generate the
    Venn diagram without all other visualizations.

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file containing calcium signals
    results : Dict[str, Dict[str, Any]]
        Results dictionary from model training
    output_path : str
        Where to save the Venn diagram
    top_n : int, optional
        Number of top neurons to analyze, by default 100
    show_plot : bool, optional
        Whether to display the plot, by default True

    Examples
    --------
    >>> create_neuron_venn_only(
    ...     mat_file_path="data/calcium_data.mat",
    ...     results=my_results,
    ...     output_path="venn_diagram.png",
    ...     top_n=100,
    ...     show_plot=True
    ... )
    """
    create_neuron_venn_diagram(
        mat_file_path=mat_file_path,
        model_or_results=results,
        top_n=top_n,
        output_path=output_path,
        show_plot=show_plot
    )


# Export all functions for backward compatibility
__all__ = [
    'create_all_visualizations',
    'create_neuron_venn_only',
    'plot_signal_comparison_top20',
    'plot_confusion_matrix_grid',
    'plot_roc_curve_grid',
    'plot_precision_recall_grid',
    'plot_performance_radar',
    'plot_model_performance_heatmap',
    'plot_feature_importance_heatmaps',
    'plot_temporal_importance_patterns',
    'plot_top_neuron_importance',
    'create_neuron_venn_diagram'
]

