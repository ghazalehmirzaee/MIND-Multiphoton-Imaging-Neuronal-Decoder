"""
Comprehensive visualization module for calcium imaging data.

This is the main entry point for creating all visualizations,
orchestrating the different visualization components.
"""
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import numpy as np

# Import visualization components
from mind.visualization.config import set_publication_style
from mind.visualization.signals import plot_signal_comparison_top
from mind.visualization.performance import (
    plot_confusion_matrix_grid,
    plot_roc_curve_grid,
    plot_precision_recall_grid,
    plot_performance_radar,
    plot_model_performance_heatmap
)
from mind.visualization.feature_importance import (
    plot_feature_importance_heatmaps,
    plot_temporal_importance_patterns,
    plot_top_neuron_importance
)

# Import new neuron activity comparison module
from mind.visualization.neuron_activity_comparison import (
    analyze_neuron_activity_importance,
    create_comparison_grid
)

# Import loader for finding active neurons
from mind.data.loader import find_most_active_neurons

logger = logging.getLogger(__name__)


def create_all_visualizations(
        results: Dict[str, Dict[str, Any]],
        calcium_signals: Dict[str, np.ndarray],
        output_dir: str,
        mat_file_path: Optional[str] = None,
        top_n: int = 100
) -> None:
    """
    Create all visualizations with modular organization and error handling.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    calcium_signals : Dict[str, np.ndarray]
        Dictionary of calcium signals
    output_dir : str
        Directory to save visualizations
    mat_file_path : Optional[str], optional
        Path to MATLAB file for neuron visualizations, by default None
    top_n : int, optional
        Number of top neurons to visualize, by default 100
    """
    logger.info("Starting comprehensive visualization creation")

    # Set publication style for all plots
    set_publication_style()

    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    signal_dir = output_dir / 'signals'
    perf_dir = output_dir / 'performance'
    feat_dir = output_dir / 'feature_importance'
    neuron_dir = output_dir / 'neuron_analysis'

    signal_dir.mkdir(exist_ok=True)
    perf_dir.mkdir(exist_ok=True)
    feat_dir.mkdir(exist_ok=True)
    neuron_dir.mkdir(exist_ok=True)

    # 1. Signal visualizations
    try:
        logger.info("Creating signal visualizations")

        # Find top 5 active neurons
        try:
            top_indices = find_most_active_neurons(calcium_signals, 5, 'deconv_signal')
        except Exception as e:
            logger.error(f"Error finding most active neurons: {e}")
            # Generate random indices as fallback
            top_indices = np.random.choice(calcium_signals['calcium_signal'].shape[1], 5, replace=False)

        # Plot signal comparison
        plot_signal_comparison_top(
            calcium_signals=calcium_signals,
            top_indices=top_indices,
            output_dir=signal_dir
        )

        logger.info("Signal visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating signal visualizations: {e}")

    # 2. Performance visualizations
    try:
        logger.info("Creating performance visualizations")

        # Create confusion matrix grid
        plot_confusion_matrix_grid(results=results, output_dir=perf_dir)

        # Create ROC curve grid
        plot_roc_curve_grid(results=results, output_dir=perf_dir)

        # Create precision-recall curve grid
        plot_precision_recall_grid(results=results, output_dir=perf_dir)

        # Create performance radar plot
        plot_performance_radar(results=results, output_dir=perf_dir)

        # Create model performance heatmap
        plot_model_performance_heatmap(results=results, output_dir=perf_dir)

        logger.info("Performance visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating performance visualizations: {e}")

    # 3. Feature importance visualizations
    try:
        logger.info("Creating feature importance visualizations")

        # Create feature importance heatmaps
        plot_feature_importance_heatmaps(results=results, output_dir=feat_dir)

        # Create temporal importance patterns
        plot_temporal_importance_patterns(results=results, output_dir=feat_dir)

        # Create top neuron importance
        plot_top_neuron_importance(results=results, output_dir=feat_dir)

        logger.info("Feature importance visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating feature importance visualizations: {e}")

    # 5. Neuron-specific visualizations with SEPARATE FIGURES (MODIFIED)
    if mat_file_path:
        try:
            logger.info("Creating neuron-specific visualizations with separate figures for each model and signal")

            # Import neuron visualization functions
            try:
                from mind.visualization.neuron_importance import plot_top_neuron_bubbles

                # Create neuron bubble charts with SEPARATE FIGURES for each model and signal
                # This will create 9 individual figures (3 models x 3 signals)
                plot_top_neuron_bubbles(
                    mat_file_path=mat_file_path,
                    model_or_results=results,
                    top_n=top_n,
                    output_path=str(neuron_dir / f"separate_figures"),  # Base path for separate figures
                    show_plot=False,
                    create_separate_figures=True  # NEW PARAMETER TO CREATE SEPARATE FIGURES
                )

                logger.info("Created separate neuron bubble charts for each model and signal type")

                # Try to create Venn diagram if matplotlib_venn is available
                try:
                    from mind.visualization.neuron_overlap import create_neuron_venn_diagram

                    create_neuron_venn_diagram(
                        mat_file_path=mat_file_path,
                        model_or_results=results,
                        top_n=top_n,
                        output_path=str(neuron_dir / f"venn_diagram_top_{top_n}.png"),
                        show_plot=False
                    )

                    logger.info("Venn diagram created successfully")
                except ImportError:
                    logger.warning("Could not import neuron_overlap module. Skipping Venn diagram.")
                except Exception as e:
                    logger.error(f"Error creating Venn diagram: {e}")

            except ImportError:
                logger.warning("Could not import neuron_importance module. Skipping neuron visualizations.")

        except Exception as e:
            logger.error(f"Error creating neuron-specific visualizations: {e}")

    logger.info(f"All visualizations created in {output_dir}")

    # Create a summary file listing all created visualizations
    create_visualization_summary(output_dir)


def create_visualization_summary(output_dir: Path):
    """
    Create a summary file listing all generated visualizations including the new separate neuron figures.
    """
    summary_path = output_dir / 'visualization_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("MIND Project - Calcium Imaging Neural Decoding Visualization Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("All visualizations use consistent color coding:\n")
        f.write("- Calcium: Blue (#356d9e)\n")
        f.write("- ΔF/F: Green (#4c8b64)\n")
        f.write("- Deconvolved: Red (#a85858)\n\n")

        f.write("Generated Visualizations:\n")
        f.write("-" * 30 + "\n\n")

        # Check which directories exist and list their contents
        subdirs = {
            'signals': 'Signal Comparisons',
            'performance': 'Performance Metrics',
            'feature_importance': 'Feature Importance',
            'neuron_analysis': 'Neuron Analysis'
        }

        for subdir_name, description in subdirs.items():
            subdir_path = output_dir / subdir_name
            if subdir_path.exists():
                f.write(f"{description} ({subdir_name}/):\n")

                # List PNG files
                png_files = sorted(subdir_path.glob('*.png'))
                for png_file in png_files:
                    f.write(f"  - {png_file.name}\n")

                # Check for model-specific subdirectories (for separate neuron figures)
                model_dirs = ['random_forest', 'svm', 'cnn']
                for model_dir in model_dirs:
                    model_path = subdir_path / model_dir
                    if model_path.exists():
                        f.write(f"  {model_dir}/:\n")
                        model_pngs = sorted(model_path.glob('*.png'))
                        for png_file in model_pngs:
                            f.write(f"    - {png_file.name}\n")

                f.write("\n")

        f.write("\nNEW: Separate Neuron Importance Visualizations\n")
        f.write("-" * 45 + "\n")
        f.write("Individual figures have been created for each model and signal type combination:\n")
        f.write("- 9 total figures (3 models × 3 signal types)\n")
        f.write("- No neuron indices for cleaner scientific presentation\n")
        f.write("- Intelligent cropping to focus on relevant neuron regions\n")
        f.write("- Bubble size indicates neuron importance for movement prediction\n")
        f.write("\nLocation: neuron_analysis/[model_name]/[model]_[signal]_top100_neurons.png\n")

        f.write("\nVisualization Details:\n")
        f.write("-" * 5 + "\n")
        f.write("1. Signal Comparisons: Shows top 5 active neurons across all three signal types\n")
        f.write("2. Performance Metrics: Confusion matrices, ROC curves, PR curves, and radar plots\n")
        f.write("3. Feature Importance: Temporal patterns and neuron-specific importance\n")
        f.write("4. Neuron Analysis: Individual figures for top 100 neurons per model/signal combination\n")

    logger.info(f"Created visualization summary at {summary_path}")


def create_neuron_activity_comparison_only(
        mat_file_path: str,
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        model_names: list = ['random_forest', 'cnn'],
        top_n: int = 5
) -> None:
    """
    Create only neuron activity vs. model importance comparison visualizations.

    This function is useful when you want to run just the activity-importance
    comparison without generating all other visualizations.

    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB file with calcium signals and ROI matrix
    results : Dict[str, Dict[str, Any]]
        Results dictionary
    output_dir : str
        Output directory
    model_names : list, optional
        Models to analyze, by default ['random_forest', 'cnn']
    top_n : int, optional
        Number of top neurons to visualize
    """
    logger.info("Creating neuron activity vs. model importance comparison only")

    # Set publication style
    set_publication_style()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create neuron analysis directory
    neuron_dir = output_dir / 'neuron_analysis'
    neuron_dir.mkdir(exist_ok=True)

    # Run comprehensive analysis
    analysis_results = analyze_neuron_activity_importance(
        mat_file_path=mat_file_path,
        results=results,
        output_dir=str(output_dir),
        model_names=model_names,
        top_n=top_n
    )

    # Create comparison grid
    create_comparison_grid(
        mat_file_path=mat_file_path,
        model_or_results=results,
        output_dir=str(neuron_dir),
        model_names=model_names,
        top_n=top_n,
        show_plot=False
    )

    logger.info(f"Neuron activity comparison visualizations created in {neuron_dir}")

    return analysis_results


