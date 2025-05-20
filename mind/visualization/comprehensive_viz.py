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
from mind.visualization.signals import plot_signal_comparison_top20
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

        # Find top 20 active neurons
        try:
            top_20_indices = find_most_active_neurons(calcium_signals, 20, 'deconv_signal')
        except Exception as e:
            logger.error(f"Error finding most active neurons: {e}")
            # Generate random indices as fallback
            top_20_indices = np.random.choice(calcium_signals['calcium_signal'].shape[1], 20, replace=False)

        # Plot signal comparison
        plot_signal_comparison_top20(
            calcium_signals=calcium_signals,
            top_20_indices=top_20_indices,
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

    # 4. Neuron activity vs. model importance comparison (NEW)
    if mat_file_path:
        try:
            logger.info("Creating neuron activity vs. model importance comparison")

            # Run comprehensive analysis for the two most important models: RF and CNN
            analyze_neuron_activity_importance(
                mat_file_path=mat_file_path,
                results=results,
                output_dir=str(output_dir),
                model_names=['random_forest', 'cnn'],
                top_n=20
            )

            logger.info("Neuron activity comparison visualizations created successfully")
        except Exception as e:
            logger.error(f"Error creating neuron activity comparison: {e}")

    # 5. Neuron-specific visualizations (if mat_file_path provided)
    if mat_file_path:
        try:
            logger.info("Creating neuron-specific visualizations")

            # Import neuron visualization functions
            try:
                from mind.visualization.neuron_importance import plot_top_neuron_bubbles

                # Create neuron bubble charts
                plot_top_neuron_bubbles(
                    mat_file_path=mat_file_path,
                    model_or_results=results,
                    top_n=top_n,
                    output_path=str(neuron_dir / f"top_{top_n}_neurons.png"),
                    show_plot=False
                )

                logger.info("Neuron bubble chart created successfully")

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


def create_neuron_activity_comparison_only(
        mat_file_path: str,
        results: Dict[str, Dict[str, Any]],
        output_dir: str,
        model_names: list = ['random_forest', 'cnn'],
        top_n: int = 20
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
        Number of top neurons to visualize, by default 20
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

