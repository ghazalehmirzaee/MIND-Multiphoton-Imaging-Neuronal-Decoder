# mind/visualization/create_all_figures.py
"""
Main orchestrator for creating all required visualizations with consistent styling.
This module serves as the central entry point for generating all figures.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from mind.data.loader import find_most_active_neurons
from .config import set_publication_style
from .signals import plot_signal_comparison_top
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

logger = logging.getLogger(__name__)


def create_all_visualizations(
        results: Dict[str, Dict[str, Any]],
        calcium_signals: Dict[str, np.ndarray],
        output_dir: Path
) -> None:
    """
    Create all required visualizations with consistent styling.

    This is the main function that orchestrates the creation of all figures
    with consistent color coding and styling throughout.

    Required visualizations:
    1. Signal comparisons for top 5 neurons
    2. Confusion matrices (5x3 grid)
    3. ROC curves (5x3 grid)
    4. Performance radar plots
    5. Feature importance heatmaps
    6. Temporal importance patterns
    7. Top neuron importance
    8. Model performance heatmap
    9. Precision-recall curves (5x3 grid)

    All visualizations use consistent color coding:
    - Calcium: Blue (#356d9e)
    - ΔF/F: Green (#4c8b64)
    - Deconvolved: Red (#a85858)

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary organized by model and signal type
    calcium_signals : Dict[str, np.ndarray]
        Dictionary containing the three signal types
    output_dir : Path
        Directory where all visualizations will be saved
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set consistent publication style for all plots
    set_publication_style()

    logger.info("Starting visualization creation with consistent color scheme...")

    # Create subdirectories for better organization
    subdirs = {
        'signals': output_dir / 'signals',
        'performance': output_dir / 'performance',
        'feature_importance': output_dir / 'feature_importance'
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    # 1. Signal comparisons for top 5 neurons
    try:
        logger.info("Creating signal comparison for top 5 neurons...")
        top_indices = find_most_active_neurons(calcium_signals, 5)
        plot_signal_comparison_top(
            calcium_signals,
            top_indices,
            output_dir=subdirs['signals']
        )
    except Exception as e:
        logger.error(f"Failed to create signal comparison: {e}")

    # 2. Confusion matrix grid (5x3)
    try:
        logger.info("Creating confusion matrix grid...")
        plot_confusion_matrix_grid(results, output_dir=subdirs['performance'])
    except Exception as e:
        logger.error(f"Failed to create confusion matrix grid: {e}")

    # 3. ROC curve grid (5x3)
    try:
        logger.info("Creating ROC curve grid...")
        plot_roc_curve_grid(results, output_dir=subdirs['performance'])
    except Exception as e:
        logger.error(f"Failed to create ROC curve grid: {e}")

    # 4. Performance radar plots
    try:
        logger.info("Creating performance radar plots...")
        plot_performance_radar(results, output_dir=subdirs['performance'])
    except Exception as e:
        logger.error(f"Failed to create performance radar: {e}")

    # 5. Feature importance heatmaps
    try:
        logger.info("Creating feature importance heatmaps...")
        plot_feature_importance_heatmaps(
            results,
            output_dir=subdirs['feature_importance']
        )
    except Exception as e:
        logger.error(f"Failed to create feature importance heatmaps: {e}")

    # 6. Temporal importance patterns
    try:
        logger.info("Creating temporal importance patterns...")
        plot_temporal_importance_patterns(
            results,
            output_dir=subdirs['feature_importance']
        )
    except Exception as e:
        logger.error(f"Failed to create temporal importance patterns: {e}")

    # 7. Top neuron importance
    try:
        logger.info("Creating top neuron importance plots...")
        plot_top_neuron_importance(
            results,
            output_dir=subdirs['feature_importance']
        )
    except Exception as e:
        logger.error(f"Failed to create top neuron importance: {e}")

    # 8. Model performance heatmap
    try:
        logger.info("Creating model performance heatmap...")
        plot_model_performance_heatmap(
            results,
            output_dir=subdirs['performance']
        )
    except Exception as e:
        logger.error(f"Failed to create model performance heatmap: {e}")

    # 9. Precision-recall curves (5x3)
    try:
        logger.info("Creating precision-recall curve grid...")
        plot_precision_recall_grid(results, output_dir=subdirs['performance'])
    except Exception as e:
        logger.error(f"Failed to create precision-recall curves: {e}")

    logger.info(f"All visualizations completed! Files saved to: {output_dir}")

    # Create a summary file listing all created figures
    create_visualization_summary(output_dir, subdirs)


def create_visualization_summary(
        output_dir: Path,
        subdirs: Dict[str, Path]
) -> None:
    """
    Create a summary file listing all generated visualizations.

    This creates a simple text file that lists all the generated figures
    and their locations for easy reference.

    Parameters
    ----------
    output_dir : Path
        Main output directory
    subdirs : Dict[str, Path]
        Dictionary of subdirectories where figures are saved
    """
    summary_path = output_dir / 'visualization_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("Calcium Imaging Neural Decoding - Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("All visualizations use consistent color coding:\n")
        f.write("- Calcium: Blue (#356d9e)\n")
        f.write("- ΔF/F: Green (#4c8b64)\n")
        f.write("- Deconvolved: Red (#a85858)\n\n")
        f.write("Generated Figures:\n")
        f.write("-" * 5 + "\n\n")

        for category, subdir in subdirs.items():
            f.write(f"{category.replace('_', ' ').title()}:\n")

            # List all PNG files in the subdirectory
            for img_file in sorted(subdir.glob('*.png')):
                relative_path = img_file.relative_to(output_dir)
                f.write(f"  - {relative_path}\n")

            f.write("\n")

    logger.info(f"Created visualization summary at {summary_path}")

