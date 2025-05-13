# mind/visualization/comprehensive_viz.py
"""
Updated comprehensive visualization module that uses modular components.
This serves as a backward-compatible wrapper for the new modular system.
"""
from .create_all_figures import create_all_visualizations

# Re-export the main function for backward compatibility
__all__ = ['create_all_visualizations']

# Re-export configuration for consistency
from .config import (
    SIGNAL_COLORS,
    SIGNAL_DISPLAY_NAMES,
    SIGNAL_GRADIENTS,
    set_publication_style
)

# Import all visualization functions for backward compatibility
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

