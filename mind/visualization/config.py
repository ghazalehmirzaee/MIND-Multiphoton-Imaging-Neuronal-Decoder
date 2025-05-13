# mind/visualization/config.py
"""
Centralized configuration for all visualizations ensuring consistency.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Define consistent scientific color scheme
SIGNAL_COLORS = {
    'calcium_signal': '#356d9e',  # Scientific blue
    'deltaf_signal': '#4c8b64',  # Scientific green
    'deconv_signal': '#a85858'  # Scientific red
}

# Display names for signals
SIGNAL_DISPLAY_NAMES = {
    'calcium_signal': 'Calcium',
    'deltaf_signal': 'ΔF/F',
    'deconv_signal': 'Deconvolved'
}

# Gradient colors for heatmaps
SIGNAL_GRADIENTS = {
    'calcium_signal': ['#f0f4f9', '#c6dcef', '#7fb0d3', '#356d9e'],
    'deltaf_signal': ['#f6f9f4', '#d6ead9', '#9dcaa7', '#4c8b64'],
    'deconv_signal': ['#fdf3f3', '#f0d0d0', '#d49c9c', '#a85858']
}

# Model display names
MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'svm': 'SVM',
    'mlp': 'MLP',
    'fcnn': 'FCNN',
    'cnn': 'CNN'
}

# Standard figure sizes
FIGURE_SIZES = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 12),
    'grid_5x3': (12, 20),
    'grid_1x3': (18, 6)
}


def set_publication_style():
    """Set publication-quality plot styling."""
    plt.style.use('seaborn-v0_8-white')
    sns.set_style("white")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def get_signal_colormap(signal_type):
    """Get a custom colormap for a specific signal type."""
    from matplotlib.colors import LinearSegmentedColormap
    gradient = SIGNAL_GRADIENTS[signal_type]
    return LinearSegmentedColormap.from_list('custom', gradient)
