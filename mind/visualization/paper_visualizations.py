"""
Additional visualizations for calcium imaging analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


def plot_signal_traces_top_20(calcium_signals: Dict[str, np.ndarray],
                              top_indices: np.ndarray,
                              time_range: tuple = (0, 3000),
                              output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create individual trace plots for all three signal types for top 20 neurons.

    This function creates three separate figures, each showing traces for top 20 neurons
    for one signal type, with neurons arranged vertically.
    """
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    signal_names = ['Raw Calcium Signal', 'ΔF/F Signal', 'Deconvolved Signal']
    colors = ['blue', 'green', 'red']

    time_points = np.arange(time_range[0], time_range[1])

    for signal_type, signal_name, color in zip(signal_types, signal_names, colors):
        fig, ax = plt.subplots(figsize=(15, 12))

        signal = calcium_signals[signal_type]
        if signal is None:
            logger.warning(f"Signal type {signal_type} not available")
            continue

        # Plot each neuron with offset for visibility
        for i, neuron_idx in enumerate(top_indices):
            # Get signal for this neuron
            neuron_signal = signal[time_range[0]:time_range[1], neuron_idx]

            # Normalize signal for better visualization
            if signal_type == 'deconv_signal':
                # For deconvolved signals, scale differently
                offset = i * 0.5
                scale = 0.4
            else:
                # For raw and ΔF/F signals
                mean_val = np.mean(neuron_signal)
                std_val = np.std(neuron_signal)
                neuron_signal = (neuron_signal - mean_val) / (std_val + 1e-8)
                offset = i * 2
                scale = 1

            # Plot with offset
            ax.plot(time_points, neuron_signal * scale + offset,
                    color=color, linewidth=0.8, alpha=0.8)

            # Add neuron label
            ax.text(-50, offset, f'N{neuron_idx}',
                    horizontalalignment='right', verticalalignment='center',
                    fontsize=8)

        ax.set_xlabel('Time (frames)', fontsize=12)
        ax.set_ylabel('Neurons', fontsize=12)
        ax.set_title(f'{signal_name} - Top 20 Active Neurons', fontsize=14, fontweight='bold')

        # Remove y-axis ticks since we have neuron labels
        ax.set_yticks([])

        # Style adjustments
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

        if output_dir:
            output_path = Path(output_dir) / f'{signal_type}_top20_traces.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved {signal_type} traces to {output_path}")

        plt.close(fig)


def plot_scatter_neuron_activity(calcium_signals: Dict[str, np.ndarray],
                                 top_20_indices: np.ndarray,
                                 top_250_indices: np.ndarray,
                                 signal_type: str = 'deconv_signal',
                                 output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create scatter plot of neuron activity levels for top 20 and top 250 neurons.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    signal = calcium_signals[signal_type]
    if signal is None:
        logger.error(f"Signal type {signal_type} not available")
        return None

    # Calculate activity metrics for all neurons
    if signal_type == 'deconv_signal':
        # For deconvolved signals, use spike count
        activity_metric = np.sum(signal > 0, axis=0)
        activity_label = 'Spike Count'
    else:
        # For other signals, use variance
        activity_metric = np.var(signal, axis=0)
        activity_label = 'Signal Variance'

    # Create scatter plot for all neurons (background)
    n_neurons = signal.shape[1]
    all_indices = np.arange(n_neurons)
    ax.scatter(all_indices, activity_metric, c='lightgray', alpha=0.3, s=20, label='All neurons')

    # Highlight top 250 neurons
    ax.scatter(top_250_indices, activity_metric[top_250_indices],
               c='skyblue', alpha=0.7, s=40, label='Top 250')

    # Highlight top 20 neurons
    ax.scatter(top_20_indices, activity_metric[top_20_indices],
               c='red', s=100, marker='*', label='Top 20')

    # Add annotations for top 20 neurons
    for i, idx in enumerate(top_20_indices[:5]):  # Annotate only top 5 to avoid clutter
        ax.annotate(f'N{idx}', (idx, activity_metric[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel(activity_label, fontsize=12)
    ax.set_title(f'Neuron Activity Distribution - {signal_type}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_dir:
        output_path = Path(output_dir) / f'neuron_activity_scatter_{signal_type}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {output_path}")

    return fig


def plot_combined_confusion_matrices(results: Dict[str, Dict[str, Any]],
                                     output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a single figure with all confusion matrices in a 5x3 grid with improved styling.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']

    fig = plt.figure(figsize=(12, 20))

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
            ax = plt.subplot(5, 3, i * 3 + j + 1)

            try:
                cm = results[model][signal]['confusion_matrix']
                accuracy = results[model][signal]['metrics']['accuracy']

                # Calculate percentages for each row
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                # Create annotation text
                annot = np.empty_like(cm, dtype=object)
                for row in range(cm.shape[0]):
                    for col in range(cm.shape[1]):
                        count = cm[row, col]
                        percentage = cm_norm[row, col] * 100
                        annot[row, col] = f'{count}\n({percentage:.1f}%)'

                # Plot heatmap
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=['No footstep', 'Contralateral'],
                            yticklabels=['No footstep', 'Contralateral'])

                ax.set_title(f'{model_name} - {signal_name}\nAccuracy: {accuracy:.3f}',
                             fontsize=11, fontweight='bold')
                ax.set_ylabel('True', fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)

            except KeyError:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.suptitle('Binary Classification Confusion Matrices\nAll Models and Signal Types',
                 fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        output_path = Path(output_dir) / 'confusion_matrices_combined.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined confusion matrices to {output_path}")

    return fig


def plot_performance_summary_table(results: Dict[str, Dict[str, Any]],
                                   output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a comprehensive performance summary table showing all metrics.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Create data for heatmap
    data = np.zeros((len(models), len(signals) * len(metrics)))
    col_labels = []

    for j, signal in enumerate(signals):
        for k, metric in enumerate(metrics):
            col_labels.append(f'{signal_names[j]}\n{metric.capitalize()}')
            for i, model in enumerate(models):
                try:
                    value = results[model][signal]['metrics'][metric]
                    data[i, j * len(metrics) + k] = value
                except KeyError:
                    data[i, j * len(metrics) + k] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Red to yellow to green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(model_names, fontsize=11)

    # Add value annotations
    for i in range(len(model_names)):
        for j in range(len(col_labels)):
            value = data[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color='white' if value < 0.7 else 'black', fontsize=9)

    # Add vertical lines to separate signal types
    for j in range(1, len(signals)):
        ax.axvline(x=j * len(metrics) - 0.5, color='black', linewidth=2)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Performance Score', fontsize=12)

    ax.set_title('Comprehensive Performance Summary', fontsize=16, fontweight='bold', pad=20)

    if output_dir:
        output_path = Path(output_dir) / 'performance_summary_table.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance summary table to {output_path}")

    return fig


def create_paper_visualizations(results: Dict[str, Dict[str, Any]],
                                calcium_signals: Dict[str, np.ndarray],
                                output_dir: Union[str, Path]) -> None:
    """
    Create all visualizations required for the paper.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating paper visualizations...")

    # Find top neurons
    from mind.data.loader import find_most_active_neurons
    top_20_indices = find_most_active_neurons(calcium_signals, 20)
    top_250_indices = find_most_active_neurons(calcium_signals, 250)

    # 1. Individual signal traces for top 20 neurons
    plot_signal_traces_top_20(calcium_signals, top_20_indices, output_dir=output_dir)

    # 2. Scatter plot of neuron activity
    plot_scatter_neuron_activity(calcium_signals, top_20_indices, top_250_indices,
                                 output_dir=output_dir)

    # 3. Combined confusion matrices
    plot_combined_confusion_matrices(results, output_dir)

    # 4. Performance summary table
    plot_performance_summary_table(results, output_dir)

    # 5. Create all other visualizations using the comprehensive viz module
    from mind.visualization.comprehensive_viz import create_all_visualizations
    create_all_visualizations(results, calcium_signals, output_dir)

    logger.info("All paper visualizations created successfully!")

