# mind/visualization/performance.py
"""
Performance visualization module with consistent styling.
Creates confusion matrices, ROC curves, and performance comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any
from sklearn.metrics import auc
import logging

from .config import (SIGNAL_COLORS, SIGNAL_DISPLAY_NAMES, MODEL_DISPLAY_NAMES,
                     set_publication_style, get_signal_colormap, FIGURE_SIZES)

logger = logging.getLogger(__name__)


def plot_confusion_matrix_grid(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a 5x3 grid of confusion matrices for all models and signal types.
    """
    set_publication_style()

    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=FIGURE_SIZES['grid_5x3'],
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, model in enumerate(models):
        for j, signal in enumerate(signals):
            ax = axes[i, j]

            # Get confusion matrix and metrics
            try:
                cm = np.array(results[model][signal]['confusion_matrix'])
                metrics = results[model][signal]['metrics']

                # Calculate percentages row-wise (for imbalanced data)
                cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

                # Create custom colormap based on signal color
                cmap = get_signal_colormap(signal)

                # Plot confusion matrix
                sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap=cmap,
                            cbar=False, ax=ax,
                            xticklabels=['No footstep', 'Contralateral'],
                            yticklabels=['No footstep', 'Contralateral'])

                # Add title with metrics
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                ax.set_title(f'{MODEL_DISPLAY_NAMES[model]} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                             f'Acc: {accuracy:.3f}, F1: {f1_score:.3f}',
                             fontsize=11)

                # Add colored border to match signal type
                color = SIGNAL_COLORS[signal]
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)

            except (KeyError, TypeError, ValueError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    if output_dir:
        output_path = Path(output_dir) / 'confusion_matrix_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix grid to {output_path}")

    return fig


def plot_roc_curve_grid(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a 5x3 grid of ROC curves for all models and signal types.

    Each ROC curve is colored according to its signal type for consistency.
    """
    set_publication_style()

    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=FIGURE_SIZES['grid_5x3'],
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, model in enumerate(models):
        for j, signal in enumerate(signals):
            ax = axes[i, j]
            signal_color = SIGNAL_COLORS[signal]

            try:
                curve_data = results[model][signal].get('curve_data', {})
                if 'roc' in curve_data:
                    fpr = np.array(curve_data['roc']['fpr'])
                    tpr = np.array(curve_data['roc']['tpr'])

                    # Plot ROC curve with signal color
                    ax.plot(fpr, tpr, color=signal_color, linewidth=2.5)
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

                    # Calculate and display AUC
                    auc_score = auc(fpr, tpr)
                    ax.set_title(f'{MODEL_DISPLAY_NAMES[model]} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                                 f'AUC: {auc_score:.3f}', fontsize=11)

                    # Fill area under curve
                    ax.fill_between(fpr, tpr, alpha=0.2, color=signal_color)
                else:
                    ax.text(0.5, 0.5, 'No ROC data',
                            ha='center', va='center', transform=ax.transAxes)

                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(signal_color)
                    spine.set_linewidth(1.5)

            except (KeyError, TypeError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    if output_dir:
        output_path = Path(output_dir) / 'roc_curve_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve grid to {output_path}")

    return fig


def plot_precision_recall_grid(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a 5x3 grid of precision-recall curves for all models and signal types.

    Each curve is colored according to its signal type for consistency.
    """
    set_publication_style()

    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=FIGURE_SIZES['grid_5x3'],
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, model in enumerate(models):
        for j, signal in enumerate(signals):
            ax = axes[i, j]
            signal_color = SIGNAL_COLORS[signal]

            try:
                curve_data = results[model][signal].get('curve_data', {})
                if 'precision_recall' in curve_data:
                    precision = np.array(curve_data['precision_recall']['precision'])
                    recall = np.array(curve_data['precision_recall']['recall'])

                    # Plot precision-recall curve with signal color
                    ax.plot(recall, precision, color=signal_color, linewidth=2.5)

                    # Calculate and display AUC
                    pr_auc = auc(recall, precision)
                    ax.set_title(f'{MODEL_DISPLAY_NAMES[model]} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                                 f'AUC: {pr_auc:.3f}', fontsize=11)

                    # Fill area under curve
                    ax.fill_between(recall, precision, alpha=0.2, color=signal_color)

                    # Add baseline (for imbalanced dataset)
                    baseline = 0.2814  # Percentage of positive class
                    ax.axhline(y=baseline, color='gray', linestyle='--',
                               alpha=0.7, label=f'Baseline: {baseline:.3f}')
                else:
                    ax.text(0.5, 0.5, 'No PR data',
                            ha='center', va='center', transform=ax.transAxes)

                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(signal_color)
                    spine.set_linewidth(1.5)

            except (KeyError, TypeError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    if output_dir:
        output_path = Path(output_dir) / 'precision_recall_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve grid to {output_path}")

    return fig


def plot_performance_radar(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create radar plots showing performance metrics for each signal type.

    Shows accuracy, precision, recall, and F1 score for all models on each
    signal type using radar charts.
    """
    set_publication_style()

    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Professional colors for models
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['grid_1x3'],
                             subplot_kw=dict(projection='polar'))

    for j, signal in enumerate(signals):
        ax = axes[j]

        # Number of metrics
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Plot for each model
        for idx, model in enumerate(models):
            try:
                values = []
                for metric in metrics:
                    value = results[model][signal]['metrics'][metric]
                    values.append(value)
                values += values[:1]

                # Plot with model color
                ax.plot(angles, values, 'o-', linewidth=2,
                        label=MODEL_DISPLAY_NAMES[model], color=model_colors[idx])
                ax.fill(angles, values, alpha=0.15, color=model_colors[idx])

            except KeyError:
                continue

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, size=12)
        ax.set_ylim(0, 1)

        # Set title with signal color
        title_color = SIGNAL_COLORS[signal]
        ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
                     size=18, weight='bold', pad=20, color=title_color)
        ax.grid(True, alpha=0.3)

        # Style
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Reference circle
        ax.plot(angles, [0.5] * len(angles), 'k--', linewidth=0.8, alpha=0.3)

        if j == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    if output_dir:
        output_path = Path(output_dir) / 'performance_radar.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance radar to {output_path}")

    return fig



def plot_model_performance_heatmap(
        results: Dict[str, Dict[str, Any]],
        output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a heatmap showing three metrics for all model-signal combinations.

    This creates a visual summary of model performance across all signal types
    with Accuracy, F1 Score, and ROC AUC grouped together.
    """
    set_publication_style()

    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    metrics = ['f1_score', 'accuracy', 'roc_auc']
    metric_names = ['F1 Score', 'Accuracy', 'ROC AUC']

    # Create subplots for three metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        matrix = np.zeros((len(models), len(signals)))

        for i, model in enumerate(models):
            for j, signal in enumerate(signals):
                try:
                    if metric == 'roc_auc':
                        matrix[i, j] = results[model][signal]['metrics'].get('roc_auc', np.nan)
                    else:
                        matrix[i, j] = results[model][signal]['metrics'][metric]
                except KeyError:
                    matrix[i, j] = np.nan

        # Create heatmap with academic color scheme
        if metric == 'f1_score':
            cmap = 'YlOrRd'  # Yellow to Red
            vmin, vmax = 0.55, 0.92
        elif metric == 'accuracy':
            cmap = 'Blues'
            vmin, vmax = 0.75, 0.925
        else:  # ROC AUC
            cmap = 'Greens'
            vmin, vmax = 0.85, 0.98

        # Plot heatmap
        im = sns.heatmap(matrix, annot=True, fmt='.3f', cmap=cmap,
                         xticklabels=[SIGNAL_DISPLAY_NAMES[s] for s in signals],
                         yticklabels=[MODEL_DISPLAY_NAMES[m] for m in models],
                         cbar_kws={'label': metric_name}, ax=ax,
                         vmin=vmin, vmax=vmax,
                         cbar=True, square=True)

        ax.set_title(f'{metric_name} Heatmap', fontsize=14, fontweight='bold')

        # Color the x-axis labels according to signal type
        for ticklabel, signal in zip(ax.get_xticklabels(), signals):
            ticklabel.set_color(SIGNAL_COLORS[signal])
            ticklabel.set_weight('bold')
            ticklabel.set_fontsize(11)

        # Style y-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

        # Add box around heatmap
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    if output_dir:
        output_path = Path(output_dir) / 'model_performance_heatmap.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model performance heatmap to {output_path}")

    return fig

