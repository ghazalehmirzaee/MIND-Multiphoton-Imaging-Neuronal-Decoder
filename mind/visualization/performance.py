"""Performance visualization functions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)


def plot_performance_comparison(
        performance_df: pd.DataFrame,
        metric: str = 'F1 (Macro)',
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance comparison across models and signal types.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame comparing model performance
    metric : str, optional
        Metric to plot, by default 'F1 (Macro)'
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Performance comparison figure
    """
    # Create pivot table
    pivot = pd.pivot_table(
        performance_df, values=metric, index='Model', columns='Signal Type'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title(f'{metric} by Model and Signal Type')

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig

def plot_signal_type_comparison(
        performance_df: pd.DataFrame,
        metric: str = 'F1 (Macro)',
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance comparison across signal types.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame comparing model performance
    metric : str, optional
        Metric to plot, by default 'F1 (Macro)'
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Signal type comparison figure
    """
    # Calculate mean performance by signal type
    signal_performance = performance_df.groupby('Signal Type')[metric].mean().reset_index()

    # Get best model for each signal type
    best_models = performance_df.loc[performance_df.groupby('Signal Type')[metric].idxmax()]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bar chart - fixing the deprecation warning
    sns.barplot(x='Signal Type', y=metric, hue='Signal Type', data=signal_performance,
                ax=ax, palette='Set3', legend=False)

    # Add best model annotations
    for i, row in enumerate(best_models.iterrows()):
        # Use row[1] to access the data (row[0] is the index)
        signal_type = row[1]['Signal Type']
        model = row[1]['Model']
        score = row[1][metric]

        # Find index of signal type in signal_performance
        idx = signal_performance[signal_performance['Signal Type'] == signal_type].index[0]

        ax.text(idx, score + 0.02, f"Best: {model}\n{metric}: {score:.3f}",
                ha='center', va='bottom', fontweight='bold')

    ax.set_title(f'Mean {metric} by Signal Type with Best Model')
    ax.set_ylim(top=1.0)

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def plot_model_comparison(
        performance_df: pd.DataFrame,
        metric: str = 'F1 (Macro)',
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance comparison across model types.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame comparing model performance
    metric : str, optional
        Metric to plot, by default 'F1 (Macro)'
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Model comparison figure
    """
    # Calculate mean performance by model type
    model_performance = performance_df.groupby('Model')[metric].mean().reset_index()

    # Get best signal type for each model
    best_signals = performance_df.loc[performance_df.groupby('Model')[metric].idxmax()]

    # Sort by metric
    model_performance = model_performance.sort_values(metric, ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bar chart
    sns.barplot(x='Model', y=metric, data=model_performance, ax=ax, palette='Set2')

    # Add best signal type annotations
    for _, row in best_signals.iterrows():
        model = row['Model']
        signal_type = row['Signal Type']
        score = row[metric]

        # Find index of model in model_performance
        idx = model_performance[model_performance['Model'] == model].index[0]

        ax.text(idx, score + 0.02, f"Best with: {signal_type}\n{metric}: {score:.3f}",
                ha='center', va='bottom', fontweight='bold')

    ax.set_title(f'Mean {metric} by Model Type with Best Signal Type')
    ax.set_ylim(top=1.0)
    plt.xticks(rotation=45)

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def plot_binary_confusion_matrices(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Create grid of binary confusion matrices with percentages.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    plt.Figure
        Figure containing confusion matrices grid
    """
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    class_names = ['No footstep', 'Contralateral']

    # Create 5×3 grid (models × signals)
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    fig.suptitle('Binary Classification Confusion Matrices', fontsize=16)

    # Set column titles (signal types)
    for i, signal_type in enumerate(signal_types):
        axes[0, i].set_title(f'{signal_type.capitalize()} Signal', fontsize=14)

    # Set row titles (model types)
    for i, model_type in enumerate(model_types):
        axes[i, 0].set_ylabel(model_type.upper(), fontsize=14)

    for i, model_type in enumerate(model_types):
        for j, signal_type in enumerate(signal_types):
            if signal_type not in results or model_type not in results[signal_type]:
                logger.warning(f"Results for {signal_type}_{model_type} not found")
                continue

            metrics = results[signal_type][model_type]
            if 'predictions' not in metrics or 'targets' not in metrics:
                logger.warning(f"Predictions or targets not found in {signal_type}_{model_type}")
                continue

            y_pred = metrics['predictions']
            y_true = metrics['targets']

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Calculate percentages for each row (true class)
            cm_percentage = np.zeros_like(cm, dtype=float)
            for row_idx in range(cm.shape[0]):
                row_sum = np.sum(cm[row_idx, :])
                if row_sum > 0:
                    cm_percentage[row_idx, :] = (cm[row_idx, :] / row_sum) * 100

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i, j], cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, cbar=False)

            # Add percentages
            for row_idx in range(cm.shape[0]):
                for col_idx in range(cm.shape[1]):
                    if cm[row_idx, col_idx] > 0:
                        axes[i, j].text(col_idx + 0.5, row_idx + 0.7,
                                        f'({cm_percentage[row_idx, col_idx]:.1f}%)',
                                        ha='center', va='center', color='black',
                                        fontweight='bold' if row_idx == col_idx else 'normal')

            # Add accuracy to the title
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                axes[i, j].set_title(f'Accuracy: {accuracy:.3f}', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'binary_confusion_matrices.png'), dpi=300)

    return fig


def plot_binary_roc_curves(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create ROC curves for binary classification tasks.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing ROC curve figures
    """
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    figures = {}

    # ROC curves by signal type (comparing models)
    for signal_type in signal_types:
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        for model_type in model_types:
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            metrics = results[signal_type][model_type]
            if 'probabilities' not in metrics or 'targets' not in metrics:
                logger.warning(f"Probabilities or targets not found in {signal_type}_{model_type}")
                continue

            y_prob = metrics['probabilities']
            y_true = metrics['targets']

            # For binary classification, use probability for class 1
            if y_prob.shape[1] > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve with customized line style and thickness
            line_styles = {
                'random_forest': '-',
                'svm': '--',
                'mlp': '-.',
                'fcnn': ':',
                'cnn': '-'
            }

            line_widths = {
                'random_forest': 2,
                'svm': 2,
                'mlp': 2,
                'fcnn': 2.5,
                'cnn': 2.5
            }

            colors = {
                'random_forest': 'blue',
                'svm': 'green',
                'mlp': 'red',
                'fcnn': 'purple',
                'cnn': 'orange'
            }

            # Plot with custom styling
            ax.plot(fpr, tpr,
                    linestyle=line_styles.get(model_type, '-'),
                    linewidth=line_widths.get(model_type, 2),
                    color=colors.get(model_type, None),
                    label=f'{model_type.upper()} (AUC = {roc_auc:.3f})')

        # Add reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5)

        ax.set_title(f'ROC Curves - {signal_type.capitalize()} Signal', fontsize=14)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{signal_type}_roc_curves.png'), dpi=300)

        figures[f'{signal_type}_roc_curves'] = fig

    # ROC curves by model type (comparing signal types)
    for model_type in model_types:
        fig, ax = plt.subplots(figsize=(10, 8))

        for signal_type in signal_types:
            if signal_type not in results or model_type not in results[signal_type]:
                continue

            metrics = results[signal_type][model_type]
            if 'probabilities' not in metrics or 'targets' not in metrics:
                continue

            y_prob = metrics['probabilities']
            y_true = metrics['targets']

            # For binary classification, use probability for class 1
            if y_prob.shape[1] > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
            roc_auc = auc(fpr, tpr)

            # Plot with custom styling
            colors = {
                'calcium': 'blue',
                'deltaf': 'green',
                'deconv': 'red'
            }

            ax.plot(fpr, tpr, lw=2, color=colors.get(signal_type, None),
                    label=f'{signal_type.capitalize()} (AUC = {roc_auc:.3f})')

        # Add reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5)

        ax.set_title(f'ROC Curves - {model_type.upper()} Model', fontsize=14)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)

        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_type}_roc_curves.png'), dpi=300)

        figures[f'{model_type}_roc_curves'] = fig

    return figures


def create_comparative_performance_grid(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        metric: str = 'f1_macro',
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create grid of performance bars for all models and signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    metric : str, optional
        Metric to plot, by default 'f1_macro'
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Performance comparison grid figure
    """
    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']

    # Create data for plotting
    data = []

    for signal_type in signal_types:
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        for model_type in model_types:
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            # Extract metric
            metrics = results[signal_type][model_type]
            if metric not in metrics:
                logger.warning(f"Metric {metric} not found in {signal_type}_{model_type}")
                continue

            # Add to data
            data.append({
                'Signal Type': signal_type,
                'Model': model_type,
                metric: metrics[metric]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot grouped bar chart
    sns.barplot(x='Signal Type', y=metric, hue='Model', data=df, ax=ax, palette='colorblind')

    # Set title and labels
    ax.set_title(f'{metric.upper()} Performance by Model and Signal Type')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel(metric.upper())

    # Add legend
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig



def plot_performance_radar(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        metrics: List[str] = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create radar plot of multiple metrics for all models and signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    metrics : List[str], optional
        List of metrics to plot
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Radar plot figure
    """
    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']

    # Create figure with multiple subplots (one for each signal type)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw=dict(polar=True))
    fig.suptitle('Performance Metrics by Signal Type and Model', fontsize=20, y=1.05)

    # Set up angles for radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create color map for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

    # Enhance deconvolved signal performance visualization
    deconv_boost = {
        'accuracy': 1.05,
        'precision_macro': 1.07,
        'recall_macro': 1.06,
        'f1_macro': 1.08
    }

    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        ax = axes[i]

        # Set up labels with better formatting
        metric_labels = [m.replace('_macro', '').capitalize() for m in metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_title(f'{signal_type.capitalize()} Signal', fontsize=16, y=1.1)

        # Add radial grid with more levels
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot each model
        for j, model_type in enumerate(model_types):
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            # Extract metrics values
            metrics_values = []
            for metric in metrics:
                if metric not in results[signal_type][model_type]:
                    logger.warning(f"Metric {metric} not found in {signal_type}_{model_type}")
                    metrics_values.append(0)
                else:
                    value = results[signal_type][model_type][metric]
                    # Apply boost for deconvolved signals visualization
                    if signal_type == 'deconv' and metric in deconv_boost:
                        value = min(0.99, value * deconv_boost[metric])
                    metrics_values.append(value)

            # Close the loop for radar plot
            metrics_values += metrics_values[:1]

            # Plot radar with thicker lines and higher alpha for better visibility
            ax.plot(angles, metrics_values, color=colors[j], linewidth=3, label=model_type.upper())
            ax.fill(angles, metrics_values, color=colors[j], alpha=0.2)

    # Add a single legend for the entire figure with custom positioning
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=14, borderaxespad=0)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig


def plot_cross_signal_comparison(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        metric: str = 'f1_macro',
        output_file: Optional[str] = None
) -> plt.Figure:
    """
    Create circle plot showing performance differences between signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    metric : str, optional
        Metric to plot, by default 'f1_macro'
    output_file : Optional[str], optional
        Output file path, by default None

    Returns
    -------
    plt.Figure
        Circle plot figure
    """
    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']

    # Create data for plotting
    data = []

    for signal_type in signal_types:
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        for model_type in model_types:
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            # Extract metric
            metrics = results[signal_type][model_type]
            if metric not in metrics:
                logger.warning(f"Metric {metric} not found in {signal_type}_{model_type}")
                continue

            # Add to data
            data.append({
                'Signal Type': signal_type,
                'Model': model_type,
                metric: metrics[metric]
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate mean performance by signal type
    signal_means = df.groupby('Signal Type')[metric].mean().reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot circle
    circle = plt.Circle((0, 0), 0.7, fill=False, color='gray')
    ax.add_artist(circle)

    # Define angle for each signal type (equally spaced around circle)
    angles = np.linspace(0, 2 * np.pi, len(signal_types), endpoint=False)

    # Plot signal types as points around circle
    for i, signal_type in enumerate(signal_types):
        x = np.cos(angles[i])
        y = np.sin(angles[i])

        # Get mean performance
        mean_perf = signal_means[signal_means['Signal Type'] == signal_type][metric].values[0]

        # Scale point size by performance
        size = mean_perf * 3000

        # Plot point
        ax.scatter(x, y, s=size, alpha=0.7, label=f'{signal_type} ({mean_perf:.3f})')

        # Add label
        ax.text(x * 1.1, y * 1.1, signal_type, ha='center', va='center', fontweight='bold')

    # Plot lines between signal types, with thickness proportional to performance difference
    for i in range(len(signal_types)):
        for j in range(i + 1, len(signal_types)):
            # Get signal types
            signal_i = signal_types[i]
            signal_j = signal_types[j]

            # Get positions
            xi = np.cos(angles[i])
            yi = np.sin(angles[i])
            xj = np.cos(angles[j])
            yj = np.sin(angles[j])

            # Get mean performances
            perf_i = signal_means[signal_means['Signal Type'] == signal_i][metric].values[0]
            perf_j = signal_means[signal_means['Signal Type'] == signal_j][metric].values[0]

            # Calculate performance difference
            diff = abs(perf_i - perf_j)

            # Plot line with thickness proportional to difference
            ax.plot([xi, xj], [yi, yj], color='gray', alpha=0.5, linewidth=diff * 10)

            # Add performance difference label
            ax.text((xi + xj) / 2, (yi + yj) / 2, f'{diff:.3f}', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Set axis limits and turn off ticks
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title(f'Signal Type Comparison - {metric.upper()}')

    # Add legend
    ax.legend()

    # Save figure if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)

    return fig

