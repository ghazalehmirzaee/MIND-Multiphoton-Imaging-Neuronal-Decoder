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

    # Plot bar chart
    sns.barplot(x='Signal Type', y=metric, data=signal_performance, ax=ax, palette='Set3')

    # Add best model annotations
    for i, row in enumerate(best_models.itertuples()):
        signal_type = row._2
        model = row.Model
        score = getattr(row, metric.replace(' ', '_'))

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


def plot_confusion_matrices(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot confusion matrices for each model and signal type.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Dictionary containing results
    output_dir : Optional[str], optional
        Output directory, by default None

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing confusion matrix figures
    """
    # Initialize figures dictionary
    figures = {}

    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']

    # Define class names
    class_names = ['No footstep', 'Contralateral', 'Ipsilateral']

    # Create figure for each signal type and model type
    for signal_type in signal_types:
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        for model_type in model_types:
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            # Extract predictions and targets
            metrics = results[signal_type][model_type]
            if 'predictions' not in metrics or 'targets' not in metrics:
                logger.warning(f"Predictions or targets not found in {signal_type}_{model_type}")
                continue

            y_pred = metrics['predictions']
            y_true = metrics['targets']

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')

            # Set title and labels
            ax.set_title(f'Confusion Matrix - {signal_type} - {model_type}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

            # Set tick labels
            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            ax.set_xticklabels(class_names[:n_classes])
            ax.set_yticklabels(class_names[:n_classes])

            # Save figure
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'{signal_type}_{model_type}_confusion_matrix.png'), dpi=300)

            # Store figure
            figures[f'{signal_type}_{model_type}_confusion_matrix'] = fig

    return figures


def plot_roc_curves(
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot ROC curves for each model and signal type.

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
    # Initialize figures dictionary
    figures = {}

    # Define signal types and model types
    signal_types = ['calcium', 'deltaf', 'deconv']
    model_types = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']

    # Define class names
    class_names = ['No footstep', 'Contralateral', 'Ipsilateral']

    # Create figure for each signal type and model type
    for signal_type in signal_types:
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        for model_type in model_types:
            if model_type not in results[signal_type]:
                logger.warning(f"Results for model type {model_type} not found in {signal_type}")
                continue

            # Extract predictions, probabilities, and targets
            metrics = results[signal_type][model_type]
            if 'probabilities' not in metrics or 'targets' not in metrics:
                logger.warning(f"Probabilities or targets not found in {signal_type}_{model_type}")
                continue

            y_prob = metrics.get('probabilities')
            y_true = metrics['targets']

            # Check if probabilities are available
            if y_prob is None:
                logger.warning(f"Probabilities not available for {signal_type}_{model_type}")
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Calculate ROC curve and AUC for each class
            classes = np.unique(y_true)
            for i, cls in enumerate(classes):
                # Create binary labels for the current class
                y_true_binary = (y_true == cls).astype(int)

                # Calculate ROC curve and AUC
                if i < y_prob.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                    roc_auc = auc(fpr, tpr)

                    # Plot ROC curve
                    ax.plot(fpr, tpr, lw=2, label=f'Class {class_names[int(cls)]} (AUC = {roc_auc:.2f})')

            # Plot random guess line
            ax.plot([0, 1], [0, 1], 'k--', lw=2)

            # Set title and labels
            ax.set_title(f'ROC Curve - {signal_type} - {model_type}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc='lower right')

            # Save figure
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'{signal_type}_{model_type}_roc_curve.png'), dpi=300)

            # Store figure
            figures[f'{signal_type}_{model_type}_roc_curve'] = fig

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
        List of metrics to plot, by default ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

    # Set up angles for radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create color map for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        if signal_type not in results:
            logger.warning(f"Results for signal type {signal_type} not found")
            continue

        ax = axes[i]

        # Set up labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_macro', '') for m in metrics])
        ax.set_title(f'{signal_type.capitalize()} Signal')

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
                    metrics_values.append(results[signal_type][model_type][metric])

            # Close the loop for radar plot
            metrics_values += metrics_values[:1]

            # Plot radar
            ax.plot(angles, metrics_values, color=colors[j], linewidth=2, label=model_type)
            ax.fill(angles, metrics_values, color=colors[j], alpha=0.1)

    # Add legend to the right of the figure
    fig.legend(model_types, loc='center right')

    # Adjust layout
    plt.tight_layout()

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

