"""
Performance visualization utilities for model evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_performance_radar(performance_df: pd.DataFrame,
                           output_dir: Optional[Union[str, Path]] = None,
                           metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                           fig_size: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Plot radar charts comparing model performance across signal types.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame containing performance metrics
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    metrics : List[str], optional
        Metrics to include in the radar chart, by default ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig_size : Tuple[int, int], optional
        Figure size, by default (15, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info("Plotting performance radar charts")

    # Get unique signal types
    signal_types = performance_df['Signal Type'].unique()

    # Create figure
    fig, axes = plt.subplots(1, len(signal_types), figsize=fig_size, subplot_kw=dict(polar=True))

    # Handle case with only one signal type
    if len(signal_types) == 1:
        axes = [axes]

    for ax, signal_type in zip(axes, signal_types):
        # Filter data for this signal type
        signal_df = performance_df[performance_df['Signal Type'] == signal_type]

        # Get number of models and metrics
        n_models = len(signal_df)
        n_metrics = len(metrics)

        # Calculate angles for radar plot
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create radar plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

        # Plot data for each model
        for _, row in signal_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the loop

            ax.plot(angles, values, linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.1)

        # Set title
        ax.set_title(f"{signal_type}")

    # Add legend to the last axis
    axes[-1].legend(loc='lower right', bbox_to_anchor=(1.2, 0))

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / "performance_radar.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance radar chart to {output_dir / 'performance_radar.png'}")

    return fig


def plot_performance_bars(performance_df: pd.DataFrame,
                          output_dir: Optional[Union[str, Path]] = None,
                          metric: str = 'Accuracy',
                          fig_size: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot bar chart comparing model performance for a specific metric.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame containing performance metrics
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    metric : str, optional
        Metric to plot, by default 'Accuracy'
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting {metric} bar chart")

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create grouped bar plot
    sns.barplot(x='Signal Type', y=metric, hue='Model', data=performance_df, ax=ax)

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    # Set title and labels
    ax.set_title(f"{metric} Comparison")
    ax.set_xlabel("Signal Type")
    ax.set_ylabel(metric)

    # Adjust legend
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"{metric.lower().replace(' ', '_')}_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved {metric} bar chart to {output_dir / f'{metric.lower().replace(' ', '_')}_comparison.png'}")

    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          model_name: str,
                          signal_type: str,
                          output_dir: Optional[Union[str, Path]] = None,
                          fig_size: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix, shape (2, 2) for binary classification
    model_name : str
        Name of the model
    signal_type : str
        Type of signal
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (8, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting confusion matrix for {model_name} on {signal_type}")

    # Calculate row-wise percentages
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot confusion matrix
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax)

    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Set title
    ax.set_title(f"{model_name} - {signal_type}")

    # Set tick labels
    ax.set_xticklabels(['No Footstep', 'Contralateral'])
    ax.set_yticklabels(['No Footstep', 'Contralateral'])

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"confusion_matrix_{model_name}_{signal_type}.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_dir / f'confusion_matrix_{model_name}_{signal_type}.png'}")

    return fig


def plot_performance_improvement(performance_df: pd.DataFrame,
                                 baseline_signal: str = 'calcium_signal',
                                 output_dir: Optional[Union[str, Path]] = None,
                                 metric: str = 'Accuracy',
                                 fig_size: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot performance improvement compared to a baseline signal type.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame containing performance metrics
    baseline_signal : str, optional
        Baseline signal type, by default 'calcium_signal'
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    metric : str, optional
        Metric to plot, by default 'Accuracy'
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 6)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting {metric} improvement compared to {baseline_signal}")

    # Get unique models and signal types
    models = performance_df['Model'].unique()
    signal_types = [s for s in performance_df['Signal Type'].unique() if s != baseline_signal]

    # Calculate improvement for each model and signal type
    improvement_data = []

    for model in models:
        model_df = performance_df[performance_df['Model'] == model]

        # Get baseline performance
        baseline_perf = model_df[model_df['Signal Type'] == baseline_signal][metric].values

        # Skip if baseline performance is missing
        if len(baseline_perf) == 0:
            continue

        baseline_perf = baseline_perf[0]

        # Calculate improvement for each signal type
        for signal_type in signal_types:
            signal_perf = model_df[model_df['Signal Type'] == signal_type][metric].values

            # Skip if signal performance is missing
            if len(signal_perf) == 0:
                continue

            signal_perf = signal_perf[0]

            # Calculate improvement (in percentage points)
            improvement = (signal_perf - baseline_perf) * 100

            improvement_data.append({
                'Model': model,
                'Signal Type': signal_type,
                'Improvement': improvement
            })

    # Create DataFrame
    improvement_df = pd.DataFrame(improvement_data)

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create grouped bar plot
    sns.barplot(x='Model', y='Improvement', hue='Signal Type', data=improvement_df, ax=ax)

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                    ha='center', va='bottom', fontsize=10)

    # Set title and labels
    ax.set_title(f"{metric} Improvement Compared to {baseline_signal} (Percentage Points)")
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{metric} Improvement (pp)")

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Adjust legend
    ax.legend(title="Signal Type")

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"{metric.lower().replace(' ', '_')}_improvement.png", dpi=300, bbox_inches='tight')
        logger.info(
            f"Saved {metric} improvement chart to {output_dir / f'{metric.lower().replace(' ', '_')}_improvement.png'}")

    return fig


def plot_training_time_vs_performance(performance_df: pd.DataFrame,
                                      output_dir: Optional[Union[str, Path]] = None,
                                      metric: str = 'Accuracy',
                                      fig_size: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot training time vs. performance.

    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame containing performance metrics and training time
    output_dir : Optional[Union[str, Path]], optional
        Directory to save the figure, by default None
    metric : str, optional
        Metric to plot, by default 'Accuracy'
    fig_size : Tuple[int, int], optional
        Figure size, by default (12, 8)

    Returns
    -------
    plt.Figure
        The figure object
    """
    logger.info(f"Plotting {metric} vs. training time")

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create scatter plot
    for signal_type in performance_df['Signal Type'].unique():
        signal_df = performance_df[performance_df['Signal Type'] == signal_type]
        ax.scatter(signal_df['Train Time'], signal_df[metric], label=signal_type, alpha=0.7, s=100)

    # Add model labels
    for _, row in performance_df.iterrows():
        ax.annotate(row['Model'], (row['Train Time'], row[metric]), fontsize=8, alpha=0.7)

    # Set labels
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. Training Time")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"{metric.lower().replace(' ', '_')}_vs_training_time.png", dpi=300,
                    bbox_inches='tight')
        logger.info(
            f"Saved {metric} vs. training time chart to {output_dir / f'{metric.lower().replace(' ', '_')}_vs_training_time.png'}")

    return fig

