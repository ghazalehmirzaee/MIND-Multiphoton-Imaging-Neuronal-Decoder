# mind/visualization/performance.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import os


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          classes: List[str],
                          title: str = 'Confusion Matrix',
                          normalize: bool = True,
                          output_dir: Optional[str] = None,
                          save_filename: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    classes : List[str]
        List of class names
    title : str, optional
        Title of the plot
    normalize : bool, optional
        Whether to normalize the confusion matrix
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        vmax = 100
    else:
        fmt = 'd'
        vmax = np.max(cm)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=vmax)
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_model_performance_comparison(results: Dict[str, Dict[str, Dict[str, float]]],
                                      metric: str = 'accuracy',
                                      title: str = 'Model Performance Comparison',
                                      output_dir: Optional[str] = None,
                                      save_filename: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot comparison of model performance across different signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary with structure {signal_type: {model_name: {metric: value}}}
    metric : str, optional
        Performance metric to plot
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract signal types and model names
    signal_types = list(results.keys())
    model_names = list(results[signal_types[0]].keys())

    # Create a DataFrame to hold the results
    import pandas as pd
    data = []
    for signal_type in signal_types:
        for model_name in model_names:
            value = results[signal_type][model_name][metric]
            data.append({
                'Signal Type': signal_type,
                'Model': model_name,
                f'{metric.capitalize()}': value
            })
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot grouped bar chart
    sns.barplot(x='Model', y=f'{metric.capitalize()}', hue='Signal Type', data=df, ax=ax)

    # Set labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(f'{metric.capitalize()}', fontsize=14)

    # Adjust legend
    ax.legend(title='Signal Type', fontsize=12)

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_roc_curves(y_true: np.ndarray,
                    y_pred_probas: Dict[str, np.ndarray],
                    title: str = 'ROC Curves',
                    output_dir: Optional[str] = None,
                    save_filename: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_probas : Dict[str, np.ndarray]
        Dictionary mapping model names to their prediction probabilities
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve for each model
    for model_name, y_pred_proba in y_pred_probas.items():
        # For binary classification, we need the probability of the positive class
        if y_pred_proba.shape[1] == 2:
            y_score = y_pred_proba[:, 1]
        else:
            # For multi-class, we can use one-vs-rest approach
            y_score = y_pred_proba

        # Compute ROC curve and area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add legend
    ax.legend(loc='lower right', fontsize=12)

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_precision_recall_curves(y_true: np.ndarray,
                                 y_pred_probas: Dict[str, np.ndarray],
                                 title: str = 'Precision-Recall Curves',
                                 output_dir: Optional[str] = None,
                                 save_filename: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot precision-recall curves for multiple models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_probas : Dict[str, np.ndarray]
        Dictionary mapping model names to their prediction probabilities
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot precision-recall curve for each model
    for model_name, y_pred_proba in y_pred_probas.items():
        # For binary classification, we need the probability of the positive class
        if y_pred_proba.shape[1] == 2:
            y_score = y_pred_proba[:, 1]
        else:
            # For multi-class, we can use one-vs-rest approach
            y_score = y_pred_proba

        # Compute precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap:.3f})')

    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add legend
    ax.legend(loc='best', fontsize=12)

    # Adjust layout
    fig.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_all_confusion_matrices(results: Dict[str, Dict[str, Dict[str, Any]]],
                                classes: List[str],
                                title_prefix: str = 'Confusion Matrix',
                                output_dir: Optional[str] = None,
                                save_filename_prefix: Optional[str] = 'confusion_matrix',
                                figsize: Tuple[int, int] = (20, 15)) -> Dict[str, Dict[str, plt.Figure]]:
    """
    Plot confusion matrices for all models and signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, Any]]]
        Nested dictionary with structure {signal_type: {model_name: {metric: value}}}
    classes : List[str]
        List of class names
    title_prefix : str, optional
        Prefix for the plot titles
    output_dir : str, optional
        Directory to save the plots
    save_filename_prefix : str, optional
        Prefix for the saved filenames
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    Dict[str, Dict[str, plt.Figure]]
        Nested dictionary with structure {signal_type: {model_name: figure}}
    """
    # Extract signal types and model names
    signal_types = list(results.keys())
    model_names = list(results[signal_types[0]].keys())

    # Create a figure with subplots for all models and signal types
    fig, axes = plt.subplots(len(model_names), len(signal_types), figsize=figsize)

    # Make sure axes is a 2D array
    if len(model_names) == 1 and len(signal_types) == 1:
        axes = np.array([[axes]])
    elif len(model_names) == 1:
        axes = np.array([axes])
    elif len(signal_types) == 1:
        axes = np.array([[ax] for ax in axes])

    # Plot confusion matrices
    for i, model_name in enumerate(model_names):
        for j, signal_type in enumerate(signal_types):
            # Get true and predicted labels
            y_true = results[signal_type][model_name]['y_true']
            y_pred = results[signal_type][model_name]['y_pred']

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Normalize
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # Plot heatmap
            ax = axes[i, j]
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=100)

            # Set labels
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes)

            # Set title
            ax.set_title(f'{model_name} - {signal_type}', fontsize=12)

            # Only set labels for the bottom and left subplots
            if i == len(model_names) - 1:
                ax.set_xlabel('Predicted label', fontsize=10)
            if j == 0:
                ax.set_ylabel('True label', fontsize=10)

            # Rotate x tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations
            thresh = cm.max() / 2.
            for i_cm in range(cm.shape[0]):
                for j_cm in range(cm.shape[1]):
                    ax.text(j_cm, i_cm, f'{cm[i_cm, j_cm]:.1f}%',
                            ha="center", va="center",
                            color="white" if cm[i_cm, j_cm] > thresh else "black",
                            fontsize=10)

    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename_prefix is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{save_filename_prefix}_all.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Also create individual plots
    figures = {}
    for signal_type in signal_types:
        figures[signal_type] = {}
        for model_name in model_names:
            # Get true and predicted labels
            y_true = results[signal_type][model_name]['y_true']
            y_pred = results[signal_type][model_name]['y_pred']

            # Create individual plot
            title = f'{title_prefix} - {model_name} - {signal_type}'
            save_filename = f'{save_filename_prefix}_{model_name}_{signal_type}.png'
            figures[signal_type][model_name] = plot_confusion_matrix(
                y_true, y_pred, classes, title=title, normalize=True,
                output_dir=output_dir, save_filename=save_filename
            )

    return figures


def plot_radar_chart(results: Dict[str, Dict[str, Dict[str, float]]],
                     metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                     title: str = 'Model Performance Radar Chart',
                     output_dir: Optional[str] = None,
                     save_filename: Optional[str] = None,
                     figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Plot radar chart of model performance across different metrics and signal types.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary with structure {signal_type: {model_name: {metric: value}}}
    metrics : List[str], optional
        List of metrics to include in the radar chart
    title : str, optional
        Title of the plot
    output_dir : str, optional
        Directory to save the plot
    save_filename : str, optional
        Filename to save the plot
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract signal types and model names
    signal_types = list(results.keys())
    model_names = list(results[signal_types[0]].keys())

    # Create a figure with subplots for each signal type
    fig, axes = plt.subplots(1, len(signal_types), figsize=figsize, subplot_kw=dict(polar=True))

    # Make sure axes is a list
    if len(signal_types) == 1:
        axes = [axes]

    # Number of metrics
    num_metrics = len(metrics)

    # Angles for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]

    # Plot radar chart for each signal type
    for i, (signal_type, ax) in enumerate(zip(signal_types, axes)):
        # Set title
        ax.set_title(f'{signal_type}', fontsize=16)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.capitalize() for metric in metrics], fontsize=12)

        # Set y-ticks
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_ylim(0, 1)

        # Plot each model
        for model_name in model_names:
            # Extract metric values
            values = [results[signal_type][model_name][metric] for metric in metrics]
            # Close the polygon
            values += values[:1]

            # Plot
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)

    # Add legend to the right of the figure
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(model_names), fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory and filename are provided
    if output_dir is not None and save_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig
