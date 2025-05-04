"""Metrics calculation functions."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : Optional[np.ndarray], optional
        Predicted probabilities, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing metrics
    """
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'predictions': y_pred,
        'targets': y_true
    }

    # Add class-specific metrics
    classes = np.unique(np.concatenate([y_true, y_pred]))
    for cls in classes:
        # Create binary labels for the current class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        # Calculate class-specific metrics
        metrics[f'precision_class_{int(cls)}'] = precision_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f'recall_class_{int(cls)}'] = recall_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f'f1_class_{int(cls)}'] = f1_score(
            y_true_binary, y_pred_binary, zero_division=0
        )

    # Add probability-based metrics if available
    if y_prob is not None:
        metrics['probabilities'] = y_prob

        # Calculate ROC AUC for each class if possible
        try:
            if y_prob.shape[1] > 1:  # Multi-class case
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )

                # Class-specific AUC
                for i, cls in enumerate(np.unique(y_true)):
                    if i < y_prob.shape[1]:
                        y_true_binary = (y_true == cls).astype(int)
                        metrics[f'roc_auc_class_{int(cls)}'] = roc_auc_score(
                            y_true_binary, y_prob[:, i]
                        )
            else:  # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")

    # Calculate confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def calculate_model_comparison(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate model comparison metrics.

    Parameters
    ----------
    metrics_list : List[Dict[str, Any]]
        List of metrics dictionaries from different models

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model comparison metrics
    """
    if not metrics_list:
        return {}

    # Initialize results
    comparison = {
        'best_model': None,
        'best_accuracy': 0,
        'best_f1_macro': 0,
        'model_rankings': {}
    }

    # Extract metrics for comparison
    model_metrics = []
    for i, metrics in enumerate(metrics_list):
        if 'accuracy' not in metrics or 'f1_macro' not in metrics:
            continue

        model_metrics.append({
            'index': i,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro']
        })

    # Rank by accuracy
    accuracy_ranks = sorted(model_metrics, key=lambda x: x['accuracy'], reverse=True)
    for rank, data in enumerate(accuracy_ranks):
        if rank == 0:
            comparison['best_model'] = data['index']
            comparison['best_accuracy'] = data['accuracy']

        comparison['model_rankings'][data['index']] = {'accuracy_rank': rank + 1}

    # Rank by F1 score
    f1_ranks = sorted(model_metrics, key=lambda x: x['f1_macro'], reverse=True)
    for rank, data in enumerate(f1_ranks):
        if rank == 0:
            comparison['best_f1_macro'] = data['f1_macro']
            if comparison['best_model'] is None:
                comparison['best_model'] = data['index']

        if data['index'] in comparison['model_rankings']:
            comparison['model_rankings'][data['index']]['f1_rank'] = rank + 1

    return comparison

