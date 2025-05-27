"""
Performance metrics with emphasis on deconvolved signal superiority.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive classification metrics."""
    # Convert tensors to numpy if needed
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and hasattr(y_prob, 'cpu'):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary')),
        'recall': float(recall_score(y_true, y_pred, average='binary')),
        'f1_score': float(f1_score(y_true, y_pred, average='binary'))
    }

    # Add ROC AUC if probabilities available
    if y_prob is not None and y_prob.shape[1] == 2:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))

    return metrics


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive results."""
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Get ROC curve data if available
    curve_data = {}
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        curve_data['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'curve_data': curve_data
    }

    logger.info(f"Model evaluation: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    return results


