"""
Performance metrics with emphasis on deconvolved signal superiority.
<<<<<<< HEAD
Now includes precision-recall curve generation to fix visualization issues.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           roc_curve, precision_recall_curve)
=======
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
>>>>>>> 57ef1b14e3841956ff736acd6e951f771b275018
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
<<<<<<< HEAD
    """
    Evaluate model and return comprehensive results including both ROC and PR curves.

    This function now generates all the curve data that visualization components expect,
    ensuring that your precision-recall grids will display properly instead of showing
    "No PR data" messages.
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available - this is crucial for curve generation
=======
    """Evaluate model and return comprehensive results."""
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
>>>>>>> 57ef1b14e3841956ff736acd6e951f771b275018
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None
<<<<<<< HEAD
        logger.warning("Model doesn't support probability prediction - curves will be limited")

    # Calculate basic metrics
=======

    # Calculate metrics
>>>>>>> 57ef1b14e3841956ff736acd6e951f771b275018
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)

<<<<<<< HEAD
    # Generate curve data - this is the key fix for your visualization issue
    curve_data = {}

    if y_prob is not None:
        # Generate ROC curve data (this was already working)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob[:, 1])
        curve_data['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }

        # Generate precision-recall curve data (this was missing!)
        # This is what fixes the "No PR data" issue in your visualizations
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob[:, 1])
        curve_data['precision_recall'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }

        logger.info(f"Generated both ROC and PR curves with {len(fpr)} and {len(precision)} points respectively")
    else:
        logger.warning("No probability predictions available - cannot generate ROC or PR curves")

    # Compile final results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'curve_data': curve_data
    }

    logger.info(f"Model evaluation complete: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

=======
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

>>>>>>> 57ef1b14e3841956ff736acd6e951f771b275018
    return results

