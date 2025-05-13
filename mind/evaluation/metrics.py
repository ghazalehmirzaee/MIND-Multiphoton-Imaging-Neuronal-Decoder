"""
Performance metrics calculation for model evaluation.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


def calculate_classification_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None,
                                     average: str = 'macro') -> Dict[str, float]:
    """
    Calculate classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : Optional[np.ndarray], optional
        Predicted probabilities, by default None
    average : str, optional
        Type of averaging for precision, recall, and F1, by default 'macro'

    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    # Convert to numpy arrays if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    if y_prob is not None and hasattr(y_prob, 'numpy'):
        y_prob = y_prob.numpy()

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

    # Calculate ROC AUC if probabilities are provided
    if y_prob is not None:
        try:
            # For binary classification
            if y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics['roc_auc'] = float(roc_auc)
            # For multi-class classification
            else:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                metrics['roc_auc'] = float(roc_auc)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")

    logger.info(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    # Convert to numpy arrays if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return cm


def get_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        False positive rate, true positive rate, thresholds
    """
    # Convert to numpy arrays if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_prob, 'numpy'):
        y_prob = y_prob.numpy()

    # For binary classification
    if y_prob.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    # For multi-class, use the first class (usually positive class in binary)
    else:
        # This is a simplified approach; for multi-class, consider one-vs-rest
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])

    return fpr, tpr, thresholds


def get_precision_recall_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precision, recall, thresholds
    """
    # Convert to numpy arrays if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_prob, 'numpy'):
        y_prob = y_prob.numpy()

    # For binary classification
    if y_prob.shape[1] == 2:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
    # For multi-class, use the first class (usually positive class in binary)
    else:
        # This is a simplified approach; for multi-class, consider one-vs-rest
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])

    return precision, recall, thresholds


def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate a model on test data and return comprehensive metrics.

    Parameters
    ----------
    model : model object
        Trained model with predict and predict_proba methods
    X_test : np.ndarray or torch.Tensor
        Test features
    y_test : np.ndarray or torch.Tensor
        Test labels

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation results
    """
    logger.info("Evaluating model performance")

    try:
        # Get predictions
        y_pred = model.predict(X_test)

        # Try to get predicted probabilities (may not be available for all models)
        try:
            y_prob = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            logger.warning("Model does not support predict_proba")
            y_prob = None

        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

        # Get confusion matrix
        cm = get_confusion_matrix(y_test, y_pred)

        # Get ROC curve data if probabilities are available
        if y_prob is not None:
            try:
                fpr, tpr, roc_thresholds = get_roc_curve_data(y_test, y_prob)
                precision, recall, pr_thresholds = get_precision_recall_curve_data(y_test, y_prob)

                # Save curve data
                curve_data = {
                    'roc': {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': roc_thresholds
                    },
                    'precision_recall': {
                        'precision': precision,
                        'recall': recall,
                        'thresholds': pr_thresholds if len(pr_thresholds) > 0 else None
                    }
                }
            except Exception as e:
                logger.warning(f"Could not calculate curve data: {e}")
                curve_data = None
        else:
            curve_data = None

        # Compile evaluation results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'curve_data': curve_data
        }

        logger.info("Model evaluation complete")

        return results

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

