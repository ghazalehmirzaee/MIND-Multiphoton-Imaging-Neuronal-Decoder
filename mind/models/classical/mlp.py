"""Multi-Layer Perceptron model implementation."""
import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def create_mlp(
        signal_type: str = None,
        random_state: int = 42
) -> MLPClassifier:
    """
    Create an MLP classifier optimized for the signal type.

    Parameters
    ----------
    signal_type : str, optional
        Signal type to optimize for ('calcium', 'deltaf', or 'deconv')
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    MLPClassifier
        Optimized MLP model
    """
    # Base parameters that work well for all signal types
    base_params = {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': random_state,
        'verbose': 0,
    }

    # Signal-specific optimizations
    if signal_type == 'deconv':
        # Deconvolved signals typically have sparse, spike-like features
        base_params['hidden_layer_sizes'] = (256, 128, 64)
        base_params['alpha'] = 0.00005
    elif signal_type == 'deltaf':
        # ΔF/F signals have normalized features with various scales
        base_params['hidden_layer_sizes'] = (192, 96)
        base_params['activation'] = 'relu'
    elif signal_type == 'calcium':
        # Raw calcium signals have high dynamic range
        base_params['hidden_layer_sizes'] = (128, 64)
        base_params['alpha'] = 0.0002

    # Create and return the model
    return MLPClassifier(**base_params)


def train_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        signal_type: Optional[str] = None
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Train an MLP model for binary classification.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    config : Dict[str, Any]
        Configuration dictionary
    optimize : bool, optional
        Whether to optimize hyperparameters, by default True
    class_weights : Optional[Dict[int, float]], optional
        Class weights for imbalanced data, by default None
    signal_type : Optional[str], optional
        Signal type for optimization, by default None

    Returns
    -------
    Tuple[MLPClassifier, Dict[str, Any]]
        Trained model and evaluation metrics
    """
    logger.info(f"Training MLP for {signal_type if signal_type else 'general'} data")

    # Create model
    model = create_mlp(signal_type, config['experiment'].get('seed', 42))

    # Train model
    model.fit(X_train, y_train.astype(int))

    # Evaluate model
    y_pred = model.predict(X_val)

    # Get probabilities if available
    try:
        y_prob = model.predict_proba(X_val)
    except:
        y_prob = None
        logger.warning("Could not obtain class probabilities from MLP model")

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        'predictions': y_pred,
        'targets': y_val
    }

    if y_prob is not None:
        metrics['probabilities'] = y_prob

    # Log metrics
    logger.info(f"MLP validation metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

    return model, metrics


def extract_feature_importance(
        model: MLPClassifier,
        window_size: int,
        n_neurons: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature importance from a trained MLP model.

    Parameters
    ----------
    model : MLPClassifier
        Trained MLP model
    window_size : int
        Window size used for data processing
    n_neurons : int
        Number of neurons

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        2D feature importance, temporal importance, and neuron importance
    """
    # Extract coefficients from the first layer as a proxy for feature importance
    if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
        # Use absolute values of first layer weights
        importance = np.abs(model.coefs_[0]).mean(axis=1)
    else:
        importance = np.ones(window_size * n_neurons)

    # Reshape to 2D (window_size, n_neurons)
    importance_2d = importance.reshape(window_size, n_neurons)

    # Calculate temporal importance (mean across neurons)
    temporal_importance = np.mean(importance_2d, axis=1)

    # Calculate neuron importance (mean across time)
    neuron_importance = np.mean(importance_2d, axis=0)

    return importance_2d, temporal_importance, neuron_importance

