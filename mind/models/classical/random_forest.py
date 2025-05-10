"""Random Forest model implementation."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def create_random_forest(
        signal_type: str = None,
        random_state: int = 42
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier optimized for the signal type.

    Parameters
    ----------
    signal_type : str, optional
        Signal type to optimize for ('calcium', 'deltaf', or 'deconv')
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    RandomForestClassifier
        Optimized Random Forest model
    """
    # Base parameters that work well for all signal types
    base_params = {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': random_state,
        'n_jobs': -1,  # Use all cores
        'bootstrap': True,
        'max_features': 'sqrt',
        'verbose': 0,
    }

    # Signal-specific optimizations
    if signal_type == 'deconv':
        # Deconvolved signals typically have sparse, spike-like features
        base_params['n_estimators'] = 250
        base_params['min_samples_split'] = 4
    elif signal_type == 'deltaf':
        # ΔF/F signals have normalized features with various scales
        base_params['max_depth'] = 25
    elif signal_type == 'calcium':
        # Raw calcium signals have high dynamic range
        base_params['min_samples_leaf'] = 3

    # Create and return the model
    return RandomForestClassifier(**base_params)


def train_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        signal_type: Optional[str] = None
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest model for binary classification.

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
    Tuple[RandomForestClassifier, Dict[str, Any]]
        Trained model and evaluation metrics
    """
    logger.info(f"Training Random Forest for {signal_type if signal_type else 'general'} data")

    # Create model
    model = create_random_forest(signal_type, config['experiment'].get('seed', 42))

    # Apply class weights if provided
    if class_weights is not None:
        model.class_weight = class_weights

    # Train model
    model.fit(X_train, y_train.astype(int))

    # Evaluate model
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        'predictions': y_pred,
        'probabilities': y_prob,
        'targets': y_val
    }

    # Log metrics
    logger.info(f"Random Forest validation metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

    return model, metrics


def extract_feature_importance(
        model: RandomForestClassifier,
        window_size: int,
        n_neurons: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature importance from a trained Random Forest model.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    window_size : int
        Window size used for data processing
    n_neurons : int
        Number of neurons

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        2D feature importance, temporal importance, and neuron importance
    """
    # Extract feature importance
    importance = model.feature_importances_

    # Reshape to 2D (window_size, n_neurons)
    importance_2d = importance.reshape(window_size, n_neurons)

    # Calculate temporal importance (mean across neurons)
    temporal_importance = np.mean(importance_2d, axis=1)

    # Calculate neuron importance (mean across time)
    neuron_importance = np.mean(importance_2d, axis=0)

    return importance_2d, temporal_importance, neuron_importance

