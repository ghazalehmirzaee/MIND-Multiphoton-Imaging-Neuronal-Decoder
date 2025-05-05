import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_efficient_random_forest(
        signal_type: str = None,
        random_state: int = 42
) -> RandomForestClassifier:
    """
    Create an efficient Random Forest classifier optimized for the signal type.

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
        'n_estimators': 50,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': random_state,
        'n_jobs': -1,  # Use all cores
        'bootstrap': True,
        'max_features': 'sqrt',
        'verbose': 0,
    }

    # Signal-specific optimizations - lightweight but effective
    if signal_type == 'deconv':
        # Enhanced parameters for deconvolved signals - more trees for better accuracy
        base_params.update({
            'n_estimators': 80,
            'max_depth': 20,
            'min_samples_split': 2,
        })
    elif signal_type == 'deltaf':
        # Moderate optimization for deltaf signals
        base_params.update({
            'n_estimators': 60,
            'max_depth': 18,
        })

    # Create and return the model
    return RandomForestClassifier(**base_params)


def train_efficient_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        signal_type: str = None,
        random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train an efficient Random Forest model.

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
    signal_type : str, optional
        Signal type to optimize for
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    Tuple[RandomForestClassifier, Dict[str, Any]]
        Trained model and evaluation metrics
    """
    # Create model
    model = create_efficient_random_forest(signal_type, random_state)

    # Train model
    model.fit(X_train, y_train.astype(int))

    # Evaluate model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

