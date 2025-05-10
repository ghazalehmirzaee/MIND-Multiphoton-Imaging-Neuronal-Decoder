"""Support Vector Machine model implementation."""
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def create_svm(
        signal_type: str = None,
        random_state: int = 42
) -> Tuple[SVC, Optional[PCA]]:
    """
    Create an SVM classifier optimized for the signal type.

    Parameters
    ----------
    signal_type : str, optional
        Signal type to optimize for ('calcium', 'deltaf', or 'deconv')
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    Tuple[SVC, Optional[PCA]]
        Optimized SVM model and PCA transformer (if needed)
    """
    # Base parameters
    base_params = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'cache_size': 200,
        'random_state': random_state,
        'verbose': 0,
    }

    # Signal-specific optimizations
    if signal_type == 'deconv':
        # Deconvolved signals typically have sparse, spike-like features
        base_params['kernel'] = 'rbf'
        base_params['C'] = 2.0
        base_params['gamma'] = 'scale'
    elif signal_type == 'deltaf':
        # ΔF/F signals have normalized features with various scales
        base_params['kernel'] = 'rbf'
        base_params['C'] = 1.5
        base_params['gamma'] = 'auto'
    elif signal_type == 'calcium':
        # Raw calcium signals have high dynamic range
        base_params['kernel'] = 'rbf'
        base_params['C'] = 1.0
        base_params['gamma'] = 'scale'

    # Create SVM model
    model = SVC(**base_params)

    # Always set up PCA for dimensionality reduction (input will be large)
    pca = PCA(n_components=0.95, random_state=random_state)

    return model, pca


def train_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        optimize: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        signal_type: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], Optional[PCA]]:
    """
    Train an SVM model for binary classification.

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
    Tuple[Any, Dict[str, Any], Optional[PCA]]
        Trained model, evaluation metrics, and PCA transformer
    """
    logger.info(f"Training SVM for {signal_type if signal_type else 'general'} data")

    # Create model with PCA
    model, pca = create_svm(signal_type, config['experiment'].get('seed', 42))

    # Apply class weights if provided
    if class_weights is not None:
        model.class_weight = class_weights

    # Apply PCA transformation
    logger.info("Applying PCA transformation for dimensionality reduction")
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    logger.info(f"PCA reduced dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}")

    # Train model
    logger.info("Training SVM model...")
    model.fit(X_train_pca, y_train.astype(int))

    # Evaluate model
    y_pred = model.predict(X_val_pca)
    y_prob = model.predict_proba(X_val_pca)

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
    logger.info(f"SVM validation metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

    return model, metrics, pca

