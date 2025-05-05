import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_svm(
        signal_type: str = None,
        n_features: int = None,
        random_state: int = 42
) -> Tuple[SVC, Optional[PCA]]:
    """
    Create an efficient SVM classifier optimized for the signal type.

    Parameters
    ----------
    signal_type : str, optional
        Signal type to optimize for ('calcium', 'deltaf', or 'deconv')
    n_features : int, optional
        Number of features to determine PCA components
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    Tuple[SVC, Optional[PCA]]
        Optimized SVM model and PCA transformer (if used)
    """
    # Base parameters that work well for all signal types
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
        # Enhanced parameters for deconvolved signals
        base_params.update({
            'C': 5.0,
            'kernel': 'rbf',
        })
    elif signal_type == 'deltaf':
        # Moderate optimization for deltaf signals
        base_params.update({
            'C': 2.0,
        })

    # Create SVM model
    model = SVC(**base_params)

    # Apply PCA for dimensionality reduction
    pca = None
    if n_features is not None and n_features > 100:
        # Determine number of components
        n_components = min(100, int(n_features * 0.5))
        pca = PCA(n_components=n_components, random_state=random_state)

    return model, pca


def optimize_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        signal_type: str = None,
        random_state: int = 42
) -> Tuple[Any, Dict[str, Any], Optional[PCA]]:
    """
    Train an efficient SVM model.

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
    Tuple[Any, Dict[str, Any], Optional[PCA]]
        Trained model, evaluation metrics, and PCA transformer (if used)
    """
    # Create model with PCA if needed
    model, pca = create_svm(signal_type, X_train.shape[1], random_state)

    # Apply PCA if needed
    if pca is not None:
        X_train_transformed = pca.fit_transform(X_train)
        X_val_transformed = pca.transform(X_val)
    else:
        X_train_transformed = X_train
        X_val_transformed = X_val

    # Train model
    model.fit(X_train_transformed, y_train.astype(int))

    # Evaluate model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_val_transformed)

    try:
        y_prob = model.predict_proba(X_val_transformed)
    except:
        y_prob = None

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

    return model, metrics, pca

