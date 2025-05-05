import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_mlp(config: Dict[str, Any], signal_type: str = None) -> MLPClassifier:
    """
    Create a Multilayer Perceptron classifier with the specified configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    MLPClassifier
        Configured MLP model
    """
    mlp_params = config['models']['classical']['mlp'].copy()

    # Enhanced parameters for deconvolved signals
    if signal_type == 'deconv':
        mlp_params.update({
            'hidden_layer_sizes': (256, 128, 64),  # Deeper network
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'early_stopping': True,
            'max_iter': 1000,  # More iterations for better convergence
            'tol': 1e-5,  # Lower tolerance for better optimization
            'learning_rate_init': 0.002  # Higher initial learning rate
        })
    elif signal_type == 'deltaf':
        # Moderate parameters for deltaf signals
        mlp_params.update({
            'hidden_layer_sizes': (180, 90),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 800
        })
    # Default parameters for calcium signals remain unchanged

    model = MLPClassifier(
        hidden_layer_sizes=mlp_params.get('hidden_layer_sizes', (128, 64)),
        activation=mlp_params.get('activation', 'relu'),
        solver=mlp_params.get('solver', 'adam'),
        alpha=mlp_params.get('alpha', 0.0001),
        learning_rate=mlp_params.get('learning_rate', 'adaptive'),
        learning_rate_init=mlp_params.get('learning_rate_init', 0.001),
        early_stopping=mlp_params.get('early_stopping', True),
        tol=mlp_params.get('tol', 1e-4),
        max_iter=mlp_params.get('max_iter', 500),
        random_state=config['experiment'].get('seed', 42)
    )

    return model


def optimize_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        signal_type: str = None
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Optimize MLP hyperparameters using efficient randomized search.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    Tuple[MLPClassifier, Dict[str, Any]]
        Optimized MLP model and best parameters
    """
    logger.info(f"Optimizing MLP hyperparameters for {signal_type if signal_type else 'general'} data")

    # Define parameter distribution for randomized search
    param_dist = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['adaptive', 'constant'],
        'early_stopping': [True],
        'solver': ['adam', 'sgd']
    }

    # Modify search space based on signal type
    if signal_type == 'deconv':
        # Biased search space for deconvolved signals
        param_dist.update({
            'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64), (128, 64, 32)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.0005],
            'learning_rate_init': [0.001, 0.002, 0.003],
            'batch_size': [64, 128, 256]
        })

    # Initialize MLP with early stopping
    mlp = MLPClassifier(
        random_state=config['experiment'].get('seed', 42),
        max_iter=800 if signal_type == 'deconv' else 500,
        early_stopping=True,
        validation_fraction=0.1
    )

    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=12,  # More iterations for deconvolved signals
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42),
        verbose=0
    )

    # Fit model
    logger.info(f"Performing randomized search for MLP with {signal_type if signal_type else 'general'} data")
    random_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters for {signal_type if signal_type else 'general'}: {best_params}")

    return random_search.best_estimator_, best_params


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
        2D feature importance (window_size, n_neurons),
        temporal importance, and neuron importance
    """
    # Extract coefficients from the first layer as a proxy for feature importance
    if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
        # Use absolute values of first layer weights as rough feature importance
        importance = np.abs(model.coefs_[0]).mean(axis=1)
    else:
        logger.warning("Model does not have coefs_ attribute or first layer")
        importance = np.ones(window_size * n_neurons)

    # Reshape to 2D (window_size, n_neurons)
    importance_2d = importance.reshape(window_size, n_neurons)

    # Calculate temporal importance (mean across neurons)
    temporal_importance = np.mean(importance_2d, axis=1)

    # Calculate neuron importance (mean across time)
    neuron_importance = np.mean(importance_2d, axis=0)

    return importance_2d, temporal_importance, neuron_importance

