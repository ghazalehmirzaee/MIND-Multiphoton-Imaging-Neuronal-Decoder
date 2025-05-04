import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple, List, Optional
import logging
from scipy.stats import uniform

logger = logging.getLogger(__name__)


def create_mlp(config: Dict[str, Any]) -> MLPClassifier:
    """
    Create a Multilayer Perceptron classifier with the specified configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    MLPClassifier
        Initialized MLP classifier
    """
    mlp_params = config['models']['classical']['mlp']

    model = MLPClassifier(
        hidden_layer_sizes=mlp_params.get('hidden_layer_sizes', (128, 64)),
        activation=mlp_params.get('activation', 'relu'),
        solver=mlp_params.get('solver', 'adam'),
        alpha=mlp_params.get('alpha', 0.0001),
        learning_rate=mlp_params.get('learning_rate', 'adaptive'),
        early_stopping=mlp_params.get('early_stopping', True),
        random_state=config['experiment'].get('seed', 42),
        max_iter=1000
    )

    return model


def optimize_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any]
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Optimize MLP hyperparameters using randomized search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Tuple[MLPClassifier, Dict[str, Any]]
        Optimized MLP classifier and best parameters
    """
    logger.info("Optimizing MLP hyperparameters")

    # Define parameter distribution for randomized search
    param_dist = {
        'hidden_layer_sizes': [(64,), (128,), (256,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(0.0001, 0.01),
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'learning_rate_init': uniform(0.0005, 0.005),
        'max_iter': [500, 1000],
        'early_stopping': [True],
        'n_iter_no_change': [10, 20],
        'solver': ['adam', 'sgd']
    }

    # Initialize MLP with early stopping
    mlp = MLPClassifier(
        random_state=config['experiment'].get('seed', 42),
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    )

    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42)
    )

    # Fit model
    logger.info("Performing randomized search for MLP")
    random_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters from randomized search: {best_params}")

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

