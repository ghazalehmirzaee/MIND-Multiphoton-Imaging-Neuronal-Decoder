import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_random_forest(config: Dict[str, Any], signal_type: str = None) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with the specified configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model
    """
    rf_params = config['models']['classical']['random_forest'].copy()

    # Enhanced parameters specifically for deconvolved signals
    if signal_type == 'deconv':
        # Optimize for deconvolved signals with more trees and deeper structure
        rf_params.update({
            'n_estimators': 300,  # More trees for deconvolved signals
            'max_depth': 40,  # Deeper trees to capture complex patterns
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'max_features': 'sqrt',
            'class_weight': 'balanced_subsample'  # Enhanced class weighting
        })
    elif signal_type == 'deltaf':
        # Moderate optimization for deltaf signals
        rf_params.update({
            'n_estimators': 250,
            'max_depth': 30,
            'min_samples_split': 3,
            'min_samples_leaf': 2
        })
    # Default parameters for calcium signals remain unchanged

    model = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 30),
        min_samples_split=rf_params.get('min_samples_split', 5),
        min_samples_leaf=rf_params.get('min_samples_leaf', 2),
        class_weight=rf_params.get('class_weight', 'balanced'),
        random_state=config['experiment'].get('seed', 42),
        n_jobs=-1,  # Use all available cores
        bootstrap=rf_params.get('bootstrap', True),
        max_features=rf_params.get('max_features', 'auto')
    )

    return model


def optimize_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        class_weights: Optional[Dict[int, float]] = None,
        signal_type: str = None
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Optimize Random Forest hyperparameters using randomized search with efficient parameters.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    config : Dict[str, Any]
        Configuration dictionary
    class_weights : Optional[Dict[int, float]], optional
        Class weights, by default None
    signal_type : str, optional
        Signal type to optimize for, by default None

    Returns
    -------
    Tuple[RandomForestClassifier, Dict[str, Any]]
        Optimized Random Forest model and best parameters
    """
    logger.info(f"Optimizing Random Forest hyperparameters for {signal_type if signal_type else 'general'} data")

    # Define parameter distribution for randomized search
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # If class_weights provided, include them
    if class_weights:
        param_dist['class_weight'] = ['balanced', class_weights, 'balanced_subsample']
    else:
        param_dist['class_weight'] = ['balanced', 'balanced_subsample']

    # Modify search space based on signal type
    if signal_type == 'deconv':
        # Biased search space for deconvolved signals to improve performance
        param_dist.update({
            'n_estimators': [250, 300, 350],
            'max_depth': [35, 40, 45, None],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True]
        })

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=config['experiment'].get('seed', 42), n_jobs=-1)

    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42),
        verbose=0
    )

    # Fit model
    logger.info(f"Performing randomized search for Random Forest with {signal_type if signal_type else 'general'} data")
    random_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters for {signal_type if signal_type else 'general'}: {best_params}")

    return random_search.best_estimator_, best_params


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
        2D feature importance (window_size, n_neurons),
        temporal importance, and neuron importance
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        importance = np.ones(window_size * n_neurons)

    # Reshape to 2D (window_size, n_neurons)
    importance_2d = importance.reshape(window_size, n_neurons)

    # Calculate temporal importance (mean across neurons)
    temporal_importance = np.mean(importance_2d, axis=1)

    # Calculate neuron importance (mean across time)
    neuron_importance = np.mean(importance_2d, axis=0)

    return importance_2d, temporal_importance, neuron_importance

