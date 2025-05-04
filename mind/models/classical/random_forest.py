import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from typing import Dict, Any, Tuple, List, Optional
import logging
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)


def create_random_forest(config: Dict[str, Any]) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with the specified configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    RandomForestClassifier
        Initialized Random Forest classifier
    """
    rf_params = config['models']['classical']['random_forest']

    model = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 30),
        min_samples_split=rf_params.get('min_samples_split', 5),
        min_samples_leaf=rf_params.get('min_samples_leaf', 2),
        class_weight=rf_params.get('class_weight', 'balanced'),
        random_state=config['experiment'].get('seed', 42),
        n_jobs=-1  # Use all available cores
    )

    return model


def optimize_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        class_weights: Optional[Dict[int, float]] = None
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Optimize Random Forest hyperparameters using randomized and grid search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    config : Dict[str, Any]
        Configuration dictionary
    class_weights : Optional[Dict[int, float]], optional
        Class weights, by default None

    Returns
    -------
    Tuple[RandomForestClassifier, Dict[str, Any]]
        Optimized Random Forest classifier and best parameters
    """
    logger.info("Optimizing Random Forest hyperparameters")

    # Define parameter distribution for randomized search
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    # If class_weights provided, include them in the search
    if class_weights:
        param_dist['class_weight'] = ['balanced', 'balanced_subsample', None, class_weights]

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=config['experiment'].get('seed', 42), n_jobs=-1)

    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,  # Number of parameter settings sampled
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42)
    )

    # Fit model
    logger.info("Performing randomized search for Random Forest")
    random_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters from randomized search: {best_params}")

    # Refine with GridSearchCV around best params
    refined_param_grid = {
        'n_estimators': [max(100, best_params['n_estimators'] - 50),
                         best_params['n_estimators'],
                         best_params['n_estimators'] + 50],
        'max_depth': [best_params['max_depth']],
        'min_samples_split': [max(2, best_params['min_samples_split'] - 1),
                              best_params['min_samples_split'],
                              best_params['min_samples_split'] + 1],
        'min_samples_leaf': [max(1, best_params['min_samples_leaf'] - 1),
                             best_params['min_samples_leaf'],
                             best_params['min_samples_leaf'] + 1]
    }

    # Grid search with cross-validation for fine-tuning
    grid_search = GridSearchCV(
        estimator=random_search.best_estimator_,
        param_grid=refined_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_weighted'
    )

    # Fit model
    logger.info("Fine-tuning RF with grid search")
    grid_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = grid_search.best_params_
    logger.info(f"Best parameters from grid search: {best_params}")

    return grid_search.best_estimator_, best_params


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
