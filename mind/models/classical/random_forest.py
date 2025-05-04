import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_random_forest(config: Dict[str, Any]) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with the specified configuration.
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
    Optimize Random Forest hyperparameters using randomized search with efficient parameters.
    """
    logger.info("Optimizing Random Forest hyperparameters")

    # Define efficient parameter distribution for randomized search
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # If class_weights provided, include them
    if class_weights:
        param_dist['class_weight'] = ['balanced', class_weights]
    else:
        param_dist['class_weight'] = ['balanced']

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=config['experiment'].get('seed', 42), n_jobs=-1)

    # Randomized search with cross-validation using efficient parameters
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15,  # Reduced number of parameter settings
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42),
        verbose=0
    )

    # Fit model
    logger.info("Performing randomized search for Random Forest")
    random_search.fit(X_train, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters: {best_params}")

    return random_search.best_estimator_, best_params


def extract_feature_importance(
        model: RandomForestClassifier,
        window_size: int,
        n_neurons: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature importance from a trained Random Forest model.
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

