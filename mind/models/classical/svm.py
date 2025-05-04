import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_svm(config: Dict[str, Any]) -> SVC:
    """
    Create a Support Vector Machine classifier with the specified configuration.
    """
    svm_params = config['models']['classical']['svm']

    model = SVC(
        kernel=svm_params.get('kernel', 'rbf'),
        C=svm_params.get('C', 1.0),
        gamma=svm_params.get('gamma', 'scale'),
        class_weight=svm_params.get('class_weight', 'balanced'),
        probability=svm_params.get('probability', True),
        random_state=config['experiment'].get('seed', 42)
    )

    return model


def optimize_svm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Dict[str, Any],
        class_weights: Optional[Dict[int, float]] = None
) -> Tuple[SVC, Dict[str, Any], Optional[PCA]]:
    """
    Optimize SVM hyperparameters using randomized search, with efficient PCA.
    """
    logger.info("Optimizing SVM hyperparameters")

    # Check if PCA should be applied
    svm_params = config['models']['classical']['svm']
    use_pca = svm_params.get('pca', True)
    pca_transformer = None

    # Apply PCA if requested
    if use_pca:
        n_features = X_train.shape[1]

        # Determine optimal number of components efficiently
        pca_components = svm_params.get('pca_components', 0.95)
        if isinstance(pca_components, float) and pca_components <= 1.0:
            # Use explained variance ratio
            n_components = pca_components
        else:
            # Use specific number of components (capped at 100 for efficiency)
            n_components = min(n_features, int(pca_components), 100)

        logger.info(f"Applying PCA to reduce dimensions from {n_features} to {n_components}")
        pca_transformer = PCA(n_components=n_components, random_state=config['experiment'].get('seed', 42))
        X_train_pca = pca_transformer.fit_transform(X_train)

        # Log explained variance
        explained_variance = np.sum(pca_transformer.explained_variance_ratio_)
        logger.info(f"PCA explained variance ratio: {explained_variance:.4f}")

        # Use transformed data for hyperparameter optimization
        X_train_opt = X_train_pca
    else:
        X_train_opt = X_train

    # Define efficient parameter distribution for randomized search
    param_dist = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear'],
        'probability': [True]
    }

    # If class_weights provided, include them
    if class_weights:
        param_dist['class_weight'] = ['balanced', class_weights]
    else:
        param_dist['class_weight'] = ['balanced']

    # Initialize SVM
    svm = SVC(random_state=config['experiment'].get('seed', 42))

    # Randomized search with cross-validation using efficient parameters
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=10,  # Reduced number of parameter settings
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=config['experiment'].get('seed', 42),
        verbose=0
    )

    # Fit model
    logger.info("Performing randomized search for SVM")
    random_search.fit(X_train_opt, y_train.astype(int))

    # Get best parameters
    best_params = random_search.best_params_
    logger.info(f"Best parameters: {best_params}")

    return random_search.best_estimator_, best_params, pca_transformer

